#!/usr/bin/env python
"""Put each sample's source image next to the .ckpt that was generated from it.

    data_<i>.ckpt   the target's full forward, already on disk
    data_<i>.img    the source image, written here

Offline branch distillation re-scores the image along with the text (see
...dataset_builder/branch_image_source.py), and the .ckpt does not contain it. Reading it back out of the source jsonl at
training time means rebuilding that dataset in every dataloader worker, which
measured at 208s per process -- worse the more workers you add. A file beside
the .ckpt is opened in microseconds and needs no index, no mapping arithmetic
and no validation.

What is written is the ORIGINAL image bytes: it is not derived data, it is the input.

Launch it the same way, and with the same arguments, as the run that produced
the .ckpt files -- that is what makes (rank, idx) address the same sample:

    torchrun --nproc_per_node=8 tools/extract_images_beside_hidden.py \
        --modal_type VLM --dataset_path <source jsonl> \
        --outdir <hidden dir> \
        --target_model_name_or_path <local snapshot> \
        --draft_model_config_path <the draft config used for generation> \
        --chat_template_type smolvlm --model_max_length 4096

Existing .img files are skipped, so it is resumable.
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoProcessor  # noqa: E402

from tools.generate_hidden_for_draft_model import (  # noqa: E402
    cleanup_distributed,
    load_dataset,
    setup_distributed,
    split_dataset_for_rank,
)
from tools.generate_hidden_for_draft_model import parse_arguments as base_parse_arguments  # noqa: E402


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--group_size", type=int, default=5000,
                       help="must match the generation run (HiddenStateGenerator's default)")
    extra.add_argument("--overwrite", action="store_true")
    extra.add_argument("--display", action="store_true")
    mine, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = base_parse_arguments()
    for k, v in vars(mine).items():
        setattr(args, k, v)
    return args


def out_path(outdir, rank, idx, group_size, suffix):
    start = (idx // group_size) * group_size
    d = Path(outdir) / f"rank_{rank}" / f"rows_{start}-{start + group_size}"
    return d / f"data_{idx}{suffix}"


def image_bytes(spec):
    """The raw file bytes for one entry of the dataset's `image_paths`.

    Entries are either a path or, in the base64-normalised datasets, the encoded
    file itself (with or without a data: URI prefix).
    """
    if os.path.exists(spec):
        return Path(spec).read_bytes()
    payload = spec.split(",", 1)[-1] if spec.startswith("data:") else spec
    return base64.b64decode(payload)


def main():
    rank, world_size, local_rank = setup_distributed()
    args = parse_args()

    # load_dataset reads target_model_type off the args; the generation run got it
    # from the draft config, so take it from there too rather than a new flag.
    from angelslim.compressor.speculative import DraftModelConfig

    cfg = DraftModelConfig.from_file(args.draft_model_config_path)
    args.target_model_type = getattr(cfg, "target_model_type", None)

    proc = AutoProcessor.from_pretrained(args.target_model_name_or_path)
    dataset = load_dataset(args, proc, rank)
    dataset = split_dataset_for_rank(dataset, rank, world_size, args.start, args.end)

    wrote = skipped = no_image = failed = 0
    it = tqdm(enumerate(dataset), total=len(dataset), desc=f"rank {rank}") if rank == 0 \
        else enumerate(dataset)
    for idx, row in it:
        dst = out_path(args.outdir, rank, idx, args.group_size, ".img")
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            paths = row.get("image_paths")
            if isinstance(paths, str):
                paths = json.loads(paths)
            if not paths or len(paths) != 1:   # no image, or several
                no_image += 1
                continue
            if not dst.parent.exists():        # only a sample with a .ckpt should land here
                failed += 1
                continue
            tmp = dst.with_suffix(".img.tmp")
            tmp.write_bytes(image_bytes(paths[0]))
            os.replace(tmp, dst)               # never leave a half-written file behind
            wrote += 1
        except Exception as e:
            print(f"[rank {rank}] sample {idx}: {type(e).__name__}: {e}", flush=True)
            failed += 1

    print(f"[rank {rank}] wrote={wrote} skipped={skipped} no_image={no_image} failed={failed}",
          flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
