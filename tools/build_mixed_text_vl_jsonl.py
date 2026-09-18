#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""
Build a mixed text / vision-language JSONL from ShareGPT and/or LLaVA Instruct JSON.

Supported VL source formats
---------------------------
1) LLaVA Instruct JSON (e.g. llava_v1_5_mix665k.json, with absolute image paths):

  [
    {
      "id": "000000033471",
      "image": "/abs/path/to/coco/train2017/000000033471.jpg",
      "conversations": [
        {"from": "human", "value": "<image>\\nWhat are the colors of the bus..."},
        {"from": "gpt", "value": "The bus in the image is white and red."}
      ]
    },
    ...
  ]

  ``image`` is used as-is when absolute. Relative paths can still be resolved with
  ``--image-root`` / ``--image-root prefix=/abs`` if needed.

  Samples without an ``image`` field are treated as text-only.

2) Legacy LLaVA-NeXT parquet (embedded image bytes) via ``--llava-parquet``.

Output format (one JSON object per line)
----------------------------------------
  # text
  {"id": "0", "conversations": [
      {"role": "user", "content": [{"type": "text", "text": "..."}]},
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]}

  # VL
  {"id": "1", "conversations": [
      {"role": "user", "content": [
          {"type": "image", "image": "/abs/path/to/coco/train2017/000000033471.jpg"},
          {"type": "text", "text": "..."}
      ]},
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]}

Examples
--------
  Filters:
    - assistant answers shorter than ``--min-answer-words`` (default 5)
    - ``<image>`` tokens without a matching image path (or more tokens than images)
    - multi-image rows, leftover ``<image>`` in text, broken role order, empty turns
    - missing local image files (unless ``--allow-missing-images``)

  # ShareGPT (text) + LLaVA Instruct (VL, absolute image paths) -> JSONL
  # Filters answers shorter than 5 words; randomly samples the requested counts.
  python dataset/build_mixed_text_vl_jsonl.py \\
    --sharegpt dataset/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \\
    --llava-json /path/to/llava_v1_5_mix665k.json \\
    --num-text 18 --num-vl 18 \\
    --min-answer-words 5 \\
    --seed 42 \\
    --output-dir dataset/mixed_text_vl_36 \\
    --output-name mixed_text_vl_36.jsonl

  # Optional: relative image paths with per-dataset absolute roots
  python dataset/build_mixed_text_vl_jsonl.py \\
    --sharegpt dataset/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \\
    --llava-json /path/to/llava_v1_5_mix665k.json \\
    --image-root coco=/data/coco \\
    --image-root gqa=/data/gqa \\
    --num-text 18 --num-vl 18 \\
    --output-dir dataset/mixed_text_vl_36 \\
    --output-name mixed_text_vl_36.jsonl

  # Legacy parquet VL source
  python dataset/build_mixed_text_vl_jsonl.py \\
    --sharegpt dataset/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \\
    --llava-parquet /data1/datasets/lmms-lab/LLaVA-NeXT-Data/data/train-00000-of-00250.parquet \\
    --num-text 18 --num-vl 18 \\
    --output-dir dataset/mixed_text_vl_36 \\
    --output-name mixed_text_vl_36.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
from PIL import Image

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "user": "user",
    "assistant": "assistant",
}


class ImageRootMap:
    """Resolve LLaVA ``image`` fields to absolute paths.

    - Absolute ``image`` paths are returned as-is.
    - ``prefix=/abs/root`` maps ``prefix/rest...`` -> ``/abs/root/rest...``
      (first path component stripped).
    - A plain default root maps ``rel`` -> ``default_root / rel``.
    """

    def __init__(
        self,
        prefix_roots: Optional[dict[str, Path]] = None,
        default_root: Optional[Path] = None,
    ):
        self.prefix_roots = {k: Path(v) for k, v in (prefix_roots or {}).items()}
        self.default_root = Path(default_root) if default_root is not None else None

    def resolve(self, image: str) -> Optional[Path]:
        p = Path(image)
        if p.is_absolute():
            return p.resolve()

        parts = p.parts
        if not parts:
            return None

        prefix = parts[0]
        if prefix in self.prefix_roots:
            root = self.prefix_roots[prefix]
            rest = Path(*parts[1:]) if len(parts) > 1 else Path()
            return (root / rest).resolve()

        if self.default_root is not None:
            return (self.default_root / p).resolve()
        return None

    def describe(self) -> str:
        bits = [f"{k}={v}" for k, v in sorted(self.prefix_roots.items())]
        if self.default_root is not None:
            bits.append(f"default={self.default_root}")
        return ", ".join(bits) if bits else "(none)"


def parse_image_root_args(values: list[str] | None) -> ImageRootMap:
    """Parse repeated ``--image-root`` values: ``/abs`` or ``prefix=/abs``."""
    prefix_roots: dict[str, Path] = {}
    default_root: Path | None = None
    for raw in values or []:
        if "=" in raw:
            key, val = raw.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key or not val:
                raise ValueError(f"Invalid --image-root '{raw}', expected prefix=/abs/path")
            prefix_roots[key] = Path(val)
        else:
            default_root = Path(raw)
    return ImageRootMap(prefix_roots=prefix_roots, default_root=default_root)


IMAGE_TOKEN = "<image>"


def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def strip_image_token(text: str) -> str:
    return text.replace(IMAGE_TOKEN, "").strip().lstrip("\n").strip()


def count_image_tokens(text: str) -> int:
    return str(text).count(IMAGE_TOKEN)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def count_images_in_field(image_field) -> int:
    if not image_field:
        return 0
    if isinstance(image_field, (list, tuple)):
        return len(image_field)
    return 1


def raw_conversation_image_tokens(convs_raw: list) -> int:
    n = 0
    for turn in convs_raw or []:
        if not isinstance(turn, dict):
            continue
        val = turn.get("value") or turn.get("content") or ""
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        n += 1
                    elif item.get("type") == "text":
                        n += count_image_tokens(item.get("text") or "")
                else:
                    n += count_image_tokens(str(item))
        else:
            n += count_image_tokens(str(val))
    return n


def validate_converted_conversation(convs: list[dict]) -> str | None:
    """
    Return a short rejection reason, or None if the conversation is usable.

    Catches format issues that break VLM processors, e.g. leftover ``<image>``
    tokens in text with no / too few attached images.
    """
    if not convs:
        return "empty_conversation"

    expected = "user"
    n_images = 0
    for turn in convs:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            return f"bad_role:{role}"
        if role != expected:
            return f"role_order_expected_{expected}_got_{role}"
        expected = "assistant" if expected == "user" else "user"

        content = turn.get("content")
        if not isinstance(content, list) or not content:
            return "empty_or_nonlist_content"

        turn_text_parts: list[str] = []
        turn_images = 0
        for item in content:
            if not isinstance(item, dict):
                return "non_dict_content_item"
            itype = item.get("type")
            if itype == "image":
                turn_images += 1
                n_images += 1
                if not item.get("image"):
                    return "empty_image_path"
            elif itype == "text":
                text = item.get("text")
                if text is None:
                    return "null_text"
                text = str(text)
                if IMAGE_TOKEN in text:
                    return "leftover_image_token_in_text"
                turn_text_parts.append(text)
            else:
                return f"unsupported_content_type:{itype}"

        text_joined = "".join(turn_text_parts).strip()
        if role == "assistant" and not text_joined:
            return "empty_assistant_text"
        if role == "user" and turn_images == 0 and not text_joined:
            return "empty_user_turn"

    if expected == "assistant":
        # ended on user without an assistant reply
        return "missing_assistant_reply"
    if n_images > 1:
        # Current mixed builder / SmolVLM smoke path expects a single image.
        return f"too_many_images:{n_images}"
    return None


def answers_meet_min_words(convs: list[dict], min_words: int) -> bool:
    """True iff every assistant reply has at least ``min_words`` words."""
    if min_words <= 0:
        return True
    found = False
    for turn in convs:
        if turn.get("role") != "assistant":
            continue
        found = True
        if word_count(content_text(turn.get("content"))) < min_words:
            return False
    return found


def random_sample(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0:
        return []
    if len(records) <= n:
        return list(records)
    return rng.sample(records, n)


def normalize_sharegpt_turns(convs: list) -> list | None:
    """Convert ShareGPT / LLaVA turns to role/content; drop leading assistant turns."""
    out: list[dict] = []
    for turn in convs:
        role = ROLE_MAP.get(turn.get("from") or turn.get("role"))
        if role is None:
            continue
        val = turn.get("value") or turn.get("content") or ""
        if isinstance(val, list):
            # already multimodal content list — keep text parts only here
            texts = [
                str(x.get("text", ""))
                for x in val
                if isinstance(x, dict) and x.get("type") == "text"
            ]
            # Reject multimodal ShareGPT rows that embed images in content lists
            # without going through the LLaVA image-field path.
            if any(isinstance(x, dict) and x.get("type") == "image" for x in val):
                return None
            val = "".join(texts)
        raw = str(val)
        # Text-only samples must not contain image placeholders.
        if count_image_tokens(raw) > 0:
            return None
        if not raw.strip():
            continue
        out.append({"role": role, "content": [text_content(raw)]})

    while out and out[0]["role"] != "user":
        out.pop(0)
    if not out or out[0]["role"] != "user":
        return None
    if not any(t["role"] == "assistant" for t in out):
        return None
    if validate_converted_conversation(out) is not None:
        return None
    return out


def normalize_llava_instruct_item(
    item: dict,
    image_roots: ImageRootMap,
    *,
    require_existing_image: bool = True,
) -> tuple[str, list[dict]] | None:
    """
    Convert one LLaVA Instruct record to (kind, conversations).

    kind is ``"vl"`` or ``"text"``.
    For VL, attaches ``{"type": "image", "image": <abs_path>}`` on the first user turn.
    """
    convs_raw = item.get("conversations") or []
    image_field = item.get("image")
    n_img_field = count_images_in_field(image_field)
    n_img_tokens = raw_conversation_image_tokens(convs_raw)

    # Text-only ShareGPT-style entries in mix665k have no ``image`` field.
    if n_img_field == 0:
        # Image tokens without an image path → processor error later.
        if n_img_tokens > 0:
            return None
        convs = normalize_sharegpt_turns(convs_raw)
        if not convs:
            return None
        return "text", convs

    # Multi-image rows are not supported by this mixed builder.
    if n_img_field != 1:
        return None
    # More placeholders than available images (e.g. 2x <image>, 1 path).
    if n_img_tokens > n_img_field:
        return None

    if isinstance(image_field, (list, tuple)):
        image_field = image_field[0]

    abs_img = image_roots.resolve(str(image_field))
    if abs_img is None:
        # No root mapping and not absolute — cannot resolve.
        if require_existing_image:
            return None
        abs_img = Path(str(image_field))
    if require_existing_image and not abs_img.is_file():
        return None

    out_convs: list[dict] = []
    first_user = True
    for turn in convs_raw:
        if not isinstance(turn, dict):
            continue
        role = ROLE_MAP.get(turn.get("from") or turn.get("role"))
        if role is None:
            continue
        text = strip_image_token(str(turn.get("value") or ""))
        if role == "user" and first_user:
            content = [
                {"type": "image", "image": str(abs_img)},
                text_content(text),
            ]
            out_convs.append({"role": "user", "content": content})
            first_user = False
        else:
            if not text and role == "user":
                continue
            out_convs.append({"role": role, "content": [text_content(text)]})

    while out_convs and out_convs[0]["role"] != "user":
        out_convs.pop(0)
    if validate_converted_conversation(out_convs) is not None:
        return None
    return "vl", out_convs


def load_pil_from_parquet_image(image_obj) -> Image.Image | None:
    if image_obj is None:
        return None
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        if image_obj.get("path"):
            try:
                return Image.open(image_obj["path"]).convert("RGB")
            except OSError:
                return None
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    return None


def sample_sharegpt(
    path: Path,
    n: int,
    *,
    min_answer_words: int = 5,
    rng: random.Random | None = None,
) -> list[dict]:
    if n <= 0:
        return []
    rng = rng or random.Random()
    print(f"Loading ShareGPT from {path} ...")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pool: list[dict] = []
    skipped_short = 0
    skipped_format = 0
    for item in data:
        convs = normalize_sharegpt_turns(item.get("conversations") or [])
        if not convs:
            skipped_format += 1
            continue
        if not answers_meet_min_words(convs, min_answer_words):
            skipped_short += 1
            continue
        pool.append({"conversations": convs})

    records = random_sample(pool, n, rng)
    print(
        f"  pool={len(pool)} eligible "
        f"(skipped {skipped_short} short answers, {skipped_format} format); "
        f"randomly selected {len(records)}"
    )
    return records


def sample_llava_instruct_json(
    path: Path,
    image_roots: ImageRootMap,
    num_text: int,
    num_vl: int,
    *,
    require_existing_image: bool = True,
    min_answer_words: int = 5,
    rng: random.Random | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Sample text / VL records from LLaVA Instruct JSON (llava_v1_5_mix665k style).

    Images are resolved via ``image_roots`` to absolute paths (files are not copied).
    Eligible entries are filtered then randomly sampled.
    """
    rng = rng or random.Random()
    print(f"Loading LLaVA Instruct JSON from {path} ...")
    print(f"  image roots: {image_roots.describe()}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    text_pool: list[dict] = []
    vl_pool: list[dict] = []
    skipped_missing = 0
    skipped_short = 0
    skipped_format = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        has_image = bool(item.get("image"))
        parsed = normalize_llava_instruct_item(
            item,
            image_roots,
            require_existing_image=require_existing_image,
        )
        if parsed is None:
            if has_image and require_existing_image:
                # Could be missing file or format; count missing only when path
                # does not resolve to an existing file.
                image_field = item.get("image")
                if isinstance(image_field, (list, tuple)):
                    image_field = image_field[0] if image_field else None
                resolved = image_roots.resolve(str(image_field)) if image_field else None
                if resolved is not None and not resolved.is_file():
                    skipped_missing += 1
                else:
                    skipped_format += 1
            else:
                skipped_format += 1
            continue
        kind, convs = parsed
        if not answers_meet_min_words(convs, min_answer_words):
            skipped_short += 1
            continue
        if kind == "vl":
            vl_pool.append({"conversations": convs})
        else:
            text_pool.append({"conversations": convs})

    text_recs = random_sample(text_pool, num_text, rng)
    vl_recs = random_sample(vl_pool, num_vl, rng)
    print(
        f"  pools: text={len(text_pool)} vl={len(vl_pool)} "
        f"(skipped {skipped_missing} missing images, {skipped_short} short answers, "
        f"{skipped_format} format); "
        f"randomly selected {len(text_recs)} text + {len(vl_recs)} vl"
    )
    return text_recs, vl_recs


def sample_llava_parquet(
    path: Path,
    n: int,
    img_dir: Path,
    *,
    min_answer_words: int = 5,
    rng: random.Random | None = None,
) -> list[dict]:
    """Reservoir-sample ``n`` VL rows after short-answer filtering."""
    rng = rng or random.Random()
    if n <= 0:
        return []
    print(f"Reading LLaVA-NeXT parquet from {path} ...")
    img_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(path)

    reservoir: list[dict] = []
    seen_eligible = 0
    skipped_short = 0

    def _build_convs(convs_raw, abs_img: str) -> list[dict] | None:
        out_convs: list[dict] = []
        first_user = True
        for turn in convs_raw:
            if not isinstance(turn, dict):
                continue
            role = ROLE_MAP.get(turn.get("from") or turn.get("role"))
            if role is None:
                continue
            text = strip_image_token(str(turn.get("value") or ""))
            if role == "user" and first_user:
                out_convs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": abs_img},
                            text_content(text),
                        ],
                    }
                )
                first_user = False
            else:
                if not text and role == "user":
                    continue
                out_convs.append({"role": role, "content": [text_content(text)]})
        while out_convs and out_convs[0]["role"] != "user":
            out_convs.pop(0)
        if not out_convs or out_convs[0]["role"] != "user":
            return None
        if "image" not in [c.get("type") for c in out_convs[0]["content"]]:
            return None
        if not any(t["role"] == "assistant" for t in out_convs):
            return None
        return out_convs

    for batch in pf.iter_batches(batch_size=64):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            img = load_pil_from_parquet_image(row.get("image"))
            if img is None:
                continue
            convs_raw = row["conversations"]
            if hasattr(convs_raw, "tolist"):
                convs_raw = convs_raw.tolist()

            # Probe answer length before writing image.
            probe = _build_convs(convs_raw, abs_img="__probe__")
            if probe is None:
                continue
            if not answers_meet_min_words(probe, min_answer_words):
                skipped_short += 1
                continue

            seen_eligible += 1
            replace_idx = None
            if len(reservoir) < n:
                replace_idx = len(reservoir)
            else:
                j = rng.randrange(seen_eligible)
                if j < n:
                    replace_idx = j
                    old = reservoir[j].get("_img_path")
                    if old is not None:
                        Path(old).unlink(missing_ok=True)
            if replace_idx is None:
                continue

            sid = str(row.get("id", f"llava_{seen_eligible}"))
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sid)
            img_path = img_dir / f"{replace_idx:04d}_{safe}.jpg"
            img.save(img_path, format="JPEG", quality=95)
            abs_img = str(img_path.resolve())
            out_convs = _build_convs(convs_raw, abs_img=abs_img)
            assert out_convs is not None
            entry = {"conversations": out_convs, "_img_path": img_path}
            if replace_idx == len(reservoir):
                reservoir.append(entry)
            else:
                reservoir[replace_idx] = entry

    records = [{"conversations": rec["conversations"]} for rec in reservoir]
    print(
        f"  eligible={seen_eligible} (skipped {skipped_short} short answers); "
        f"randomly selected {len(records)}; images -> {img_dir}"
    )
    return records


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sharegpt",
        type=Path,
        default=None,
        help="Optional ShareGPT JSON list (文生文). If omitted, text may come from --llava-json.",
    )
    p.add_argument(
        "--llava-json",
        type=Path,
        default=None,
        help=(
            "LLaVA Instruct JSON (e.g. llava_v1_5_mix665k.json). "
            "Uses absolute image paths as-is; relative paths need --image-root."
        ),
    )
    p.add_argument(
        "--image-root",
        action="append",
        default=None,
        metavar="ROOT",
        help=(
            "Image root. Repeatable. Use absolute dataset roots as "
            "prefix=/abs/path (e.g. coco=/data/coco) so "
            "coco/train2017/x.jpg -> /data/coco/train2017/x.jpg. "
            "A plain /abs/path is a default parent for relative paths. "
            "If JSON already has absolute image paths, this can be omitted."
        ),
    )
    p.add_argument(
        "--llava-parquet",
        type=Path,
        default=None,
        help="Optional legacy LLaVA-NeXT-Data parquet shard (图生文; copies images out).",
    )
    p.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Keep VL samples even if resolved image file is missing.",
    )
    p.add_argument("--num-text", type=int, default=18, help="Number of text samples to randomly select")
    p.add_argument("--num-vl", type=int, default=18, help="Number of VL samples to randomly select")
    p.add_argument(
        "--min-answer-words",
        type=int,
        default=5,
        help="Drop samples whose any assistant answer has fewer than this many words (0 disables).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random sampling",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "dataset/mixed_text_vl_36",
        help="Output directory (writes JSONL; parquet mode also writes images/)",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="mixed_text_vl_36.jsonl",
        help="JSONL filename inside --output-dir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    img_dir = out_dir / "images"
    out_jsonl = out_dir / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if not args.sharegpt and not args.llava_json and not args.llava_parquet:
        raise SystemExit("Provide at least one of --sharegpt, --llava-json, --llava-parquet")

    try:
        image_roots = parse_image_root_args(args.image_root)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    text_recs: list[dict] = []
    vl_recs: list[dict] = []

    # 1) Prefer dedicated ShareGPT for text if provided.
    if args.sharegpt is not None:
        text_recs.extend(
            sample_sharegpt(
                args.sharegpt,
                args.num_text,
                min_answer_words=args.min_answer_words,
                rng=rng,
            )
        )

    # 2) LLaVA Instruct JSON (text and/or VL from one file).
    if args.llava_json is not None:
        need_text = max(0, args.num_text - len(text_recs))
        need_vl = max(0, args.num_vl - len(vl_recs))
        t, v = sample_llava_instruct_json(
            args.llava_json,
            image_roots,
            num_text=need_text,
            num_vl=need_vl,
            require_existing_image=not args.allow_missing_images,
            min_answer_words=args.min_answer_words,
            rng=rng,
        )
        text_recs.extend(t)
        vl_recs.extend(v)

    # 3) Legacy parquet VL filler.
    if args.llava_parquet is not None and len(vl_recs) < args.num_vl:
        if img_dir.exists():
            for old in img_dir.glob("*"):
                old.unlink()
        img_dir.mkdir(parents=True, exist_ok=True)
        need_vl = args.num_vl - len(vl_recs)
        vl_recs.extend(
            sample_llava_parquet(
                args.llava_parquet,
                need_vl,
                img_dir,
                min_answer_words=args.min_answer_words,
                rng=rng,
            )
        )

    text_recs = text_recs[: args.num_text]
    vl_recs = vl_recs[: args.num_vl]

    records = []
    for i, rec in enumerate(text_recs + vl_recs):
        records.append({"id": str(i), "conversations": rec["conversations"]})

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_vl = sum(
        any(c.get("type") == "image" for t in r["conversations"] for c in t["content"])
        for r in records
    )
    print(
        f"Wrote {len(records)} lines ({len(records) - n_vl} text + {n_vl} vl) "
        f"[min_answer_words={args.min_answer_words}, seed={args.seed}] -> {out_jsonl}"
    )


if __name__ == "__main__":
    main()
