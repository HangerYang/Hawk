# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.image_utils import load_image

from angelslim.utils import rank0_print

__all__ = [
    "process_token_dict_to_mappings",
    "convert_sharegpt_data",
    "convert_ultrachat_data",
    "convert_openai_vl_data",
    "DataCollatorWithPadding",
    "VLMDataCollatorWithPadding",
    "VLMHunyuanDataCollatorWithPadding",
    "VLMSmolVLMDataCollatorWithPadding",
    "AudioDataCollatorWithPadding",
    "CosyVoice3DataCollatorWithPadding",
    "build_image_processor_kwargs",
    "stable_hf_map_cache_files",
]


def stable_hf_map_cache_files(
    datapath: Union[str, List[str]],
    *,
    builder_tag: str,
    max_length: int,
    shuffle: bool,
    shuffle_seed: int,
    sample_num: Optional[int] = None,
    min_loss_tokens: Optional[int] = None,
    cache_version: str = "v1",
) -> Dict[str, str]:
    """Stable on-disk paths for HuggingFace ``datasets.map`` / ``filter`` caches.

    Default HF fingerprints bound methods (tokenizer object id, etc.), so
    ``load_from_cache_file=True`` still remaps every restart. Pinning
    ``cache_file_name`` under ``<data_dir>/.map_cache/`` makes reuse real.

    Key includes data path + size + mtime, builder, max_length, shuffle, etc.
    Delete the ``.map_cache`` dir (or set ``load_from_cache_file=false``) after
    changing preprocessing code.
    """
    paths = datapath if isinstance(datapath, list) else [datapath]
    abs_paths = [os.path.abspath(p) for p in paths]
    meta_parts: List[str] = []
    for p in abs_paths:
        if os.path.isfile(p):
            st = os.stat(p)
            meta_parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
        else:
            meta_parts.append(f"{p}:missing")
    raw = "|".join(
        [
            *meta_parts,
            f"builder={builder_tag}",
            f"ver={cache_version}",
            f"L={max_length}",
            f"sh={int(bool(shuffle))}",
            f"seed={shuffle_seed}",
            f"n={sample_num}",
            f"mlt={min_loss_tokens}",
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    root = os.path.join(os.path.dirname(abs_paths[0]), ".map_cache")
    os.makedirs(root, exist_ok=True)
    stem = f"{builder_tag}_{digest}"
    return {
        "root": root,
        "map": os.path.join(root, f"{stem}_map.arrow"),
        "filter_empty": os.path.join(root, f"{stem}_filter_empty.arrow"),
        "filter_loss": os.path.join(root, f"{stem}_filter_loss.arrow"),
    }


def build_image_processor_kwargs(image_processor, max_pixels=None, min_pixels=None):
    """
    convert max_pixels/min_pixels to the format required by the specific image_processor.
      - Qwen2.5-VL: directly use max_pixels / min_pixels
      - Qwen3-VL:   convert to size={"longest_edge": max_pixels, "shortest_edge": min_pixels}

    Args:
        image_processor: model's image_processor instance
        max_pixels: maximum pixels (total area), None means no limit
        min_pixels: minimum pixels (total area), None means no limit

    Returns:
        dict: can be directly passed to image_processor(...)
    """
    if max_pixels is None and min_pixels is None:
        return {}

    processor_class = type(image_processor).__name__
    # Qwen3-VL uses size={"longest_edge": ..., "shortest_edge": ...}
    if "Qwen3" in processor_class:
        size = {}
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        return {"size": size}
    if "Idefics3" in processor_class or "SmolVLM" in processor_class:
        # Idefics3/SmolVLM has no pixel-area knob at all: it resizes to
        # size["longest_edge"] and THEN tiles into 512px crops, so max_pixels /
        # min_pixels are not just differently spelled here, they are rejected
        # outright. Returning {} leaves the processor at its own default
        # resolution -- which is the resolution the target was trained at and
        # the one the existing baseline checkpoint's visual token count assumes,
        # so a cap here would silently make arms incomparable.
        return {}
    else:
        # Qwen2.5-VL's accept max_pixels and min_pixels
        kwargs = {}
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        return kwargs


def convert_sharegpt_data(row, dataset_column="conversations"):
    converted_messages = []

    role_mapping = {"human": "user", "gpt": "assistant"}
    messages = row[dataset_column]
    for message in messages:
        converted_messages.append(
            {"role": role_mapping[message["from"]], "content": message["value"]}
        )

    return {"conversations": converted_messages, "id": row["id"]}


def convert_ultrachat_data(row, dataset_column="messages"):
    converted_messages = []

    messages = row[dataset_column]
    for message in messages:
        converted_messages.append({"role": message["role"], "content": message["content"]})
    return {"conversations": converted_messages, "id": row["prompt_id"]}


def _pil_file_to_data_url(path: str) -> str:
    """Encode a local image file as an OpenAI-compatible data URL."""
    import base64
    import mimetypes

    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def convert_openai_vl_data(row, dataset_column="conversations"):
    """
    Passthrough for OpenAI-style VLM conversations used by AngelSlim Eagle docs.

    Input content parts may use:
      {"type": "text", "text": "..."}
      {"type": "image", "image": "/abs/path.jpg"}

    Images are converted to OpenAI-style image_url data URLs.
    """
    converted_messages = []
    for message in row[dataset_column]:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            converted_messages.append({"role": role, "content": content})
            continue

        new_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                new_parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image":
                img = part.get("image") or part.get("image_url")
                if isinstance(img, dict):
                    # already {"url": "..."}
                    new_parts.append({"type": "image_url", "image_url": img})
                elif isinstance(img, str) and img.startswith(("http://", "https://", "data:")):
                    new_parts.append({"type": "image_url", "image_url": {"url": img}})
                elif isinstance(img, str):
                    new_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _pil_file_to_data_url(img)},
                        }
                    )
            elif ptype == "image_url":
                new_parts.append(part)
        converted_messages.append({"role": role, "content": new_parts})

    return {"conversations": converted_messages, "id": str(row["id"])}


def process_token_dict_to_mappings(
    token_dict,
    draft_vocab_size: int,
    target_vocab_size: int,
):
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.

    Args:
        token_dict: A Counter object mapping token ids to their frequencies.
        draft_vocab_size: The size of the draft vocabulary.
        target_vocab_size: The size of the target vocabulary.

    Returns:
        A tuple containing:
            - d2t: A tensor mapping draft token ids to target token ids.
            - t2d: A tensor mapping target token ids to draft token ids.
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
    rank0_print(f"Added missing tokens to reach draft vocab size: {draft_vocab_size}")
    rank0_print(f"Total tokens after addition: {len(token_dict)}")
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)

    if total_frequency == 0:
        rank0_print("Warning: Total token frequency is zero. All tokens will have zero ratio.")
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency

    rank0_print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()

    used_set = set(used_tokens)
    d2t = torch.tensor(
        [used_tokens[i] - i for i in range(len(used_tokens))],
        dtype=torch.int64,  # must match register_buffer dtype in Eagle3LlamaForCausalLM
    )
    t2d = torch.tensor(
        [i in used_set for i in range(target_vocab_size)],
        dtype=torch.bool,  # must match register_buffer dtype in Eagle3LlamaForCausalLM
    )

    assert d2t.shape == (draft_vocab_size,), f"d2t shape {d2t.shape} != ({draft_vocab_size},)"
    assert t2d.shape == (target_vocab_size,), f"t2d shape {t2d.shape} != ({target_vocab_size},)"
    assert (
        t2d.sum().item() == draft_vocab_size
    ), f"t2d has {t2d.sum().item()} True entries, expected {draft_vocab_size}"

    return d2t, t2d


def paddingtensor(intensors, N):
    B, n, S = intensors.shape
    padding_tensor = torch.zeros(
        B, N - n, S, dtype=intensors.dtype, device=intensors.device
    )
    return torch.cat((intensors, padding_tensor), dim=1)


def paddingtensor2D(intensors, N):
    B, n = intensors.shape
    padding_tensor = torch.zeros(
        B, N - n, dtype=intensors.dtype, device=intensors.device
    )
    return torch.cat((intensors, padding_tensor), dim=1)


def paddingtensor3D_CBN(tensor_list):
    if all(tensor is None for tensor in tensor_list):
        return None
    N = max(tensor.shape[-1] for tensor in tensor_list if tensor is not None)
    out_tensor_list = []
    for tensor in tensor_list:
        c, b, n = tensor.shape
        outtensor = torch.zeros(c, b, N, dtype=tensor_list[0].dtype)
        if tensor is not None:
            outtensor[:, :, :n] = tensor
        out_tensor_list.append(outtensor)
    return torch.cat(out_tensor_list, dim=1)


def paddingtensor3D_BCN(tensor_list):
    if all(tensor is None for tensor in tensor_list):
        return None
    N = max(tensor.shape[-1] for tensor in tensor_list if tensor is not None)
    out_tensor_list = []
    for tensor in tensor_list:
        b, c, n = tensor.shape
        outtensor = torch.zeros(b, c, N, dtype=tensor_list[0].dtype)
        if tensor is not None:
            outtensor[:, :, :n] = tensor
        out_tensor_list.append(outtensor)
    return torch.cat(out_tensor_list, dim=0)


def paddingtensor3D_BHW(tensor_list):
    if all(tensor is None for tensor in tensor_list):
        return None
    max_h = max(tensor.shape[-2] for tensor in tensor_list if tensor is not None)
    max_w = max(tensor.shape[-1] for tensor in tensor_list if tensor is not None)
    out_tensor_list = []
    for tensor in tensor_list:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        b, h, w = tensor.shape
        outtensor = torch.zeros(b, max_h, max_w, dtype=tensor.dtype)
        if tensor is not None:
            outtensor[:, :h, :w] = tensor
        out_tensor_list.append(outtensor)
    return torch.cat(out_tensor_list)


class DataCollatorWithPadding:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target_hiddens": None,
        }

        # Handle hidden_states and target_hiddens independently
        if all("hidden_states" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )

        if all("target_hiddens" in item for item in features):
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        return batch


# ---------------------------------------------------------------------------
# Draft-side visual-token row reduction (offline VLM training).
#
# The target's prefill ran at FULL image resolution -- it has to, or the thing
# the draft is trying to predict changes underneath it -- and `target_hiddens`
# in the .ckpt is that full-image forward. What this reduces is only what the
# DRAFT sees: the aux hidden streams and the ids/positions that go with them.
# So it is a deterministic function of the stored ckpt plus the image mask,
# which is why it lives here in the data path and costs no extra generation
# and no extra disk.
#
# Positions are kept at their ORIGINAL ABSOLUTE values by default rather than
# renumbered contiguously. Two reasons. (1) `target_hiddens` was computed by
# the target at those absolute positions; renumbering would ask the draft at
# position p to predict a target hidden computed at p+delta, folding a rope
# shift into what is supposed to be an image-information ablation. (2) With
# positions held fixed, a reduced run differs from the full-image run in
# exactly one thing -- how coarse the image rows are -- so the comparison is
# a clean dilution test rather than a dilution+context-length test.
# LlamaRotaryEmbedding already returns its table uncut so that callers may
# index it by gapped absolute positions (see llama_eagle3.py).
# ---------------------------------------------------------------------------


def _image_row_segments(mask, factor, how):
    """Map each original row to an output group id (-1 = dropped).

    Segments are built per contiguous RUN of image rows, not over the image
    block as a whole: SmolVLM interleaves ``<row_i_col_j>`` separators between
    its tiles, so one run is one tile, and reducing runs independently leaves
    the tile structure and the separators exactly where they were.
    """
    n = int(mask.numel())
    dev = mask.device
    seg = torch.full((n,), -1, dtype=torch.long, device=dev)
    if n == 0:
        return seg, 0
    # Run boundaries in one kernel, and the group ids of each run in one more.
    # This used to scan the sequence element by element in Python, which is
    # O(n) tensor reads -- fine on a stored .ckpt, but one device sync per row
    # once the same code reduces the drafter's live CUDA sequence at inference.
    m = mask.reshape(-1).to(torch.bool)
    chg = torch.ones(n, dtype=torch.bool, device=dev)
    chg[1:] = m[1:] != m[:-1]
    starts = chg.nonzero().flatten()
    bounds = starts.tolist() + [n]
    is_img = m[starts].tolist()
    g = 0
    for i, j, img in zip(bounds[:-1], bounds[1:], is_img):
        if not img:
            seg[i:j] = torch.arange(g, g + (j - i), device=dev)
            g += j - i
        else:
            length = j - i
            k = max(1, length // factor)
            if how == "pool":
                # Same linspace cuts as before; `p` lands in group t exactly
                # when cut[t] <= p < cut[t+1], which is what searchsorted gives.
                cut = torch.linspace(0, length, k + 1, device=dev).round().long()
                pos = torch.arange(length, device=dev)
                seg[i:j] = g + torch.searchsorted(
                    cut[1:].contiguous(), pos, right=True).clamp_(max=k - 1)
            else:
                sel = torch.linspace(0, length - 1, k, device=dev).round().long()
                seg[i + sel] = torch.arange(g, g + k, device=dev)
            g += k
    return seg, g


def reduce_vlm_image_rows(item, image_token_id, factor, how="pool", keep_positions=True):
    """Shrink the DRAFT's image rows in one offline .ckpt feature by `factor`.

    Reduces `hidden_states` (the aux streams the draft consumes),
    `target_hiddens`, `input_ids`, `loss_mask` and `attention_mask` together --
    they are all length-N and a partial reduction is a silent misalignment --
    and emits an explicit `position_ids`.

    ``pool`` averages contiguous groups of image rows (nothing is discarded,
    every region still contributes); ``subset`` keeps evenly spaced rows and
    drops the rest; ``drop`` removes every image row, leaving the drafter the
    text alone. Every image row carries the same token id, so the reduced
    rows keep that id and `input_ids` stays well-formed with no invented tokens.
    """
    ids = item["input_ids"]
    mask = ids[0] == image_token_id
    if not bool(mask.any()):
        return item

    n = ids.shape[1]
    if how == "drop":
        # No image at all: the rows are removed, not summarised. The text rows
        # keep the absolute positions they were computed at, so the drafter
        # sees the same gap an arm with a reduced image would leave -- the only
        # difference is that nothing occupies it.
        keep = ~mask
        kept = int(keep.sum())
        out = dict(item)
        for key in ("hidden_states", "target_hiddens", "inputs_embeds"):
            if key in item and item[key] is not None:
                out[key] = item[key][:, keep]
        out["input_ids"] = ids[:, keep]
        if item.get("loss_mask") is not None:
            out["loss_mask"] = item["loss_mask"][:, keep]
        out["attention_mask"] = torch.ones(1, kept, dtype=ids.dtype, device=ids.device)
        idx = torch.arange(n, device=ids.device)[keep]
        out["position_ids"] = (
            idx[None].clone() if keep_positions
            else torch.arange(kept, device=ids.device)[None]
        )
        return out

    dev = ids.device
    seg, groups = _image_row_segments(mask, factor, how)
    keep = seg >= 0
    seg_keep = seg[keep]
    src_idx = torch.arange(n, device=dev)[keep]

    # First original row of each group: the id/loss_mask/position representative.
    first = torch.full((groups,), n, dtype=torch.long, device=dev)
    first.scatter_reduce_(0, seg_keep, src_idx, reduce="amin", include_self=True)

    counts = torch.zeros(groups, device=dev).index_add_(
        0, seg_keep, torch.ones(seg_keep.numel(), device=dev))

    def seg_mean(x):  # (1, N, D) -> (1, G, D)
        out = torch.zeros(groups, x.shape[-1], dtype=torch.float32, device=x.device)
        out.index_add_(0, seg_keep.to(x.device), x[0][keep.to(x.device)].float())
        return (out / counts.to(x.device)[:, None]).to(x.dtype)[None]

    out = dict(item)
    # inputs_embeds is stored too and is length-N like the rest. Every image
    # row carries the same token id, so its embedding rows are identical and
    # pooling them is an identity on the values -- but the length has to shrink
    # with everything else or the collator pads to a negative width.
    for key in ("hidden_states", "target_hiddens", "inputs_embeds", "target_logits"):
        if key in item and item[key] is not None:
            out[key] = seg_mean(item[key])
    out["input_ids"] = ids[:, first]
    if "loss_mask" in item and item["loss_mask"] is not None:
        # Image rows are prompt, never supervised, so a group's loss_mask is
        # constant and taking the representative row is exact. amax rather
        # than the representative anyway, so a violation of that would show up
        # as a loud shape/loss change instead of silently dropped supervision.
        lm = item["loss_mask"]
        out["loss_mask"] = torch.zeros(
            1, groups, dtype=lm.dtype, device=lm.device
        ).scatter_reduce_(
            1, seg_keep.to(lm.device)[None], lm[0][keep.to(lm.device)][None],
            reduce="amax", include_self=True
        )
    out["attention_mask"] = torch.ones(1, groups, dtype=ids.dtype, device=dev)
    out["position_ids"] = (
        first[None].clone() if keep_positions
        else torch.arange(groups, device=dev)[None]
    )
    return out


class VLMDataCollatorWithPadding:

    def __init__(self, processor=None, image_processor_kwargs=None):
        """
        Args:
            processor: VLM processor (e.g. AutoProcessor for qwen3_vl).
                       When provided, image_paths in features will be decoded
                       on-the-fly to pixel_values (used in online training).
            image_processor_kwargs: Additional kwargs passed to image_processor,
                       e.g. {"max_pixels": 1003520, "min_pixels": 200704}.
        """
        self.processor = processor
        # Draft-side image-row reduction (see reduce_vlm_image_rows). Off by
        # default; an arm turns it on through the environment so that every
        # arm reads the SAME generated .ckpt files.
        self.img_reduce = os.environ.get("VISTOKEN_REDUCE", "none").lower()
        self.img_factor = int(os.environ.get("VISTOKEN_FACTOR", "4"))
        self.img_keeppos = os.environ.get("VISTOKEN_KEEPPOS", "1") != "0"
        _tok = os.environ.get("VISTOKEN_IMAGE_TOKEN_ID", "")
        self.img_token_id = int(_tok) if _tok else None
        if self.img_reduce not in ("none", "pool", "subset", "drop"):
            raise ValueError(
                f"VISTOKEN_REDUCE must be none|pool|subset|drop, got {self.img_reduce}")
        if self.img_reduce != "none":
            if self.img_token_id is None:
                raise ValueError("VISTOKEN_REDUCE is set but VISTOKEN_IMAGE_TOKEN_ID is not")
            rank0_print(
                f"[vistoken] draft image rows "
                f"{'removed' if self.img_reduce == 'drop' else '/' + str(self.img_factor)}"
                f" ({self.img_reduce}), "
                f"positions={'absolute' if self.img_keeppos else 'renumbered'}, "
                f"image_token_id={self.img_token_id}"
            )
        if image_processor_kwargs is None:
            image_processor_kwargs = {}
        max_pixels = image_processor_kwargs.get("max_pixels", None)
        min_pixels = image_processor_kwargs.get("min_pixels", "1024")
        if (
            processor is not None
            and (max_pixels is not None or min_pixels is not None)
            and hasattr(processor, "image_processor")
        ):
            self._resolved_image_processor_kwargs = build_image_processor_kwargs(
                processor.image_processor, max_pixels, min_pixels
            )
        else:
            self._resolved_image_processor_kwargs = {}
        rank0_print(f"_resolved_image_processor_kwargs: {self._resolved_image_processor_kwargs}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.img_reduce != "none":
            # Offline .ckpt features already have teacher hidden_states, so
            # pooling here only shrinks the DRAFT sequence. Online features
            # have input_ids/images only -- pooling those would also shrink
            # the teacher's prefill. Online pooling happens after the target
            # forward instead (see OnlineVLMEagle3Trainer).
            features = [
                reduce_vlm_image_rows(
                    item,
                    self.img_token_id,
                    self.img_factor,
                    how=self.img_reduce,
                    keep_positions=self.img_keeppos,
                )
                if "hidden_states" in item
                else item
                for item in features
            ]
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target_hiddens": None,
            "inputs_embeds": None,
            "position_ids": None,
        }

        # Online training: decode image_paths -> pixel_values on-the-fly

        if self.processor is not None and "image_paths" in features[0]:
            all_pixel_values, all_image_grid_thw = [], []
            all_pixel_values_videos, all_video_grid_thw = [], []
            for item in features:
                image_paths = json.loads(item["image_paths"])
                if image_paths:
                    images = [load_image(p) for p in image_paths]
                    if hasattr(self.processor, "image_processor"):
                        vision_enc = self.processor.image_processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )
                    else:
                        vision_enc = self.processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )
                    all_pixel_values.append(vision_enc["pixel_values"])
                    if "image_grid_thw" in vision_enc:
                        all_image_grid_thw.append(vision_enc["image_grid_thw"])
                    if "pixel_values_videos" in vision_enc:
                        all_pixel_values_videos.append(vision_enc["pixel_values_videos"])
                    if "video_grid_thw" in vision_enc:
                        all_video_grid_thw.append(vision_enc["video_grid_thw"])
            if all_pixel_values:
                batch["pixel_values"] = paddingtensor3D_BHW(all_pixel_values)
            if all_image_grid_thw:
                batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
            if all_pixel_values_videos:
                batch["pixel_values_videos"] = paddingtensor3D_BHW(all_pixel_values_videos)
            if all_video_grid_thw:
                batch["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)
        else:
            if "pixel_values" in features[0]:
                batch["pixel_values"] = paddingtensor3D_BHW(
                    [item["pixel_values"] for item in features]
                )
            if "pixel_values_videos" in features[0]:
                batch["pixel_values_videos"] = paddingtensor3D_BHW(
                    [item["pixel_values_videos"] for item in features]
                )
            if all(
                "image_grid_thw" in item and item["image_grid_thw"] is not None
                for item in features
            ):
                batch["image_grid_thw"] = torch.cat(
                    [item["image_grid_thw"] for item in features], dim=0
                )
            if all(
                "video_grid_thw" in item and item["video_grid_thw"] is not None
                for item in features
            ):
                batch["video_grid_thw"] = torch.cat(
                    [item["video_grid_thw"] for item in features], dim=0
                )

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        if all("inputs_embeds" in item and item["inputs_embeds"] is not None for item in features):
            batch["inputs_embeds"] = torch.cat(
                [paddingtensor(item["inputs_embeds"], max_length) for item in features]
            )
        if all("position_ids" in item and item["position_ids"] is not None for item in features):
            # reduce_vlm_image_rows emits plain (1, N) positions, which is what
            # LlamaRotaryEmbedding's `cos[position_ids]` indexing wants; stored
            # Qwen mrope positions are (3, B, N) and keep their own path.
            if features[0]["position_ids"].dim() == 2:
                batch["position_ids"] = torch.cat(
                    [paddingtensor2D(item["position_ids"], max_length) for item in features]
                )
            else:
                batch["position_ids"] = paddingtensor3D_CBN(
                    [item["position_ids"] for item in features]
                )

        # Branch distillation: the trainer re-scores a token-substituted copy of
        # each sequence with a live target. Carried per sample, not batched:
        # the tile count follows the aspect ratio, and padding tiles would hand
        # the vision tower embeddings that no <image> token claims.
        if any("branch_image" in item for item in features):
            batch["branch_image"] = [item.get("branch_image") for item in features]

        return batch


class VLMSmolVLMDataCollatorWithPadding:
    """Collator for SmolVLM / Idefics3 online Eagle3 training.

    Recomputes ``pixel_values`` / ``pixel_attention_mask`` from ``image_paths``
    (input_ids are already image-token-expanded in the dataset builder).
    """

    def __init__(self, processor=None, image_processor_kwargs=None):
        self.processor = processor
        self._resolved_image_processor_kwargs = image_processor_kwargs or {}

    @staticmethod
    def _pad_tile_batch(tensor_list, pad_value=0):
        """Pad list of (1, N_i, ...) tensors along the tile dim, then cat batch."""
        if not tensor_list:
            return None
        max_n = max(t.shape[1] for t in tensor_list)
        padded = []
        for t in tensor_list:
            if t.shape[1] == max_n:
                padded.append(t)
                continue
            pad_shape = list(t.shape)
            pad_shape[1] = max_n - t.shape[1]
            pad = t.new_full(pad_shape, pad_value)
            padded.append(torch.cat([t, pad], dim=1))
        return torch.cat(padded, dim=0)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch = {
            "input_ids": torch.cat(
                [paddingtensor2D(item["input_ids"], max_length) for item in features]
            ),
            "attention_mask": torch.cat(
                [paddingtensor2D(item["attention_mask"], max_length) for item in features]
            ),
            "loss_mask": torch.cat(
                [paddingtensor2D(item["loss_mask"], max_length) for item in features]
            ),
            "hidden_states": None,
            "target_hiddens": None,
            "inputs_embeds": None,
            "position_ids": None,
        }


        if self.processor is not None and "image_paths" in features[0]:
            all_pixel_values, all_pixel_attention_mask = [], []
            for item in features:
                image_paths = json.loads(item["image_paths"])
                if not image_paths:
                    continue
                images = [load_image(p) for p in image_paths]
                if hasattr(self.processor, "image_processor"):
                    vision_enc = self.processor.image_processor(
                        images=images,
                        return_tensors="pt",
                        **self._resolved_image_processor_kwargs,
                    )
                else:
                    vision_enc = self.processor(
                        images=images,
                        return_tensors="pt",
                        **self._resolved_image_processor_kwargs,
                    )
                if "pixel_values" in vision_enc:
                    all_pixel_values.append(vision_enc["pixel_values"])
                if "pixel_attention_mask" in vision_enc:
                    all_pixel_attention_mask.append(vision_enc["pixel_attention_mask"])
            if all_pixel_values:
                batch["pixel_values"] = self._pad_tile_batch(all_pixel_values, pad_value=0)
            if all_pixel_attention_mask:
                batch["pixel_attention_mask"] = self._pad_tile_batch(
                    all_pixel_attention_mask, pad_value=0
                )

        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        return batch


class VLMHunyuanDataCollatorWithPadding:

    def __init__(self, processor=None, image_processor_kwargs=None):
        """
        Args:
            processor: VLM processor (e.g. AutoProcessor for hunyuan_vl).
                       When provided, image_paths in features will be decoded
                       on-the-fly to pixel_values (used in online training).
            image_processor_kwargs: Additional kwargs passed to image_processor,
                       e.g. {"max_pixels": 1003520, "min_pixels": 200704}.
        """
        self.processor = processor
        if image_processor_kwargs is None:
            image_processor_kwargs = {}
        max_pixels = image_processor_kwargs.get("max_pixels", None)
        min_pixels = image_processor_kwargs.get("min_pixels", "1024")
        if (
            processor is not None
            and (max_pixels is not None or min_pixels is not None)
            and hasattr(processor, "image_processor")
        ):
            self._resolved_image_processor_kwargs = build_image_processor_kwargs(
                processor.image_processor, max_pixels, min_pixels
            )
        else:
            self._resolved_image_processor_kwargs = {}
        rank0_print(f"_resolved_image_processor_kwargs: {self._resolved_image_processor_kwargs}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_states": None,
            "target_hiddens": None,
            "inputs_embeds": None,
            "position_ids": None,
            "input_position_ids": None,
        }

        # Online training: decode image_paths -> pixel_values on-the-fly
        if self.processor is not None and "image_paths" in features[0]:
            all_pixel_values, all_image_grid_thw = [], []
            for item in features:
                image_paths = json.loads(item["image_paths"])
                if image_paths:
                    images = [load_image(p) for p in image_paths]
                    if hasattr(self.processor, "image_processor"):
                        vision_enc = self.processor.image_processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )
                    else:
                        vision_enc = self.processor(
                            images=images,
                            return_tensors="pt",
                            **self._resolved_image_processor_kwargs,
                        )
                    all_pixel_values.append(vision_enc["pixel_values"])
                    if "image_grid_thw" in vision_enc:
                        all_image_grid_thw.append(vision_enc["image_grid_thw"])
            if all_pixel_values:
                batch["pixel_values"] = paddingtensor3D_BHW(all_pixel_values)
            if all_image_grid_thw:
                batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        else:
            if "pixel_values" in features[0]:
                batch["pixel_values"] = paddingtensor3D_BHW(
                    [item["pixel_values"] for item in features]
                )
            if all(
                "image_grid_thw" in item and item["image_grid_thw"] is not None
                for item in features
            ):
                batch["image_grid_thw"] = torch.cat(
                    [item["image_grid_thw"] for item in features], dim=0
                )

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        if all("inputs_embeds" in item and item["inputs_embeds"] is not None for item in features):
            batch["inputs_embeds"] = torch.cat(
                [paddingtensor(item["inputs_embeds"], max_length) for item in features]
            )
        if all(
            "input_position_ids" in item and item["input_position_ids"] is not None
            for item in features
        ):
            batch["input_position_ids"] = paddingtensor3D_BCN(
                [item["input_position_ids"] for item in features]
            )
        if all("position_ids" in item and item["position_ids"] is not None for item in features):
            batch["position_ids"] = torch.cat(
                [paddingtensor2D(item["position_ids"], max_length) for item in features]
            )
        return batch


class AudioDataCollatorWithPadding:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [paddingtensor2D(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.cat(
            [paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "feature_attention_mask": None,
            "input_features": None,
            "hidden_states": None,
            "target_hiddens": None,
            "inputs_embeds": None,
            "position_ids": None,
        }

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        if all("inputs_embeds" in item and item["inputs_embeds"] is not None for item in features):
            batch["inputs_embeds"] = torch.cat(
                [paddingtensor(item["inputs_embeds"], max_length) for item in features]
            )
        if all("position_ids" in item and item["position_ids"] is not None for item in features):
            batch["position_ids"] = torch.cat(
                [paddingtensor2D(item["position_ids"], max_length) for item in features]
            )
        if all(
            "feature_attention_mask" in item and item["feature_attention_mask"] is not None
            for item in features
        ):
            batch["feature_attention_mask"] = torch.cat(
                [(item["feature_attention_mask"]) for item in features]
            )
        if all(
            "input_features" in item and item["input_features"] is not None for item in features
        ):
            batch["input_features"] = torch.cat([(item["input_features"]) for item in features])
        return batch


class CosyVoice3DataCollatorWithPadding:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["text"].shape[-1] for item in features)
        batch_text_tokens = torch.cat(
            [paddingtensor2D(item["text"].unsqueeze(0), max_length) for item in features]
        )
        max_length = max(item["speech_token"].shape[-1] for item in features)
        batch_speech_tokens = torch.cat(
            [paddingtensor2D(item["speech_token"].unsqueeze(0), max_length) for item in features]
        )
        max_length = max(item["prompt_text"].shape[-1] for item in features)
        batch_prompt_text = torch.cat(
            [paddingtensor2D(item["prompt_text"].unsqueeze(0), max_length) for item in features]
        )
        max_length = max(item["prompt_speech_token"].shape[-1] for item in features)
        batch_prompt_speech_tokens = torch.cat(
            [
                paddingtensor2D(item["prompt_speech_token"].unsqueeze(0), max_length)
                for item in features
            ]
        )
        batch_text_token_lens = torch.stack([item["text_len"] for item in features])
        batch_speech_token_lens = torch.stack([item["speech_token_len"] for item in features])
        batch_prompt_text_lens = torch.stack([item["prompt_text_len"] for item in features])
        batch_prompt_speech_token_lens = torch.stack(
            [item["prompt_speech_token_len"] for item in features]
        )

        batch = {
            "text": batch_text_tokens,
            "text_len": batch_text_token_lens,
            "speech_token": batch_speech_tokens,
            "speech_token_len": batch_speech_token_lens,
            "prompt_speech_token": batch_prompt_speech_tokens,
            "prompt_speech_token_len": batch_prompt_speech_token_lens,
            "prompt_text": batch_prompt_text,
            "prompt_text_len": batch_prompt_text_lens,
            "hidden_states": None,
            "target_hiddens": None,
        }

        # Check if both hidden_states and target_hiddens exist in all features
        if all("hidden_states" in item and "target_hiddens" in item for item in features):
            batch["hidden_states"] = torch.cat(
                [paddingtensor(item["hidden_states"], max_length) for item in features]
            )
            batch["target_hiddens"] = torch.cat(
                [paddingtensor(item["target_hiddens"], max_length) for item in features]
            )
        return batch
