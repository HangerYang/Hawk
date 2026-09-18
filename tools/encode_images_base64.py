#!/usr/bin/env python3
"""Rewrite a mixed text/VL JSONL, replacing relative image paths with
inline base64 data URLs (same format as data_utils.py's
_pil_file_to_data_url: "data:<mime>;base64,<...>").

Input conversations use the OpenAI-style content-part format:
  {"type": "image", "image": "coco/train2017/000000454979.jpg"}
Relative image paths are resolved against --image-root.
"""

import argparse
import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_line(line: str, image_root: Path) -> tuple[str, int, int]:
    row = json.loads(line)
    n_images = 0
    n_missing = 0
    for message in row.get("conversations", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            image_field = part.get("image", "")
            if isinstance(image_field, str) and image_field.startswith("data:"):
                continue  # already encoded
            abs_path = image_root / image_field
            if not abs_path.is_file():
                n_missing += 1
                continue
            part["image"] = to_data_url(abs_path)
            n_images += 1
    return json.dumps(row), n_images, n_missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source JSONL with relative image paths")
    parser.add_argument("--output", required=True, help="Destination JSONL with embedded base64 images")
    parser.add_argument("--image-root", required=True, help="Directory relative image paths resolve against")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    image_root = Path(args.image_root)
    with open(args.input, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    total_images = 0
    total_missing = 0
    with open(args.output, "w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (encoded_line, n_images, n_missing) in enumerate(
            pool.map(lambda line: encode_line(line, image_root), lines)
        ):
            out.write(encoded_line + "\n")
            total_images += n_images
            total_missing += n_missing
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{len(lines)} rows, {total_images} images encoded, {total_missing} missing")

    print(
        f"Done: {len(lines)} rows -> {args.output} "
        f"({total_images} images encoded, {total_missing} missing)"
    )


if __name__ == "__main__":
    main()
