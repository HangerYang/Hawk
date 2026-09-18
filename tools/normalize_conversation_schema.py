#!/usr/bin/env python3
"""Pad every content part of a mixed text/VL JSONL with both "text" and "image".

`datasets` infers the Arrow schema from the first rows it reads. A corpus whose
text parts are {type, text} and whose image parts are {type, text, image} fails
to cast as soon as the first image row arrives, which -- because the corpora
here are ordered text-first -- happens tens of thousands of rows in.

Padding both keys everywhere is inert: the dataset builders dispatch on `type`
and treat an empty `image` as absent. A content part given as a bare string is
wrapped into one text part.

    python tools/normalize_conversation_schema.py --in train.jsonl --out train.norm.jsonl
"""

import argparse
import json
import os


def normalize(row: dict) -> dict:
    for message in row.get("conversations", []):
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content, "image": ""}]
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                part.setdefault("text", "")
                part.setdefault("image", "")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    n = 0
    part = args.dst + ".part"
    with open(args.src) as f, open(part, "w") as g:
        for line in f:
            if not line.strip():
                continue
            g.write(json.dumps(normalize(json.loads(line))) + "\n")
            n += 1
    os.replace(part, args.dst)
    print(f"{n} rows -> {args.dst}")


if __name__ == "__main__":
    main()
