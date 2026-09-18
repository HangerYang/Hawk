#!/usr/bin/env python3
"""Re-layout result JSONs written by an older run_angelslim_eval.py.

Nothing is re-measured. Every number in the output is either copied from the
input or re-derived from the same stored ``per_prompt`` records, so a table
built from converted files is identical to one built from the originals.

What changes is only which number sits under which name:

  old ``tau``            -> ``tau_round_pooled``   (same value, HiViS's convention:
                                                    every round weighs the same)
  new ``tau``            =  mean over prompts of (accepted draft tokens + 1),
                            which is what tools/eval_table.py prints under
                            --filter none

and these are dropped, each being a duplicate or derivable:

  ``metrics``            duplicated ``cfg``
  ``mean_accept_length`` equals tau - 1 in the other convention
  ``tok_per_s_macro``, ``tok_per_s_macro_retok``, ``total_output_tokens``,
  ``total_time_s``, ``avg_input_tokens``, ``avg_output_tokens``
                         all recomputable from ``per_prompt``

A file is refused, not written, when its stored header and its per-prompt
records disagree: that means the file was edited by hand and neither number can
be trusted.

    python tools/convert_eval_json.py <dir> --out <dir>
    python tools/convert_eval_json.py <dir> --in-place
"""

import argparse
import json
import shutil
from pathlib import Path

TOLERANCE = 1e-9


def pooled_tau(rows):
    """Every round weighs the same -- what the old files stored as `tau`."""
    acc = [x for r in rows for x in r.get("accept_lengths", [])]
    return sum(x + 1 for x in acc) / len(acc) if acc else 1.0


def macro_tau(rows):
    """One number per prompt, then the mean -- what eval_table.py prints."""
    if not rows:
        return 1.0
    per_prompt = [
        1 + sum(r["accept_lengths"]) / len(r["accept_lengths"])
        if r.get("accept_lengths") else 1.0
        for r in rows
    ]
    return sum(per_prompt) / len(per_prompt)


def convert(doc, path):
    rows = doc.get("per_prompt")
    if not isinstance(rows, list):
        raise ValueError("no per_prompt records")

    pooled = pooled_tau(rows)
    stored = doc.get("tau", doc.get("metrics", {}).get("tau"))
    if stored is not None and abs(stored - pooled) > TOLERANCE:
        raise ValueError(
            "stored tau %.6f does not match the per-prompt records (%.6f); "
            "refusing to rewrite a file whose header was edited" % (stored, pooled)
        )

    metrics = doc.get("metrics", {})
    total_tok = sum(r["tokens"] for r in rows)
    total_time = sum(r["time"] for r in rows)
    tok_per_s = doc.get("tok_per_s", metrics.get("tok_per_s"))
    if tok_per_s is None:
        tok_per_s = total_tok / total_time if total_time else 0.0

    for r in rows:
        r.pop("mean_accept_length", None)

    return {
        "tau": macro_tau(rows),
        "tau_convention": "mean over prompts of (accepted draft tokens + 1); "
                          "matches tools/eval_table.py --filter none",
        "tau_round_pooled": pooled,
        "tok_per_s": tok_per_s,
        "rounds": doc.get("rounds", metrics.get("rounds", sum(len(r.get("accept_lengths", [])) for r in rows))),
        "acceptance_rates": metrics.get("acceptance_rates", doc.get("acceptance_rates", [])),
        "per_prompt": rows,
        "cfg": doc.get("cfg", {k: v for k, v in metrics.items() if k in (
            "draft", "draft_method", "base", "dataset", "total_token", "depth",
            "top_k", "max_new_tokens", "naive", "temperature", "dtype", "seed",
            "warmup", "prompt_style", "max_input_tokens")}),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory of result JSONs")
    ap.add_argument("--out", help="write converted copies here")
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite src, keeping each original as <name>.json.bak")
    args = ap.parse_args()

    if bool(args.out) == bool(args.in_place):
        raise SystemExit("choose exactly one of --out DIR or --in-place")

    src = Path(args.src)
    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit("no .json files in %s" % src)
    out_dir = Path(args.out) if args.out else src
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = skipped = 0
    for path in files:
        doc = json.loads(path.read_text())
        if "tau_convention" in doc and "metrics" not in doc:
            print("  already converted, skipping  %s" % path.name)
            skipped += 1
            if args.out:
                shutil.copy2(path, out_dir / path.name)
            continue
        try:
            new = convert(doc, path)
        except ValueError as exc:
            print("  SKIPPED %s: %s" % (path.name, exc))
            skipped += 1
            continue
        if args.in_place:
            shutil.copy2(path, path.with_suffix(".json.bak"))
        (out_dir / path.name).write_text(
            json.dumps(new, indent=2, ensure_ascii=False))
        print("  %-42s tau %.4f (was reported as %.4f round-pooled)"
              % (path.name, new["tau"], new["tau_round_pooled"]))
        converted += 1

    print("%d converted, %d skipped -> %s" % (converted, skipped, out_dir))


if __name__ == "__main__":
    main()
