#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Acceptance length and speedup from run_angelslim_eval.py result files.

One convention, applied the same way to every arm:

* **macro** -- each prompt contributes one number, then those are averaged.
  This is what hivis/evaluation/speed.py does (it means `np.mean` over
  per-prompt tok/s), and tau is averaged the same way for consistency.
* **filtered** (`--filter`) -- which prompts count. Speculative output is not
  identical to autoregressive under bf16, and tok/s rises with output length
  because the prefill amortises, so an arm that happened to generate more text
  would otherwise look faster for that reason alone. Three ways to handle that:

  * `length10` (default) -- keep a prompt when every arm's answer is within
    +-10% of the baseline's length. Lengths are comparable, so tok/s is, and
    almost every prompt survives. On a long-answer benchmark the stricter rule
    keeps nothing at all: under `exact`, COCO-Caption (435 tokens on average)
    keeps 0 of 80 prompts and OmniDocBench keeps 4, because one differing token
    diverges the rest.
  * `exact` -- keep only prompts where the baseline and EVERY arm produced
    byte-identical text.
  * `none` -- keep every prompt. Required when the arms are not meant to agree,
    as in an image ablation where one arm is run with `--draft_image_control
    drop`: its answers differ by construction, so any agreement filter would
    throw the comparison away.

  The filter is decided on rep 1 and the same prompts are then used for every
  repeat, so the repeats of one arm are averaged over one prompt set.
* **the baseline's tokens are counted HiViS's way** -- by re-tokenising the
  answer text (`retok_tokens`), not by the generation loop. That undercounts by
  about two tokens per answer on tokenisers that prepend no BOS, which inflates
  every ratio; it is kept so a number here is comparable to a HiViS number. It
  cancels out of arm-vs-arm comparisons, which is why those are reported as
  percentages rather than as a difference of x-factors.

Tau carries EAGLE's +1 (accepted draft tokens plus the one the target's own
verification produces); hivis/evaluation reports it without.

    python tools/eval_table.py results/ --arms full_4ep,pool16x_4ep
    python tools/eval_table.py results/ --arms a,b --reps 2 --relative-to a
    python tools/eval_table.py results/ --arms with_image,no_image --filter none
"""
import argparse
import json
import statistics
from pathlib import Path

# Every benchmark run_angelslim_eval.py accepts. A table that leaves some out
# is a subset chosen after the fact, so the default is all of them.
BENCHMARKS = ("MathVista", "ScienceQA", "mmmu", "mmvet", "vqav2", "mme", "textvqa",
              "MMStar", "seedbench", "gqa", "DocVQA", "coco_caption", "omnidocbench")
FILTERS = ("length10", "exact", "none")
LENGTH_TOLERANCE = 0.10


def load(path):
    with open(path) as f:
        return {row["index"]: row for row in json.load(f)["per_prompt"]}


def find(results_dir, benchmark, arm, rep, reps):
    """One result file, tolerating both `__arm.json` and `__arm_r1.json`."""
    stem = f"{benchmark}__{arm}"
    for name in ([f"{stem}_r{rep}.json"] if reps > 1 else [f"{stem}.json", f"{stem}_r{rep}.json"]):
        path = Path(results_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{Path(results_dir)/stem}*.json")


def tau(rows, keys):
    """Mean over prompts of (accepted draft tokens + 1) for that prompt."""
    return statistics.mean(
        1 + sum(rows[k]["accept_lengths"]) / len(rows[k]["accept_lengths"])
        if rows[k]["accept_lengths"] else 1.0
        for k in keys
    )


def keep(base, arms, how):
    """The prompts a table is computed over, under one filter."""
    present = [k for k in base if all(k in a for a in arms.values())]
    if how == "none":
        return present
    if how == "exact":
        return [k for k in present
                if all(a[k]["generated_text"] == base[k]["generated_text"]
                       for a in arms.values())]
    return [k for k in present
            if all(abs(a[k]["tokens"] - base[k]["tokens"])
                   <= LENGTH_TOLERANCE * base[k]["tokens"] for a in arms.values())]


def tokens_per_second(rows, keys, field="tokens"):
    return statistics.mean(rows[k][field] / rows[k]["time"] for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--baseline", default="naive", help="arm name of the AR baseline")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--reps", type=int, default=1,
                    help="repeats per arm; speedup averages them, tau uses rep 1")
    ap.add_argument("--relative-to", default=None,
                    help="also printeach arm relative to this one, in percent")
    ap.add_argument("--filter", default="length10", choices=FILTERS,
                    help="which prompts count: length10 (default) keeps answers "
                         "within +-10%% of the baseline's length, exact keeps "
                         "only byte-identical ones, none keeps all -- use none "
                         "when the arms are not meant to agree")
    args = ap.parse_args()

    arms = args.arms.split(",")
    benchmarks = args.benchmarks.split(",")
    taus = {a: [] for a in arms}
    speedups = {a: [] for a in arms}
    subset = []

    for benchmark in benchmarks:
        base = load(find(args.results_dir, benchmark, args.baseline, 1, 1))
        first = {a: load(find(args.results_dir, benchmark, a, 1, args.reps)) for a in arms}
        keys = keep(base, first, args.filter)
        if not keys:
            raise SystemExit(
                f"{benchmark}: --filter {args.filter} keeps no prompt. On a "
                "long-answer benchmark try --filter length10, or none if the "
                "arms are not meant to agree.")
        subset.append(f"{len(keys)}/{len(base)}")
        denominator = tokens_per_second(base, keys, "retok_tokens")
        for a in arms:
            taus[a].append(tau(first[a], keys))
            reps = [
                tokens_per_second(load(find(args.results_dir, benchmark, a, r, args.reps)), keys)
                for r in range(1, args.reps + 1)
            ]
            speedups[a].append(statistics.mean(reps) / denominator)

    print(f"filter={args.filter}")
    width = max(len(a) for a in arms) + 10
    header = "".join(f"{b[:9]:>10}" for b in benchmarks)
    print(f"{'':<{width}}{header}{'MEAN':>10}")
    print(f"{'subset':<{width}}" + "".join(f"{s:>10}" for s in subset))
    for a in arms:
        print(f"{a + ' tau':<{width}}" + "".join(f"{v:10.3f}" for v in taus[a])
              + f"{statistics.mean(taus[a]):10.3f}")
        print(f"{a + ' speedup':<{width}}" + "".join(f"{v:9.2f}x" for v in speedups[a])
              + f"{statistics.mean(speedups[a]):9.2f}x")
    if args.relative_to:
        ref = args.relative_to
        print(f"\nrelative to {ref}")
        for a in arms:
            dt = (statistics.mean(taus[a]) / statistics.mean(taus[ref]) - 1) * 100
            ds = statistics.mean(
                (speedups[a][i] / speedups[ref][i] - 1) * 100 for i in range(len(benchmarks))
            )
            print(f"  {a:<{width}} tau {dt:+6.1f}%   speedup {ds:+6.1f}%")


if __name__ == "__main__":
    main()
