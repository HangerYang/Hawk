#!/bin/bash
# Evaluate one drafter across the benchmark set, at the settled protocol.
#
# Defaults are the protocol, not suggestions: n=80, greedy, tree
# total_token=60 depth=5 top_k=10, two repeats. Repeats matter -- the same
# configuration measured twice on a busy machine has moved by several ms per
# prompt, which is the size of the differences being resolved.
#
# Run the baseline once per result directory before any arm; tools/eval_table.py
# needs it as the speedup denominator and as the text every arm is filtered
# against.
#
#   ARM=naive       bash scripts/speculative/smolvlm/run_eval.sh
#   ARM=full_4ep    DRAFT=output/vistoken/full_4ep bash .../run_eval.sh
#   ARM=pool16x_4ep DRAFT=output/vistoken/pool16x_4ep POOL_FACTOR=13 bash .../run_eval.sh
#   ARM=hivis_smolvlm DRAFT_METHOD=hivis DRAFT=output/hivis_official/<run> bash .../run_eval.sh
#   ARM=hivis_qwen7b  DRAFT_METHOD=hivis BASE=Qwen/Qwen2.5-VL-7B-Instruct \
#       DRAFT=~/HiViS/models/HiViS-Qwen2.5-VL-7B-Instruct bash .../run_eval.sh
#
# Then:  python tools/eval_table.py $OUT_DIR --arms full_4ep,pool16x_4ep --reps 2
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT/HiViS" || exit 1
export PYTHONPATH="$ROOT/HiViS${PYTHONPATH:+:$PYTHONPATH}"

ARM=${ARM:?set ARM to the name this run is filed under}
DRAFT_METHOD=${DRAFT_METHOD:-angelslim_eagle3}
BASE=${BASE:-HuggingFaceTB/SmolVLM-256M-Instruct}
OUT_DIR=${OUT_DIR:-$ROOT/results/$(date +%Y%m%d)}
BENCHMARKS=${BENCHMARKS:-"textvqa ScienceQA vqav2 mme mmvet MathVista mmmu MMStar seedbench gqa DocVQA coco_caption omnidocbench"}
N=${N:-80}
REPS=${REPS:-1}
DEVICE=${DEVICE:-cuda:0}
# Draft-side image-row pooling: unset = the drafter sees every row the target
# prefilled. The factor must be the one the checkpoint was TRAINED with.
POOL_FACTOR=${POOL_FACTOR:-}
PY=${EVAL_PY:-python}

EXTRA=()
[ "$ARM" = "naive" ] && EXTRA+=(--naive)
if [ -n "$POOL_FACTOR" ]; then
  EXTRA+=(--draft_image_reduce pool --draft_image_factor "$POOL_FACTOR" --draft_image_root "$ROOT")
fi
# --naive ignores the drafter but the evaluator still wants a path for it.
DRAFT=${DRAFT:-$ROOT/output/vistoken/full_4ep}

mkdir -p "$OUT_DIR/logs"
echo "arm=$ARM method=$DRAFT_METHOD base=$BASE draft=$DRAFT pool=${POOL_FACTOR:-none} -> $OUT_DIR"
for BENCH in $BENCHMARKS; do
  for REP in $(seq 1 "$REPS"); do
    [ "$ARM" = "naive" ] && [ "$REP" -gt 1 ] && continue   # the denominator is deterministic
    JSON=$OUT_DIR/${BENCH}__${ARM}_r${REP}.json
    [ -s "$JSON" ] && { echo "  skip $BENCH r$REP"; continue; }
    HIVIS_POOL_FAST=${HIVIS_POOL_FAST:-1} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
      "$PY" run_angelslim_eval.py \
        --draft_method "$DRAFT_METHOD" --base "$BASE" --draft "$DRAFT" \
        --dataset "$BENCH" --n "$N" --device "$DEVICE" "${EXTRA[@]}" \
        --out "$JSON" > "$OUT_DIR/logs/${BENCH}__${ARM}_r${REP}.log" 2>&1
    echo "  $BENCH $ARM r$REP exit=$? $(date '+%T')"
  done
done
echo "done: $OUT_DIR"
