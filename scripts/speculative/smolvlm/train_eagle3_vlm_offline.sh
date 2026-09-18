#!/bin/bash
# Offline EAGLE3 training for SmolVLM / Idefics3: a 1-layer draft reading
# pre-computed hidden states instead of running the target every step.
#
# Two arms, one script. ARM=method (the default) is banded aux mix and
# branch-change distillation -- both in the draft config -- plus each image
# tile's rows mean-pooled 4x for the draft, through the VISTOKEN_* variables
# below. The target's rows are untouched either way.
#
#   bash scripts/speculative/smolvlm/train_eagle3_vlm_offline.sh              # the method
#   ARM=baseline bash scripts/speculative/smolvlm/train_eagle3_vlm_offline.sh # stock EAGLE-3
#
# ARM=baseline is the same architecture with all three removed: the stock
# fused_fc over aux layers [1, 14, 26], every image row, no branch loss. Both
# arms read the same generated .ckpt files.
#
# Requires, first:
#   generate_vlm_hidden_for_draft_model.sh      TRAIN_HIDDEN_PATH / EVAL_HIDDEN_PATH
#   tools/extract_images_beside_hidden.py       the .img files branch distillation
#                                               re-scores images from
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR=angelslim/compressor/speculative/train/configs
ARM=${ARM:-method}
case "$ARM" in
  method)   DEFAULT_CONFIG=$CONFIG_DIR/smolvlm-256m-eagle3-banded-mix-fc-3.1-bands-1-2-8-10_18-20_23-26-28-branch-change-top1-w01.json ;;
  baseline) DEFAULT_CONFIG=$CONFIG_DIR/smolvlm-256m-eagle3.json ;;
  *) echo "ARM must be method or baseline, got '$ARM'" >&2; exit 2 ;;
esac
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
export DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$DEFAULT_CONFIG}
TRAIN_HIDDEN_PATH=${TRAIN_HIDDEN_PATH:-dataset/smolvlm_256m_eagle3_offline_hidden/train}
EVAL_HIDDEN_PATH=${EVAL_HIDDEN_PATH:-dataset/smolvlm_256m_eagle3_offline_hidden/eval}
OUTPUT_DIR=${OUTPUT_DIR:-output/smolvlm_256m_eagle3_offline_1layer}
RUN_NAME=${RUN_NAME:-smolvlm_256m_eagle3_offline_1layer}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
EMBED_WEIGHT_KEY=${EMBED_WEIGHT_KEY:-model.text_model.embed_tokens.weight}
LM_HEAD_KEY=${LM_HEAD_KEY:-lm_head.weight}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
# train_eagle3_offline.py parses --warmup_ratio but does not forward it
# (transformers 5.x has no TrainingArguments.warmup_ratio), so it has no effect.
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-5000}
EVAL_STRATEGY=${EVAL_STRATEGY:-no}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
REPORT_TO=${REPORT_TO:-none}
NPROC=${NPROC:-8}
NUM_PROC=${NUM_PROC:-8}
# Each sample is one ~7 MB .ckpt read; with 0 workers that read runs in the
# training process and the GPUs idle (~3% utilisation measured).
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

if [[ "$ARM" == "method" ]]; then
  # Draft-side image rows: pool each tile's rows 4x, keeping absolute positions.
  export VISTOKEN_REDUCE=${VISTOKEN_REDUCE:-pool}
  export VISTOKEN_FACTOR=${VISTOKEN_FACTOR:-4}
  export VISTOKEN_KEEPPOS=${VISTOKEN_KEEPPOS:-1}
  export VISTOKEN_IMAGE_TOKEN_ID=${VISTOKEN_IMAGE_TOKEN_ID:-49190}
  # Branch distillation re-scores token-substituted sequences with a live target;
  # it must be the snapshot the hidden states were generated from.
  export BRANCH_DISTILL_TARGET=${BRANCH_DISTILL_TARGET:-$TARGET_MODEL_NAME_OR_PATH}
  export BRANCH_DISTILL_IMAGE_TOKEN_ID=${BRANCH_DISTILL_IMAGE_TOKEN_ID:-49190}
fi
# The baseline arm exports neither family. BRANCH_DISTILL_TARGET alone makes the
# dataset builder open every .img, which costs time the baseline never uses.

ARGS=(
  --modal_type VLM
  --training_mode offline
  --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH"
  --draft_model_config_path "$DRAFT_MODEL_CONFIG_PATH"
  --train_hidden_path "$TRAIN_HIDDEN_PATH"
  --eval_hidden_path "$EVAL_HIDDEN_PATH"
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler_type "$LR_SCHEDULER_TYPE"
  --warmup_ratio "$WARMUP_RATIO"
  --save_strategy "$SAVE_STRATEGY"
  --save_steps "$SAVE_STEPS"
  --eval_strategy "$EVAL_STRATEGY"
  --logging_steps "${LOGGING_STEPS:-100}"
  --model_max_length "$MODEL_MAX_LENGTH"
  --chat_template_type "$CHAT_TEMPLATE_TYPE"
  --embed_weight_key "$EMBED_WEIGHT_KEY"
  --lm_head_key "$LM_HEAD_KEY"
  --num_proc "$NUM_PROC"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --report_to "$REPORT_TO"
  --run_name "$RUN_NAME"
  --bf16
)
if [[ -n "$DEEPSPEED_CONFIG" ]]; then ARGS+=(--deepspeed "$DEEPSPEED_CONFIG"); fi

mkdir -p "$OUTPUT_DIR"
echo "ARM=$ARM config=$DRAFT_MODEL_CONFIG_PATH out=$OUTPUT_DIR"
torchrun --nproc_per_node="$NPROC" tools/train_eagle3_offline.py "${ARGS[@]}"
