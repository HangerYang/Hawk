#!/bin/bash
# Online EAGLE3 training for SmolVLM: the live target runs every step.
# Same two arms as the offline launcher. ARM=method (default) is banded aux mix,
# branch-change distillation, and pooled draft image rows. ARM=baseline is stock
# EAGLE-3, every image row, no branch loss.
#
#   bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#   ARM=baseline bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
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
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-dataset/train.jsonl}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-dataset/eval.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-output/smolvlm_256m_eagle3_online}
RUN_NAME=${RUN_NAME:-smolvlm_256m_eagle3_online}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
EMBED_WEIGHT_KEY=${EMBED_WEIGHT_KEY:-model.text_model.embed_tokens.weight}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-5000}
EVAL_STRATEGY=${EVAL_STRATEGY:-no}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
REPORT_TO=${REPORT_TO:-none}
NPROC=${NPROC:-8}
NUM_PROC=${NUM_PROC:-8}

if [[ "$ARM" == "method" ]]; then
  export VISTOKEN_REDUCE=${VISTOKEN_REDUCE:-pool}
  export VISTOKEN_FACTOR=${VISTOKEN_FACTOR:-4}
  export VISTOKEN_KEEPPOS=${VISTOKEN_KEEPPOS:-1}
  export VISTOKEN_IMAGE_TOKEN_ID=${VISTOKEN_IMAGE_TOKEN_ID:-49190}
fi

ARGS=(
  --modal_type VLM
  --training_mode online
  --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH"
  --draft_model_config_path "$DRAFT_MODEL_CONFIG_PATH"
  --train_data_path "$TRAIN_DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler_type "$LR_SCHEDULER_TYPE"
  --save_strategy "$SAVE_STRATEGY"
  --save_steps "$SAVE_STEPS"
  --eval_strategy "$EVAL_STRATEGY"
  --logging_steps "${LOGGING_STEPS:-100}"
  --model_max_length "$MODEL_MAX_LENGTH"
  --chat_template_type "$CHAT_TEMPLATE_TYPE"
  --embed_weight_key "$EMBED_WEIGHT_KEY"
  --num_proc "$NUM_PROC"
  --report_to "$REPORT_TO"
  --run_name "$RUN_NAME"
  --bf16
)
if [[ -n "$EVAL_DATA_PATH" ]]; then ARGS+=(--eval_data_path "$EVAL_DATA_PATH"); fi
if [[ -n "$DEEPSPEED_CONFIG" ]]; then ARGS+=(--deepspeed "$DEEPSPEED_CONFIG"); fi
[[ -n "${SAMPLE_NUM:-}" ]] && ARGS+=(--sample_num "$SAMPLE_NUM")

mkdir -p "$OUTPUT_DIR"
echo "ARM=$ARM config=$DRAFT_MODEL_CONFIG_PATH out=$OUTPUT_DIR"
torchrun --nproc_per_node="$NPROC" tools/train_eagle3_online.py "${ARGS[@]}"
