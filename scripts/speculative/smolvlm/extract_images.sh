#!/bin/bash
# Write data_<i>.img next to each hidden-state .ckpt. Branch distillation
# re-scores the image and the ckpt does not contain it.
#
#   SPLIT=train bash scripts/speculative/smolvlm/extract_images.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR=angelslim/compressor/speculative/train/configs
SPLIT=${SPLIT:-train}
DATASET_PATH=${DATASET_PATH:-dataset/smolvlm256m_offline_src/${SPLIT}.jsonl}
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3-gen-10layer.json}
OUTPUT_DIR=${OUTPUT_DIR:-dataset/smolvlm_256m_eagle3_offline_hidden/${SPLIT}}
NPROC=${NPROC:-8}

ARGS=(
  --modal_type VLM
  --dataset_path "$DATASET_PATH"
  --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH"
  --draft_model_config_path "$DRAFT_MODEL_CONFIG_PATH"
  --chat_template_type smolvlm
  --model_max_length "${MODEL_MAX_LENGTH:-4096}"
  --outdir "$OUTPUT_DIR"
)
[[ -n "${SAMPLE_NUM:-}" ]] && ARGS+=(--sample_num "$SAMPLE_NUM")

echo "SPLIT=$SPLIT DATASET_PATH=$DATASET_PATH OUTPUT_DIR=$OUTPUT_DIR"
torchrun --nproc_per_node="$NPROC" tools/extract_images_beside_hidden.py "${ARGS[@]}"
