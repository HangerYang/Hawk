#!/bin/bash
# Offline EAGLE3 hidden-state generation for SmolVLM / Idefics3.
#
# Produces one .ckpt per sample (input_ids, loss_mask, the hidden states of the
# layers listed in the draft config's aux_hidden_states_layer_ids, and the
# target hidden) plus vocab_mapping.pt and aux_layers.json, which is what
# train_eagle3_vlm_offline.sh consumes.
#
# The default config generates ten layers -- the union of what the baseline arm
# wants ([1, 14, 26]) and what the method wants (its nine) -- so one corpus
# trains both. The trainer slices the layers its own config asks for by reading
# aux_layers.json beside the shards. Generation itself runs through the
# ONLINE dataset builder -- that is where SmolVLM's processor-driven image
# token expansion happens -- so images must be openable by PIL at the paths
# stored in the jsonl (absolute paths are safest).
#
#   SPLIT=train bash scripts/speculative/smolvlm/generate_vlm_hidden_for_draft_model.sh
#   SAMPLE_NUM=64 OUTPUT_DIR=<dir> bash .../generate_vlm_hidden_for_draft_model.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR="$ROOT/angelslim/compressor/speculative/train/configs"
SPLIT=${SPLIT:-train}
DATASET_PATH=${DATASET_PATH:-dataset/smolvlm256m_offline_src/${SPLIT}.jsonl}
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3-gen-10layer.json}
TARGET_BACKEND=${TARGET_BACKEND:-hf}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
OUTPUT_DIR=${OUTPUT_DIR:-dataset/smolvlm_256m_eagle3_offline_hidden/${SPLIT}}
NPROC=${NPROC:-8}
NUM_PROC=${NUM_PROC:-8}

ARGS=(
  --modal_type VLM
  --dataset_path "$DATASET_PATH"
  --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH"
  --draft_model_config_path "$DRAFT_MODEL_CONFIG_PATH"
  --target_backend "$TARGET_BACKEND"
  --torch_dtype bfloat16
  --model_max_length "$MODEL_MAX_LENGTH"
  --chat_template_type "$CHAT_TEMPLATE_TYPE"
  --outdir "$OUTPUT_DIR"
  --num_proc "$NUM_PROC"
)
[[ -n "${SAMPLE_NUM:-}" ]] && ARGS+=(--sample_num "$SAMPLE_NUM")

mkdir -p "$OUTPUT_DIR"
echo "SPLIT=$SPLIT DATASET_PATH=$DATASET_PATH OUTPUT_DIR=$OUTPUT_DIR SAMPLE_NUM=${SAMPLE_NUM:-all}"
torchrun --nproc_per_node="$NPROC" tools/generate_hidden_for_draft_model.py "${ARGS[@]}"
