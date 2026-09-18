# Commands

Run everything from this repo root. `ARM=method` is the paper method.
`ARM=baseline` is stock EAGLE-3. Same scripts either way.

Two envs. Do not mix them.

```bash
python3.12 -m venv .venv-train && . .venv-train/bin/activate && pip install -r requirements.txt
export PATH="$PWD/.venv-train/bin:$PATH"   # torchrun must be this env

python3.9 -m venv .venv-eval && . .venv-eval/bin/activate && pip install -r HiViS/requirements.txt
```

Set these once:

```bash
TARGET=HuggingFaceTB/SmolVLM-256M-Instruct   # or a local snapshot
NPROC=8                                      # same GPU count for dump + train
EVAL_PY=$PWD/.venv-eval/bin/python
```

---

## 1. Data

JSONL in, one conversation per line. Then:

```bash
python tools/build_mixed_text_vl_jsonl.py \
    --sharegpt raw/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
    --llava-json raw/llava_v1_5_mix665k.json \
    --image-root raw/images --num-text 70000 --num-vl 70000 \
    --output-dir dataset --output-name mixed.jsonl
python tools/normalize_conversation_schema.py --in dataset/mixed.jsonl --out dataset/train.jsonl
```

Or use `WindUpHanger/angelslim-smolvlm-eagle3-artifacts` (`data/smolvlm256m/*_path_b64.jsonl`).

---

## 2. Offline train

Dump the target once. Then train on the `.ckpt` files. Use the same `NPROC`
for dump, image extract, and train.

```bash
# dump train
SPLIT=train DATASET_PATH=dataset/train.jsonl OUTPUT_DIR=dataset/hidden/train \
  TARGET_MODEL_NAME_OR_PATH=$TARGET NPROC=$NPROC \
  bash scripts/speculative/smolvlm/generate_vlm_hidden_for_draft_model.sh
SPLIT=train DATASET_PATH=dataset/train.jsonl OUTPUT_DIR=dataset/hidden/train \
  TARGET_MODEL_NAME_OR_PATH=$TARGET NPROC=$NPROC \
  bash scripts/speculative/smolvlm/extract_images.sh

# dump eval (same, different paths)
SPLIT=eval DATASET_PATH=dataset/eval.jsonl OUTPUT_DIR=dataset/hidden/eval \
  TARGET_MODEL_NAME_OR_PATH=$TARGET NPROC=$NPROC \
  bash scripts/speculative/smolvlm/generate_vlm_hidden_for_draft_model.sh
SPLIT=eval DATASET_PATH=dataset/eval.jsonl OUTPUT_DIR=dataset/hidden/eval \
  TARGET_MODEL_NAME_OR_PATH=$TARGET NPROC=$NPROC \
  bash scripts/speculative/smolvlm/extract_images.sh

# method
TARGET_MODEL_NAME_OR_PATH=$TARGET \
TRAIN_HIDDEN_PATH=dataset/hidden/train EVAL_HIDDEN_PATH=dataset/hidden/eval \
OUTPUT_DIR=output/method NUM_TRAIN_EPOCHS=20 NPROC=$NPROC \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_offline.sh

# baseline
ARM=baseline TARGET_MODEL_NAME_OR_PATH=$TARGET \
TRAIN_HIDDEN_PATH=dataset/hidden/train EVAL_HIDDEN_PATH=dataset/hidden/eval \
OUTPUT_DIR=output/baseline NUM_TRAIN_EPOCHS=20 NPROC=$NPROC \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_offline.sh
```

`ARM=method` turns on pooling (`VISTOKEN_REDUCE=pool`) and branch distillation.
You do not pass those by hand.

---

## 3. Online train

No dump. JSONL in, target runs every step. Same `ARM` split.

```bash
# method
TARGET_MODEL_NAME_OR_PATH=$TARGET \
TRAIN_DATA_PATH=dataset/train.jsonl EVAL_DATA_PATH=dataset/eval.jsonl \
OUTPUT_DIR=output/method_online NUM_TRAIN_EPOCHS=20 NPROC=$NPROC \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

# baseline
ARM=baseline TARGET_MODEL_NAME_OR_PATH=$TARGET \
TRAIN_DATA_PATH=dataset/train.jsonl EVAL_DATA_PATH=dataset/eval.jsonl \
OUTPUT_DIR=output/baseline_online NUM_TRAIN_EPOCHS=20 NPROC=$NPROC \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

Smoke: `SAMPLE_NUM=64 NUM_TRAIN_EPOCHS=1`.

---

## 4. Eval

One GPU. Run `naive` first, into the same `OUT_DIR`. Method was trained with
pool factor 4; baseline was not.

```bash
OUT=results/run1

ARM=naive OUT_DIR=$OUT EVAL_PY=$EVAL_PY \
  bash scripts/speculative/smolvlm/run_eval.sh

ARM=method DRAFT=output/method POOL_FACTOR=4 OUT_DIR=$OUT EVAL_PY=$EVAL_PY \
  bash scripts/speculative/smolvlm/run_eval.sh

ARM=baseline DRAFT=output/baseline OUT_DIR=$OUT EVAL_PY=$EVAL_PY \
  bash scripts/speculative/smolvlm/run_eval.sh

python tools/eval_table.py $OUT --arms method,baseline --baseline naive --filter exact
```

One benchmark: `BENCHMARKS=textvqa`.

---

## 5. Paths the launchers print

A finished train dir has `model.safetensors` at the top. A run still going has
only `checkpoint-*`. Point `--draft` / `DRAFT=` at either.
