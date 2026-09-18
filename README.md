# Offline EAGLE-3 drafting for SmolVLM-256M

A speculative-decoding drafter for a frozen vision-language target. The target is
never fine-tuned and always prefills the full image; every reduction happens on
what reaches the drafter. The drafter is one decoder layer, 51.9M parameters, and
it differs from a stock EAGLE-3 drafter in exactly three things:

| | |
|---|---|
| **banded aux mix** | EAGLE-3 hands the drafter three target layers, hard-concatenated, chosen by a fixed rule. Instead take nine layers, group them into three bands, and let a softmax over each band learn the mixture. Each band collapses to one vector, so the drafter's input width is unchanged: nine layers cost what three did. |
| **branch-change distillation** | A drafter trained only on the correct continuation never learns what follows its own mistakes. At each training-time-test step, a position *branches* when the draft's top-1 differs from the teacher's top-1 but sits inside the teacher's top-k. Those tokens are substituted at their absolute positions into one copy of the sequence, which gets a single teacher forward, and the drafter is trained to match the **change** in the teacher's centred logits relative to the real path. |
| **pooled draft image rows** | SmolVLM turns one image into 832 rows (12 crops plus a thumbnail tile, 64 rows each). Average each tile's rows in contiguous groups, keeping `max(1, len // factor)` per tile. Nothing is discarded, every surviving row keeps a **real absolute position** so its RoPE angle is the one the target computed it at, and the target's own rows are untouched. Factor 4 leaves the drafter 208 rows. |

The baseline this is measured against is the same architecture with all three
removed, trained on the same data, evaluated through the same loop.

```
tools/          entry points: data generation, training, evaluation tables
angelslim/      the drafter, the trainers, the dataset builders
scripts/        launchers, one per stage
HiViS/          the evaluator, vendored; benchmark metadata included
```

---

## 1. Two environments

They do not mix. The evaluator pins transformers 4.54; training needs 5.x.

| | training and data generation | evaluation |
|---|---|---|
| python | 3.12 | 3.9 |
| torch | 2.11 | 2.6 |
| transformers | 5.16.1 | **4.54.0** (pinned) |
| accelerate | 1.14 | 1.9 |
| requirements | `requirements.txt` | `HiViS/requirements.txt` |

```bash
python3.12 -m venv .venv-train && . .venv-train/bin/activate && pip install -r requirements.txt
python3.9  -m venv .venv-eval  && . .venv-eval/bin/activate  && pip install -r HiViS/requirements.txt
```

Any environment manager will do; nothing in the scripts assumes one. The eval
scripts take the interpreter from `EVAL_PY`, so the two never have to be active
at the same time.

**The most common way to lose a run:** the training launcher ends in a bare
`torchrun`, which resolves against whatever is first on `PATH`, not against the
interpreter you checked the version with. A checkpoint written under 5.x cannot
be resumed under 4.x — it dies with `_pickle.UnpicklingError: Weights only load
failed`. Put the training environment's `bin` first on `PATH` in the launcher
itself and confirm with `which torchrun`.

No GPU-side pin is required beyond this; nothing here needs flash-attention.

---

## 2. Data

### 2.1 The source corpus

One JSONL, one conversation per line, image parts carrying either an absolute
path or an inline `data:` URI:

```json
{"id": "70000",
 "conversations": [
   {"role": "user", "content": [
      {"type": "image", "text": "", "image": "/abs/path/coco/train2017/000000213364.jpg"},
      {"type": "text",  "text": "What is on the table?", "image": ""}]},
   {"role": "assistant", "content": [{"type": "text", "text": "A spray nozzle ...", "image": ""}]}]}
```

Text-only rows are the same minus the image part. Both `text` and `image` must be
present on **every** content part: `datasets` infers the Arrow schema from the
first rows it reads, and these corpora are ordered text-first, so a file whose
text parts lack `image` fails to cast tens of thousands of rows in.

```bash
python tools/build_mixed_text_vl_jsonl.py \
    --sharegpt  raw/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
    --llava-json raw/llava_v1_5_mix665k.json \
    --image-root raw/images --num-text 70000 --num-vl 70000 \
    --output-dir dataset --output-name mixed.jsonl
python tools/normalize_conversation_schema.py --in dataset/mixed.jsonl --out dataset/train.jsonl
```

`tools/encode_images_base64.py` inlines the images if you would rather move one
file than an image tree. The builder drops rows with short answers, missing
images, multi-image prompts, or a broken role order.

The published run trained on 132,943 rows from the Hub repo
`WindUpHanger/angelslim-smolvlm-eagle3-artifacts` (`data/smolvlm256m/*_path_b64.jsonl`,
base64 images, 6,997 held out), which is the same schema after normalisation.

### 2.2 Hidden states

The target runs once, offline, and its hidden states are stored. Training never
runs the target again except for branch distillation's single extra forward.

```bash
SPLIT=train DATASET_PATH=dataset/train.jsonl OUTPUT_DIR=dataset/hidden/train \
  bash scripts/speculative/smolvlm/generate_vlm_hidden_for_draft_model.sh
SPLIT=train DATASET_PATH=dataset/train.jsonl OUTPUT_DIR=dataset/hidden/train \
  bash scripts/speculative/smolvlm/extract_images.sh
```

Repeat for the eval split. What lands on disk:

```
dataset/hidden/train/
  rank_<0..7>/rows_<a>-<b>/data_<i>.ckpt    one sample: input_ids, loss_mask,
                                            hidden_states (1,N,K*H), target_hiddens (1,N,H)
  rank_<0..7>/rows_<a>-<b>/data_<i>.img     the original image bytes, for branch distillation
  vocab_mapping.pt                          d2t (draft->target), t2d (target->draft)
  aux_layers.json                           which target layers the K blocks are
```

Four things that bite:

- **`--nproc_per_node` must be the same for generation, `extract_images.sh` and
  training.** `(rank, idx)` is the sample's address; change the process count and
  the `.img` beside a `.ckpt` belongs to a different sample. The reference run
  used 8 throughout.
- **Generation runs through the *online* dataset builder**, because that is where
  SmolVLM's per-tile image-token expansion happens. Images must therefore be
  openable at the paths in the JSONL.
- **The default generation config writes ten layers**
  (`smolvlm-256m-eagle3-gen-10layer.json`), the union of what the baseline wants
  (`[1, 14, 26]`) and what the method wants (its nine). One corpus trains both
  arms; the trainer slices the layers its own config asks for by reading
  `aux_layers.json`. A layer a config asks for and the data does not have is
  named in the error rather than silently mis-sliced.
- **Size.** About 7 MB per sample at ten layers, so a 130k-row corpus is roughly
  900 GB. Generation is resumable: existing files are skipped.

`dataset/**/.map_cache/` pins the `datasets` map cache so restarts hit it. Delete
it after changing anything about preprocessing.

---

## 3. Training

Both arms, same data, one script:

```bash
# the method
TARGET_MODEL_NAME_OR_PATH=<local SmolVLM-256M snapshot or HuggingFaceTB/SmolVLM-256M-Instruct> \
TRAIN_HIDDEN_PATH=dataset/hidden/train EVAL_HIDDEN_PATH=dataset/hidden/eval \
OUTPUT_DIR=output/method NUM_TRAIN_EPOCHS=20 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_offline.sh

# the baseline
ARM=baseline OUTPUT_DIR=output/baseline ... bash scripts/speculative/smolvlm/train_eagle3_vlm_offline.sh
```

`ARM=method` (the default) selects the method config and exports the `VISTOKEN_*`
and `BRANCH_DISTILL_*` variables; `ARM=baseline` selects
`smolvlm-256m-eagle3.json` and exports neither. The log line at the top of a run
states which arm and which config it is using; `[vistoken] draft image rows /4
(pool)` and `[branch-distill] re-score images from ...` appear only in the method
arm.

Branch distillation re-scores substituted sequences with a **live** target, so
`BRANCH_DISTILL_TARGET` must be the same snapshot the `.ckpt` files were
generated from. A different copy of the same model silently pollutes the logit
difference the loss is built on. Pin a snapshot path, not a floating repo id, for
anything you intend to publish.

### What the config keys mean

`angelslim/compressor/speculative/train/configs/` ships three files: the
generation config, the baseline, and the method. The method adds, over the
baseline:

```json
"eagle_aux_injection_mode": "banded_mix_fc",
"aux_hidden_states_layer_ids": [1, 2, 8, 10, 18, 20, 23, 26, 28],
"eagle_aux_layer_bands": [[1, 2, 8, 10], [18, 20], [23, 26, 28]],
"eagle_aux_band_init_layer_ids": [1, 18, 23],
"fc_norm": true, "norm_output": true,

"branch_distill_loss_weight": 0.1,
"branch_distill_objective": "change",
"branch_distill_top_k": 1,
"branch_distill_target_top_k": 3,
"branch_distill_steps": 1,
"branch_distill_prob_ratio_threshold": 0.0
```

`eagle_aux_band_init_layer_ids` is where each band's softmax starts, so
initialisation reproduces a three-layer concatenation and the mixture is learned
away from it. `fc_norm` / `norm_output` are EAGLE 3.1: RMSNorm on the fusion-FC
inputs and on the draft output. `branch_distill_loss_weight: 0` disables the fork
entirely, and with it the extra teacher forward, which costs about 1.75x a plain
offline step. `branch_distill_top_k` must be 1 — any other value is rejected.

Image-row pooling is not in the config; it is environment, because it is applied
in the collator and both arms read the same `.ckpt` files:

```bash
VISTOKEN_REDUCE=pool      # none | pool | subset | drop
VISTOKEN_FACTOR=4         # must match the factor used at eval
VISTOKEN_KEEPPOS=1        # keep real absolute positions
VISTOKEN_IMAGE_TOKEN_ID=49190    # SmolVLM; required whenever REDUCE != none
```

### Where each piece is implemented

| | file | switch |
|---|---|---|
| banded aux mix | `angelslim/.../models/draft/llama_eagle3.py` — band params in `__init__`, the mix in `combine_hidden_states`, the learned weights dumped by `get_banded_aux_mix_weights` | `eagle_aux_injection_mode` |
| branch-change distillation | `angelslim/.../trainer/eagle3_trainer.py` (`_branch_loss`, the fork bookkeeping in the TTT loop); the offline teacher forward in `trainer/offline_eagle3_trainer.py` (`_OfflineBranchMixin`); the `.img` in `data/dataset_builder/branch_image_source.py` | `branch_distill_loss_weight` + `BRANCH_DISTILL_TARGET` |
| pooled draft image rows | `angelslim/.../data/data_utils.py` — `reduce_vlm_image_rows` and `_image_row_segments`, called from the VLM collator | `VISTOKEN_*` |
| image-token expansion | `angelslim/.../data/dataset_builder/online_dataset_builder.py` | — |
| reading the stored layers | `angelslim/.../data/dataset_builder/offline_dataset_builder.py` (`_aux_layer_columns`) | `aux_hidden_states_layer_ids` |
| the drafter at eval time | `HiViS/hivis/model/angelslim_drafter.py`, and `draft_image_rows.py` for pooling | `--draft_method angelslim_eagle3`, `--draft_image_reduce pool` |

A finished run has `model.safetensors` at the top of `OUTPUT_DIR`; a run still
going has only `checkpoint-*` subdirectories. The method arm also writes
`banded_aux_mix_weights.json`, the learned per-layer softmax weights.

Two notes on the trainer itself. `--warmup_ratio` is parsed but never forwarded
(transformers 5.x dropped `TrainingArguments.warmup_ratio`), so use
`--warmup_steps` if you want warmup. And training is not bit-reproducible
run to run — the same code and seed twice gives slightly different per-step
losses — so compare arms, never single runs.

---

## 4. Evaluation

One evaluator, one protocol, one analysis. Every arm goes through the same
`EaModel.eagenerate` loop — the same target forward, the same accept/reject — so
a number from one arm is comparable to a number from another. The only per-family
branch is which drafter gets built.

```bash
cd HiViS && export PYTHONPATH=$PWD
HIVIS_POOL_FAST=1 CUDA_VISIBLE_DEVICES=0 $EVAL_PY run_angelslim_eval.py \
    --draft_method angelslim_eagle3 \
    --base HuggingFaceTB/SmolVLM-256M-Instruct \
    --draft <checkpoint dir or HF repo id> \
    --draft_image_reduce pool --draft_image_factor 4 --draft_image_root <this package> \
    --dataset textvqa --n 80 --device cuda:0 \
    --out $OUT/textvqa__method.json
```

`scripts/speculative/smolvlm/run_eval.sh` wraps exactly that over the benchmark
set (`ARM=` names the arm, `DRAFT=`, `POOL_FACTOR=`, `OUT_DIR=`, `BENCHMARKS=`).

**The protocol is the defaults. Do not move them:** `n=80`, greedy,
`--max_new_tokens 500`, `--max_input_tokens 4000` (a prompt over the limit stops
the run rather than silently shrinking the sample), `--warmup 3`, `--seed 42`,
tree `total_token=60 depth=5 top_k=10`, one GPU, serial, with the other GPUs
idle — run arms in parallel and you are measuring system load. One run per arm
per benchmark; a 1–2% difference is not a result.

Rules that are not optional:

- **Run `naive` first, into the same directory.** It is the speedup denominator
  and the text every arm is filtered against: `--naive`, everything else
  unchanged, with any valid path for `--draft` (it is not used).
- **File names must be `${BENCH}__${ARM}.json`** — `tools/eval_table.py` finds
  files by that pattern. `run_eval.sh` appends `_r<n>` for the repeat number;
  the table reads either spelling.
- **`POOL_FACTOR` must be the factor the checkpoint was trained with.** A
  mismatch silently evaluates a drafter on an input distribution it never saw.
- **One `OUT_DIR` per target.** Arms with different targets cannot share a table;
  the denominator is that target's own autoregressive run.
- **Never compare speedup or wall-clock across machines.** They are wall-clock
  quantities and depend on the card and the load; every machine runs its own
  `naive`. τ *is* comparable across machines under greedy, provided the target,
  `n`, seed, `max_new_tokens`, depth and `total_token` all match.

### The arms

| arm | `--draft_method` | `--draft` |
|---|---|---|
| ours | `angelslim_eagle3` | the method checkpoint, plus `POOL_FACTOR=4` |
| EAGLE-3 baseline | `angelslim_eagle3` | the baseline checkpoint, no pooling |
| autoregressive | — | `--naive` |
| HiViS | `hivis` | `Irisssme/HiViS-Qwen2.5-VL-7B-Instruct` with `--base Qwen/Qwen2.5-VL-7B-Instruct`, or a local SmolVLM run |
| ViSpec | `vispec` | `JLKang/ViSpec-Qwen2.5-VL-7B-Instruct` (3B also published) |

A 7B target wants `--cache_len`
set to roughly prompt + `max_new_tokens`, otherwise the static KV cache is sized
off `max_position_embeddings` (128k on Qwen2.5-VL) and OOMs a 24GB card.

### The table

```bash
python tools/eval_table.py $OUT --arms method,baseline --baseline naive --filter exact
```

All thirteen benchmarks by default: `textvqa ScienceQA vqav2 mme mmvet MathVista
mmmu MMStar seedbench gqa DocVQA coco_caption omnidocbench`. A table over a
subset is a choice — say which subset and why.

Three quantities, three different meanings. Never mix them in one column:

| | |
|---|---|
| **τ** | per prompt, `mean(accept_length) + 1`, then averaged over prompts. The `+1` is the token the target verifies itself. This is the `tau` in the result JSON and the `tau` the table prints under `--filter none` — the same number by construction. Comparable across machines. |
| **speedup** | the arm's tokens/time over `naive`'s re-tokenised tokens/time, per prompt then averaged. The denominator counts the re-tokenised answer, which is systematically short, so the absolute value is inflated — only the arm-to-arm percentage means anything. |
| **wall clock** | per prompt, `naive` time over arm time, then averaged. The most conservative reading. `eval_table.py` does not print it; compute it from the JSONs. |

Each result JSON carries `tau` (the above), `tau_round_pooled` (HiViS pools every
round instead, so a long answer outweighs a short one — its published numbers use
this), `tok_per_s`, `rounds`, `acceptance_rates`, the per-prompt records and the
full `cfg`. Quote one τ convention per table and say which. The two can differ by
a few percent under greedy and by tens of percent under sampling, and by a
different amount per arm, so they are not interchangeable.

Result files written before the headline was renamed store the round-pooled value
under `tau`. `tools/convert_eval_json.py` re-lays a directory of them out into the
current shape without re-measuring anything — the old `tau` becomes
`tau_round_pooled` unchanged, the new `tau` is derived from the same per-prompt
records, and a table built from converted files is identical to one built from the
originals:

```bash
python tools/convert_eval_json.py <results dir> --out <dir>   # or --in-place
```

`--filter` decides which prompts count, and it is the one knob that changes a
table without changing a single run. **State it next to any number.**

| `--filter` | keeps a prompt when | use it for |
|---|---|---|
| `exact` | `naive` and every arm produced byte-identical text | the default; asks only that the answers agree |
| `length10` | every arm's token count is within ±10% of `naive`'s | long-answer benchmarks, where `exact` empties out |
| `none` | always | arms that are not meant to agree, such as an image ablation |

Two facts to know before choosing. `exact` compares the **decoded answer** (cut
at the first EOS, special tokens stripped) while `length10` compares the
generation loop's `tokens`, which includes whatever was written in the same
accepted block as the EOS — so identical answers with different token counts are
normal, not a bug. And `tokens` is an integer, so ±10% is "may differ by
`floor(baseline/10)`": on a benchmark whose answers are under ten tokens the
tolerance is zero, which makes `length10` the *stricter* filter there, and it
penalises whichever arm overruns more.

### Benchmark data

Everything except two benchmarks needs nothing: the Hub-hosted ones download
their split on first use, and `gqa`, `textvqa`, `seedbench`, `mme` and `mmvet`
keep their metadata in this package and pull only the sampled rows' images out of
the Hub parquet, writing them where the loaders expect. `vqav2` downloads a large
split; if you have a saved 500-row prefix, point `VQAV2_SLICE` at it, otherwise
the loader fetches `test[:n]`, which is the same rows in the same order.

The image fetchers skip files that already exist, so a wrong image on disk stays
wrong: delete the directory and let it re-pull.

`HiViS/` is vendored as-is, down to comments referring to tools that are not part
of this package. That is deliberate: it is the code every published number came
from, and editing an evaluator is the one change that can move a number without
raising an error.

---

## 5. Reference numbers

20 passes over the corpus, both arms, same data, same evaluator, twelve
benchmarks, `--filter exact`:

| | τ | |
|---|---:|---|
| EAGLE-3 baseline | 3.450 | |
| this method | **4.077** | +18.2% |

Eleven of the twelve improve; `coco_caption` is level. Under a depth sweep the
advantage holds and grows slightly with depth (+5.2% at depth 5, +6.3% at depth
9, measured on the acceptance length with each prompt's last round dropped).

Those two checkpoints are not in this package. On the machine this was built on
they are:

```
output/vistoken/full_20ep/checkpoint-332340                   the baseline
output/vistoken/final_banded_branch_pool4x/checkpoint-332340  the method (POOL_FACTOR=4)
```

---

## 6. Scope

The target is never trained, and it always sees the full image. That is not a
detail: reducing the target's input changes what the drafter is trying to
predict, and the acceptance numbers stop being comparable across arms.

Pooling is measured on SmolVLM, whose 832 rows include a thumbnail tile that
duplicates the crops, so some of the headroom is this target's own redundancy.
Qwen2.5-VL has no tiling and no thumbnail — one grid, every token distinct — and
pooling does not run there as written: the row segmentation groups along the
sequence in one dimension, which crosses grid rows, and the Qwen drafter uses
M-RoPE, so it needs 2D pooling over `image_grid_thw` and three-dimensional
positions.

`--draft_image_reduce subset` (evenly spaced rows instead of averaged ones) is
inherited from the ablations that led here and is not part of the method. So is
`--draft_image_control`, which corrupts the drafter's image rows without
changing their count, ids or positions — the control for whether the drafter
reads the image at all.
