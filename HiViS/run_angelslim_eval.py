"""Acceptance length + throughput for a speculative drafter, measured inside
HiViS's own EaModel loop.

Works for both drafter families, so a paper table can be filled from one
backend:

    --draft_method angelslim_eagle3   our AngelSlim EAGLE-3 drafts (SmolVLM)
    --draft_method hivis|vispec     HiViS's / ViSpec's own checkpoints

and for both benchmark families (see hivis/evaluation/benchmark_data.py):
our vLLM-matched rows (`omnidocbench`, `mmmu_history`) and HiViS's eleven.
Rows and prompts come from the shared adapter, so "their model on our
benchmark" and "our model on their benchmark" are the same code path.

Every mode -- tree, chain, --naive -- runs through the same target backbone and
the same generation loop, so tok/s ratios across runs are a real speedup and not
a comparison of kernel stacks. See README_ANGELSLIM.md.

    # our drafter, our benchmark
    python run_angelslim_eval.py --draft <ckpt> --n 40 --max_new_tokens 1024 \
        --total_token 60 --depth 5 --top_k 10           # tree
        --total_token 5  --depth 4 --top_k 1            # chain (K=4)
        --naive                                         # AR baseline

    # their drafter, our benchmark
    python run_angelslim_eval.py --draft_method hivis \
        --base Qwen/Qwen2.5-VL-7B-Instruct \
        --draft Irisssme/HiViS-Qwen2.5-VL-7B-Instruct --dataset omnidocbench
"""
import argparse, json, random, time
import numpy as np
import torch
from hivis.model.model_hivis import EaModel
from hivis.evaluation.benchmark_data import (
    PROMPT_STYLES, load_benchmark, prepare_inputs, row_message_and_image,
    set_prompt_style, supported_benchmarks,
)


def setup_seed(seed):
    """Verbatim from hivis/evaluation/ge_hivis_answer.py."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def truncate_list(values, stop_token_id):
    """Verbatim from hivis/evaluation/ge_hivis_answer.py."""
    if stop_token_id not in values:
        return values
    return values[: values.index(stop_token_id) + 1]

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="HuggingFaceTB/SmolVLM-256M-Instruct",
                help="target VLM; must match the drafter it was trained against")
ap.add_argument("--draft", required=True, help="drafter checkpoint dir or HF repo id")
ap.add_argument("--draft_method", default="angelslim_eagle3",
                choices=["angelslim_eagle3", "hivis", "vispec"])
ap.add_argument("--dataset", default="omnidocbench", choices=supported_benchmarks())
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--start", type=int, default=0,
                help="measure rows [start, n) only, so a sample can be extended "
                     "past an earlier --n without re-running its prompts")
# Protocol defaults are HiViS's own (hivis/evaluation/ge_hivis_answer.py
# build_parser), so a number from here is comparable to one from there. The
# only deliberate departure is --device: HiViS uses device_map="auto".
ap.add_argument("--max_new_tokens", type=int, default=500,
                help="HiViS's --max-new-token default.")
ap.add_argument("--max_input_tokens", type=int, default=4000,
                help="skip a prompt longer than this, as HiViS does. Its 80 "
                     "samples are the first 80 that FIT, not the first 80 rows.")
ap.add_argument("--warmup", type=int, default=3,
                help="untimed prompts before the measured loop. HiViS runs 3; "
                     "without them the first prompt carries CUDA init (472ms "
                     "against 84-100ms steady state, measured on SmolVLM-256M).")
ap.add_argument("--seed", type=int, default=42, help="HiViS's --seed default.")
ap.add_argument("--prompt_style", default="raw", choices=list(PROMPT_STYLES),
                help="answer_then_describe re-wraps the question as 'Answer "
                     "this question: ... Then describe the image in detail to "
                     "justify your answer.', which lengthens the output 6-10x. "
                     "Most of these benchmarks answer in a handful of tokens, "
                     "and a per-prompt cost dominates there -- a speedup "
                     "measured on a 5-token answer says almost nothing about "
                     "the drafter. Changes the task, so accuracy under it is "
                     "not comparable with published numbers. omnidocbench has "
                     "no question to re-wrap and mmmu_history already uses "
                     "this wording; both ignore the flag.")
ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
                help="bfloat16, matching the AngelSlim side and every arm already "
                     "measured. NOT aligned to HiViS's evaluators, which load "
                     "float16 -- HiViS itself trains in bf16 and model_hivis.py "
                     "casts pixel_values to bf16 regardless of model dtype, so "
                     "its own fp16 eval is the inconsistent one.")
ap.add_argument("--total_token", type=int, default=60)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--top_k", type=int, default=10,
                help="draft tree width -- NOT the sampling top-k.")
ap.add_argument("--temperature", type=float, default=0.0,
                help="0 = greedy (the gold-standard protocol). >0 samples, "
                     "and then no arm reproduces another byte for byte, so "
                     "only --filter none is meaningful downstream.")
ap.add_argument("--top_p", type=float, default=0.0,
                help="sampling nucleus; 0 disables it. Ignored when "
                     "--temperature is 0.")
ap.add_argument("--sample_top_k", type=int, default=0,
                help="sampling top-k; 0 disables it. Ignored when "
                     "--temperature is 0.")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max_pixels", type=int, default=None,
                help="cap the image resolution the processor emits, in pixels "
                     "(Qwen2.5-VL: ~1 visual token per 28x28 px). Its default "
                     "turns one OmniDocBench page into ~16.3k tokens, and this "
                     "harness's attention is eager/quadratic, so a 7B target "
                     "OOMs on a 24GB card. 1605632 gives ~2k tokens. Changes "
                     "what the model sees -- hold it fixed across compared runs.")
ap.add_argument("--cache_len", type=int, default=None,
                help="cap the preallocated static KV cache (default: the model's "
                     "max_position_embeddings). Qwen2.5-VL's 128k default costs "
                     "~7.3GB and OOMs a 7B target on a 24GB card; 4096 is ample "
                     "for these prompts. Must be >= prompt + max_new_tokens.")
# Drafter-side image-row reduction. The TARGET always prefills the full image;
# these only change what the drafter is handed, and must match how the
# checkpoint was trained or the number measures the mismatch instead.
ap.add_argument("--draft_image_reduce", choices=["pool", "subset"], default=None,
                help="reduce the full forward's image rows by "
                     "--draft_image_factor, the way the checkpoint was trained.")
ap.add_argument("--draft_image_control",
                choices=["zero", "shuffle", "random", "wrong", "drop"], default=None,
                help="Corrupt the drafter's image rows without changing their "
                     "count, ids or positions -- a control for whether the "
                     "drafter reads the image at all.")
ap.add_argument("--draft_image_control_seed", type=int, default=0)
ap.add_argument("--draft_image_factor", type=int, default=4,
                help="pool/subset only; matches VISTOKEN_FACTOR at training time.")
ap.add_argument("--draft_image_token_id", type=int, default=None,
                help="defaults to the target config's image token id.")
ap.add_argument("--draft_image_root", default=None,
                help="checkout holding the training-side reduction code "
                     "(reduce_vlm_image_rows). Separate from the drafter's own "
                     "root; defaults to $ANGELSLIM_TRAIN_ROOT.")
ap.add_argument("--out", default=None)
ap.add_argument("--naive", action="store_true",
                help="autoregressive baseline in the same harness (speedup denominator)")
a = ap.parse_args()
setup_seed(a.seed)
set_prompt_style(a.prompt_style)
DTYPE = getattr(torch, a.dtype)

model = EaModel.from_pretrained(
    base_model_path=a.base, ea_model_path=a.draft,
    total_token=a.total_token, depth=a.depth, top_k=a.top_k,
    draft_method=a.draft_method,
    torch_dtype=DTYPE, low_cpu_mem_usage=True, device_map=a.device,
)
model.eval()
model.cache_max_len = a.cache_len
if a.max_pixels is not None:
    image_processor = getattr(model.processor, "image_processor", None)
    if image_processor is None:
        raise SystemExit("--max_pixels: this processor has no image_processor")
    image_processor.max_pixels = a.max_pixels
    if isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["longest_edge"] = a.max_pixels
    print("max_pixels ->", a.max_pixels)
print("drafter:", type(model.ea_layer).__name__,
      "| method:", a.draft_method, "| aux layers:", getattr(model, "aux_layer_ids", None))

reducer = None
if a.draft_image_reduce is not None:
    from hivis.model.draft_image_rows import DraftImageReducer
    from hivis.model.utils_hivis import _target_image_token_id

    img_tok = a.draft_image_token_id or _target_image_token_id(model)
    reducer = DraftImageReducer(
        a.draft_image_reduce, img_tok, factor=a.draft_image_factor,
        control=a.draft_image_control, control_seed=a.draft_image_control_seed,
        angelslim_root=a.draft_image_root,
    )
    model.draft_image_reducer = reducer
    print("draft image rows: %s (factor=%s image_token_id=%d)"
          % (a.draft_image_reduce, a.draft_image_factor, img_tok))

dataset = load_benchmark(a.dataset, sample_count=a.n)
tokenizer = getattr(model.processor, "tokenizer", None) or model.processor


def row_metadata(row):
    """Everything about a benchmark row that is worth keeping beside the
    generation -- the question and the gold answer, so two runs can be diffed
    on what they actually produced, not only on tau. Images and any other
    non-JSON value are dropped; the row schema differs per benchmark, so this
    keeps whatever scalar/list fields there are rather than naming them."""
    keep = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            keep[k] = v
        elif isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
            keep[k] = v
    return keep


def generate(inputs):
    """One generation, speculative or not, returning a uniform tuple."""
    if a.naive:
        out_ids, new_token, idx = model.naivegenerate(
            inputs, temperature=a.temperature, top_p=a.top_p,
            top_k=a.sample_top_k, max_new_tokens=a.max_new_tokens, log=True)[:3]
        return out_ids, new_token, idx, []
    return model.eagenerate(
        inputs, temperature=a.temperature, top_p=a.top_p,
        top_k=a.sample_top_k, max_new_tokens=a.max_new_tokens, log=True)


def hivis_text(out_ids, prompt_len):
    """Decode the way hivis/evaluation does: clamp ids past the vocab, cut at
    the first EOS, then decode with its spacing flags. The old path here
    decoded the whole tail with default flags, which keeps whatever the run
    emitted after EOS -- and a speculative round emits several tokens at once,
    so it routinely overshoots."""
    gen = out_ids[0][prompt_len:].clone()
    gen[gen > tokenizer.vocab_size] = 0
    ids = truncate_list(gen.tolist(), tokenizer.eos_token_id)
    return ids, tokenizer.decode(
        ids, skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True)


# Untimed warm-up. HiViS's two evaluators disagree on how to do it and this
# reproduces each rather than tidying the difference away: the speculative side
# walks prompts 0..2 (ge_hivis_answer.py), the baseline runs prompt 0 three
# times (ge_baseline_answer_hivis.py). Both re-seed before every call and both
# prepare with truncation=True, which the measured loop does not.
for w in range(min(a.warmup, len(dataset)) if not a.naive else a.warmup):
    warm_index = 0 if a.naive else w
    torch.manual_seed(0)
    warm_inputs = prepare_inputs(model, dataset, warm_index, a.dataset, truncation=True)
    generate(warm_inputs)
if a.warmup:
    print("Warmup done")

rows, all_acc, skipped = [], [], []
with torch.inference_mode():
    for i in range(a.start, len(dataset)):
        inputs = prepare_inputs(model, dataset, i, a.dataset)
        prompt_len = inputs["input_ids"].shape[1]
        if prompt_len > a.max_input_tokens:
            # HiViS's wording, so its logs and these are greppable the same way.
            print("Skipping candidate %d: input length %d exceeds %d"
                  % (i, prompt_len, a.max_input_tokens), flush=True)
            skipped.append(i)
            continue
        if a.cache_len is not None and prompt_len + a.max_new_tokens > a.cache_len:
            # The cache is preallocated, so overflowing it surfaces deep inside
            # KVCache as "start (0) + length (N) exceeds dimension size". Say what
            # is actually wrong. Qwen2.5-VL does not downsample image tokens the way
            # SmolVLM does -- one OmniDocBench page is ~16k visual tokens.
            raise SystemExit(
                "--cache_len %d is too small for prompt %d (row %d) + --max_new_tokens %d. "
                "Use --cache_len %d or more."
                % (a.cache_len, prompt_len, i, a.max_new_tokens,
                   prompt_len + a.max_new_tokens)
            )
        torch.cuda.synchronize(); t0 = time.time()
        out_ids, new_token, idx, acc = generate(inputs)
        torch.cuda.synchronize(); dt = time.time() - t0
        # `accept_length` from evaluate_posterior counts the DRAFTED tokens
        # that survived, so a round emits accept_length + 1. One convention
        # here and at the top of the file: the +1 is always included.
        tau = sum(x + 1 for x in acc) / len(acc) if acc else 1.0
        all_acc += acc
        decode_ids, text = hivis_text(out_ids, prompt_len)
        rows.append({
            "index": i,
            "tokens": int(new_token), "rounds": len(acc), "tau": tau,
            "accept_lengths": [int(x) for x in acc],
            "time": dt, "prompt_tokens": int(prompt_len),
            # speed.py counts the BASELINE's tokens by re-tokenising its text
            # rather than by the generation loop. Stored per prompt so that
            # convention can be reproduced from this file without the model.
            "retok_tokens": len(tokenizer(text).input_ids) - 1,
            "generated_text": text,
            "row": row_metadata(dataset[i]),
        })
        print("  [%d] tokens=%d rounds=%d tau=%.3f  %.2fs"
              % (i, new_token, len(acc), tau, dt))

if reducer is not None and reducer.control_report():
    print("\n" + reducer.control_report())
if skipped:
    # HiViS raises when it cannot reach --target-samples. Same here: a table
    # whose cells silently rest on different prompt counts is worse than a stop.
    raise SystemExit(
        "%d of %d prompts exceeded --max_input_tokens %d (rows %s). HiViS stops "
        "here rather than report a short sample; raise --max_input_tokens or "
        "lower --max_pixels if this is expected."
        % (len(skipped), len(dataset), a.max_input_tokens, skipped))

# THE headline number, and the only one this file reports as `tau`: one value
# per prompt, then the mean over prompts. It is what tools/eval_table.py prints
# for this arm under --filter none, so the file and the table agree by
# construction. Subtract 1 for HiViS's convention (drafted tokens only).
tau = sum(1 + sum(r["accept_lengths"]) / len(r["accept_lengths"])
          if r["accept_lengths"] else 1.0 for r in rows) / len(rows)
# HiViS pools every round instead, so a long answer outweighs a short one.
# Kept under its own name because HiViS's published numbers use it -- never
# quote the two in one column.
tau_round_pooled = sum(x + 1 for x in all_acc) / len(all_acc) if all_acc else 1.0
tok_per_s = sum(r["tokens"] for r in rows) / sum(r["time"] for r in rows)
print("\n%s | %s | total_token=%d depth=%d top_k=%d | %s"
      % (a.draft_method, a.dataset, a.total_token, a.depth, a.top_k, a.dtype))
print("tau = %.4f  (per-prompt mean, matches eval_table.py --filter none)" % tau)
print("     %.4f  (round-pooled, HiViS's convention)  over %d rounds"
      % (tau_round_pooled, len(all_acc)))
print("%.2f tok/s" % tok_per_s)
# Acceptance rate at each speculative position, the way vLLM's
# `acceptance_rates` reports it: fraction of rounds in which at least k drafted
# tokens were accepted, k = 1..max. A tau can be reached either by a short
# chain that almost always lands or a long one that usually breaks at depth 1,
# and only this curve tells those apart.
max_k = max(all_acc) if all_acc else 0
acceptance_rates = [
    sum(1 for x in all_acc if x >= k) / len(all_acc) for k in range(1, max_k + 1)
] if all_acc else []

if a.out:
    json.dump({
        "tau": tau,
        "tau_convention": "mean over prompts of (accepted draft tokens + 1); "
                          "matches tools/eval_table.py --filter none",
        "tau_round_pooled": tau_round_pooled,
        "tok_per_s": tok_per_s,
        "rounds": len(all_acc),
        "acceptance_rates": acceptance_rates,
        "per_prompt": rows,
        "cfg": vars(a),
    }, open(a.out, "w"), indent=2, ensure_ascii=False)
