"""Run an AngelSlim EAGLE-3 draft checkpoint inside HiViS's EaModel.

HiViS's EaModel drives its drafter through EAGLE-2's interface -- init_tree(),
reset_kv(), topK_genrate(hidden_states, input_ids, ...) returning
(draft_tokens, retrieve_indices, tree_mask, tree_position_ids). AngelSlim
already ships a drafter with exactly that interface in
``angelslim/compressor/speculative/inference/models/eagle3/draft``, so this
module reuses it rather than reimplementing tree decoding.

Two things stand between that class and our checkpoint:

1. ``import angelslim`` executes angelslim/__init__.py, which reaches
   qat/modules/quantizer.py and evaluates a py3.10-only ``X | None`` annotation
   at class-body time. HiViS runs on py3.9, so importing normally raises
   TypeError. The draft modules themselves are py3.9-clean, so
   ``load_angelslim_draft_module`` registers empty stand-in packages and
   imports the leaf module directly, skipping every __init__ on the way.

2. Our drafts use ``eagle_aux_injection_mode: banded_mix_fc``: the target's 9
   aux streams are first collapsed to one stream per band by a learned softmax
   (band{i}_mix_logits), and only those band streams reach the stock EAGLE-3.1
   fc_norm + nH->H fusion FC. The stock drafter sizes fc and fc_norm from
   len(aux_hidden_states_layer_ids) == 9, so it builds Linear(9H, H) and 9
   norms where the checkpoint holds Linear(3H, H) and 3 norms.

(2) is handled without touching the stock forward: the band mix runs in
topK_genrate before delegating upward, and the config handed to the base class
advertises the BAND count as its aux-stream count. The base forward already
applies fc_norm+fc only when the incoming hidden is wider than the embedding
(i.e. only on the first draft step, exactly as training does) and feeds
_next_step_hidden -- post-norm under EAGLE 3.1 ``norm_output`` -- to later
steps, so nothing downstream needs to change.
"""

import json
import os
import sys
import types

import torch

from .phase_timing import phase as _ph
from torch import nn

_ANGELSLIM_ROOT_ENV = "ANGELSLIM_ROOT"
_DRAFT_PKG = "angelslim.compressor.speculative.inference.models.eagle3.draft"


def _default_root():
    """Walk up from this file looking for the repo that contains `angelslim/`.

    HiViS is vendored inside the AngelSlim checkout, so the repo root is a
    couple of levels up -- but not at a fixed absolute path, since this runs on
    several machines. Fall back to the cwd's repo if the layout ever changes.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    for candidate in (here, os.getcwd()):
        while True:
            if os.path.isdir(os.path.join(candidate, "angelslim")):
                return candidate
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent
    return here


def load_angelslim_draft_module(angelslim_root=None):
    """Import the AngelSlim inference drafter without running angelslim/__init__."""
    import importlib

    root = angelslim_root or os.environ.get(_ANGELSLIM_ROOT_ENV) or _default_root()
    if not os.path.isdir(os.path.join(root, "angelslim")):
        raise RuntimeError(
            "Could not find an angelslim checkout at %r. Pass angelslim_root= or set %s."
            % (root, _ANGELSLIM_ROOT_ENV)
        )
    if root not in sys.path:
        sys.path.insert(0, root)

    parts = _DRAFT_PKG.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        stub.__path__ = [os.path.join(root, *parts[:i])]
        sys.modules[name] = stub
    return importlib.import_module(_DRAFT_PKG + ".llama3_eagle3")


class DrafterConfig(object):
    """Plain attribute bag for the drafter.

    Deliberately not transformers' LlamaConfig: HiViS pins transformers 4.54
    while the checkpoints were written by 5.x, and the two disagree on where
    RoPE settings live. Translating explicitly here keeps that disagreement
    from silently changing the RoPE base -- the checkpoint carries
    ``rope_parameters: {"rope_theta": 100000.0}`` (5.x), while the drafter
    reads ``config.rope_theta`` and would otherwise fall back to 10000.
    """

    def __init__(self, raw, num_aux_streams):
        self.hidden_size = raw["hidden_size"]
        self.intermediate_size = raw["intermediate_size"]
        self.num_attention_heads = raw["num_attention_heads"]
        self.num_key_value_heads = raw["num_key_value_heads"]
        self.head_dim = raw.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = raw.get("hidden_act", "silu")
        # Read by the drafter's tensor-parallel branch, which is inert at tp=1.
        self.pretraining_tp = raw.get("pretraining_tp", 1)
        self.rms_norm_eps = raw.get("rms_norm_eps", 1e-5)
        self.max_position_embeddings = raw.get("max_position_embeddings", 8192)
        self.vocab_size = raw["vocab_size"]
        self.draft_vocab_size = raw.get("draft_vocab_size", self.vocab_size)
        self.pad_token_id = raw.get("pad_token_id", 0)
        self.target_hidden_size = raw.get("target_hidden_size", self.hidden_size)

        rope = raw.get("rope_parameters") or {}
        self.rope_theta = raw.get("rope_theta", rope.get("rope_theta", 10000.0))
        scaling = raw.get("rope_scaling")
        if scaling is None:
            rope_type = rope.get("rope_type", "default")
            scaling = None if rope_type in (None, "default") else dict(rope)
        self.rope_scaling = scaling

        # EAGLE 3.1 switches.
        self.fc_norm = bool(raw.get("fc_norm", False))
        self.norm_output = bool(raw.get("norm_output", False))
        # The base class sizes fc / fc_norm off this list's LENGTH. Under
        # banded_mix_fc the streams reaching the FC are the BANDS, not the raw
        # aux layers, so advertise the band count here. The real 9-layer list
        # stays on the wrapper as `aux_hidden_states_layer_ids`.
        self.aux_hidden_states_layer_ids = list(range(num_aux_streams))


class HiViSInterfaceMixin(object):
    """Adapts AngelSlim's drafter to what HiViS's tree loop expects.

    Two adjustments, both at the topK_genrate boundary:
      * arity -- AngelSlim returns a 5th value (early_stop_signal); HiViS
        unpacks exactly four.
      * banded_mix_fc -- collapse the raw aux streams to one per band before
        the stock fc_norm + FC sees them (no-op when the checkpoint has no
        bands).
    """

    def init_banded_mix(self, bands):
        self.aux_layer_bands = tuple(tuple(b) for b in bands)
        for i, band in enumerate(self.aux_layer_bands):
            self.register_parameter(
                "band%d_mix_logits" % i,
                nn.Parameter(torch.zeros(len(band))),
            )

    def mix_aux(self, hidden_states):
        """Collapse the 9 raw aux streams to one per band via a learned softmax.

        Pass-through when the width is not the raw aux concat: later draft
        steps hand back a single H-wide hidden, which must not be re-mixed.
        """
        bands = getattr(self, "aux_layer_bands", ())
        if not bands:
            return hidden_states
        h = self.config.target_hidden_size
        n_raw = sum(len(b) for b in bands)
        if hidden_states.shape[-1] != h * n_raw:
            return hidden_states
        chunks = hidden_states.split(h, dim=-1)
        mixed, off = [], 0
        for i, band in enumerate(bands):
            stack = torch.stack(chunks[off : off + len(band)], dim=-1)
            w = torch.softmax(getattr(self, "band%d_mix_logits" % i).float(), dim=0)
            mixed.append(torch.matmul(stack, w.to(stack.dtype)))
            off += len(band)
        return torch.cat(mixed, dim=-1)

    # ---- absolute draft positions -------------------------------------
    # A drafter fed fewer image rows than the target saw was trained on
    # position ids that are ABSOLUTE and therefore not contiguous: the rows the
    # reduction removed leave a gap, and every text row after the image keeps
    # the position the target computed it at. The stock drafter infers
    # positions from sequence length, which silently closes that gap and shifts
    # all post-image text down by (rows removed) -- measuring how the drafter
    # copes with unseen positions rather than how well the reduction works.
    #
    # `set_draft_positions` is called once per prompt, from initialize_tree.
    # Rounds after the first re-enter topK_genrate without positions, so they
    # are extended from the stored prefill array: past the image the mapping is
    # a constant offset, absolute = index + gap.

    def _widen_rotary(self, min_len):
        """Let RoPE be indexed past the sequence length.

        LlamaRotaryEmbedding.forward returns cos_cached[:, :, :seq_len] and
        apply_rotary_pos_emb then does cos[position_ids]. With contiguous
        positions the largest index is seq_len-1 and the truncation is
        invisible; with absolute positions the sequence is SHORTER than its
        highest position, so indexing runs off the end of the table -- as a
        device-side assert, not a Python error. Hand back the whole table
        instead: for contiguous positions this changes nothing, because the
        rows actually indexed are the same rows.
        """
        # Patching the forward is a one-time job, and the table only ever has
        # to grow. Re-walking 21 modules for every prompt to re-assert the same
        # min_len is pure overhead on the first-tree path, which only the
        # reduced arms take.
        import os as _os
        if (_os.environ.get("HIVIS_POOL_SLOWPREP", "0") in ("0", "", "false")
                and min_len <= getattr(self, "_widen_done_upto", 0)):
            return
        self._widen_done_upto = min_len
        for module in self.modules():
            rot = getattr(module, "rotary_emb", None)
            if rot is None:
                continue
            rot._hivis_min_len = max(int(min_len), getattr(rot, "_hivis_min_len", 0))
            if getattr(rot, "_hivis_widened", False):
                continue

            def forward(x, seq_len=None, _rot=rot):
                need = max(int(seq_len or 0), int(getattr(_rot, "_hivis_min_len", 0)))
                if need > _rot.max_seq_len_cached:
                    _rot._set_cos_sin_cache(seq_len=need, device=x.device, dtype=x.dtype)
                return (_rot.cos_cached.to(dtype=x.dtype),
                        _rot.sin_cached.to(dtype=x.dtype))

            rot.forward = forward
            rot._hivis_widened = True

    def set_draft_positions(self, position_ids):
        with _ph("ft_setpos"):
            return self._set_draft_positions(position_ids)

    def _set_draft_positions(self, position_ids):
        if position_ids is None:
            self._draft_positions = None
            self._position_gap = 0
            return
        # `_hivis_last` is the last absolute position, read by the reducer
        # BEFORE the target's prefill was queued. Without it these three reads
        # each wait for that prefill to drain, once per prompt, for a scalar
        # that was known before it started.
        last = getattr(position_ids, "_hivis_last", None)
        pos = position_ids.reshape(-1).long()
        if last is None:
            last = int(pos[-1])
        self._draft_positions = pos
        # The same scalar, kept host-side. _positions_for needs the LAST
        # absolute position of whatever slice it hands back, and reading it
        # off the device is a synchronisation queued behind the draft prefill.
        self._draft_last = int(last)
        self._position_gap = last + 1 - int(pos.numel())
        # + the tree depth, which continues past the last prefill position.
        self._widen_rotary(last + 1 + int(getattr(self, "depth", 8)) + 2)

    def _positions_for(self, n, device):
        base = getattr(self, "_draft_positions", None)
        if base is None:
            return None
        base = base.to(device)
        if n <= base.numel():
            # This is the FIRST tree, and it is the branch that runs: the
            # drafter drops the leading row (`input_ids = input_ids[:, 1:]` in
            # base_model.topK_genrate) and appends the sampled token, so n
            # comes back exactly equal to the stored prefill length. The
            # earlier code left _last_position None here, and
            # _get_initial_hidden then read pos[0, -1] off the device -- one
            # synchronisation, queued directly behind the target prefill and
            # the draft prefill, once per prompt. It cost 6 ms of GPU idle,
            # which is most of what the pool arm was losing.
            # HIVIS_POS_SYNC=1 restores the read, to A/B it in one session.
            if os.environ.get("HIVIS_POS_SYNC", "0") not in ("0", "", "false"):
                self._last_position = None
            else:
                self._last_position = (self._draft_last if n == base.numel()
                                       else None)
            return base[:n][None]
        # Past the stored prefill the mapping is a constant offset, so the last
        # absolute position is arithmetic on Python ints. Keeping it lets
        # _get_initial_hidden set initial_position_id without reading back a
        # device tensor right after the draft prefill was queued.
        self._last_position = int(n - 1 + self._position_gap)
        grown = torch.arange(base.numel(), n, device=device) + self._position_gap
        return torch.cat([base, grown])[None]

    def _get_initial_hidden(self, hidden_states, input_ids, inputs_embeds=None):
        with _ph("ft_draft_prefill"):
            return self._get_initial_hidden_inner(hidden_states, input_ids, inputs_embeds)

    def _get_initial_hidden_inner(self, hidden_states, input_ids, inputs_embeds=None):
        """Prefill the draft cache on absolute positions when they were set.

        Also corrects `initial_position_id`, which the caller has just set to
        the sequence length: the tree levels continue from the last REAL
        position, which is further along than the row count.
        """
        pos = self._positions_for(input_ids.shape[1], input_ids.device)
        if pos is None:
            return super(HiViSInterfaceMixin, self)._get_initial_hidden(
                hidden_states, input_ids, inputs_embeds
            )
        if getattr(self, "stable_kv", None) is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            outputs = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                inputs_embeds=(inputs_embeds[:, kv_len:] if inputs_embeds is not None else None),
                past_key_values=self.stable_kv,
                position_ids=pos[:, kv_len:],
                use_cache=True,
            )
        else:
            outputs = self(
                hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=pos,
                use_cache=True,
            )
        out_hidden, past_key_values, early_stop_signal = outputs
        _lp = getattr(self, "_last_position", None)
        self.initial_position_id = (_lp if _lp is not None else int(pos[0, -1])) + 1
        return out_hidden[:, -1], past_key_values, early_stop_signal

    def topK_genrate(self, hidden_states, input_ids, inputs_embeds=None,
                     logits_processor=None, position_ids=None):
        if position_ids is not None:
            self.set_draft_positions(position_ids)
        with _ph("ft_mix_aux"):
            _mixed = self.mix_aux(hidden_states)
        with _ph("ft_tree_total"):
            out = super(HiViSInterfaceMixin, self).topK_genrate(
                _mixed, input_ids, inputs_embeds, logits_processor
            )
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = out[:4]
        # AngelSlim's _finalize_results builds the tree with torch.eye() and no
        # device, then returns tree_position_ids on the HOST. HiViS's own
        # drafters end topK_genrate with
        #     tree_position_ids = tree_position_ids.to(hidden_states.device)
        # (cnets_hivis.py:1000, cnets_eagle.py:817, cnets_vispec.py:1263); the
        # line was lost when that code was refactored into _build_tree_mask /
        # _generate_retrieve_indices / _finalize_results. Without it the target
        # does cos[position_ids] and sin[position_ids] with a CPU index in all
        # 30 layers -- 60 implicit synchronisations per target forward, which
        # measures -1.6 ms per ROUND. Put the line back.
        #
        # tree_mask stays on the host, which is also what HiViS does, and
        # deliberately: moving it makes `combined[...][tree_mask == 0] = ...`
        # in _prepare_decoder_attention_mask a device-side boolean index_put_,
        # and that runs nonzero() internally -- one synchronisation traded for
        # another, plus two copies. Measured +0.46 ms/prompt, i.e. worse.
        # HIVIS_TREE_CPU=1 restores AngelSlim's behaviour (move neither),
        # "mask"/"both" are the variants that were measured and rejected.
        _tc = os.environ.get("HIVIS_TREE_CPU", "0")
        if _tc in ("0", "", "false", "pos", "both", "mask"):
            dev = _mixed.device
            if (_tc in ("0", "", "false", "pos", "both")
                    and tree_position_ids is not None
                    and tree_position_ids.device != dev):
                tree_position_ids = tree_position_ids.to(dev, non_blocking=True)
            if (_tc in ("mask", "both") and tree_mask is not None
                    and tree_mask.device != dev):
                tree_mask = tree_mask.to(dev, non_blocking=True)
        if len(out) > 4 and out[4] is not None:
            raise NotImplementedError(
                "early-stop drafts are not wired into HiViS's tree loop "
                "(early_stop_signal was not None)"
            )
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def build_drafter(checkpoint_dir, total_tokens=60, depth=5, top_k=10,
                  threshold=1.0, angelslim_root=None, dtype=torch.bfloat16,
                  device=None):
    """Load an AngelSlim EAGLE-3 checkpoint as a HiViS-compatible drafter."""
    mod = load_angelslim_draft_module(angelslim_root)

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        raw = json.load(f)

    mode = raw.get("eagle_aux_injection_mode", "fused_fc")
    bands = raw.get("eagle_aux_layer_bands")
    if mode == "banded_mix_fc":
        if not bands:
            raise ValueError("banded_mix_fc checkpoint without eagle_aux_layer_bands")
        n_streams = len(bands)
    elif mode in ("fused_fc", None):
        bands = None
        n_streams = len(raw.get("aux_hidden_states_layer_ids") or [0, 1, 2])
    else:
        raise NotImplementedError(
            "eagle_aux_injection_mode=%r is not supported by this adapter yet "
            "(supported: fused_fc, banded_mix_fc)" % mode
        )

    n_layers = raw.get("num_hidden_layers", 1)
    if n_layers != 1:
        raise NotImplementedError(
            "this adapter targets the single-layer drafter (Llama3Eagle3Drafter has "
            "one `midlayer`); checkpoint has num_hidden_layers=%d" % n_layers
        )

    cfg = DrafterConfig(raw, n_streams)

    cls = type("HiViSLlama3Eagle3Drafter",
               (HiViSInterfaceMixin, mod.Llama3Eagle3Drafter), {})
    drafter = cls(cfg, load_emb=False, path=None, total_tokens=total_tokens,
                  depth=depth, top_k=top_k, threshold=threshold)
    drafter.aux_layer_bands = ()
    if bands:
        drafter.init_banded_mix(bands)

    state = _remap_train_keys(_load_state(checkpoint_dir))
    missing, unexpected = drafter.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(("t2d", "d2t"))]
    # gist_norm is a dead weight in checkpoints trained with gist_conditioning
    # off; it has no consumer here.
    unexpected = [k for k in unexpected if "gist_norm" not in k]
    if missing:
        raise RuntimeError("drafter is missing %d checkpoint keys: %s"
                           % (len(missing), missing[:8]))

    # The real target layers whose hidden states must be captured and fed in.
    drafter.aux_hidden_states_layer_ids = list(raw["aux_hidden_states_layer_ids"])
    drafter.eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    if device is not None:
        drafter.to(device=device)
    drafter.to(dtype=dtype)
    return drafter, raw, unexpected


def _load_state(checkpoint_dir):
    from safetensors.torch import load_file

    state = {}
    shards = sorted(
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".safetensors")
    )
    for shard in shards:
        state.update(load_file(shard))
    if not state:
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.isfile(bin_path):
            raise FileNotFoundError("no weights under %s" % checkpoint_dir)
        state = torch.load(bin_path, map_location="cpu")
    return state


def _remap_train_keys(state):
    """Training names the single draft block `layers.0.*`; inference calls it `midlayer.*`.

    Same module, same shapes -- only the attribute path differs between the
    training stack (an nn.ModuleList sized by num_hidden_layers) and the
    inference drafter (one fixed block).
    """
    out = {}
    for k, v in state.items():
        out["midlayer." + k[len("layers.0."):] if k.startswith("layers.0.") else k] = v
    return out
