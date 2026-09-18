"""Give the drafter fewer image rows than the target saw, the way it was trained.

The target ALWAYS runs its own full prefill over the full-resolution image;
nothing here touches that. What changes is only what the drafter is handed.

``pool`` (and its ``subset`` variant) averages contiguous runs of image rows in
the hidden states the target's full prefill already produced. No second forward.

It is imported from the training code rather than reimplemented here
(``reduce_vlm_image_rows``), because a drafter evaluated against a slightly
different reduction than it was trained on measures the mismatch, not the
reduction.

POSITIONS. The reduction keeps ABSOLUTE position ids: the text rows after the
image were computed by the full forward and carry the positions they were
computed at, so the reduced image rows must keep theirs too. The resulting
sequence is shorter than its highest position id -- there is a gap where the
image was -- which is exactly what training fed the drafter, and why the
drafter cannot be left to infer positions from sequence length here.
"""

import importlib
import time
import os
import sys
import types

import torch


_TRAIN_PKG = "angelslim.compressor.speculative.train"


_TRAIN_ROOT_ENV = "ANGELSLIM_TRAIN_ROOT"


def _load(module, angelslim_root=None):
    """Import one AngelSlim training module without running angelslim/__init__.

    Same stand-in-package trick as angelslim_drafter.load_angelslim_draft_module,
    and for the same reason: angelslim/__init__.py evaluates py3.10-only
    annotations at import time and HiViS runs on py3.9. The leaf modules
    themselves are py3.9-clean.

    Deliberately a SEPARATE root from the one the drafter is loaded from. The
    reduction code lives on the branch the checkpoints were trained on, which
    need not be the checkout HiViS is vendored in; keeping the two independent
    means adding it here cannot quietly change which drafter implementation the
    already-measured arms ran against.
    """
    from .angelslim_drafter import _default_root, _ANGELSLIM_ROOT_ENV

    root = (angelslim_root or os.environ.get(_TRAIN_ROOT_ENV)
            or os.environ.get(_ANGELSLIM_ROOT_ENV) or _default_root())
    leaf = os.path.join(root, *module.split(".")) + ".py"
    if not os.path.isfile(leaf):
        raise RuntimeError(
            "%s not found under %r. The draft-side image reduction lives on the "
            "branch these checkpoints were trained on; point %s at that checkout "
            "(or pass --draft_image_root)." % (module, root, _TRAIN_ROOT_ENV)
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    parts = module.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name in sys.modules:
            continue
        stand_in = types.ModuleType(name)
        stand_in.__path__ = [os.path.join(root, *parts[:i])]
        sys.modules[name] = stand_in
    return importlib.import_module(module)


# --------------------------------------------------------------------------
# pool
# --------------------------------------------------------------------------

def pool_image_rows(input_ids, hidden_states, image_token_id, factor,
                    keep_positions=True, how="pool", angelslim_root=None):
    """Reduce the drafter's image rows by `factor`, averaging contiguous runs.

    Delegates to the collator's own reduce_vlm_image_rows, so this is the same
    grouping, the same representative row, and the same position rule the
    checkpoint was trained with.

    The tensors stay on the GPU. They used to make a round trip to host memory
    so that the collator -- written for a dataloader worker -- could do the
    averaging in CPU float32; that cost ~8 ms per prompt against 0.04 ms for the
    same arithmetic on device, and at 48 OMP threads it had a tail to 130 ms.
    reduce_vlm_image_rows now allocates on its input's device, so training (CPU
    tensors) and this call (CUDA tensors) still run the one code path.

    Returns (input_ids, hidden_states, position_ids), all batch-1.
    """
    data_utils = _load(_TRAIN_PKG + ".data.data_utils", angelslim_root)
    item = data_utils.reduce_vlm_image_rows(
        {"input_ids": input_ids, "hidden_states": hidden_states},
        image_token_id, factor, how=how, keep_positions=keep_positions,
    )
    return (
        item["input_ids"].to(input_ids.device),
        item["hidden_states"].to(hidden_states.device, hidden_states.dtype),
        item["position_ids"].to(input_ids.device),
    )


_TIME_REDUCER = os.environ.get("HIVIS_TIME_REDUCER", "0") not in ("0", "", "false")
_TIMING = {"outer_calls": 0, "outer_s": 0.0}

if _TIME_REDUCER:
    import atexit

    @atexit.register
    def _report_timing():
        m = _TIMING["outer_calls"]
        if m:
            print(f"[reducer timing] outer: {m} calls, {_TIMING['outer_s']:.2f}s "
                  f"total, {1000 * _TIMING['outer_s'] / m:.1f} ms/call", flush=True)


# --------------------------------------------------------------------------
class DraftImageReducer(object):
    """Callable held on the EaModel as `draft_image_reducer`.

    Returns (draft_input_ids, draft_hidden_states, absolute_position_ids), or
    None to leave the full rows in place (a prompt with no image, or one whose
    image region cannot be located).

    """

    def __init__(self, mode, image_token_id, factor=4,
                 keep_positions=True, angelslim_root=None,
                 control=None, control_seed=0):
        self.mode = mode
        # A control replaces the CONTENT of the drafter's image rows while
        # leaving their count, their token ids and their absolute positions
        # exactly as the arm produced them. If tau survives one of these, the
        # drafter was not reading the image in the first place.
        if control not in (None, "zero", "shuffle", "random", "wrong", "drop"):
            raise ValueError(
                "control must be zero, shuffle, random, wrong or drop")
        self.control = control
        self._gen = torch.Generator().manual_seed(int(control_seed))
        self._bank = []
        self._swapped = 0
        self._control_calls = 0
        self.image_token_id = int(image_token_id)
        self.factor = int(factor)
        self.keep_positions = bool(keep_positions)
        self.angelslim_root = angelslim_root
        self._reported = False
        # HIVIS_POOL_FAST: the same reduction with no device synchronisation on
        # the per-prompt path. The segment layout is a function of input_ids
        # alone, and every prompt in a benchmark carries the same image
        # structure, so it is computed once per sequence length and cached; a
        # hit leaves only index_select + index_add_ + div, all async. The slow
        # path syncs several times (mask.any(), two .tolist() in
        # _image_row_segments, boolean-mask indexing) at the point where the
        # target's prefill is still draining, which is what the phase probe
        # showed costs ~8 ms of GPU idle rather than ~8 ms of work.
        # On by default: every measurement since the fix ran on it, and
        # HIVIS_POOL_VERIFY found it bit-identical to the original path.
        # HIVIS_POOL_FAST=0 restores that path.
        self.fast = os.environ.get("HIVIS_POOL_FAST", "1") not in ("0", "", "false")
        self.verify_fast = os.environ.get("HIVIS_POOL_VERIFY", "0") not in ("0", "", "false")
        self._layout = {}
        self._pending = None
        self._du = None
        if mode not in ("pool", "subset"):
            raise ValueError("mode must be pool or subset, got %r" % mode)

    def _apply_control(self, input_ids, hidden_states, out):
        """Corrupt only the drafter's image rows; keep ids, count and positions."""
        new_ids, new_hidden, new_pos = out
        img = (new_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if img.numel() == 0:
            return out
        self._control_calls += 1
        if self.control == "drop":
            # Not a corruption: the image rows are REMOVED. The drafter is left
            # with the text rows alone, at their original absolute positions, so
            # every arm gets a byte-identical sequence and any remaining
            # difference is the weights.
            keep = new_ids[0] != self.image_token_id
            return new_ids[:, keep], new_hidden[:, keep], new_pos[:, keep]
        rows = new_hidden[0, img]
        if self.control == "zero":
            rows = torch.zeros_like(rows)
        elif self.control == "shuffle":
            perm = torch.randperm(rows.shape[0], generator=self._gen)
            rows = rows[perm.to(rows.device)]
        elif self.control == "random":
            # Same count, drawn at random from the rows the FULL forward made.
            full_img = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            k = min(rows.shape[0], full_img.numel())
            pick = torch.randperm(full_img.numel(), generator=self._gen)[:k].sort().values
            drawn = hidden_states[0, full_img[pick.to(full_img.device)]]
            if k < rows.shape[0]:
                drawn = torch.cat([drawn, drawn[-1:].expand(rows.shape[0] - k, -1)], 0)
            rows = drawn.to(rows.dtype)
        elif self.control == "wrong":
            # Rows this same arm computed for a DIFFERENT prompt's image.
            same = [t for t in self._bank if t.shape == rows.shape]
            if same:
                j = int(torch.randint(len(same), (1,), generator=self._gen))
                rows = same[j].to(rows.device, rows.dtype)
                self._swapped += 1
            self._bank.append(new_hidden[0, img].detach().to("cpu").clone())
            if len(self._bank) > 32:
                self._bank.pop(0)
        new_hidden = new_hidden.clone()
        new_hidden[0, img] = rows.to(new_hidden.dtype)
        return new_ids, new_hidden, new_pos

    def control_report(self):
        if self.control is None:
            return ""
        extra = ""
        if self.control == "wrong":
            extra = "  (%d/%d prompts got another prompt's rows)" % (
                self._swapped, self._control_calls)
        return "control=%s applied to %d prompts%s" % (
            self.control, self._control_calls, extra)

    def __call__(self, input_ids, hidden_states):
        if _TIME_REDUCER:
            # Both edges, or this measures the reducer PLUS however much of the
            # target's prefill was still queued when it was entered -- which is
            # not a fixed quantity: the same probe read 2.6 ms on MathVista and
            # 24.5 ms on textvqa for the same arithmetic.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return self._call(input_ids, hidden_states)
            finally:
                torch.cuda.synchronize()
                _TIMING["outer_calls"] += 1
                _TIMING["outer_s"] += time.perf_counter() - t0
        return self._call(input_ids, hidden_states)

    def prepare(self, input_ids):
        """Build this prompt's segment layout BEFORE the target's prefill.

        The layout is a function of input_ids alone, so it can be built at a
        point where the GPU queue is empty and the several reads it needs cost
        nothing. Doing it after the prefill -- which is where the reducer
        itself runs -- makes every one of those reads wait for the prefill to
        drain, and that idle GPU time, not the arithmetic, is what the pool arm
        was paying. Cached on a key that identifies the image layout, because
        SmolVLM tiles by aspect ratio and two prompts of equal length need not
        have equal layouts.
        """
        if not (self.fast and self.mode in ("pool", "subset")
                and self.control is None):
            return
        n = int(input_ids.shape[1])
        # On the CPU, deliberately. _image_row_segments walks the sequence one
        # contiguous run at a time in Python -- ~28 runs for a 13-tile SmolVLM
        # image -- and on CUDA each run costs a few kernel launches, ~90 in
        # total, for arithmetic that is microseconds of actual work. The same
        # function over 909 CPU elements is 2.8x faster measured, with no
        # launches and no synchronisation. Only the finished index tensors go
        # to the device.
        if os.environ.get("HIVIS_POOL_SLOWPREP", "0") not in ("0", "", "false"):
            mask = input_ids[0] == self.image_token_id      # A/B: 旧行为，GPU 上算
        else:
            mask = input_ids[0].to("cpu") == self.image_token_id
        hit = mask.nonzero().flatten()
        if hit.numel() == 0:
            self._pending = False
            return
        key = (n, int(hit.numel()), int(hit[0]), int(hit[-1]))
        lay = self._layout.get(key)
        if lay is None:
            if self._du is None:
                self._du = _load(_TRAIN_PKG + ".data.data_utils", self.angelslim_root)
            dev = input_ids.device
            seg, groups = self._du._image_row_segments(mask, self.factor, self.mode)
            keep_idx = (seg >= 0).nonzero().flatten()
            seg_keep = seg[keep_idx]
            first = torch.full((groups,), n, dtype=torch.long)
            first.scatter_reduce_(0, seg_keep, keep_idx, reduce="amin",
                                  include_self=True)
            counts = torch.zeros(groups).index_add_(
                0, seg_keep, torch.ones(seg_keep.numel()))
            keep_idx = keep_idx.to(dev); seg_keep = seg_keep.to(dev)
            first = first.to(dev); counts = counts.to(dev)
            del dev
            # The drafter needs the last absolute position as a Python int
            # (set_draft_positions). Read it here, where the queue is empty,
            # and carry it on the tensor so the drafter need not sync for it.
            lay = (keep_idx, seg_keep, first, counts, groups, int(first[-1]))
            self._layout[key] = lay
        self._pending = lay

    def _pool_fast(self, input_ids, hidden_states):
        """pool/subset off the prepared layout; nothing here synchronises."""
        lay = self._pending
        if lay is None:
            raise RuntimeError("HIVIS_POOL_FAST: prepare() was not called for "
                               "this prompt")
        if lay is False:
            return None
        keep_idx, seg_keep, first, counts, groups, last_pos = lay
        rows = hidden_states[0].index_select(0, keep_idx).float()
        acc = torch.zeros(groups, rows.shape[-1], dtype=torch.float32,
                          device=rows.device)
        acc.index_add_(0, seg_keep, rows)
        hs = (acc / counts[:, None]).to(hidden_states.dtype)[None]
        pos = first[None].clone()
        pos._hivis_last = last_pos
        return input_ids.index_select(1, first), hs, pos

    def _call(self, input_ids, hidden_states):
        if self.fast and self.mode in ("pool", "subset") and self.control is None:
            out = self._pool_fast(input_ids, hidden_states)
            if self.verify_fast and out is not None:
                ref = pool_image_rows(
                    input_ids, hidden_states, self.image_token_id, self.factor,
                    keep_positions=self.keep_positions, how=self.mode,
                    angelslim_root=self.angelslim_root)
                for name, a, b in zip(("ids", "hidden", "pos"), out, ref):
                    if a.shape != b.shape:
                        raise RuntimeError("fast/slow %s shape %s vs %s"
                                           % (name, tuple(a.shape), tuple(b.shape)))
                    d = (a.float() - b.float()).abs().max().item()
                    if d > 0:
                        print("  [verify] %s max|fast-slow| = %.3e" % (name, d))
            if out is not None and not self._reported:
                self._report(input_ids, out)
            return out
        if not bool((input_ids[0] == self.image_token_id).any()):
            return None
        out = pool_image_rows(
            input_ids, hidden_states, self.image_token_id, self.factor,
            keep_positions=self.keep_positions, how=self.mode,
            angelslim_root=self.angelslim_root,
        )
        if out is not None and self.control is not None:
            out = self._apply_control(input_ids, hidden_states, out)
        # Say once what actually happened. A reduction that silently did nothing
        # (image token id wrong, region not found) still produces a plausible
        # tau -- of the unreduced drafter -- and nothing else would show it.
        if out is not None and not self._reported:
            self._report(input_ids, out)
        return out

    def _report(self, input_ids, out):
        self._reported = True
        n_img = int((input_ids[0] == self.image_token_id).sum())
        kept = int((out[0][0] == self.image_token_id).sum())
        print("  draft rows: %d -> %d (image %d -> %d), positions %d..%d"
              % (input_ids.shape[1], out[0].shape[1], n_img, kept,
                 int(out[2][0, 0]), int(out[2][0, -1])))
        if self.control is not None:
            print("  draft image rows CONTROL: %s" % self.control)
