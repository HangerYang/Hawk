"""Phase-resolved timing for one generation, gated by HIVIS_PHASE.

Two modes, and the difference between them is the whole point.

HIVIS_PHASE=1        every phase synchronises on both edges. The numbers add
                     up to the generation, but the probe SERIALISES CPU and
                     GPU: it fills in exactly the pipeline gap one would be
                     looking for, and measured here it reversed the winner.
                     Kept only to reproduce that.

HIVIS_PHASE=events   CUDA events on both edges (record() does not block) plus
                     the host wall clock over the same region, read once at
                     the end of the generation. Per phase this gives
                       gpu  -- device-side span from entering to leaving
                       cpu  -- host-side span over the same region
                     GPU idle inside a phase shows up as cpu >> gpu with no
                     device work to account for it. Nothing is serialised, so
                     the timeline being measured is the timeline that runs.

HIVIS_SYNC_DEBUG=<k> arm torch's implicit-synchronisation detector for
                     generation k alone (0-based, warmups included) and print
                     each offending call site once with a count. This is how
                     to tell "this path still syncs" from "this path is slow".
"""
import atexit
import json
import os
import time
import traceback
import warnings

import torch

_MODE = os.environ.get("HIVIS_PHASE", "0")
ON = _MODE not in ("0", "", "false")
EVENTS = _MODE == "events"
OUT = os.environ.get("HIVIS_PHASE_OUT")
_SYNC_AT = os.environ.get("HIVIS_SYNC_DEBUG")
_PROF_AT = os.environ.get("HIVIS_PROFILE")
_cur = {}
_rows = []

# --- event pool -----------------------------------------------------------
# Events are recycled across generations: cudaEventCreate is only a couple of
# microseconds, but 40-odd of them per prompt on a 4 ms question is not a
# rounding error.
_free = []
_used = []
_pairs = []
_gen_start = None


def _event():
    if _free:
        e = _free.pop()
    else:
        e = torch.cuda.Event(enable_timing=True)
    _used.append(e)
    return e


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class phase(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if EVENTS:
            global _gen_start
            if _gen_start is None:
                _gen_start = _event()
                _gen_start.record()
                _cur["_wall0"] = time.perf_counter()
            self.e0 = _event()
            self.e0.record()
            self.t = time.perf_counter()
        elif ON:
            _sync()
            self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        if EVENTS:
            dt = time.perf_counter() - self.t
            e1 = _event()
            e1.record()
            _pairs.append((self.name, self.e0, e1))
            k = "cpu_" + self.name
            _cur[k] = _cur.get(k, 0.0) + dt
        elif ON:
            _sync()
            _cur[self.name] = _cur.get(self.name, 0.0) + (time.perf_counter() - self.t)
        return False


# --- implicit-sync detector ------------------------------------------------
_sync_hits = {}
_sync_armed = False
_orig_showwarning = None


def _catch(message, category, filename, lineno, file=None, line=None):
    if "synchroniz" in str(message).lower() or "Synchron" in str(message):
        stack = traceback.extract_stack()[:-1]
        frames = [f for f in stack
                  if "/torch/" not in f.filename and "warnings.py" not in f.filename]
        site = " <- ".join("%s:%d(%s)" % (os.path.basename(f.filename), f.lineno, f.name)
                           for f in reversed(frames[-4:]))
        _sync_hits[site] = _sync_hits.get(site, 0) + 1
        return
    if _orig_showwarning is not None:
        _orig_showwarning(message, category, filename, lineno, file, line)


def _arm_sync_debug(on):
    global _sync_armed, _orig_showwarning
    if on and not _sync_armed:
        _orig_showwarning = warnings.showwarning
        warnings.showwarning = _catch
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        _sync_armed = True
    elif not on and _sync_armed:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = _orig_showwarning
        _sync_armed = False
        print("\n[sync] implicit synchronisations in ONE generation:")
        for site, n in sorted(_sync_hits.items(), key=lambda kv: -kv[1]):
            print("  %4d  %s" % (n, site))
        print("[sync] %d total\n" % sum(_sync_hits.values()))


_prof = None


def _arm_profile(on):
    """Profile generation k alone. Same arming as the sync detector."""
    global _prof
    from torch.profiler import profile, ProfilerActivity
    if on:
        _prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=False, with_stack=False)
        _prof.__enter__()
    elif _prof is not None:
        _prof.__exit__(None, None, None)
        print(_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        out = os.environ.get("HIVIS_PROFILE_OUT")
        if out:
            _prof.export_chrome_trace(out)
            print("[prof] wrote %s" % out)
        _prof = None


_gen_n = 0


def commit(**extra):
    global _gen_n, _gen_start
    n = _gen_n
    _gen_n += 1
    if _SYNC_AT is not None:
        k = int(_SYNC_AT)
        if n == k - 1:
            _arm_sync_debug(True)
        elif n == k:
            _arm_sync_debug(False)
    if _PROF_AT is not None:
        k = int(_PROF_AT)
        if n == k - 1:
            _arm_profile(True)
        elif n == k:
            _arm_profile(False)
    if not ON:
        return
    row = dict(_cur)
    if EVENTS:
        wall0 = row.pop("_wall0", None)
        gen_end = _event()
        gen_end.record()
        torch.cuda.synchronize()
        if wall0 is not None:
            row["cpu_generation"] = time.perf_counter() - wall0
            row["gpu_generation"] = _gen_start.elapsed_time(gen_end) / 1000.0
        for name, e0, e1 in _pairs:
            key = "gpu_" + name
            row[key] = row.get(key, 0.0) + e0.elapsed_time(e1) / 1000.0
        _free.extend(_used)
        del _used[:]
        del _pairs[:]
        _gen_start = None
    row.update(extra)
    _rows.append(row)
    _cur.clear()


@atexit.register
def _dump():
    if not ON or not _rows:
        return
    keys = sorted({k for r in _rows for k in r})
    n = len(_rows)
    print("\n[phase] %d generations, mean ms" % n)
    if EVENTS:
        names = sorted({k[4:] for k in keys if k.startswith(("cpu_", "gpu_"))})
        print("  %-16s %9s %9s %9s" % ("phase", "gpu", "cpu", "cpu-gpu"))
        for name in names:
            g = 1000.0 * sum(r.get("gpu_" + name, 0.0) for r in _rows) / n
            c = 1000.0 * sum(r.get("cpu_" + name, 0.0) for r in _rows) / n
            print("  %-16s %9.3f %9.3f %9.3f" % (name, g, c, c - g))
        for k in ("rounds", "new_token"):
            if k in keys:
                print("  %-16s %9.2f" % (k, sum(r.get(k, 0) for r in _rows) / n))
    else:
        for k in keys:
            v = [r.get(k, 0.0) for r in _rows]
            unit = "" if k in ("rounds", "new_token") else " ms"
            scale = 1.0 if k in ("rounds", "new_token") else 1000.0
            print("  %-12s %8.2f%s" % (k, scale * sum(v) / n, unit))
    if OUT:
        with open(OUT, "w") as f:
            json.dump(_rows, f)
        print("[phase] wrote %s" % OUT)
