#!/usr/bin/env python3
"""Record the per-step routed-expert request stream of the offload cache.

FREETOKEN_TRACE_OUT=/path.npz   enable; writes an int32 array [steps, num_layers, width]
FREETOKEN_TRACE_N=3000          decode steps to record (default 3000)

CUDA-graph safe: ensure_experts() runs inside the captured decode graph, so it only writes
into a fixed device buffer at a static layer index; the host copy happens in forward_batch
after the replay returns. Unused slots are -1.

Feeds bin/sim_cache.py, which replays the trace against LRU / LFU / Belady-optimal to bound
how much any smarter admission or eviction policy (or a learned expert predictor) could win.
Run from the FreeToken source root.
"""

def patch(path, old, new):
    s = open(path).read()
    n = s.count(old)
    assert n == 1, f"anchor count {n} in {path}"
    open(path, "w").write(s.replace(old, new))
    print("patched", path)


W = 64  # max expert ids recorded per (step, layer)

patch("python/freetoken/moe/offload_cache.py",
      """    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:""",
      f"""    def _trace_record(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        \"\"\"Graph-safe: static layer index, fixed-size device buffer, no host sync.\"\"\"
        buf = getattr(self, "trace_buf", None)
        if buf is None:
            buf = self.trace_buf = torch.full(
                (self.num_layers, {W}), -1, dtype=torch.int32, device=self.device
            )
        flat = expert_ids.flatten()
        n = min(int(flat.numel()), {W})
        buf[layer_id].fill_(-1)
        buf[layer_id, :n].copy_(flat[:n])

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        if _TRACE_ON:
            self._trace_record(layer_id, expert_ids)""")

patch("python/freetoken/moe/offload_cache.py",
      """class OffloadMoeCache""",
      """_TRACE_ON = bool(os.environ.get("FREETOKEN_TRACE_OUT"))


class OffloadMoeCache""")

patch("python/freetoken/engine/engine.py",
      """class ForwardOutput(NamedTuple):""",
      '''_TRACE: dict | None = None


def _trace_state() -> dict | None:
    """Expert-request trace (FREETOKEN_TRACE_OUT); None when disabled or finished."""
    global _TRACE
    if _TRACE is None:
        out = os.environ.get("FREETOKEN_TRACE_OUT")
        if not out:
            return None
        _TRACE = {"out": out, "n": int(os.environ.get("FREETOKEN_TRACE_N", "3000")),
                  "rows": [], "done": False}
        logger.info_rank0(f"expert trace: recording {_TRACE['n']} decode steps to {out}")
    return None if _TRACE["done"] else _TRACE


def _trace_step(cache) -> None:
    tr = _trace_state()
    if tr is None or cache is None or getattr(cache, "trace_buf", None) is None:
        return
    tr["rows"].append(cache.trace_buf.to("cpu", copy=True).numpy())
    if len(tr["rows"]) >= tr["n"]:
        import numpy as _np

        tr["done"] = True
        arr = _np.stack(tr["rows"])
        _np.savez_compressed(tr["out"], trace=arr)
        logger.info_rank0(f"expert trace: wrote {arr.shape} to {tr['out']}")


class ForwardOutput(NamedTuple):''')

patch("python/freetoken/engine/engine.py",
      """        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)""",
      """        if batch.is_decode and batch.size == 1:
            _trace_step(self.moe_offload_cache)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)""")
print("done")
