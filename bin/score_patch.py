#!/usr/bin/env python3
"""Add an env-gated teacher-forcing / logits-capture mode to the engine.

FREETOKEN_SCORE_OUT=/path/prefix   enable; writes <prefix>.logits.npy (fp16 [N, vocab])
                                   and <prefix>.ids.json (the emitted continuation)
FREETOKEN_SCORE_N=512              rows to capture (default 512)
FREETOKEN_SCORE_IDS=/path.json     optional: force this continuation instead of sampling,
                                   so two configurations are scored on an identical sequence

Capture happens on every single-request forward that produces a real next token (the final
prefill chunk and each decode step), i.e. row i is the distribution that predicts
continuation token i. Measurement mode only: it syncs on the sampled token each step.
Run from the FreeToken source root.
"""

def patch(path, old, new):
    s = open(path).read()
    n = s.count(old)
    assert n == 1, f"anchor count {n} in {path}"
    open(path, "w").write(s.replace(old, new))
    print("patched", path)


patch("python/freetoken/engine/engine.py",
      """        batch_logits = logits[: batch.size]
        next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)""",
      """        batch_logits = logits[: batch.size]
        _sc = _score_state()
        if (
            _sc is not None
            and not _sc["done"]
            and batch.size == 1
            and type(batch.reqs[0]).__name__ != "ChunkedReq"
        ):
            _sc["rows"].append(batch_logits[0].detach().to(torch.float16).cpu())
        else:
            _sc = None
        next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
        if _sc is not None:
            _i = len(_sc["rows"]) - 1
            if _sc["ids"] is not None and _i < len(_sc["ids"]):
                next_tokens_gpu[0] = int(_sc["ids"][_i])
            _sc["emitted"].append(int(next_tokens_gpu[0].item()))
            if len(_sc["rows"]) >= _sc["n"]:
                _score_flush(_sc)""")

patch("python/freetoken/engine/engine.py",
      """class ForwardOutput(NamedTuple):""",
      '''_SCORE: dict | None = None


def _score_state() -> dict | None:
    """Teacher-forcing / logits-capture state (FREETOKEN_SCORE_OUT); None when disabled."""
    global _SCORE
    if _SCORE is None:
        out = os.environ.get("FREETOKEN_SCORE_OUT")
        if not out:
            return None
        ids = None
        ids_path = os.environ.get("FREETOKEN_SCORE_IDS")
        if ids_path and os.path.exists(ids_path):
            import json as _json

            ids = _json.load(open(ids_path))
        _SCORE = {
            "out": out,
            "n": int(os.environ.get("FREETOKEN_SCORE_N", "512")),
            "ids": ids,
            "rows": [],
            "emitted": [],
            "done": False,
        }
        logger.info_rank0(
            f"scoring mode: capturing {_SCORE['n']} logits rows to {out}"
            + (f", forcing {len(ids)} tokens" if ids else ", sampling freely")
        )
    return None if _SCORE["done"] else _SCORE


def _score_flush(sc: dict) -> None:
    import json as _json

    import numpy as _np

    sc["done"] = True
    arr = torch.stack(sc["rows"]).numpy()
    _np.save(sc["out"] + ".logits.npy", arr)
    _json.dump(sc["emitted"], open(sc["out"] + ".ids.json", "w"))
    logger.info_rank0(
        f"scoring mode: wrote {arr.shape} to {sc['out']}.logits.npy "
        f"and {len(sc['emitted'])} ids"
    )


class ForwardOutput(NamedTuple):''')
print("done")
