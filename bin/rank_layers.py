#!/usr/bin/env python3
"""Merge [moe-stats] dumps from one or more serve logs and rank checkpoint layers by decode
misses per step (bytes over PCIe). usage: rank_layers.py <n_resident> <serve.log> [<serve.log> ...]
Each log contributes its LAST [moe-stats] line; offload-bank ids map to checkpoint layers via the
dump's bank_to_layer list. Layers that were resident in a run are unmeasured in that run."""
import json, sys

n_res = int(sys.argv[1])
per_layer = {}
for path in sys.argv[2:]:
    last = None
    for line in open(path, errors="replace"):
        i = line.find("[moe-stats] ")
        if i >= 0:
            last = line[i + len("[moe-stats] "):].strip()
    if not last:
        print(f"{path}: no [moe-stats] line")
        continue
    d = json.loads(last)
    m = d["bank_to_layer"]
    for row in d["per_layer"]:
        b = row["layer"]
        if m is None or b >= len(m):
            continue
        L = m[b]
        if row["steps"] and row["active_per_step"] > 0:
            per_layer[L] = (row["missing_per_step"], row["miss_rate"], row["steps"], path.split("/")[-2])

rows = sorted(per_layer.items(), key=lambda kv: -kv[1][0])
hdr_layer, hdr_mps, hdr_mr, hdr_steps = "layer", "miss/step", "miss_rate", "steps"
print(f"{hdr_layer:>5} {hdr_mps:>9} {hdr_mr:>9} {hdr_steps:>6}  source")
for L, (mps, mr, st, src) in rows:
    print(f"{L:>5} {mps:>9.2f} {mr:>9.3f} {st:>6}  {src}")
unmeasured = [L for L in range(3, 45) if L not in per_layer]
print("unmeasured layers:", unmeasured)


def ranges(xs):
    out = []
    s = p = None
    for x in xs:
        if s is None:
            s = p = x
        elif x == p + 1:
            p = x
        else:
            out.append(f"{s}-{p}" if s != p else f"{s}")
            s = p = x
    if s is not None:
        out.append(f"{s}-{p}" if s != p else f"{s}")
    return ",".join(out)


top = sorted(L for L, _ in rows[:n_res])
tot = sum(v[0] for v in per_layer.values())
saved = sum(per_layer[L][0] for L in top)
pct = 100 * saved / tot if tot else 0
print(f"top-{n_res} by miss/step: RES={ranges(top)}  (covers {saved:.1f} of {tot:.1f} measured misses/step = {pct:.0f}%)")
