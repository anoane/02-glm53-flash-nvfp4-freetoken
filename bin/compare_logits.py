#!/usr/bin/env python3
"""Compare two logits captures (see score_patch.py) on an identical token sequence.

usage: compare_logits.py <ref_prefix> <test_prefix> [vocab_size] [--json out.json]

Reports, mirroring the exl3 bitrate-ladder instrument: NLL and perplexity of the scored
continuation under each configuration, KL(ref || test) mean/median/percentiles, top-1
agreement, and both split by how confident the reference was at that position.
"""
import json, sys
import numpy as np

argv = sys.argv[1:]
out_json = None
if "--json" in argv:
    i = argv.index("--json")
    out_json = argv[i + 1]
    del argv[i:i + 2]
args = argv
ref_p, test_p = args[0], args[1]
vs = int(args[2]) if len(args) > 2 else 0

ref = np.load(ref_p + ".logits.npy").astype(np.float32)
test = np.load(test_p + ".logits.npy").astype(np.float32)
ids = json.load(open(ref_p + ".ids.json"))
n = min(len(ref), len(test), len(ids))
ref, test, ids = ref[:n], test[:n], np.array(ids[:n])
if vs:
    ref, test = ref[:, :vs], test[:, :vs]
print(f"scored positions: {n}, vocab width: {ref.shape[1]}")


def logsoftmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


lr, lt = logsoftmax(ref), logsoftmax(test)
pr = np.exp(lr)
rows = np.arange(n)

nll_ref = -lr[rows, ids]
nll_test = -lt[rows, ids]
kld = (pr * (lr - lt)).sum(axis=-1)          # KL(ref || test), nats
top_ref, top_test = lr.argmax(-1), lt.argmax(-1)
agree = (top_ref == top_test)
conf = pr.max(axis=-1)

def stats(name, v):
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"metric": name, "mean": float(v.mean()), "p10": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}

summary = {
    "positions": int(n),
    "nll_ref": float(nll_ref.mean()), "ppl_ref": float(np.exp(nll_ref.mean())),
    "nll_test": float(nll_test.mean()), "ppl_test": float(np.exp(nll_test.mean())),
    "kld_mean": float(kld.mean()), "kld_median": float(np.median(kld)),
    "kld_p90": float(np.percentile(kld, 90)), "kld_max": float(kld.max()),
    "top1_agree": float(agree.mean()),
    "ref_mean_confidence": float(conf.mean()),
}
print("\n=== overall ===")
print(f"NLL      ref {summary['nll_ref']:.5f}   test {summary['nll_test']:.5f}   "
      f"delta {summary['nll_test'] - summary['nll_ref']:+.5f}")
print(f"PPL      ref {summary['ppl_ref']:.5f}   test {summary['ppl_test']:.5f}   "
      f"ratio {summary['ppl_test'] / summary['ppl_ref']:.4f}x")
print(f"KLD(ref||test)  mean {summary['kld_mean']:.6f}  median {summary['kld_median']:.6f}  "
      f"p90 {summary['kld_p90']:.6f}  max {summary['kld_max']:.4f}")
print(f"top-1 agreement {summary['top1_agree']:.4f}   (ref mean confidence {summary['ref_mean_confidence']:.4f})")

print("\n=== by reference confidence ===")
print(f"{'bucket':>12} {'share':>7} {'top1':>7} {'kld_mean':>10} {'kld_med':>10}")
buckets = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95), (0.95, 1.0001)]
rows_out = []
for lo, hi in buckets:
    m = (conf >= lo) & (conf < hi)
    if not m.any():
        rows_out.append({"lo": lo, "hi": hi, "share": 0.0}); continue
    r = {"lo": lo, "hi": hi, "share": float(m.mean()), "top1": float(agree[m].mean()),
         "kld_mean": float(kld[m].mean()), "kld_median": float(np.median(kld[m])),
         "n": int(m.sum())}
    rows_out.append(r)
    print(f"{lo:>5.2f}-{hi:<6.2f} {r['share']:>7.1%} {r['top1']:>7.4f} "
          f"{r['kld_mean']:>10.6f} {r['kld_median']:>10.6f}")
summary["buckets"] = rows_out

dis = np.where(~agree)[0]
print(f"\ndisagreements: {len(dis)} of {n}")
if len(dis):
    worst = dis[np.argsort(-kld[dis])][:5]
    print("worst by KLD (position, ref confidence, kld):",
          [(int(i), round(float(conf[i]), 3), round(float(kld[i]), 4)) for i in worst])
if out_json:
    json.dump(summary, open(out_json, "w"), indent=1)
    print("wrote", out_json)
