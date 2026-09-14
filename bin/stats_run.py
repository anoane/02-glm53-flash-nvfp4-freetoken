#!/usr/bin/env python3
"""Drive ~N output tokens of real greedy decode on the corpus prompt so the per-layer decode
miss counters (FREETOKEN_MOE_STATS=1) accumulate. usage: stats_run.py <url> <corpus> [max_tokens]"""
import json, sys, time, requests

url, path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
txt = open(path).read()
prompt = ("Review this C code for security vulnerabilities. For each finding give file, "
          "function, CWE and a one-line reason.\n\n" + txt)
body = {"model": "model", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
t0 = time.time(); usage = None; first = None
with requests.post(url, json=body, stream=True, timeout=3600) as r:
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            break
        d = json.loads(p)
        if d.get("usage"):
            usage = d["usage"]
        if first is None and d.get("choices"):
            first = time.time() - t0
dt = time.time() - t0
ct = (usage or {}).get("completion_tokens", 0)
dec = ct / max(dt - (first or 0.0), 1e-6)
print(f"done: usage={usage} ttft={first:.1f}s total={dt:.1f}s decode~{dec:.1f} tok/s", flush=True)
