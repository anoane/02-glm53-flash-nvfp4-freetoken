#!/usr/bin/env python3
"""Time-to-first-byte vs prompt length against a FreeToken OAI-compatible endpoint.
usage: ttft_probe.py <url> <model_id> <corpus.c.txt>
Sends max_tokens=1 requests with the first N characters of the corpus for several N
(distinct prefixes, so the radix cache cannot serve them) and reports wall time to the
first streamed byte and to completion. Run only on an idle server."""
import json, sys, time, requests

url, mid, path = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(path).read()
# character budgets ~ token counts x3 for C source
for label, nchar in (("~300 tok", 900), ("~1k tok", 3000), ("~4k tok", 12000), ("~8k tok", 24000), ("~16k tok", 48000), ("full", len(txt))):
    body = {"model": mid, "messages": [{"role": "user", "content": f"/* probe {label} */\n" + txt[:nchar]}],
            "max_tokens": 1, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); first = None; usage = None
    with requests.post(url, json=body, stream=True, timeout=900) as r:
        for line in r.iter_lines():
            if not line: continue
            if first is None: first = time.time() - t0
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                d = json.loads(line[5:]); usage = d.get("usage") or usage
    dt = time.time() - t0
    pt = (usage or {}).get("prompt_tokens")
    print(f"{label:>9} prompt_tokens={pt} first_byte={first:.2f}s total={dt:.2f}s", flush=True)
