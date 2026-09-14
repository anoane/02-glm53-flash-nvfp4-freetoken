#!/usr/bin/env python3
"""Prefill peak-VRAM probe vs prompt length. usage: long_probe.py <url> <corpus.c.txt> [reps...]
Each rep repeats the corpus N times behind a unique prefix (radix-cache miss), sends max_tokens=1,
and reports wall time plus the peak of nvidia-smi memory.used sampled every 0.5 s during the request.
Run on an idle server. Token counts come back from the usage block."""
import json, subprocess, sys, threading, time, requests

url, path = sys.argv[1], sys.argv[2]
reps = [int(x) for x in sys.argv[3:]] or [1, 3, 5]
txt = open(path).read()

def sample(stop, out):
    peak = 0
    while not stop.is_set():
        try:
            v = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])
            peak = max(peak, v)
        except Exception:
            pass
        time.sleep(0.5)
    out.append(peak)

for n in reps:
    content = "\n".join(f"/* probe rep {n} part {i} */\n" + txt for i in range(n))
    body = {"model": "model", "messages": [{"role": "user", "content": content}], "max_tokens": 1,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    stop = threading.Event(); peak = []
    th = threading.Thread(target=sample, args=(stop, peak)); th.start()
    t0 = time.time(); usage = None; status = None; err = ""
    try:
        with requests.post(url, json=body, stream=True, timeout=1800) as r:
            status = r.status_code
            for line in r.iter_lines():
                if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                    try:
                        d = json.loads(line[5:]); usage = d.get("usage") or usage
                    except Exception:
                        pass
    except Exception as e:
        err = repr(e)[:120]
    dt = time.time() - t0; stop.set(); th.join()
    pt = (usage or {}).get("prompt_tokens")
    print(f"reps={n} prompt_tokens={pt} time={dt:.1f}s http={status} peak_vram={peak[0] if peak else 0} MiB {err}", flush=True)
