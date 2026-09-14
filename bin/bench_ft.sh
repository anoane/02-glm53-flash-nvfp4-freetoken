#!/usr/bin/env bash
# usage: bench_ft.sh smoke|full [label]
set -u; cd /root/workspace/bench_c_review
URL=http://127.0.0.1:1919/v1/chat/completions
MID=$(curl -s http://127.0.0.1:1919/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])")
echo "served model id: $MID"
case "${1:-smoke}" in
  smoke)
    python3 - "$URL" "$MID" <<'PY'
import sys,json,time,requests
url,mid=sys.argv[1],sys.argv[2]
body={"model":mid,"messages":[{"role":"user","content":"In C, what does the function strncpy guarantee about NUL termination? Answer in two sentences."}],"max_tokens":400,"temperature":0.0,"stream":True,"stream_options":{"include_usage":True}}
t0=time.time(); first=None; n=0; reason=""; ans=""; usage=None
with requests.post(url,json=body,stream=True,timeout=600) as r:
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"): continue
        p=line[5:].strip()
        if p==b"[DONE]": break
        d=json.loads(p)
        if d.get("usage"): usage=d["usage"]
        for c in d.get("choices",[]):
            delta=c.get("delta",{})
            rc=delta.get("reasoning_content") or delta.get("reasoning"); ct=delta.get("content")
            if rc or ct:
                if first is None: first=time.time()-t0
                n+=1
            if rc: reason+=rc
            if ct: ans+=ct
dt=time.time()-t0
print(f"ttft={first:.2f}s total={dt:.1f}s chunks={n} usage={usage}")
print("REASONING[:300]:",reason[:300].replace("\n"," "))
print("ANSWER:",ans[:600])
PY
    ;;
  full)
    LABEL=${2:-freetoken-pr270}
    nohup python3 run_bench.py --url "$URL" --model "$MID" --corpus corpus.c.txt --out "results/${LABEL}_perf.json" --label "$LABEL" --max-tokens 200000 > "results/${LABEL}_run.log" 2>&1 &
    echo "full benchmark started pid $! -> results/${LABEL}_perf.json (log results/${LABEL}_run.log)"
    ;;
esac
