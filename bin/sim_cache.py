#!/usr/bin/env python3
"""Replay a routed-expert trace against cache policies to bound what any predictor can win.

usage: sim_cache.py <trace.npz> [slots ...]

The engine keys its VRAM slot cache by (offload-bank layer, expert) globally, so this
simulates the same thing: one pool of S slots, each step requesting the union of every
layer's routed experts. Policies:

  LRU     what the engine implements today (cache_policy = "lru")
  LFU     evict the least frequently requested entry (no prediction needed)
  BELADY  evict the entry whose next use is furthest away -- requires the future, so it is
          the upper bound on ANY admission/eviction policy or learned expert predictor

Misses are what cost PCIe bytes, so the LRU-to-Belady gap is the prize available without
touching the model's routing (i.e. losslessly). Heaps with lazy invalidation keep all three
policies near O(1) per access.
"""
import heapq
import sys
from collections import OrderedDict, defaultdict

import numpy as np

trace = np.load(sys.argv[1])["trace"]          # [steps, layers, width], -1 padded
slots_list = [int(x) for x in sys.argv[2:]] or [576]
steps, n_layers, width = trace.shape

reqs = []
for s in range(steps):
    seen, keys = set(), []
    for L in range(n_layers):
        for e in trace[s, L]:
            if e < 0:
                continue
            k = L * 100000 + int(e)
            if k not in seen:
                seen.add(k)
                keys.append(k)
    reqs.append(keys)

lookups = sum(len(r) for r in reqs)
universe = len(set(k for r in reqs for k in r))
print(f"steps {steps}, layers {n_layers}, lookups/step {lookups / steps:.1f}, "
      f"distinct (layer,expert) touched {universe}")

# occurrence lists per key, for Belady's next-use queries
occ = defaultdict(list)
for s, keys in enumerate(reqs):
    for k in keys:
        occ[k].append(s)
ptr = {k: 0 for k in occ}
INF = 1 << 30


def next_use_after(k, s):
    lst = occ[k]
    i = ptr[k]
    while i < len(lst) and lst[i] <= s:
        i += 1
    ptr[k] = i
    return lst[i] if i < len(lst) else INF


def sim_lru(S):
    cache, misses = OrderedDict(), 0
    for keys in reqs:
        for k in keys:
            if k in cache:
                cache.move_to_end(k)
                continue
            misses += 1
            if len(cache) >= S:
                cache.popitem(last=False)
            cache[k] = 1
    return misses


def sim_lfu(S):
    cache, freq, heap, misses, tick = set(), defaultdict(int), [], 0, 0
    for keys in reqs:
        for k in keys:
            freq[k] += 1
            tick += 1
            if k in cache:
                heapq.heappush(heap, (freq[k], tick, k))
                continue
            misses += 1
            if len(cache) >= S:
                while heap:
                    f, _, victim = heapq.heappop(heap)
                    if victim in cache and f == freq[victim]:
                        cache.discard(victim)
                        break
            cache.add(k)
            heapq.heappush(heap, (freq[k], tick, k))
    return misses


def sim_belady(S):
    for k in ptr:
        ptr[k] = 0
    cache, heap, misses = set(), [], 0
    for s, keys in enumerate(reqs):
        for k in keys:
            nu = next_use_after(k, s)
            if k in cache:
                heapq.heappush(heap, (-nu, k))
                continue
            misses += 1
            if len(cache) >= S:
                while heap:
                    negnu, victim = heapq.heappop(heap)
                    if victim not in cache:
                        continue
                    if -negnu < next_use_after(victim, s):
                        continue          # stale entry, its real next use is later
                    cache.discard(victim)
                    break
            cache.add(k)
            heapq.heappush(heap, (-nu, k))
    return misses


EXPERT_MIB = 15.0
PCIE_GBPS = 56.0
print(f"\n{'slots':>6} {'GiB':>5} {'policy':>7} {'miss/step':>10} {'hit':>7} "
      f"{'GB/token':>9} {'DMA ms':>7} {'vs LRU':>8}")
for S in slots_list:
    base = None
    for name, fn in (("lru", sim_lru), ("lfu", sim_lfu), ("belady", sim_belady)):
        mps = fn(S) / steps
        gb = mps * EXPERT_MIB / 1024
        ms = gb / PCIE_GBPS * 1000
        if base is None:
            base = mps
        rel = "--" if name == "lru" else f"{(base - mps) / base:+.0%}"
        print(f"{S:>6} {S * EXPERT_MIB / 1024:>5.1f} {name:>7} {mps:>10.1f} "
              f"{1 - mps / (lookups / steps):>6.1%} {gb:>9.2f} {ms:>7.1f} {rel:>8}")
