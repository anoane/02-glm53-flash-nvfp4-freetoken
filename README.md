# GLM-5.3-Flash on one RTX PRO 6000 with FreeToken

Serving config, build recipe, systemd unit and measurements for GLM-5.3-Flash (NVFP4)
on a single RTX PRO 6000 Blackwell (96 GB, sm_120) with DDR5 host offload and no swap,
using FreeToken's expert offload with the sm_120 `b12x` kernel on every MoE layer.

Checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4` (ModelOpt layout, 181.3 GiB); the PR's loader
does not read the compressed-tensors layout.

### Engine lineage

```
upstream   https://github.com/FlashML-org/FreeToken    branch pr270, head 857c167
fork       https://github.com/anoane/FreeToken         branch anoane, head e77b0dc
```

Branch `anoane` is **five commits** on top of `857c167`, shipped verbatim in `patches/`:

| | commit | what |
|---|---|---|
| 0001 | `ebff881` | serve the GLM-5.3 clamped SwiGLU on the b12x (SM12x W4A16) backend |
| 0002 | `040fdff` | run the VRAM-resident expert layers on the b12x backend |
| 0003 | `95daca8` | wire the MTP self-speculative decoder (**off by default** — measured negative) |
| 0004 | `d515df1` | `FREETOKEN_MOE_STATS=1` per-layer decode miss counters |
| 0005 | `e77b0dc` | cache-aware router knobs for the offloaded layers (env-gated, default off) |

Nothing in the fork is required to *load* the model — 0001 is the only commit on the
serving-correctness path. 0003 and 0005 are inert unless explicitly enabled.

### Hardware this was built and measured on

| | |
|---|---|
| Host | Proxmox VE 9.2.x, kernel 7.0.x-pve, Secure Boot **disabled** |
| CPU | AMD Ryzen 9 9950X3D (16C/32T) — 24 vCPU passed to the guest |
| RAM | 160 GiB DDR5 allocated to the guest (157 GiB usable) — **no swap, deliberately** |
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation — 97,887 MiB, `sm_120`, `10de:2bb1`, PCIe Gen5 x16 (~42 GB/s H2D measured), 400 W default limit / 600 W max |
| GPU 1 | NVIDIA CMP 170HX — 65,536 MiB after the [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) unlock (8 GB stock), `sm_80` (GA100), `10de:20c2`, 200 W default limit / 250 W max. The unlocked card does **Gen2 x16** (tested in the x16 slot); in *this* build it occupies a PCIe 3.0 **x1** slot, so the link runs **Gen2 x1, ~0.38 GB/s** — see repo 01 §5 |
| Guest | Ubuntu 24.04.4 LTS, kernel 6.8.0-139-generic, NVIDIA driver 610.43.02 |
| Storage | 5.8 TB NVMe (~93 GB free with all packs resident) |

**This recipe uses GPU 0 only.** The CMP 170HX is not involved. Host DDR5 carries the expert
bank (~95 GiB pinned after 17 resident layers). Note that this recipe was originally tuned
with 128 GB allocated to the guest (`MEMCAP=118g`); the box now runs 160 GiB, so there is
more headroom than the memory plan below assumes.

> **This is a point-in-time recipe and may be slightly outdated or incomplete.**
> It was transcribed from a working system rather than written as a clean-room guide: driver,
> engine and image versions move quickly, some steps that were obvious in the moment are
> under-documented, and a few numbers were measured once rather than averaged. Every measured
> figure below is specific to the hardware in the table above — on a different PCIe topology,
> a different RAM size, or a card without the CMP's x1 bottleneck, the tuning will differ.
> Read it as a worked example with its reasoning shown, not as a turnkey script.

## Layout

```
config/serve.env            all knobs (paths, resident layers, chunk, cache, ports, caps)
bin/build_venv.sh           clone + venv inside the CUDA 13 toolchain image, installs fla
bin/serve_ft.sh             the server (foreground for systemd, --detach for hand runs)
bin/memguard.sh             kills the container below MIN_AVAIL_GB host memory (no swap here)
bin/bench_ft.sh             smoke test / full 22,818-token C-review benchmark
bin/ttft_probe.py           time-to-first-token vs prompt length (one output token)
bin/long_probe.py           prefill time and peak VRAM vs prompt length (23k / 68k / 114k tokens)
bin/install_service.sh      installs and enables systemd/freetoken-glm53.service
patches/                    the five commits of branch `anoane`, as diffs against 857c167
results/                    perf + score JSON per benchmark row, ft bench bw profile
```

## Bring-up

```
bin/build_venv.sh --bench        # ~5 min; --bench needs an idle GPU (PCIe/CPU calibration)
bin/install_service.sh --start   # enables at boot and starts now; READY in ~6 min (cold JIT +1 min)
journalctl -fu freetoken-glm53
bin/bench_ft.sh smoke            # 31-token prompt: TTFT ~2 s, ~24 tok/s
```

OAI-compatible endpoint on `127.0.0.1:1919`, model id `model`, reasoning in
`reasoning_content`. One request at a time (`--max-running-requests 1`).

## Memory plan (the whole game)

| item | host | VRAM |
|---|---|---|
| expert bank, layers 3-44, 42 x 3.80 GiB | 159.5 GiB pinned by default | |
| 17 resident layers (`RES=3-10,36-44`) | -64.6 -> ~95 GiB pinned (shows as Shmem) | +64.6 |
| non-expert weights, FP8 at load | | ~10 |
| KV 262,144 tokens | | 3.44 |
| expert cache 620 slots | | 8.4 |
| steady state under load | 8 GiB available, guard at 4 | 94.9 of 97.9 GB |

16 resident layers leaves ~5 GiB of host; 18 leaves no VRAM for the prefill double buffer.
`--memory-ratio 0.95` is required (0.9 cannot fit the minimum plan). `--moe-cache-size` turns
off the expert auto-sizer only, so `--num-tokens` must cap the KV or it takes the freed VRAM.
`CHUNK=8192` with the 576-slot cache is the standing config: TTFT 8 s at 23k tokens and a peak of
96.6 of 97.9 GB VRAM that does not grow with prompt length (measured with `bin/long_probe.py`):

| prompt tokens | prefill time | peak VRAM |
|---|---|---|
| 22,839 | 8.0 s warm (22 s with the one-time JIT of the 8192 bucket) | 96,410 MiB |
| 68,493 | 24.2 s (~2,830 tok/s) | 96,410 MiB |
| 114,147 | 39.0 s (~2,930 tok/s) | 96,570 MiB |

With 620 slots the same chunk size peaked at 97,110 MiB; `CHUNK=4096` peaks at ~94,900 MiB if more headroom is ever needed.

## Measured (same 22,818-token C security review, 16 planted CWEs, greedy, uncapped reasoning)

| run | TTFT s | prefill tok/s | decode tok/s | delivered | CWE labels |
|---|---|---|---|---|---|
| exllamav3, exl3 4.05 bpw, CPU expert tail | 45.6 | 505 | 17.6 | 16/16 | 13/16 |
| FreeToken, Triton kernels | 32.7 | 703 | 22.2 | 16/16 | 14/16 |
| FreeToken, b12x on the 25 offloaded layers | 21.9 | 1,050 | 22.1 | 16/16 | 14/16 |
| FreeToken, b12x on all 42 layers, 4096-token chunks | 13.6 | 1,695 | 22.5 | 16/16 | 14/16 |
| same, 8192-token chunks | 8.1 | 2,857 | 22.7 | 16/16 | 15/16 |
| production service (systemd, fresh build of branch anoane) | 13.6 | 1,698 | 22.4 | 16/16 | 14/16 |
| production service, 8192-token chunks, 576 slots (standing config) | 8.1 | 2,853 | 22.2 | 16/16 | 13/16 |

Decode is bound by expert-cache misses over PCIe (56 GB/s) and does not move with the
kernel. MTP speculation was wired and measured at 12 tok/s with corrupted output
(verifying k+1 rows multiplies cold-expert traffic); it stays off.

## Gotchas

- `flash-linear-attention` is imported by the KDA layers but not declared; build_venv installs it.
- A fresh install compiles its kernels on first use: the first request took ~77 s and the first long prompt ~15 s extra; afterwards a new prefill bucket costs ~3 s once. The JIT cache lives in `home/` and survives restarts of the same install.
- The exl3 unit `glm53-sec.service` uses the same GPU; the unit declares `Conflicts=` on it.
- Never add swap: a host OOM here is a clean container kill; swap turns it into a machine hang.
