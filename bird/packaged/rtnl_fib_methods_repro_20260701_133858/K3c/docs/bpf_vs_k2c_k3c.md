# BPF Dataplane Optimized Result Compared With K2c/K3c

Date: 2026-06-29/30

This document records the optimized BPF dataplane run on the 3083-scale
SeedEMU topology and compares it with the K2c/K3c Linux-FIB writers.

## Scope

BPF dataplane is not equivalent to K2c/K3c:

- BPF writes selected BIRD routes into pinned BPF LPM maps and forwards packets
  at TC ingress.
- K2c/K3c write routes into Linux main FIB through custom kernel route APIs.
- BPF validation uses packet reachability and BPF map counters; `ip route | wc`
  is not a valid success check for BPF because the Linux main FIB intentionally
  does not contain the remote routes.

## Optimizations Added

Files:

- `bird/bpf_dataplane/seedemu_bpf_dataplane.py`
- `bird/bpf_dataplane/bpf_map_apply.c`

Changes:

- Added `--apply-mode syscall`, using a small C helper with direct `bpf(2)`
  `BPF_MAP_UPDATE_ELEM` / `BPF_MAP_DELETE_ELEM` calls.
- Added binary route-op files under `bpf_ops/`, avoiding bpftool text parsing
  in optimized runs.
- Added `--no-text-batches` so optimized runs can skip bpftool batch text files.
- Added JSON route-cache files under `route_cache/`, making delta compare
  cheaper than reparsing previous bpftool batch text.
- Added interface/ifindex cache reuse through `--interface-cache-run-dir`
  and defaulted it to `--delta-from-run-dir`.
- Replaced slower route-key construction with direct IPv4 prefix parsing in the
  hot path.

## 3083 Setup

The VM was reset before this round. The project was started with BIRD only;
`start-kernel` was skipped because this experiment does not write Linux main
FIB routes.

```bash
python3 bird/run_all.py bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output \
  --vm 192.168.122.60 \
  --project-name bpf-3083 \
  --no-start-kernel \
  --entry-method docker-exec \
  --start-stage-jobs 1 \
  --start-bird-delay 0.1 \
  --build-jobs 16 \
  --up-jobs 4
```

Control-plane timing:

| Stage | Time | Result |
| --- | ---: | --- |
| build | `169.878s` | success |
| docker compose up | `3063.188s` | 62 batches succeeded, 0 failed |
| start-bird | `454.772s` | 3083 triggered, 0 failed |
| start-kernel | `0s` | skipped |

Local summary:

```text
bird/runs/topology-3085_20260629_234636/remote_summary.json
```

## Optimized BPF Full Install

Command on the VM:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name bpf-3083 \
  --jobs 8 \
  --apply-mode syscall \
  --no-text-batches \
  --skip-rp-filter-disable \
  --force-compile \
  --run-dir /home/lxl/bird/runs/bpf_opt_full_bpf_3083_20260629_165027
```

Result:

| Metric | Value |
| --- | ---: |
| brdnodes | `2757/2757` |
| BIRD prefixes seen | `6,131,994` |
| BPF routes installed | `5,938,022` |
| route ops applied | `5,938,022` |
| writer wall time | `228.919s` |
| old BPF full baseline | `347.125s` |
| improvement over old BPF full | about `34%` |

Note: this command used `--skip-rp-filter-disable` on a freshly reset VM, so the
first packet validation failed until `rp_filter` was disabled afterward. The
one-time `rp_filter` disable pass took `25.477s`. A clean first-install
end-to-end estimate is therefore about `254.396s`, still faster than the old
`347.125s` path.

Full-install phase totals across all brdnodes:

| Phase | Total seconds | Average per brdnode |
| --- | ---: | ---: |
| Dump BIRD routes | `641.577` | `0.233` |
| Read interfaces | `448.513` | `0.163` |
| Load BPF program/map | `213.255` | `0.077` |
| Attach TC program | `211.635` | `0.077` |
| Build entries | `103.085` | `0.037` |
| Apply BPF map ops | `94.203` | `0.034` |
| Delta/cache compare | `82.924` | `0.030` |

## Optimized BPF Reconvergence Delta

After the full install, 10 brdnode containers were stopped to emulate a network
wave. Then the VM waited until load was stable below the gate and the BPF delta
writer was run:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name bpf-3083 \
  --jobs 8 \
  --apply-mode syscall \
  --no-text-batches \
  --skip-rp-filter-disable \
  --delta-from-run-dir /home/lxl/bird/runs/bpf_opt_full_bpf_3083_20260629_165027 \
  --run-dir /home/lxl/bird/runs/bpf_opt_delta_bpf_3083_20260629_165936/writer
```

Result:

| Metric | Value |
| --- | ---: |
| stopped brdnodes | `10` |
| stop duration | `23.931s` |
| load gate before writer | `90.013s` |
| writer brdnodes | `2747/2747` |
| BIRD prefixes seen | `5,261,198` |
| BPF route snapshot | `5,068,520` |
| route ops applied | `1,106,971` |
| updates | `257,674` |
| deletes | `849,297` |
| route-cache hits | `2747/2747` |
| interface-cache hits | `2747/2747` |
| writer wall time | `118.626s` |

Continuous ping during the writer saw `4075` replies and no unreachable,
timeout, or packet-loss lines in the captured log. Post-writer validation also
passed in both directions:

- forward ping: `20/20`, 0% loss, average RTT `0.252ms`
- reverse ping: `20/20`, 0% loss, average RTT `0.139ms`

Delta phase totals across all brdnodes:

| Phase | Total seconds | Average per brdnode |
| --- | ---: | ---: |
| Dump BIRD routes | `611.491` | `0.223` |
| Apply BPF map ops | `124.970` | `0.045` |
| Delta/cache compare | `99.958` | `0.036` |
| Build entries | `74.726` | `0.027` |
| Read interfaces | `0.021` | `0.000008` |
| Load BPF program/map | `0.004` | `0.000001` |
| Attach TC program | `0.004` | `0.000001` |

## Zero-Change Delta Floor

Immediately after the reconvergence delta, a second delta run used the previous
delta run as baseline without introducing new topology changes:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name bpf-3083 \
  --jobs 8 \
  --apply-mode syscall \
  --no-text-batches \
  --skip-rp-filter-disable \
  --delta-from-run-dir /home/lxl/bird/runs/bpf_opt_delta_bpf_3083_20260629_165936/writer \
  --run-dir /home/lxl/bird/runs/bpf_opt_zero_delta_bpf_3083_20260629_170649
```

Result:

| Metric | Value |
| --- | ---: |
| brdnodes | `2747/2747` |
| BPF route snapshot | `5,068,520` |
| route ops applied | `0` |
| route-cache hits | `2683` |
| route-cache baseline misses | `64` |
| interface-cache hits | `2747/2747` |
| writer wall time | `94.261s` |

This is the important floor: even with zero route ops, the current controller
still spends about `94s` scanning all running brdnodes with
`birdc show route primary`, rebuilding route entries, and comparing snapshots.
Therefore the next major BPF optimization cannot be another map-write micro
optimization; it must avoid all-container full RIB dumps.

## Comparison With K2c/K3c

K2c/K3c results are from the documented real 3083 runs in `0626kernelDesign.md`
and `design.md`.

| Method | Semantics | Full initial write | Reconvergence delta | Notes |
| --- | --- | ---: | ---: | --- |
| old BPF | TC/BPF map dataplane, no Linux FIB routes | `347.125s` for `5,938,022` BPF routes | `305.481s` for `285,184` ops | bpftool text batches |
| optimized BPF | TC/BPF map dataplane, no Linux FIB routes | `228.919s` map writer, or `~254.396s` clean first install with rp_filter | `118.626s` for `1,106,971` ops | direct syscall helper, caches |
| K2c | Linux main FIB, custom kernel bulk path | `71.162s` for `6,131,994` routes | `55.026s` for `198,869` ops | per-netns route lock/no-notify upper bound |
| K3c | Linux main FIB, cached fib_info custom kernel path | `10.552s` total, `7.867s` kernel write for `6,131,994` routes | `6.791s` for `198,860` ops | fastest current kernel prototype |

Interpretation:

- The optimized BPF path is much faster than the previous BPF controller, and it
  avoids RTNL entirely.
- It is still slower than K2c/K3c for route installation time, especially K3c.
- BPF's current bottleneck is not just BPF map syscall speed. The zero-change
  floor shows that route discovery through `birdc show route primary` across
  thousands of containers is already around `94s`.
- K3c is much faster because it writes Linux FIB through a bulk kernel path and
  avoids repeated `fib_info` construction. BPF avoids Linux FIB semantics, so it
  solves a different deployment problem.

## Next Optimization Direction

The next BPF target should be event-driven route changes:

```text
BIRD selected-route event
  -> persistent controller/agent
  -> dirty router set + changed prefix keys
  -> direct bpf(2) map update/delete only for dirty routers
```

This would remove the all-container `birdc show route primary` scan from the
reconvergence path. With the current optimized writer, map apply is no longer
the only cost; the controller needs to know which routers changed without
dumping every RIB again.

Lower-priority optimizations:

- Test whether `BPF_MAP_*_BATCH` works for this LPM trie kernel/map type. If it
  does, replace per-element helper syscalls with kernel batch calls.
- Keep interface cache as the default after the first install.
- Tune `--jobs`; higher jobs may help map apply but can amplify Docker/BIRD
  command pressure during route dumps.
- Move more parsing into a long-lived process if BIRD event export is delayed,
  but this is secondary to eliminating full RIB dumps.

