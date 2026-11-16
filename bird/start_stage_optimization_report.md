# Start-BIRD / Start-Kernel Optimization Report

Date: 2026-06-10

## Context

The slow `run_all.py` stages are not primarily caused by RTNL congestion. In the
observed 1897-container topology, load average stayed far below the high-load
reset threshold while start-bird/start-kernel were slow. The expensive part was
entering each container repeatedly, especially through `sudo nsenter`.

## Entry Method Benchmark

Environment:

- VM: `lxl@192.168.122.60`
- Compose project: `topology-291e63c1`
- Running containers: 1897
- Benchmark sample: 300 containers
- Parallel jobs: 16

| Test | Wall Time | Avg Per Command | Failures |
| --- | ---: | ---: | ---: |
| `docker exec true` sequential | 49.382s | 164.598ms | 0 |
| `sudo nsenter true` sequential | 93.808s | 312.684ms | 0 |
| `docker exec probe` sequential | 48.988s | 163.285ms | 0 |
| `sudo nsenter probe` sequential | 95.042s | 316.798ms | 0 |
| `docker exec true` parallel | 2.663s | 137.721ms | 0 |
| `sudo nsenter true` parallel | 59.407s | 3136.739ms | 0 |
| `docker exec probe` parallel | 2.838s | 147.927ms | 0 |
| `sudo nsenter probe` parallel | 58.927s | 3105.865ms | 0 |

Conclusion: in this VM and Docker setup, `docker exec` is the correct default
entry path. Parallel `sudo nsenter` performs badly because every command pays
sudo/nsenter process overhead and contends on local process scheduling.

## Simulation-Semantic Integration

The best fit for SeedEMU is not to treat start-kernel as a runtime control-plane
operation. The generated Docker output already contains each router's static
`/etc/bird/conf/kernel.conf` source file. Therefore the kernel protocol should
be rewritten in the build context before image build:

1. Find every Dockerfile `COPY <source> /etc/bird/conf/kernel.conf`.
2. Rewrite that source file to `export all` with a stable jittered scan time.
3. Build images with the corrected config.
4. Start BIRD once with `docker exec -d`.
5. Skip runtime `birdc configure`.

This preserves the emulation semantics: every router still owns its own BIRD
process and FIB in its own container network namespace, but we avoid thousands
of post-start config rewrites and repeated entry calls.

## Implemented Defaults

- `run_all.py` default `--entry-method docker-exec`
- `run_all.py` default `--kernel-config-mode build-context`
- start-bird uses parallel `docker exec -d`
- runtime start-kernel remains available with `--kernel-config-mode runtime`
- runtime start-kernel now uses one combined entry per router instead of
  separate probe/write/check/configure entries
- fast Docker archive `RepoTags` now include explicit `:latest`, fixing
  Docker 29 `docker load` failures

## Validation Results

Existing up topology, no rebuild/up, optimized start-bird only:

- Command: `python3 bird/run_all.py output/ --vm 192.168.122.60 --no-build --no-up --no-start-kernel`
- Local summary: `bird/runs/topology-1899_20260610_091032/summary.json`
- start-bird: 1897 eligible, 1897 triggered, 0 failed
- start-bird duration: 42.115s
- Full BIRD process verification: 1897/1897 running, 11.488s

Build-context prepare-kernel validation:

- Remote summary: `/home/lxl/bird/runs/prepare-kernel-test/summary.json`
- kernel.conf files found: 1897
- updated: 1897
- failed: 0
- duration: 1.266s

Clean VM full run after optimization:

- Command: `python3 bird/run_all.py output/ --vm 192.168.122.60 --build-jobs 16 --up-jobs 8`
- Local summary: `bird/runs/topology-1899_20260610_091938/summary.json`
- Remote summary copy: `bird/runs/topology-1899_20260610_091938/remote_summary.json`
- Remote run dir: `/home/lxl/bird/runs/topology-1899_20260610_091938_topology-291e63c1`
- prepare-kernel: 1897 files updated, 0 failed, 1.243s
- build: success, 1897 images, 102.923s
- up: success, 38/38 batches, 0 failed, 793.549s
- start-bird: success, 1897/1897 triggered, 0 failed, 41.892s
- start-kernel: skipped because `kernel.conf` was prepared in build contexts
- Post-run reset: completed with containers=0, volumes=0, custom networks=0,
  and `handsonsecurity/seedemu-base` / `handsonsecurity/seedemu-router`
  present as both `:2.0` and `:latest`

## Recommendation

Use the build-context kernel configuration plus docker-exec BIRD start as the
default experiment path. It is faster and better aligned with the SeedEMU model
than runtime start-kernel. The earlier RTNL solution ideas still matter for
actual route convergence after BIRD is running, but they should not be used to
solve start-bird/start-kernel orchestration overhead.
