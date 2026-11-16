# SeedEMU BPF Dataplane Prototype

This directory is intentionally separate from the existing F/K1/K2c/K3c
experiments. It tests a different idea:

```text
BIRD selected-route delta
  -> host/controller
  -> update BPF LPM trie / nexthop maps
  -> TC program forwards packets
  -> no massive Linux main-FIB write, no RTNL contention
```

## Current Stage

`smoke_bpf_dataplane.py` is a self-contained feasibility test. It creates:

```text
h1 -- r1 -- r2 -- h2
```

`r1` and `r2` keep `net.ipv4.ip_forward=0` and do not have remote Linux FIB
routes. TC BPF programs on router interfaces do IPv4 LPM lookup in BPF maps and
redirect packets with `bpf_redirect_neigh()`.

Run on the VM:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends clang llvm bpftool iproute2 iputils-ping
sudo python3 /home/lxl/bird/bpf_dataplane/smoke_bpf_dataplane.py
```

Success means packet-level forwarding can work without installing remote routes
into Linux main FIB.

As of 2026-06-29, the SeedEMU container prototype also passed a 3083-scale VM
run with `bpf-3083`: 2757 brdnode routers were programmed with BPF routes and a
cross-AS ping succeeded while the source router's Linux main FIB still reported
the destination as unreachable.

## What A Real SeedEMU Integration Still Needs

- A BIRD route exporter that converts selected routes or delta events into BPF
  LPM map updates.
- A per-router pinned map layout, not per-interface duplicated maps.
- A nexthop resolver for each route: BIRD nexthop IP -> egress ifindex and
  neighbor entry.
- A verification flow based on packet reachability and BPF map consistency, not
  `ip route | wc`.
- Fallback behavior for local/control-plane traffic that should still use Linux
  FIB.

The smoke test validates only the kernel dataplane mechanism. The SeedEMU
prototype validates packet-level compatibility for basic IPv4 routes, but it is
still a prototype and does not yet cover IPv6, ECMP, policy routing, or full TTL
semantics.

## Running Against SeedEMU Containers

After a SeedEMU project has been built, started, and BIRD has converged:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name <compose-project> \
  --jobs 4
```

For a small pilot:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name <compose-project> \
  --limit-containers 20 \
  --jobs 4
```

The script:

- reads `birdc show route primary` from each running brdnode;
- installs one pinned BPF LPM map per router;
- attaches the TC program to all non-loopback router interfaces;
- writes only `via ... on <dev>` routes into the BPF map;
- leaves connected/dev-only routes to the normal Linux connected FIB path.
- disables `rp_filter` inside brdnode containers before attaching BPF, because
  reverse-path filtering can drop packets that intentionally bypass Linux FIB
  route lookup.

The default mode does not decrement TTL. The BPF program has an experimental
`--decrement-ttl` path, but the current SeedEMU-compatible mode is the default
no-TTL-decrement mode.

Useful refresh/reconvergence options:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name <compose-project> \
  --jobs 8 \
  --skip-rp-filter-disable \
  --delta-from-run-dir /home/lxl/bird/runs/<previous-bpf-run>
```

- `--apply-mode syscall` uses the direct `bpf(2)` helper instead of bpftool
  batch parsing. This is the current fastest update path.
- `--no-text-batches` skips bpftool text batch generation and uses route-cache
  JSON plus binary op files for the syscall helper.
- `--skip-rp-filter-disable` should be used after the first BPF install has
  already disabled `rp_filter` in every brdnode. A 20-node probe improved from
  `11.373s` to `9.271s`.
- `--delta-from-run-dir` compares the previous `bpftool_batches` with the new
  selected-route snapshot and applies only changed BPF map keys. It also keeps
  the already-attached TC program and pinned map, so it avoids program reload
  and reattach work when the baseline exists.
- The full refresh path now loads/fills the new BPF map before attaching the TC
  filter. This avoids a short empty-map window during refresh.

## 3083-Scale Result On 2026-06-29

VM preparation and control-plane startup:

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

Observed timings:

- build: `167.664s`
- docker compose up: `2893.833s`
- start-bird: `482.100s`
- brdnode BIRD socket check: `2757/2757` passed

BPF dataplane install:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name bpf-3083 \
  --jobs 8 \
  --force-compile
```

Observed result:

- brdnode success: `2757/2757`
- routes installed in BPF maps: `5,938,022`
- BIRD prefixes seen: `6,131,994`
- duration: `347.125s`
- remote run directory:
  `/home/lxl/bird/runs/bpf_dataplane_bpf_3083_20260629_102620`

Packet validation used temporary host containers:

```bash
docker run -d --name bpf3083-hsrc \
  --network bpf-3083_net_1277_net_4_5 \
  --ip 5.253.68.100 \
  --cap-add NET_ADMIN \
  handsonsecurity/seedemu-base:latest sleep infinity

docker run -d --name bpf3083-hdst \
  --network bpf-3083_net_1439_net_3_4 \
  --ip 6.159.1.100 \
  --cap-add NET_ADMIN \
  handsonsecurity/seedemu-base:latest sleep infinity

docker exec bpf3083-hsrc ip route replace default via 5.253.68.254
docker exec bpf3083-hdst ip route replace default via 6.159.1.253
```

Validation facts:

- `docker exec as1277brd-r4-1.4.4.253 ip route get 6.159.1.100` returned
  `Network is unreachable`.
- `docker exec bpf3083-hsrc ping -c 5 -W 1 6.159.1.100` returned `5/5`, 0%
  packet loss.
- `docker exec as1439brd-r4-1.4.5.159 ip route get 5.253.68.100` returned
  `Network is unreachable`.
- `docker exec bpf3083-hdst ping -c 5 -W 1 5.253.68.100` returned `5/5`, 0%
  packet loss.
- Source BPF stats after the first ping showed `redirects=5`, `drops=0`,
  `ttl_drops=0`.

## Throughput And Loss On 2026-06-29

Temporary iperf hosts were created with brdnode images because the base image
does not include `iperf3`:

```text
bpf3083-iperf-src: 5.253.68.101/24 via 5.253.68.254
bpf3083-iperf-dst: 6.159.1.101/24 via 6.159.1.253
```

Local result directory:

```text
/data/lxl/seed-emulator/topology/bird/runs/bpf_perf_bpf_3083_20260629_190012
```

Results:

| Test | Sender | Receiver | Loss / Retransmits |
| --- | ---: | ---: | ---: |
| TCP 1 stream, 15s | `10.182 Gbit/s` | `10.180 Gbit/s` | `994` retransmits |
| TCP 4 streams, 15s | `36.395 Gbit/s` | `36.394 Gbit/s` | `4205` retransmits |
| TCP reverse, 15s | `9.849 Gbit/s` | `9.847 Gbit/s` | `749` retransmits |
| UDP target 1G, 15s | `0.716 Gbit/s` | `0.709 Gbit/s` | `0.881%` loss |
| UDP target 5G, 15s | `0.702 Gbit/s` | `0.696 Gbit/s` | `0.867%` loss |

BPF stats across the iperf run:

| Router side | Packets | Redirects | Misses | BPF drops |
| --- | ---: | ---: | ---: | ---: |
| Source AS1277 router | `6,317,240` | `4,165,594` | `2,151,636` | `0` |
| Destination AS1439 router | `6,317,101` | `2,151,283` | `4,165,811` | `0` |

Interpretation:

- The BPF program did not report dataplane drops, checksum drops, TTL drops, or
  map store drops during the throughput test.
- The UDP generator did not reach the requested 1G/5G rates in this container
  setup, so the UDP number should be read as a userspace/container generation
  limit plus path behavior, not as a pure BPF forwarding ceiling.

## Reconvergence Behavior On 2026-06-29

Two network-wave experiments were run by stopping 10 brdnode containers each
time and then refreshing the BPF dataplane. Continuous ping was left running
while the writer updated maps.

Local full-refresh result directory:

```text
/data/lxl/seed-emulator/topology/bird/runs/bpf_reconv_bpf_3083_20260629_111118
```

Full refresh after stopping 10 brdnodes:

- stopped containers: `10`
- writer success: `2747/2747` brdnodes
- BPF route snapshot: `5,656,710`
- writer wall time: `444.751s`
- writer summary duration: `443.891s`
- continuous ping during writer: `8681/8681`, `0%` loss, average RTT `0.136ms`
- Linux main FIB validation: `ip route get` still returned `Network is
  unreachable`, so forwarding was still coming from BPF rather than main FIB.

Full-refresh phase totals across all successful brdnodes:

| Phase | Total seconds | Average per brdnode |
| --- | ---: | ---: |
| Build bpftool commands | `901.728` | `0.328` |
| Dump BIRD routes | `634.266` | `0.231` |
| Attach TC program | `597.812` | `0.218` |
| Read interfaces | `482.840` | `0.176` |
| Load BPF program/map | `448.726` | `0.163` |
| Apply bpftool batch | `432.965` | `0.158` |

Local delta-refresh result directory:

```text
/data/lxl/seed-emulator/topology/bird/runs/bpf_reconv_delta_bpf_3083_20260629_112337
```

Delta map update after stopping another 10 brdnodes:

- stopped containers in this wave: `10`
- total stopped brdnodes after both waves: `20`
- writer success: `2737/2737` brdnodes
- BPF route snapshot: `5,425,003`
- changed BPF map operations applied: `285,184`
- updates: `75,441`
- deletes: `209,743`
- missing baselines: `0`
- writer wall time: `306.302s`
- writer summary duration: `305.481s`
- continuous ping during writer: `5968/5968`, `0%` loss, average RTT `0.125ms`

Delta-refresh phase totals:

| Phase | Total seconds | Average per brdnode |
| --- | ---: | ---: |
| Dump BIRD routes | `793.504` | `0.290` |
| Read interfaces | `550.512` | `0.201` |
| Build candidate commands | `543.461` | `0.199` |
| Compare against baseline | `272.567` | `0.100` |
| Apply changed bpftool ops | `244.478` | `0.089` |
| Load BPF program/map | `0.005` | `0.000002` |
| Attach TC program | `0.004` | `0.000001` |

Delta mode reduced reconvergence writer time from `443.891s` to `305.481s`
for this run, about `31%` faster. It avoided TC reload/reattach and avoided
rewriting millions of unchanged map entries. The remaining bottleneck is still
the all-container selected-route snapshot: `birdc show route primary`, interface
collection, and command construction are paid for every running brdnode.

The next major optimization should therefore avoid full route dumps on unchanged
routers. The practical path is a persistent BIRD/agent delta feed that records
selected-route changes as they happen, plus cached ifindex/nexthop information
so a reconvergence writer can skip brdnodes with no route-map changes.

## Optimized BPF Writer On 2026-06-29/30

The optimized controller path adds a direct syscall helper, route-cache JSON,
interface-cache reuse, binary op files, and a fast IPv4 prefix parser.

Optimized full install:

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

- brdnode success: `2757/2757`
- BPF routes installed: `5,938,022`
- route ops applied: `5,938,022`
- writer duration: `228.919s`
- previous BPF full baseline: `347.125s`

Because this first optimized run used `--skip-rp-filter-disable` after a fresh
VM reset, packet validation only succeeded after the one-time rp_filter pass was
run separately. That pass took `25.477s`, so a clean first-install estimate is
about `254.396s`.

Optimized reconvergence delta after stopping 10 brdnodes:

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

- writer success: `2747/2747`
- route ops applied: `1,106,971`
- updates: `257,674`
- deletes: `849,297`
- route-cache hits: `2747/2747`
- interface-cache hits: `2747/2747`
- writer duration: `118.626s`
- continuous ping during writer: `4075` replies, no unreachable/timeout/loss
  lines in the captured log

Zero-change delta floor:

- writer success: `2747/2747`
- route ops applied: `0`
- writer duration: `94.261s`

The zero-change result is the key bottleneck signal. Even when no BPF map entry
changes, the current controller still pays about `94s` to run
`birdc show route primary` across all running brdnodes, rebuild route entries,
and compare snapshots. The next meaningful BPF optimization is therefore a
persistent BIRD/agent event feed that tells the controller which routers and
prefixes changed without a full RIB dump.

Full comparison with K2c/K3c is recorded in:

```text
bird/bpf_vs_k2c_k3c.md
```

Cleanup:

```bash
sudo python3 /home/lxl/bird/bpf_dataplane/seedemu_bpf_dataplane.py \
  --project-name <compose-project> \
  --cleanup
```
