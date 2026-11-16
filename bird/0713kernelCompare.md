# K3c Kernel Version Runtime Comparison - 2026-07-13

## Scope

This note compares the 3083-scale K3c experiments currently available in the local run artifacts, including the new Ubuntu 24.04 HWE 6.17 port tested on 2026-07-13.

The comparison uses the same high-level workflow:

1. build / load delta-fast SeedEMU images;
2. `docker compose up` the 3083-scale topology;
3. start BIRD in brdnode containers with load gating;
4. wait for RIB stability;
5. write the selected BIRD RIB routes into Linux main FIB with the K3c/K1 bulk writer;
6. verify FIB correctness on one deterministic brdnode per AS;
7. simulate reconvergence by stopping 10 brdnodes;
8. collect BIRD selected-route delta events and write the delta FIB update with the same bulk path;
9. verify post-delta FIB correctness again.

The deterministic FIB correctness gate is:

```text
birdc show route count networks < ip route | wc -l
```

The check is run after load1 drops below 50 and uses one fixed representative brdnode per AS.

## Version Set

| Label | Kernel release | Role in comparison | Local artifact status |
|---|---|---|---|
| Old K3c | `6.8.12-birdrtbulk-k3c` | Earlier fast K3c baseline before the later fibref safety fix | Has local 3083 results |
| K3c fibref | `6.8.12-birdrtbulk-k3c-fibref` | Previous fixed K3c kernel after the `fib_info` / netdevice reference fix | Has local 3083 results |
| K3c 6.14 rtnlfix | `6.14.11-birdrtbulk-k3c-fibref-rtnlfix` | Compatibility port to 6.14 with RTNL assertion fixes | Has local 3083 results |
| K3c 6.17 Ubuntu 24.04 HWE | `6.17.13-35-birdrtbulk-k3c-fibref-rtnlfix` | Ubuntu 24.04 HWE source based K3c port | Built, booted, and measured locally |
| K3c 7.1 rtnlfix | `7.1.3-birdrtbulk-k3c-fibref-rtnlfix` | Compatibility port to a newer upstream stable branch tested earlier | Has local 3083 results |

The 6.17 port was built from Ubuntu HWE source package `linux-hwe-6.17_6.17.0-35.35~24.04.1`; the VM booted kernel was:

```text
6.17.13-35-birdrtbulk-k3c-fibref-rtnlfix
```

The previous bootable kernels and VM XML backups were preserved before direct-booting this kernel.

## 3083-Scale Runtime Table

All rows use the same topology scale class: `topology_services = 3085`. The current K3c workflow records `2757` brdnode routers. The FIB write target is Linux main FIB.

| Kernel | Run path | Build | Up | Start BIRD | RIB stable wait | Full FIB writer wall | Full writer details | Reconvergence delta | Post-delta verification |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| `6.8.12-birdrtbulk-k3c` | `bird/runs/k3c-rerun-3083_20260701_114246` | 191.845s | 2997.117s | 919.520s | 42.620s | 10.226s | dump 2.554s, write 7.668s, ops 6,131,994, fail 0 | writer 6.386s for 198,832 commands, fail 0 | Not from current deterministic verifier; run itself succeeded |
| `6.8.12-birdrtbulk-k3c-fibref` | `bird/packaged/.../K3c/runs/k3c-fibref-amd-3083-clean_20260706_180623` | 79.375s | 2271.337s | 765.596s | 0.001s | 100.944s | dump 2.624s, write 98.317s, ops 6,131,994, fail 0 | writer 9.691s for 198,877 commands, fail 0 | Run-level reconvergence succeeded |
| `6.14.11-birdrtbulk-k3c-fibref-rtnlfix` | `bird/packaged/.../K3c/runs/k3c-614fix-3083-direct_20260712_121826` | 210.553s | 3514.613s | 829.745s | 4.314s initially | 5.619s after retry | retry dump 2.289s, write 3.328s, ops 6,131,994, fail 0 | dump 11.050s, writer 38.095s, 6,326,537 commands, fail 0 | 312/312 AS passed after retry and after delta |
| `6.17.13-35-birdrtbulk-k3c-fibref-rtnlfix` | `bird/packaged/.../K3c/runs/k3c-617fix-3083-direct_20260713_104122` | 185.945s | 3552.599s | 937.346s | 245.433s | 7.199s | dump 2.333s, write 4.862s, ops 6,131,994, fail 0 | duration 167.759s: stop 39.832s, wait 69.265s, dump 11.487s, writer 38.356s, 6,318,189 commands, fail 0 | Initial 312/312 passed; post-delta 312/312 passed |
| `7.1.3-birdrtbulk-k3c-fibref-rtnlfix` | `bird/packaged/.../K3c/runs/k3c-713fix-3083-direct_20260712_135300` | 170.801s | 1685.380s | 797.257s | 285.699s | 7.061s | dump 2.216s, write 4.843s, ops 6,131,994, fail 0 | dump 11.283s, writer 37.135s, 6,297,634 commands, fail 0 | 312/312 AS passed after full and after delta |

## 6.17 Detailed Result

Run:

```text
/data/lxl/seed-emulator/topology/bird/packaged/rtnl_fib_methods_repro_20260701_133858/K3c/runs/k3c-617fix-3083-direct_20260713_104122
```

Command shape:

```bash
VM_HOST=192.168.122.60 VM_USER=lxl \
PROJECT_NAME=topology-3083-k3c-617fix \
OUTPUT_DIR=/data/lxl/seed-emulator/topology/output_3083 \
RUN_NAME=k3c-617fix-3083-direct_20260713_104122 \
K3C_KERNEL=6.17.13-35-birdrtbulk-k3c-fibref-rtnlfix \
K1_KERNEL=6.17.13-35-birdrtbulk-k3c-fibref-rtnlfix \
RESET_VM=0 SKIP_BUILD_UP=0 BUILD_MODE=delta-fast \
INITIAL_LOAD_ONLY_STABLE=0 INITIAL_STABLE_SECONDS=120 \
RESTART_JOBS=1 RESTART_DELAY=0.1 \
INITIAL_WRITE_JOBS=16 K1_DUMP_JOBS=16 K1_BATCH=16000 \
DELTA_WRITE_JOBS=8 K1_DELTA_BATCH=16000 K1_OPS_PER_ATTR=2000 \
RUN_FULL_COMPARISON=0 \
./run_3083_k3c_delta_experiment.sh
```

Full initial FIB writer:

```text
containers:                  2757
ops_requested:               6,131,994
wall_seconds:                7.1985
dump_wall_seconds:           2.3328
write_wall_seconds:          4.8619
kernel_apply_seconds_sum:    28.1152
fail_count:                  0
multipath_first:             223,843
```

Initial deterministic FIB verification:

```text
as_count:        312
failure_count:  0
duration:       122.257s
load1 at gate:  4.47
```

Reconvergence delta:

```text
total duration:                 167.759s
stop_victims:                   39.832s
wait_rib_stable:                69.265s
delta dump wall:                11.487s
raw route events:               14,653,041
collapsed events / commands:    6,318,189
delta writer wall:              38.356s
writer success_count:           6,110,234
writer fail_count:              0
ignored delete miss:            207,955
kernel_apply_seconds_sum:       3.016s
writer jobs:                    8
```

Post-delta deterministic FIB verification:

```text
as_count:        312
failure_count:  0
duration:       118.982s
load1 at gate:  9.32
```

Kernel safety grep on the 6.17 boot:

```text
BUG / Oops / WARNING / ASSERT_RTNL / refcount / use-after-free /
unregister_netdevice / soft lockup / hung task

matched lines: 0
log copy: bird/packaged/.../K3c/runs/k3c-617fix-3083-direct_20260713_104122/kernel_safety_grep_617.log
```

## Interpretation

### 1. 6.17 preserves K3c full-write performance

The new 6.17 HWE port writes the 6.13M-route full FIB in `7.199s`, essentially the same order as the previously measured 7.1.3 port (`7.061s`) and the 6.14 retry result (`5.619s`). This indicates the K3c bulk route path and the fibref/rtnlfix safety changes remain effective on the Ubuntu 24.04 HWE 6.17 source base.

### 2. The 6.17 `up` time is slow, but not due to the K3c route writer

The 6.17 run spent `3552.599s` in Docker compose up. During monitoring, the slow section was dominated by Docker creating thousands of networks. The VM load stayed far below the high-load recovery threshold, and the final up stage succeeded.

Observed non-fatal Docker symptoms during up:

```text
Network ... Created
Network ... Error
```

and repeated Docker daemon messages:

```text
Error handling plugin refcount operation ... legacy plugin: Unable to find plugin without name
```

These did not prevent the topology from starting, but they explain why the up stage should not be interpreted as K3c kernel write-path performance. The relevant K3c dataplane writer metric is the full writer and delta writer timing after BIRD RIB convergence.

### 3. 6.17 passes both correctness gates

The run passed:

```text
initial FIB verify:    312/312 AS representatives
post-delta FIB verify: 312/312 AS representatives
```

Both used the deterministic route-count condition after load gating. This makes the 6.17 row directly comparable to the 6.14 and 7.1 rtnlfix runs.

### 4. Reconvergence delta volume is large in the newer workflow

The old 6.8 K3c reconvergence run collapsed about `198k` delta commands, while the 6.14/6.17/7.1 rtnlfix workflow collapses about `6.3M` commands. Therefore, old 6.8 delta writer time is not directly comparable to the newer runs. For the comparable newer workflow:

```text
6.14.11: 38.095s writer for 6,326,537 commands
6.17.13: 38.356s writer for 6,318,189 commands
7.1.3:   37.135s writer for 6,297,634 commands
```

The 6.17 result is consistent with 6.14 and 7.1.

### 5. Safety status across tested rtnlfix ports

The current checked safety signatures found zero matched kernel error lines for the newer rtnlfix ports:

```text
6.14.11: 0 matched lines
6.17.13: 0 matched lines
7.1.3:   0 matched lines
```

This does not prove complete kernel safety for all Linux networking paths, but it is strong smoke-test evidence for the SeedEMU/BIRD route-write workload we care about.

## Current Conclusion

The 6.17 Ubuntu 24.04 HWE K3c port is now a real measured row, not a pending target. At 3083 scale it:

1. booted successfully in the VM as `6.17.13-35-birdrtbulk-k3c-fibref-rtnlfix`;
2. completed the full SeedEMU K3c workflow;
3. wrote 6,131,994 full FIB route ops in `7.199s`;
4. completed reconvergence delta writing in `38.356s` for 6,318,189 commands;
5. passed deterministic initial and post-delta FIB verification on `312/312` AS representatives;
6. produced zero matched kernel safety warning/error lines in the checked signatures.

For paper-quality claims, the strongest current statement is:

```text
K3c-fibref-rtnlfix has been validated at 3083 scale on 6.14.11,
Ubuntu-HWE-based 6.17.13, and 7.1.3. Across these tested kernels, the
full main-FIB write stays within the 5.6-7.2s range after RIB stabilization,
the newer reconvergence delta writer stays around 37-38s for roughly 6.3M
commands, deterministic FIB verification passes 312/312 AS representatives,
and the checked kernel safety signatures show no BUG/Oops/WARNING/
ASSERT_RTNL/refcount/use-after-free/unregister_netdevice/soft-lockup/hung-task
lines.
```

## Addendum: Clean Upstream 7.1.3 Per-Netns RTNL

On 2026-07-15 I also tested a clean upstream-derived
`7.1.3-upstream-pernet-rtnl` kernel. This kernel contains upstream
`RTNL_FLAG_DOIT_PERNET` support for IPv4 `RTM_NEWROUTE` and `RTM_DELROUTE`, but
does not contain our private `BIRD_RT_BULK` / K3c path.

Run path:

```text
bird/runs/upstream-pernet-713-F-3083_20260715_213944
project: topology-3083-upstream713
full result: bird/runs/realfib-F_20260715_225119_topology-3083-upstream713/summary.json
```

Result:

```text
build:                  185.592s
up:                     2251.902s
start-bird:             1304s, 2757/2757 brdnodes
full standard F writer: 408.185s for 6,131,994 route ops
writer throughput:      15,022 ops/s
max load1 in writer:    15.507
post-FIB verifier:      312/312 AS representatives passed
```

This is much slower than K3c on the same scale:

```text
clean upstream 7.1.3 standard rtnetlink: 408.185s
K3c 7.1.3 private bulk/cached fib_info:   7.061s
K3c 6.17 Ubuntu HWE port:                 7.199s
```

Therefore, upstream per-netns RTNL for standard route messages is a useful
kernel direction, but it does not replace K3c for this workload. The 3083-scale
route install still needs bulk submission and cached/batched `fib_info`
construction; otherwise standard `ip -force -batch` remains hundreds of seconds
slower than K3c.
