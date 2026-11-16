# Clean Upstream 7.1.3 Per-Netns RTNL Comparison - 2026-07-15

## Purpose

This run checks whether an unmodified upstream kernel that already marks IPv4
`RTM_NEWROUTE` / `RTM_DELROUTE` handlers as per-netns RTNL operations can replace
our K3c kernel bulk route path for the 3083-scale SeedEMU BIRD/FIB experiment.

The tested kernel is `7.1.3-upstream-pernet-rtnl`. It was built from the local
clean upstream `linux-7.1.3.tar.xz` source with only `LOCALVERSION` changed.
There is no private `BIRD_RT_BULK`, `bird_rtbulk`, or `fib_bulk` code in this
kernel. The relevant upstream route handlers are:

```text
RTM_NEWROUTE -> inet_rtm_newroute -> RTNL_FLAG_DOIT_PERNET
RTM_DELROUTE -> inet_rtm_delroute -> RTNL_FLAG_DOIT_PERNET
```

## Workflow

Run path:

```text
bird/runs/upstream-pernet-713-F-3083_20260715_213944
project: topology-3083-upstream713
VM kernel: 7.1.3-upstream-pernet-rtnl
```

The experiment used the standard F writer:

```text
birdc show route primary
  -> parse selected IPv4 routes
  -> ip -force -batch -
  -> Linux main FIB
```

This is the normal rtnetlink path. It does not use K1/K2c/K3c private kernel
bulk APIs.

## Script Fix

During the first `writeFIB_F` attempt, `rtnl_solution_lab.py` still had
`--limit-containers` defaulting to `32`. That produced a valid smoke test but
not a full 3083-scale run. I changed the default to `0` and made
`run_f_real_main_fib_vm.sh` always pass `--limit-containers`.

The 32-container smoke test succeeded:

```text
containers: 32
route_ops: 71,552
F write wall: 4.701s
post-FIB sample: 10/10 passed
```

The full run below is after that fix.

## 3083-Scale Result

| Stage | Result |
|---|---:|
| Fast image build/load | 185.592s |
| Docker compose up | 2251.902s |
| Start BIRD | 1304s |
| brdnode BIRD started | 2757/2757 |
| Pre-FIB load gate | 138.144s, load1 stayed < 50 |
| Full F writer containers | 2757 |
| Full F route ops | 6,131,994 |
| Full F writer wall | 408.185s |
| Full F throughput | 15,022 ops/s |
| Full F max load1 | 15.507 |
| Post-FIB per-AS verifier | 312/312 AS passed |
| Post-FIB verifier time | 116.384s |

The F runner dumps and writes per container in one worker task, so it does not
record a separate global dump-wall metric. The per-container dump cost was very
small:

```text
sum of per-task birdc dump time: 39.516s
average per-container dump time: 0.014s
p95 per-container dump time: 0.026s
```

The observed bottleneck is still route installation, not RIB extraction.

## Comparison With K3c

| Kernel / method | Route path | Full route ops | Full FIB writer wall | Notes |
|---|---|---:|---:|---|
| clean upstream `7.1.3-upstream-pernet-rtnl` | standard rtnetlink `ip -batch` | 6,131,994 | 408.185s | upstream per-netns RTNL standard route path |
| K1 | private bulk + global RTNL | 6,131,994 | 402.388s | batching helps, but still global serialization |
| K2c | private bulk + per-netns route lock + fib_info mutex | 6,131,994 | 71.162s | shows global RTNL was a major cross-netns bottleneck |
| K3c `6.17.13-35` | K2c + cached fib_info fast path | 6,131,994 | 7.199s | Ubuntu 24.04 HWE K3c port |
| K3c `7.1.3` | K2c + cached fib_info fast path | 6,131,994 | 7.061s | same upstream generation with our K3c fast path |

## Interpretation

Upstream per-netns RTNL for `RTM_NEWROUTE` / `RTM_DELROUTE` is useful, but it is
not enough for this workload. On the real 3083-scale BIRD routes, clean 7.1.3
standard rtnetlink writes 6.13M routes in `408.185s`, only slightly better than
the older F/K1 range and still about `57.8x` slower than K3c 7.1.3.

This result strengthens the K3c conclusion: the bottleneck is not just whether
`RTM_NEWROUTE` can enter a per-netns RTNL section. The workload also needs:

1. a route bulk API that avoids per-route netlink command overhead;
2. controlled per-netns route serialization without global notifier costs;
3. batched/cached `fib_info` construction for many routes sharing the same
   nexthop/device/metric structure.

Clean upstream 7.1.3 improves the locking model in the standard path, but it
does not provide the bulk/cached FIB-object primitive that makes K3c reach the
5-7s full-write range.
