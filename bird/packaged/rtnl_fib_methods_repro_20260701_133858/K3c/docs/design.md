# BIRD/RTNL K1-K2c-K3c Design And Reproduction Guide

更新时间：2026-06-22 15:20 CST

本文记录当前 K1/K2c/K3a/K3b/K3c 内核方案的思路、实现位置、复现实验命令，以及 K4 的可行性分析。目标是让后续实验能快速判断：

- 为什么 K2c 有明显效果。
- K3a 如何把受控 route-only 快路径接入标准 rtnetlink。
- K3b/K3c 如何继续优化私有 bulk API 的内核 apply 成本。
- 如何复现 3083 规模 full FIB sync 和 reconvergence delta。
- K4 要改哪里，为什么当前还不能直接进入 3083 全流程。

## 1. 问题背景

SeedEMU 大规模拓扑中，如果每个 router 容器里的 BIRD 同时通过 kernel protocol 把 selected routes 写入 Linux main FIB，会形成大量 `RTM_NEWROUTE/RTM_DELROUTE` 请求。Linux route 配置路径依赖全局 RTNL 以及若干 FIB 共享结构，许多容器同时写会造成：

- `rtnl_mutex` 等待累积。
- 大量进程 D-state。
- SSH/docker exec 尾延迟变长。
- VM load 长时间很高，控制面/数据面收敛和验证都被拖慢。

当前工程里有两类解决路线：

| 路线 | 代表 | 解决点 |
|---|---|---|
| 用户态调度 | F 方法、delta event | 减少同时写 FIB 的 writer 数，或减少需要写入的 route ops |
| 内核路径优化 | K1、K2c | 降低同样 route ops 下的内核写入成本 |

K2c 不是替代 delta 的方案。更合理的组合是：

```text
start-bird 只做 RIB
  -> 初始 full RIB-to-FIB 使用 K2c bulk writer
  -> 启用 BIRD delta event
  -> 故障后只写 collapsed delta routes
  -> delta 写入也使用 K2c bulk writer
```

## 2. K1：Global-RTNL Bulk Route API

### 2.1 思路

K1 不改变内核语义边界。它新增一个私有 generic-netlink family：`BIRD_RT_BULK`，用户态一次提交一组 IPv4 route ops，内核在一次 global RTNL 临界区内循环调用 `fib_table_insert()` / `fib_table_delete()`。

K1 解决的是这些成本：

- 每条 route 一个 netlink message。
- 每条 route 一个 ACK/解析/调度开销。
- 每条 route 一次 `rtnl_lock()` / `rtnl_unlock()`。

K1 没解决的是：

- 不同 network namespace 之间仍共用 global RTNL。
- 多 writer 并发时仍会抢同一个全局锁。

K1 调用链：

```text
kernel_bulk_*_writer.py
  -> setns(target container netns)
  -> generic netlink BIRD_RT_BULK/APPLY_V4
  -> net/ipv4/fib_bulk.c
  -> rtnl_lock()
  -> loop fib_table_insert/delete()
  -> rtnl_unlock()
```

### 2.2 实现文件

| 文件 | 作用 |
|---|---|
| `bird/kernel/patches/0001-bird-rtbulk-global-rtnl.patch` | K1 内核 patch，新增 `include/uapi/linux/bird_rtbulk.h` 和 `net/ipv4/fib_bulk.c` |
| `bird/kernel/tools/kernel_bulk_writer.py` | synthetic bulk writer |
| `bird/kernel/tools/kernel_bulk_birdc_writer.py` | 直接从 `birdc show route primary` dump 并写 K1/K2c bulk |
| `bird/kernel/tools/kernel_bulk_command_writer.py` | 从 delta command 文件写 K1/K2c bulk |
| `bird/run_3083_k1_delta_reconvergence.sh` | 3083 K1 full + reconvergence delta runner |

K1 route op 结构是紧凑二进制数组，字段包括：

- op: add / replace / del
- prefix / prefix length
- gateway
- output ifindex
- table
- metric
- protocol
- scope

当前范围：

- IPv4 only。
- 当前 network namespace only。
- 当前实验里 multipath policy 默认 `first`，即 ECMP 取第一个 nexthop。
- 不处理 IPv6、nexthop object、VRF、lwtunnel、完整 ECMP 语义。

### 2.3 K1 实测结果

Synthetic：

| case | route ops | jobs,batch | wall | throughput | 结果 |
|---|---:|---:|---:|---:|---|
| K1 `100x600k_j4` | 60,000,000 | `4,1024` | `585.122s` | `102,543 routes/s` | success |

真实 3083 safe-jobs：

| 阶段/指标 | 结果 |
|---|---:|
| build | `165.913s` |
| docker compose up | `2929.003s` |
| restart_bird | `933.490s` |
| initial full writer | `402.388s` |
| initial full route ops | `6,131,994 / 6,131,994` |
| reconvergence total | `622.785s` |
| prepare delta offsets | `519.594s` |
| delta writer | `63.639s` |
| delta route ops | `198,804 / 198,804` |
| ignored delete miss | `0` |

K1 的重要教训：

- `jobs=8,batch=2000` full write 能写完，但可能让 VM load 长时间不可恢复。
- `jobs=1,batch=256` 能稳定跑完整 3083 full + delta。
- K1 delta writer 本身不慢，`63.639s` 很好；总时间长主要是 offset 准备当时被串行化到 `519.594s`。
- K1 是可靠的 global-RTNL bulk baseline，不是最终跨 namespace 并发方案。

### 2.4 K1 复现命令

前置：确认 VM 可 SSH，且 `/home/lxl/bird/reset_docker_vm.sh` 可用。

```bash
cd /data/lxl/seed-emulator/topology

RUN_NAME=k1-delta-reconv-3083_safej1_repro_$(date +%Y%m%d_%H%M%S) \
RESET_VM=1 \
INITIAL_WRITE_JOBS=1 \
DELTA_WRITE_JOBS=1 \
OFFSET_JOBS=16 \
DELTA_JOBS=16 \
K1_BATCH=256 \
K1_DELTA_BATCH=256 \
RUN_FULL_COMPARISON=0 \
LOAD_THRESHOLD=50 \
MAX_WAIT_SECONDS=7200 \
POLL_INTERVAL=30 \
SSH_CONNECT_TIMEOUT=30 \
bird/run_3083_k1_delta_reconvergence.sh
```

输出位置：

```text
bird/runs/<RUN_NAME>/main_summary.json
bird/runs/<RUN_NAME>/reconvergence/summary.json
bird/runs/<RUN_NAME>/verify_initial_fib.json
```

## 3. K2c：Per-netns Bulk Route Lock + no-notify Upper-bound

### 3.1 从 K2 到 K2c 的演进

K2 的第一步是把 K1 的 global RTNL 换成每个 network namespace 一个私有 route mutex：

```text
BIRD_RT_BULK/APPLY_V4
  -> mutex_lock(per-netns route_lock)
  -> loop fib_table_insert/delete()
  -> mutex_unlock(per-netns route_lock)
```

这一步直接失败，因为 `fib_table_insert()` 会调用 FIB notifier，而 notifier 路径里有 `ASSERT_RTNL()`。证据点：

```text
net/ipv4/fib_notifier.c: call_fib4_notifiers() -> ASSERT_RTNL()
net/ipv4/fib_trie.c: fib_table_insert/delete 注释均为 "Caller must hold RTNL."
```

K2b 为了测性能上限，新增 `fib_config.fc_skip_notify`，在 BIRD_RT_BULK 私有路径里跳过：

- `call_fib_entry_notifiers()`
- `fib_notify_alias_delete()`
- `rtmsg_fib()`

K2b 说明 no-notify per-netns route update 很快，但在 `100x600k` 规模触发 `fib_create_info()` Oops。原因是 `fib_info` hash/list 这类共享结构也隐式依赖 RTNL。

K2c 在 K2b 上加了 `fib_info_hash_mutex`，显式保护：

- `fib_find_info_nh()`
- `fib_info_hash_move()`
- `fib_find_info()`
- fib_info hash/list 插入
- `fib_release_info()` removal

### 3.2 K2c 思路

K2c 当前是 controlled experiment，不是 production-safe 内核方案：

```text
user writer
  -> setns(target container netns)
  -> BIRD_RT_BULK/APPLY_V4
kernel
  -> per-netns route_lock
  -> skip route notifier/rtnetlink multicast
  -> explicit fib_info_hash_mutex for shared fib_info structures
  -> fib_table_insert/delete
```

K2c 为什么快：

- 不同 netns 的 route update 不再争 global RTNL。
- 每个 netns 内仍串行，避免同一 FIB trie 并发写。
- 同一 batch 内还是 bulk，减少 netlink 和 lock/unlock 成本。
- `fib_info_hash_mutex` 只保护跨 netns 共享的 `fib_info` hash/list，不把整个 route update 拉回 global RTNL。

K2c 的限制：

- no-notify：不会发送 route notifier，也不发 rtnetlink route multicast。
- 当前只验证 IPv4、main table、controlled `proto bird metric 232` 这类实验写入。
- 没有完整处理 IPv6、ECMP/nexthop object、VRF/lwtunnel、device unregister、fib rules mutation、netns teardown 的生产语义。
- 因此它适合做 SeedEMU/BIRD RTNL controlled experiment，不应直接上游或作为通用内核行为。

### 3.3 K2c 实现文件

| 文件 | 作用 |
|---|---|
| `bird/kernel/patches/0002-bird-rtbulk-pernetns-lock.patch` | 给 BIRD_RT_BULK 增加 per-netns route mutex |
| `bird/kernel/patches/0003-bird-rtbulk-pernetns-no-notify-upper-bound.patch` | 增加 `fc_skip_notify`，跳过 notifier/rtnetlink multicast |
| `bird/kernel/patches/0004-bird-rtbulk-k2c-fib-info-hash-mutex.patch` | 给 `fib_info` hash/list 加互斥，修复 K2b Oops |
| `bird/kernel/tools/kernel_bulk_birdc_writer.py` | 初始 full RIB-to-FIB K2c writer |
| `bird/kernel/tools/kernel_bulk_command_writer.py` | reconvergence delta K2c writer，支持 mixed batch |
| `bird/run_3083_k2c_delta_optimized.sh` | 当前推荐 3083 K2c full + delta wrapper |

### 3.4 K2c Mixed-batch Delta

2026-06-20 之后的 K2c 推荐用 mixed-batch delta writer：

- `K1_DELTA_MIXED_BATCHES=1`
- `OFFSET_JOBS=16`
- `DELTA_WRITE_JOBS=1`
- `K1_DELTA_BATCH=2000`

原因：

- sparse reconvergence delta 的单容器命令文件很小，replace/delete 拆开会产生很多小 batch。
- mixed batch 把同一容器的 replace/delete 合成一条 op 序列，减少 bulk submit 次数。
- K2c delta 写入高并发反而放大 `fib_info_hash_mutex` 和 route path 等待；真实语义实验中 `jobs=1` 更稳。
- offset 准备是 Docker exec/文件 I/O 成本，和 FIB writer 并发不同，因此用 `OFFSET_JOBS=16` 单独提速。

### 3.5 K2c 实测结果

Synthetic：

| case | route ops | jobs,batch | wall | throughput | 结果 |
|---|---:|---:|---:|---:|---|
| K2c `100x600k_j4` | 60,000,000 | `4,1024` | `132.815s` | `451,755 routes/s` | success |

真实 3083 K2c mixed：

| 阶段/指标 | 结果 |
|---|---:|
| kernel | `6.8.12-birdrtbulk-k2c` |
| build | `205.432s` |
| docker compose up | `3019.109s` |
| restart_bird | `1230.731s` |
| initial full writer | `71.162s` |
| initial full route ops | `6,131,994 / 6,131,994` |
| post-full verify | `312/312` AS reps passed |
| reconvergence total | `157.028s` |
| stop 10 victims | `28.112s` |
| wait RIB/load stable | `30.003s` |
| prepare delta offsets | `32.334s` |
| delta dump/collapse | `8.596s` |
| delta writer | `55.026s` |
| delta route ops | `198,869 / 198,869` |
| ignored delete miss | `0` |
| dmesg grep | no `BUG/Oops/panic/fib_create_info/bird_rtbulk/rtbulk/WARNING` |

与上一轮 K2c 非 mixed 相比：

| 指标 | 旧 K2c | K2c mixed | 改善 |
|---|---:|---:|---:|
| full writer | `95.537s` | `71.162s` | `1.34x` |
| reconvergence total | `338.218s` | `157.028s` | `2.15x` |
| delta writer | `236.353s` | `55.026s` | `4.30x` |
| delta kernel apply sum | `1826.924s` | `41.680s` | `43.83x` |

Replay sweep 只用于观察并发拐点，不作为语义正确性结果：

| jobs | replay wall |
|---:|---:|
| 2 | `45.688s` |
| 4 | `51.146s` |
| 8 | `61.906s` |
| 12 | `164.108s` |
| 16 | `179.387s` |

结论：K2c mixed 的真实 reconvergence 采用 `DELTA_WRITE_JOBS=1` 是合理的；盲目增大 jobs 会放大内核同步等待。

### 3.6 K2c 复现命令

推荐复现命令：

```bash
cd /data/lxl/seed-emulator/topology

RUN_NAME=k2c-delta-mixed-3083_repro_$(date +%Y%m%d_%H%M%S) \
K1_GRUB_ENTRY_ID='1>2' \
RESET_VM=1 \
INITIAL_WRITE_JOBS=8 \
DELTA_WRITE_JOBS=1 \
OFFSET_JOBS=16 \
K1_DELTA_MIXED_BATCHES=1 \
K1_DELTA_BATCH=2000 \
RUN_FULL_COMPARISON=0 \
LOAD_THRESHOLD=50 \
MAX_WAIT_SECONDS=7200 \
POLL_INTERVAL=30 \
SSH_CONNECT_TIMEOUT=30 \
bird/run_3083_k2c_delta_optimized.sh
```

`K1_GRUB_ENTRY_ID='1>2'` 是当前 VM 的实用写法，用于启动 `6.8.12-birdrtbulk-k2c`。之前 GRUB saved-entry 字符串在此 VM 上不稳定。

输出位置：

```text
bird/runs/<RUN_NAME>/main_summary.json
bird/runs/<RUN_NAME>/reconvergence/summary.json
bird/runs/<RUN_NAME>/comparison_summary.json
bird/runs/<RUN_NAME>/verify_initial_fib.json
```

复现后建议执行：

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR lxl@192.168.122.60 \
  "sudo -n dmesg -T | egrep -i 'BUG:|Oops|panic|fib_create_info|bird_rtbulk|rtbulk|WARNING:' | tail -n 80 || true"
```

实验结束后按固定流程清理 VM：

```bash
virsh destroy bird-rtnl-dev || true
virsh start bird-rtnl-dev

for i in $(seq 1 60); do
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR -o ConnectTimeout=5 lxl@192.168.122.60 "echo ready" && break
  sleep 5
done

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=20 \
  lxl@192.168.122.60 "/home/lxl/bird/reset_docker_vm.sh"
```

## 4. 验证规则

任何 full 或 delta RIB-to-FIB 实验都必须区分“命令成功”和“状态正确”。

### 4.1 start-bird 后

先等 VM `load1 < 50`，再抽固定 10 个不同 AS 的第一个 brdnode：

```bash
docker exec <container> birdc show protocols
docker exec <container> birdc show route count
```

要求：

- `birdc` 命令成功。
- BGP-like protocol 应为 Established。
- OSPF/device/direct/kernel 不应 failed/down/start-looping。

### 4.2 RIB 写入 FIB 后

先等 VM `load1 < 50`，再每个 AS 固定选一个 brdnode：

```bash
docker exec <container> birdc show route count
docker exec <container> sh -lc "ip route | wc -l"
```

要求：

```text
bird_network_count < ip_route_wc_count
```

当前脚本里常用：

```text
bird/verify_bird_fib_vm.py
```

## 5. K3a：标准 RTM_NEWROUTE/DELROUTE Route-only 化

### 5.1 想达到什么

K3a 的目标是让标准 rtnetlink route update 路径在受控实验 route 上避开 global RTNL，类似：

```text
ip -batch / BIRD kernel protocol
  -> RTM_NEWROUTE / RTM_DELROUTE
  -> inet_rtm_newroute / inet_rtm_delroute
  -> route-only per-netns lock
  -> fib_table_insert/delete
```

如果 K3a 成功，用户态不需要使用私有 `BIRD_RT_BULK` API；现有 `ip route replace ... proto bird metric 232` 这类标准 netlink route 写入就能获益。当前 3083 实验使用 host writer 生成标准 `ip route` batch，未直接切换 BIRD kernel protocol。

### 5.2 当前源码事实

Ubuntu 6.8 的 rtnetlink 支持 handler 标记：

```text
RTNL_FLAG_DOIT_UNLOCKED
```

rtnetlink 收到非 dump 请求时，如果 handler 有该 flag，就不在外层拿 global RTNL，直接调用 `doit()`。

但 IPv4 route update 当前注册为：

```text
rtnl_register(PF_INET, RTM_NEWROUTE, inet_rtm_newroute, NULL, 0);
rtnl_register(PF_INET, RTM_DELROUTE, inet_rtm_delroute, NULL, 0);
```

也就是默认由 rtnetlink 外层持 global RTNL 调用。

同时，route update 内部仍有硬依赖：

```text
net/ipv4/fib_trie.c:
  fib_table_insert(): /* Caller must hold RTNL. */
  fib_table_delete(): /* Caller must hold RTNL. */

net/ipv4/fib_notifier.c:
  call_fib4_notifiers(): ASSERT_RTNL()

net/core/fib_rules.c:
  fib_rules_seq_read(): ASSERT_RTNL()
  fib_rules_event(): ASSERT_RTNL()
```

### 5.3 K3a 实现

K3a 没有直接把所有 route handler 都变成 unlocked。当前实现是受控分流：

1. 给 `RTM_NEWROUTE/DELROUTE` 注册 `RTNL_FLAG_DOIT_UNLOCKED`。
2. `inet_rtm_newroute()` / `inet_rtm_delroute()` 先解析 `fib_config`，再判断是否是实验 route：

```text
fc_protocol == RTPROT_BIRD
fc_priority == 232
fc_table == RT_TABLE_MAIN
IPv4 only
no nexthop object
no multipath
no lwtunnel
not local route
```

3. 如果不是实验 route，则函数内部手动 `rtnl_lock()`，走原始语义。
4. 如果是实验 route，则：
   - 不拿 global RTNL。
   - 拿 per-netns route mutex。
   - 设置 `fc_skip_notify=true`。
   - 调用 `fib_table_insert/delete()`。
   - 依赖 K2c 的 `fib_info_hash_mutex`。
5. 对同一 netns 的 route writes 串行；不同 netns 可以并发。

这个方案本质上是“把 K2c 的 no-notify route-only 逻辑接到标准 rtnetlink 上”，不是完整生产 K3。

实现位置：

| 文件 | 作用 |
|---|---|
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_frontend.c` | 标准 `RTM_NEWROUTE/DELROUTE` unlocked 分流，非实验 route 回退 global RTNL |
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_bulk.c` | per-netns route lock helper、K3a route gate、复用 K2c fib_info 互斥 |
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/include/net/ip_fib.h` | `fc_skip_notify` 和 K3a helper 声明 |
| `bird/kernel/patches/0005-bird-k3a-route-unlocked-pernetns-experiment.patch` | K3a patch 记录 |
| `bird/synthetic_fib_bench.py` | synthetic 标准 `ip route` writer，支持 `--metric 232` |
| `bird/run_3083_loadonly_delta_vs_full.sh` | 本轮 3083 K3a 实验 wrapper |

部署内核：

```text
6.8.12-birdrtbulk-k3a
```

包路径：

```text
/data/lxl/kernel-work/6.8.12-birdrtbulk-k3a.tar.gz
```

### 5.4 K3a 实测结果

Synthetic：

| case | route ops | jobs | wall | throughput | 结果 |
|---|---:|---:|---:|---:|---|
| K3a `1x1k_j1` | 1,000 | 1 | `0.528s` | writer `0.036s` | success |
| K3a `4x10k_j4` | 40,000 | 4 | `0.520s` | `76,887 routes/s` | success |
| K3a `10x600k_j4` | 6,000,000 | 4 | `36.039s` | `166,485 routes/s` | success |
| K3a `100x600k_j8` | 60,000,000 | 8 | `204.735s` | `293,062 routes/s` | success |

真实 3083 K3a 主流程：

| 阶段/指标 | 结果 |
|---|---:|
| kernel | `6.8.12-birdrtbulk-k3a` |
| build / fast image patch | `169.973s` |
| docker compose up | `2825.982s` |
| restart_bird | `937.512s` |
| wait RIB stable | `0.494s`，load-only |
| validate_bird | `2.911s` |
| initial full RIB-to-FIB | `94.535s` stage |
| initial writer wall | `92.670s` |
| initial route ops | `6,131,994 / 6,131,994` |
| enable_delta | `33.763s` |
| start_agents | `36.917s` |
| main flow | success，`4115.784s` |

真实 3083 K3a reconvergence：

| 阶段/指标 | 结果 |
|---|---:|
| stop 10 victims | `26.700s` |
| wait RIB/load stable | `0.002s`，load-only，final load1 `27.86` |
| prepare delta offsets | `59.312s` |
| delta raw events | `347,647` |
| delta collapsed ops | `198,877` |
| delta writer/stage | `153.829s / 154.745s` |
| delta result | `2747/2747` brdnodes success，`0` failures |
| full birdc-primary compare | failed after `1944.350s` |

Full compare 失败含义：

- `write_full_fib_compare` 使用同样 K3a `metric 232` 路径，但全量 `birdc-primary` dump 后重写 2747 个 brdnode。
- 该阶段超过 `FULL_COMMAND_TIMEOUT=1800s`，被判失败。
- timeout 后 Python multiprocessing worker 被 reparent 到 PID 1，需手动 kill orphan worker 才能让上层脚本收尾。后续脚本应改为独立 process group kill。
- 这个失败不影响 K3a 初始 full 写入和 delta event 写入的成功结论；它说明故障后 full birdc-primary 对照在当前脚本/负载下不可接受。

### 5.5 当前可行性结论

K3a 已通过 controlled experiment 门槛：

| 检查项 | 当前状态 |
|---|---|
| rtnetlink unlocked handler 支持 | 有，`RTNL_FLAG_DOIT_UNLOCKED` |
| 非实验 route 保留原始 RTNL 语义 | 已实现，非匹配 route 手动回退 global RTNL |
| 实验 route 识别 | 已实现，严格限制 `proto bird metric 232 main table` 的简单 IPv4 route |
| per-netns route lock | 已实现 helper，供 `fib_frontend.c` 调用 |
| notifier 语义 | 仍需 no-notify 才能避开 `ASSERT_RTNL()`，这不是完整语义 |
| fib_info shared state | 复用 K2c `fib_info_hash_mutex`，synthetic 和 3083 未见 Oops |
| device/nexthop/rules/netns teardown | 仍未做生产级完整审计 |

当前结论：

1. K3a 证明了“标准 rtnetlink route path + per-netns route lock”的受控可行性。
2. K3a 性能强于原 F 方法和 K1，但当前 3083 full/delta 结果弱于 K2c mixed。
3. K3a 价值在于不需要私有 bulk API，更贴近最初“每个 namespace 独立 route lock”的目标。
4. K3a 仍不是 production-safe K3：no-notify、IPv4-only、无 ECMP/nexthop/IPv6/device lifecycle 完整语义。
5. 后续若继续推进，应做 K3b：bulk/delayed notifier、process-group timeout 修复、更多 route 类型语义测试。

只有这些通过后，才考虑 3083。

## 6. K4：Multi-netns Bulk Transaction

### 6.1 想达到什么

K4 的目标是一次系统调用或一次 generic-netlink 请求提交多个 network namespace 的 route ops，例如：

```text
K4_APPLY_MULTI_V4
  [netns_fd_1, ops...]
  [netns_fd_2, ops...]
  ...
```

理想收益：

- 减少用户态多进程调度和多次 netlink 往返。
- 内核统一排序 netns，避免死锁。
- 可以在内核里更好地做 batch 统计、错误处理和限速。

### 6.2 两种实现路线

#### K4-global

```text
global RTNL once
  -> loop all netns
  -> loop all ops
```

优点：

- 语义最接近现有内核。
- notifier、device、rules 等路径风险最低。

缺点：

- 没有跨 namespace 真并发。
- 会形成超长 global RTNL 临界区，可能让其它网络控制操作长时间等待。
- 对我们最初的 RTNL 拥塞问题帮助有限。

结论：不值得作为主线。

#### K4-pernetns

```text
sort netns by stable id
  -> lock per-netns route locks in order
  -> update each netns FIB
  -> unlock
```

优点：

- 继承 K2c 的跨 netns 并发潜力。
- 内核可以统一处理 deadlock order 和错误 summary。

缺点：

- 依赖 K2c 语义正确性。
- multi-netns 请求需要处理 netns fd 引用、权限、生命周期、部分失败和回滚。
- 如果仍 no-notify，它依旧不是生产语义。
- 如果补 notifier，就进入 K2d/K3 级别工作量。

### 6.3 当前可行性结论

K4 当前也不通过“可直接 3083 全流程”的判断：

- K4-global 不解决跨 netns RTNL 并发，本质只是更大 batch。
- K4-pernetns 必须先有语义完整的 K2d，否则只是把 K2c no-notify 原型扩大成 multi-netns API。
- 当前已有 K2c mixed 在 3083 上 full `71.162s`、delta `55.026s`，用户态多次调用的调度成本不是当前最大瓶颈。

下一步如果要推进 K4，应先做 K2d：

```text
K2d = K2c + bulk notifier / delayed notifier flush + lifecycle audit
```

K2d 通过 synthetic 和 3083 后，再做 K4-pernetns 才有意义。

## 7. K3b/K3c：Bulk API 内部继续降本

### 7.1 K3b：multi-attr bulk request

K3b 的目标是减少用户态到内核态的 generic-netlink request 次数。K1/K2c 的 `BIRD_RTBULK_A_OPS` 原来每个 netlink attr 最多约 2000 条 route op，超过后用户态需要多次发 request。K3b 改为一个 generic-netlink request 中携带多个 repeated `BIRD_RTBULK_A_OPS` attr：

```text
GENL request
  -> attr OPS[0..1999]
  -> attr OPS[2000..3999]
  -> attr OPS[4000..5999]
  -> ...
```

内核侧 `bird_rtbulk_apply_v4()` 逐个 attr 调用 apply loop；用户态 `kernel_bulk_writer.py` / `kernel_bulk_command_writer.py` / `kernel_bulk_birdc_writer.py` 增加 `--ops-per-attr`，默认仍是 `2000`，但 `--batch` 可以是更大的逻辑 batch。

K3b 结论：

- 能正常编译、启动、写入。
- 10x600k synthetic 与 K2c 同环境几乎持平，而不是明显更快。
- 说明此时瓶颈主要不再是 generic-netlink request 次数，而是每条 route 仍要走 `fib_create_info() -> fib_table_insert()` 的 per-route 成本。

K3b synthetic 结果：

| case | route ops | jobs,batch | wall | 结果 |
|---|---:|---:|---:|---|
| K3b `10x600k_j4_b1024` | 6,000,000 | `4,1024` | `17.549s` | success |
| K3b `10x600k_j4_b16000` | 6,000,000 | `4,16000` | `18.044s` | success |
| K2c recheck `10x600k_j4_b1024` | 6,000,000 | `4,1024` | `17.538s` | success |

### 7.2 K3c：cached fib_info fast path

K3c 的目标是跳过重复的 `fib_create_info()`。在 SeedEMU/BIRD 写入场景里，同一容器内大量 route 往往共享同一组 nexthop/device/metric/proto/table，只是 prefix 不同。K2c/K3b 每条 route 都会重新构造并查找 `fib_info`，这在大批量写入中成为主要成本。

K3c 新增：

```text
bird_rtbulk_apply_many_v4()
  -> 检查一个 attr 内所有 ops 是否共享同一 fib_info key
  -> fib_create_info(base_cfg) 一次
  -> 对每条 prefix 调 fib_table_insert_info(..., cached_fi)
  -> 每条 alias 持有 cached fib_info 的 ref
```

核心实现点：

| 文件 | 修改 |
|---|---|
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_bulk.c` | 增加 `bird_rtbulk_apply_many_v4()`，同一 attr 内共享 `fib_info` 时走快路径，否则回退 K2c/K3b per-route apply |
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_trie.c` | 增加 `fib_table_insert_info()`，复用 caller 提供的 held `struct fib_info *` |
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/include/net/ip_fib.h` | 声明 `fib_table_insert_info()` |
| `bird/kernel/tools/kernel_bulk_command_writer.py` | 默认 `--group-fib-info`，delta command writer 按 nexthop/device/metric/table 分组提交 |
| `bird/kernel/tools/kernel_bulk_birdc_writer.py` | 默认 `--group-fib-info`，full birdc writer 按 nexthop/device/metric/table 分组提交 |

K3c 仍然是 controlled experiment：

- IPv4 main FIB 为主。
- no-notify 语义继承自 K2c。
- 依赖 `BIRD_RT_BULK` 私有 generic-netlink API。
- 还不是生产语义完整的 per-netns RTNL 替代。

### 7.3 K3c 实测结果

当前 VM 内核：

```text
6.8.12-birdrtbulk-k3c
```

Smoke：

| case | route ops | wall | kernel apply | 结果 |
|---|---:|---:|---:|---|
| K3c smoke | 1,000 | `0.00165s` | `0.00083s` | success |

10 router synthetic：

| case | route ops | jobs,batch | wall | throughput | 结果 |
|---|---:|---:|---:|---:|---|
| K3c `10x600k_j4_b1024` | 6,000,000 | `4,1024` | `7.054s` | `850,629 routes/s` | success |
| K3c `10x600k_j4_b16000` | 6,000,000 | `4,16000` | `7.025s` | `854,074 routes/s` | success |
| K3c `10x600k_j4_b64000` | 6,000,000 | `4,64000` | `7.522s` | `797,685 routes/s` | success |

100 router synthetic：

| case | route ops | jobs,batch | wall | throughput | max load1 | CPU util sample | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| K3c `100x600k_j4_b16000` | 60,000,000 | `4,16000` | `56.095s` | `1.070M routes/s` | `3.71` | `~9%` | success |
| K3c `100x600k_j8_b16000` | 60,000,000 | `8,16000` | `30.567s` | `1.963M routes/s` | `5.47` | `~17%` | success |
| K3c `100x600k_j16_b16000` | 60,000,000 | `16,16000` | `18.098s` | `3.315M routes/s` | `7.25` | `~35%` | success |
| K3c `100x600k_j24_b16000` | 60,000,000 | `24,16000` | `13.608s` | `4.409M routes/s` | `8.80` | `~50%` | success |
| K3c `100x600k_j32_b16000` | 60,000,000 | `32,16000` | `11.621s` | `5.163M routes/s` | `13.15` | `~54-63%` | success |

与旧结果对比：

| 方法 | case | wall | 说明 |
|---|---|---:|---|
| K1 | `100x600k_j4` | `585.122s` | global RTNL bulk baseline |
| K2c | `100x600k_j4` | `132.815s` | per-netns no-notify + fib_info hash mutex |
| K3a standard rtnetlink | `100x600k_j8` | `204.735s` | 标准 rtnetlink 快路径，仍有每条 rtnetlink 成本 |
| K3b | `10x600k_j4` | `17.549s` | multi-attr 后仍 per-route `fib_create_info()` |
| K3c | `10x600k_j4` | `7.025s` | cached fib_info |
| K3c | `100x600k_j32` | `11.621s` | 目前 synthetic 最优 |

### 7.4 推荐参数

Synthetic 压测：

```bash
sudo python3 /home/lxl/bird/synthetic_fib_bench.py \
  --routers 100 \
  --routes-per-router 600000 \
  --jobs 32 \
  --writer bulk \
  --bulk-writer /home/lxl/bird/kernel_bulk_writer.py \
  --bulk-batch 16000 \
  --bulk-ops-per-attr 2000 \
  --bulk-route-type dev \
  --bulk-op replace \
  --summary-path /home/lxl/bird/runs/k3c_synthetic_20260622/k3c_100x600k_j32_b16000.json
```

真实 full writer / delta writer：

- 运行 K3c 内核。
- 保持 `--ops-per-attr 2000`。
- 保持默认 `--group-fib-info`。
- 初始建议 full/delta writer jobs 从 `8/16/24` 扫，不再默认 `1/2`，因为 K3c synthetic 在 `32` 仍稳定，但真实 BIRD dump 和 Docker 进程数会引入额外成本。

### 7.5 当前 K3c 决策

K3c 已经替代 K2c 成为 synthetic 性能主线：`100x600k_j32` 把 6000 万 route 写入压到 `11.621s`，远低于最初目标 `300s`。

下一步应跑真实 3083 full + reconvergence delta：

1. 启动 3083 网络，start-bird 后只等 load gate。
2. 用 `kernel_bulk_birdc_writer.py --group-fib-info` 写 initial full FIB。
3. 故障后用 BIRD delta event 生成 command files。
4. 用 `kernel_bulk_command_writer.py --group-fib-info` 写 delta FIB。
5. 与 K2c mixed 的 full `71.162s`、delta `55.026s` 对比。

## 8. 当前决策

当前推荐路线：

1. Synthetic 性能主线切到 K3c：它在 `100x600k_j32` 达到 `11.621s`，证明 cached `fib_info` 是当前最大收益点。
2. 真实 3083 结果暂时仍以 K2c mixed 为已验证基线：full `71.162s`、delta `55.026s`。
3. 下一轮真实实验应使用 K3c 内核 + `--group-fib-info` full/delta writer 复测 3083。
4. 语义方向继续推进 K3a/K3c：K3a 更贴近标准 rtnetlink/per-netns route lock；K3c 更适合高性能 controlled experiment。
5. 不把 K3a/K3c 当 production kernel：当前只覆盖简单 IPv4 main-table `proto bird metric 232` route，并跳过 notifier。
6. K4 只有在 K2d/K3c 的 notifier/lifecycle 语义补齐后再设计；否则只是把 no-notify 原型扩大成 multi-netns API。
7. 下次真实 full 对照前优先修 `FULL_COMMAND_TIMEOUT` 的 process-group kill，避免 timeout 后 orphan worker 持有管道。

## 9. 快速状态检查命令

检查 VM 当前内核和 Docker：

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=10 \
  lxl@192.168.122.60 \
  "uname -r; docker version --format '{{.Server.Version}}'; docker compose version; docker buildx version"
```

检查 Docker 是否干净：

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=10 \
  lxl@192.168.122.60 \
  "docker ps -aq | wc -l; docker volume ls -q | wc -l; docker network ls --filter type=custom -q | wc -l; docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | sort"
```

检查最近 K2c mixed 结果：

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path("bird/runs/k2c-delta-mixed-3083_20260620_1150")
main = json.load(open(base / "main_summary.json"))
reconv = json.load(open(base / "reconvergence/summary.json"))

full = main["stages"]["write_initial_fib"]["writer_stats"]
delta = reconv["stages"]["write_delta_fib"]["writer"]["writer"]

print("main success:", main["success"])
print("full writer:", full["wall_seconds"], full["success_count"], full["fail_count"])
print("reconv success:", reconv["success"])
print("delta writer:", delta["wall_seconds"], delta["success_count"], delta["fail_count"], delta["ignored_delete_miss"])
PY
```
