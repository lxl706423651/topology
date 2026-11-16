# 2026-06-18 内核 RTNL/FIB 优化实验记录

更新时间：2026-06-20 13:32 CST

设计来源：[0618ExperimentDesign.md](./0618ExperimentDesign.md)

## 当前完成度

| 阶段 | 目标 | 状态 | 证据 |
|---|---|---|---|
| K0 | stock kernel 可观测性和基线，不改变行为 | 已完成第一轮合成观测 | `bird/runs/k0_stock_observe_20260618/` |
| K1 | 新增 bulk route API，内部仍使用全局 RTNL | 已完成合成实验、真实 3083 初始 full FIB、3083 reconvergence delta；delete scope bug 已在用户态 writer 修复；`jobs=1,batch=256` 可稳定跑完全流程 | `bird/kernel/`，`bird/runs/k1_synthetic_*`，`bird/runs/k1-delta-reconv-3083_safej1_20260619_194605/` |
| K2 | bulk API 内 route-only per-netns lock | K2c 修复 K2b 的 `fib_create_info` 并发崩溃；synthetic `100x600k_j4` 跑完 60M route ops，用时 `132.815s`；真实 3083 full + delta reconvergence 已通过；mixed-batch delta writer 已把 3083 reconvergence delta writer 从 `236.353s` 降到 `55.026s` | `bird/kernel/patches/0002-*` 到 `0004-*`，`bird/runs/k2c_*` |
| K3 | 标准 `RTM_NEWROUTE/DELROUTE` route-only 化 | 暂停实现；K2b 证明绕开 RTNL 后 `fib_create_info`/FIB shared state 仍不安全 | 依赖 K2c 重新设计 |
| K4 | multi-netns bulk transaction | 暂停实现；若仍拿全局 RTNL 只会形成超长临界区，若 per-netns 则依赖 K2c 正确性 | 依赖 K2c 锁顺序、notifier、事务语义 |

本文件目前记录 K0、K1 的真实实测结果、K1 参数 sweep、K2/K2b 失败证据、K2c synthetic 修复结果，以及 K3/K4 为什么不能在缺少完整语义审计前直接实现。

## 实验环境

| 项 | 值 |
|---|---|
| VM | `lxl@192.168.122.60` |
| libvirt domain | `bird-rtnl-dev` |
| VM stock kernel | `6.8.0-124-generic` |
| VM K1 test kernel | `6.8.12-birdrtbulk` |
| Host kernel | `6.8.0-124-generic` |
| K0 run dir, VM | `/home/lxl/bird/runs/k0_stock_observe_20260618` |
| K0 run dir, host | `bird/runs/k0_stock_observe_20260618` |
| K0 工具 | `bird/synthetic_fib_bench.py` + `perf stat` + `perf lock` |

注意：K1 正式测试前已在 VM 内执行 `/home/lxl/bird/reset_docker_vm.sh`，Docker runtime 被清空并重建 SeedEMU base/router images。测试时 Docker 容器数为 `0`，避免旧容器影响 load、内存和 route path 观测。

## K0：Stock Kernel 合成观测

### 实验目的

K0 不改内核，只验证当前 Linux route 写入路径是否存在 RTNL/FIB 竞争特征：

- 并发 writer 增加时，吞吐是否继续增长。
- `rtnl_mutex` 等待时间是否明显增长。
- CPU/load 是否已经饱和。
- 合成 netns 写入是否能在不依赖 BIRD/Docker 的情况下复现 FIB 写入瓶颈。

### 实验方法

在 VM 内执行三组 synthetic FIB 写入：

| case | netns/router 数 | 每 netns routes | jobs | 写入方式 |
|---|---:|---:|---:|---|
| `case1_1x100k_j1` | 1 | 100,000 | 1 | `ip -n <ns> -4 -force -batch -` |
| `case2_4x100k_j4` | 4 | 100,000 | 4 | 每个 netns 一个 batch writer |
| `case3_8x100k_j8` | 8 | 100,000 | 8 | 每个 netns 一个 batch writer |

每组都：

1. 创建临时 network namespace，prefix 为 `k0<case>-r`。
2. 写入极简 IPv4 route，绕过 BIRD 与 Docker 编排成本。
3. 用 `perf stat` 记录 CPU/上下文切换等事件。
4. 用 `perf lock record/report` 记录锁竞争。
5. 结束后删除临时 netns。

原始命令形态：

```bash
sudo -n perf stat -x, -o "$RUN/${name}_perf_stat.csv" -- \
  perf lock record -o "$RUN/${name}_perf_lock.data" -- \
  python3 /home/lxl/bird/synthetic_fib_bench.py \
    --routers "$routers" \
    --routes-per-router "$routes" \
    --jobs "$jobs" \
    --prefix "$prefix" \
    --sample-interval 2 \
    --summary-path "$RUN/${name}_summary.json" \
    --cleanup-before \
    --cleanup-after
```

### K0 写入结果

| case | 总 route ops | wall time | 总吞吐 | 单 netns 平均耗时 | load1 min/max | 结果 |
|---|---:|---:|---:|---:|---:|---|
| `1x100k_j1` | 100,000 | 7.541s | 13,261.7 routes/s | 7.280s | 4.03 / 4.12 | 成功 |
| `4x100k_j4` | 400,000 | 25.578s | 15,638.4 routes/s | 25.074s | 3.79 / 5.02 | 成功 |
| `8x100k_j8` | 800,000 | 54.116s | 14,783.1 routes/s | 53.718s | 5.55 / 13.00 | 成功 |

直接结论：

- 从 jobs=1 到 jobs=4，总吞吐只从 13.26k/s 提升到 15.64k/s。
- 从 jobs=4 到 jobs=8，总吞吐反而下降到 14.78k/s。
- 每个 netns 都写同样 100k route，但单 netns 平均耗时从 jobs=4 的 25.07s 增加到 jobs=8 的 53.72s，说明并发 writer 会显著拖慢单 writer。
- load1 最高只有 13.00，未达到 CPU/load 饱和型瓶颈。

### K0 `perf stat` 摘要

| case | task-clock | CPUs utilized | context switches | cycles | instructions |
|---|---:|---:|---:|---:|---:|
| `1x100k_j1` | 8,240.41 ms | 0.868 | 2,682 | 26.01B | 25.02B |
| `4x100k_j4` | 92,085.64 ms | 3.335 | 26,141 | 306.74B | 134.83B |
| `8x100k_j8` | 369,867.58 ms | 6.537 | 123,476 | 1,220.80B | 386.93B |

CPU 利用并没有随着 jobs 成比例转化为 route 吞吐。jobs=8 使用了更多 CPU 时间和上下文切换，但整体吞吐低于 jobs=4。

### K0 `perf lock` 摘要

| case | 关键锁 | acquired/contended | avg wait | total wait | max wait |
|---|---|---:|---:|---:|---:|
| `1x100k_j1` | `rtnl_mutex` | 589 / 589 | 64.39 us | 37.93 ms | 4.28 ms |
| `4x100k_j4` | `rtnl_mutex` | 394,947 / 394,947 | 174.04 us | 1.15 m | 340.96 ms |
| `8x100k_j8` | `rtnl_mutex` | 792,205 / 792,205 | 454.45 us | 6.00 m | 357.17 ms |

辅助锁现象：

| case | 锁 | acquired/contended | avg wait | total wait |
|---|---|---:|---:|---:|
| `4x100k_j4` | `pcpu_lock` | 230,422 / 230,422 | 3.77 us | 869.42 ms |
| `8x100k_j8` | `pcpu_lock` | 528,891 / 528,891 | 6.88 us | 3.64 s |
| `4x100k_j4` | `namespace_sem` | 34 / 34 | 9.00 ms | 305.95 ms |
| `8x100k_j8` | `namespace_sem` | 65 / 65 | 24.51 ms | 1.59 s |

`perf lock report` 对单 writer 的 acquired 计数远小于 route ops，说明该报告更适合看 contended lock 的等待趋势，而不是直接当作每条 route 的完整加锁次数统计。即便如此，jobs=4 到 jobs=8 的趋势非常明确：`rtnl_mutex` 平均等待从 174.04us 上升到 454.45us，总等待从 1.15 分钟上升到 6.00 分钟。

## K0 分析

K0 支持 `0618ExperimentDesign.md` 中的判断：

1. 当前瓶颈不是单纯 CPU 算力。jobs=8 的 load1 最高只有 13.00，但吞吐比 jobs=4 更差。
2. 当前瓶颈不是 BIRD RIB dump。K0 完全绕过 BIRD，只用 `ip -batch` 写极简 route，仍出现并发 writer 退化。
3. 当前瓶颈不是 Docker 编排。K0 使用临时 netns，仍能看到 `rtnl_mutex` 等待随 jobs 增长显著上升。
4. `ip -batch` 只能降低用户态进程和命令解析成本，不能把内核 route 更新变成“一次 RTNL 锁写 N 条”的 bulk transaction。
5. K1 的价值在于减少同一 netns 内逐条 netlink/ACK/lock 开销；但只要 K1 仍拿全局 RTNL，它不应被期待解决跨 netns 真并行。
6. K2 的价值在于验证 route-only per-netns lock 是否能让不同 netns 的 FIB 更新真正并行，这是更接近原始目标的内核方向。

## K1：Bulk Route API With Global RTNL

### K1 实现内容

K1 已按设计完成第一版实验性内核 API：

| 文件 | 作用 |
|---|---|
| `bird/kernel/patches/0001-bird-rtbulk-global-rtnl.patch` | Ubuntu 6.8 source patch，新增 `CONFIG_BIRD_RT_BULK`、uapi header、`net/ipv4/fib_bulk.c` |
| `bird/kernel/tools/kernel_bulk_writer.py` | Python generic-netlink writer，支持 `setns()` 后向当前 netns 提交 bulk route ops |
| `bird/kernel/tests/synthetic_1x600k.sh` | 单 netns 60 万 route 合成测试 |
| `bird/kernel/tests/synthetic_100x600k.sh` | 多 netns 合成测试，可通过 `ROUTERS`、`ROUTES_PER_ROUTER`、`JOBS`、`BATCH` 调参 |

K1 内核版本：

| 项 | 值 |
|---|---|
| kernel source | `/data/lxl/kernel-work/linux-source-6.8.0-124.124` |
| test kernel release | `6.8.12-birdrtbulk` |
| config | `CONFIG_BIRD_RT_BULK=y` |
| kernel package | `/data/lxl/kernel-work/6.8.12-birdrtbulk.tar.gz` |
| VM default fallback | GRUB saved default 仍为 `Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-124-generic` |

K1 的处理方式：

```text
user writer
  -> setns(target netns)
  -> generic netlink BIRD_RT_BULK/APPLY_V4
kernel
  -> parse binary op array
  -> rtnl_lock() once per bulk message
  -> loop fib_table_insert/delete for N route ops
  -> rtnl_unlock()
  -> return success/fail summary
```

本轮 K1 限制：

- IPv4 only。
- 单请求只作用于当前 netns。
- `dev lo` synthetic route 为主，未接入真实 BIRD main-table route。
- 不支持 IPv6、ECMP、nexthop object、VRF、复杂 route attribute。
- 仍使用全局 RTNL，不是 per-netns RTNL；因此 K1 的收益来自减少 netlink message、ACK、parse 和 lock/unlock 次数，而不是跨 netns 真并行。

### K1 部署和修复记录

部署过程中遇到两个环境问题：

1. VM 使用 `sudo tar -C /` 解包 host staging 时，archive 中的 `./lib` owner 来自 host 用户 `lxl`，导致 VM `/lib` owner 被改成 `lxl`。已修正为 `root:root`，并将 `/lib`、`/lib/modules`、`/lib/modules/6.8.12-birdrtbulk` 权限改回 `755`。
2. VM 的 initramfs hook 依赖 `/lib/cryptsetup` 和 `/lib/udev`，但实际包文件在 `/usr/lib/cryptsetup` 和 `/usr/lib/udev`。已补：

```bash
sudo ln -s /usr/lib/cryptsetup /lib/cryptsetup
sudo ln -s /usr/lib/udev /lib/udev
sudo ln -s /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
```

修复后 `update-initramfs -c -k 6.8.12-birdrtbulk` 和 `update-grub` 成功。VM 当前运行 `6.8.12-birdrtbulk`，但 GRUB saved default 仍指向 stock `6.8.0-124-generic`，便于后续 fallback。

### K1 Smoke 和 Sanity

K1 generic-netlink smoke：

| 测试 | 结果 |
|---|---|
| `sudo python3 kernel_bulk_writer.py --routes 10 --batch 10` | 成功 |
| generic netlink family id | `23` |
| success/fail | `10 / 0` |
| writer wall | `0.000144s` |
| kernel apply | `0.0000757s` |

K1 多 netns sanity：

| case | netns | 每 netns routes | jobs | batch | 总 routes | wall | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| `k1_debug` | 2 | 100 | 1 | 50 | 200 | 约 1s | 成功 |
| `k1_sanity4` | 4 | 10,000 | 4 | 1000 | 40,000 | <1s | 成功 |

`k1_sanity4` 证明并发 writer、`setns()`、bulk API、summary 聚合和 cleanup 流程可用。短测早期出现 `wall_seconds_outer=0`，原因是脚本使用 `date +%s` 秒级计时；随后已改为 `time.monotonic()` 高精度计时。

### K1 正式合成结果

| case | netns/router 数 | 每 netns routes | jobs | batch | 总 route ops | wall time | 吞吐 | fail | loadavg after | 证据 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `1x600k` | 1 | 600,000 | 1 | 1024 | 600,000 | 3.794s outer / 3.690s writer | 162,605 routes/s | 0 | `0.42 29.06 28.11` | `bird/runs/k1_synthetic_1x600k_20260617_165558/` |
| `10x600k_j4` | 10 | 600,000 | 4 | 1024 | 6,000,000 | 56.846s | 105,548 routes/s | 0 | `2.61 22.98 26.04` | `bird/runs/k1_synthetic_10x600k_j4_20260617_165625/` |
| `100x600k_j4` | 100 | 600,000 | 4 | 1024 | 60,000,000 | 585.122s | 102,543 routes/s | 0 | `4.22 6.75 15.68` | `bird/runs/k1_synthetic_100x600k_j4_20260617_165735/` |

`1x600k` 额外校验：

```text
ip route | wc -l = 600000
writer batches = 586
kernel_apply_seconds = 3.080337459
```

`100x600k_j4` 运行期间资源采样：

| elapsed | load1 | used memory | netns | writer/process count |
|---:|---:|---:|---:|---:|
| 1 min | 3.66 | 3.2 GiB | 100 | 8 |
| 2 min | 4.14 | 4.2 GiB | 100 | 8 |
| 3 min | 4.36 | 5.4 GiB | 100 | 8 |
| 4 min | 4.29 | 6.6 GiB | 100 | 8 |
| 5 min | 4.23 | 7.6 GiB | 100 | 8 |
| 6 min | 4.31 | 8.7 GiB | 100 | 8 |
| 7 min | 4.31 | 9.7 GiB | 100 | 8 |

### K1 对比分析

设计文档记录的 stock synthetic 对照为：

```text
100 netns * 600,000 routes = 60M route ops
stock ip-batch wall = 3456.237s
```

K1 在同规模 synthetic 上：

```text
100 netns * 600,000 routes = 60M route ops
K1 bulk wall = 585.122s
```

速度提升：

```text
3456.237 / 585.122 = 5.91x
```

结论：

1. K1 明显降低了 synthetic route install 时间，60M route ops 从约 57.6 分钟降到约 9.75 分钟。
2. `100x600k_j4` 的 load1 一直在 3-5 左右，说明 K1 没有复现 stock `ip -batch` 那类高 RTNL wait/load 问题。
3. 吞吐从单 netns `162.6k routes/s` 降到 100 netns `102.5k routes/s`，说明全局 RTNL 和 FIB 插入成本仍然存在，但已经远低于 stock `ip -batch` 的逐条 netlink 路径。
4. K1 仍不能证明“跨 netns 真并行”，因为它仍在 bulk message 内持全局 RTNL。要验证这一点必须进入 K2。

## K2：Private Per-Netns Route Lock Smoke

### K2 实现内容

K2 做了一个最小实验 patch：

| 文件 | 作用 |
|---|---|
| `bird/kernel/patches/0002-bird-rtbulk-pernetns-lock.patch` | 相对 K1 的增量 patch，只修改 `net/ipv4/fib_bulk.c` |

K2 内核版本：

| 项 | 值 |
|---|---|
| test kernel release | `6.8.12-birdrtbulk-k2` |
| kernel package | `/data/lxl/kernel-work/6.8.12-birdrtbulk-k2.tar.gz` |
| VM fallback after test | 已回到 stock `6.8.0-124-generic` |

K2 只改实验 API，不改标准 rtnetlink：

```text
BIRD_RT_BULK/APPLY_V4
  K1: rtnl_lock() once per bulk request
  K2: mutex_lock(per-netns private route_lock) once per bulk request

RTM_NEWROUTE / RTM_DELROUTE
  unchanged, still uses upstream RTNL path
```

K2 目标是验证：如果只在受控 `BIRD_RT_BULK` 路径绕开 global RTNL，不同 netns 的 synthetic route writes 是否能真正并行。

### K2 Smoke 结果

K2 smoke 命令：

```bash
sudo dmesg -C
sudo python3 /home/lxl/bird/kernel/tools/kernel_bulk_writer.py \
  --routes 10 \
  --batch 10 \
  --base 10.251.0.0 \
  --summary-path /tmp/k2_smoke.json
sudo dmesg --level=err,warn,crit,alert,emerg | tail -50
```

用户态 summary 显示 10 条 route 写入成功：

| 指标 | 值 |
|---|---:|
| routes requested | 10 |
| success/fail | 10 / 0 |
| writer wall | `0.000481557s` |
| kernel apply | `0.000411974s` |

但 kernel dmesg 立即出现 RTNL assertion：

```text
RTNL: assertion failed at net/ipv4/fib_notifier.c (22)
WARNING: CPU: 29 PID: 1843 at net/ipv4/fib_notifier.c:22 call_fib4_notifiers+0x89/0x90
Call Trace:
  fib_table_insert+0x501/0x630
  bird_rtbulk_apply_one_v4+0x1a3/0x1c0
  bird_rtbulk_apply_v4+0xdc/0x310
```

证据路径：

| 文件 | 内容 |
|---|---|
| `bird/runs/k2_smoke_pernetns_lock_20260617_172249/k2_smoke_observed.json` | K2 smoke 用户态写入 summary |
| `bird/runs/k2_smoke_pernetns_lock_20260617_172249/k2_rtnl_assertion_dmesg.txt` | 上一轮 K2 boot 的 kernel journal 中 RTNL assertion |
| `bird/runs/k2_smoke_pernetns_lock_20260617_172249/vm_post_fallback_status.txt` | VM fallback 后状态 |

### K2 结论

K2 smoke 证明：不能简单把 `BIRD_RT_BULK` 内的 global RTNL 替换成 per-netns mutex。

直接原因：

- `fib_table_insert()` 会调用 FIB notifier。
- `net/ipv4/fib_notifier.c:22` 的 notifier 路径含 `ASSERT_RTNL()`。
- 即使 synthetic `dev lo` route 的用户态写入成功，kernel 仍明确认为该路径违反 RTNL 保护约束。

因此本轮没有继续跑 K2 `1x600k`、`10x600k`、`100x600k`。继续跑大规模只会在 tainted kernel 上积累无效数据。

K2 后续如果要继续，需要改设计，而不是继续放大这个 patch：

1. 保留 global RTNL，但优化 notifier 或批量 notifier 发送。
2. 拆出 route trie 更新和 notifier 发送：per-netns lock 下更新 FIB，随后在 global RTNL 下批量发送 notifier。但这需要证明 FIB reader、rules、nexthop、device lifetime 都安全。
3. 新增“no-notifier synthetic-only”模式，只用于纯性能上限估计，不能作为真实方案。
4. 深入审计并重构 `fib_notifier`，使其有 per-netns 或 RCU-safe 通知语义。这已经接近 K3/K4 级别风险。

## K3-K4 当前缺口

当前 worktree 已具备 K1/K2 的 patch、writer 和 synthetic tests，但还缺少 K3-K4 交付物：

```text
bird/kernel/
  patches/
    0003-bird-rtm-newroute-route-only.patch
    0004-bird-rtbulk-multinetns-transaction.patch
  tools/
    parse_f_writer_commands.py
  tests/
    strict_3083_bulk.sh
```

K2 已证明不能简单把全局 `rtnl_mutex` 替换成 per-netns lock。下一步需要围绕 notifier 和 route update 语义重新设计，至少明确以下对象仍由全局 RTNL 保护还是可以被拆分：

- `net->ipv4.fib_seq`
- `fib_rules`
- nexthop object / nh id
- device unregister/notifier
- route cache/exception path
- VRF/lwtunnel/multipath

源码观察：`net/ipv4/fib_trie.c` 中 `fib_table_insert()` 和 `fib_table_delete()` 的函数注释均明确写着 `Caller must hold RTNL`。K2 smoke 进一步证明，不只是注释要求，`fib_notifier` 路径会直接 `ASSERT_RTNL()`。

因此 K3/K4 不能建立在“直接绕开 RTNL”的假设上，必须先解决 FIB notifier 和相关对象生命周期。

## 下一步实验计划

1. 在 K1 内核上补 tracepoint 或 ftrace/eBPF 观测，记录 bulk 请求的 RTNL hold time、单条 `fib_table_insert()` 时间分布和 batch size 影响。
2. 做 K1 batch/jobs sweep：
   - batch：`256/512/1024/2000`
   - jobs：`1/2/4/8`
   - 目标：判断 K1 当前瓶颈是 FIB trie insert、全局 RTNL 还是 Python writer 构造 op 数组。
3. 重新设计 K2：
   - K2a：global RTNL + bulk notifier，验证 notifier 批处理是否能减少剩余开销。
   - K2b：synthetic-only no-notifier 模式，估计 per-netns FIB trie update 的性能上限，但不作为真实方案。
   - K2c：per-netns update + global RTNL notifier flush，审计是否存在 reader/device/nexthop 生命周期风险。
4. 只有 K2c 通过 smoke、sanity、K1 对照，并且没有 RTNL assertion/WARN/Oops，才考虑 `1x600k`、`10x600k_j4`、`100x600k_j4`。
5. K3/K4 暂不建议直接实现；它们需要完整审计 rtnetlink、fib rules、nexthop、netdevice notifier、VRF/lwtunnel 以及错误回滚语义。

## 当前结论

K0 已经证明：在 stock `6.8.0-124-generic` 下，单纯增加并发 `ip -batch` writer 不能线性提升 FIB 写入吞吐；jobs=8 相比 jobs=4 出现退化，并伴随 `rtnl_mutex` wait 明显增加。

K1 已经证明：新增 bulk route API 即使仍使用全局 RTNL，也能显著降低 synthetic route install 时间。`100 netns * 600,000 routes` 从 stock `ip -batch` 的 `3456.237s` 降到 K1 的 `585.122s`，约 `5.91x` 提升。

K2 已经给出负面结果：直接把 K1 的 global RTNL 换成 per-netns mutex，会在 10 条 route 的 smoke 阶段触发 `fib_notifier.c` 的 RTNL assertion。因此“每个 namespace 一个 route lock”不能按这个简单方式落地。下一步应围绕 FIB notifier 批处理、notifier 延后 flush 或 synthetic-only no-notifier 上限测试重新设计。

## 2026-06-18 K1 真实 3083 BIRD RIB -> FIB 实验

### 实验目的

把 K1 `BIRD_RT_BULK` 从 synthetic route 扩展到真实 BIRD RIB：

1. 启动 3083 规模拓扑。
2. 串行 start-bird，避免 start 阶段把 load 推高。
3. start-bird 后执行固定 10 AS BIRD 协议验证。
4. 从每个 brdnode 执行 `birdc show route primary`，生成 K1 bulk route ops。
5. 通过 K1 generic-netlink bulk API 写入 main FIB。
6. FIB 写入后等待 `load1 < 50`，再对每个 AS 固定一个 brdnode 做 `birdc show route count` 与 `ip route | wc -l` 验证。

### 代码与证据路径

| 类型 | 路径 |
|---|---|
| K1 真实 writer | `bird/kernel/tools/kernel_bulk_birdc_writer.py` |
| 验证脚本 | `bird/verify_bird_fib_vm.py` |
| 3083 K1 runner | `bird/run_3083_k1_bulk_real.sh` |
| build/up summary | `bird/runs/topology-3085_20260618_091704/remote_summary.json` |
| start-bird summary | `bird/runs/start-bird_20260618_095209_topology-3083/summary.json` |
| K1 skip-multipath summary | `bird/runs/k1-real-3083_20260618_091600_topology-3083/k1_bulk_writer_summary.json` |
| K1 multipath-first summary | `bird/runs/k1-real-3083_20260618_091600_topology-3083/k1_bulk_writer_first_summary.json` |
| K1 clean-FIB multipath-first summary | `bird/runs/k1-real-3083_20260618_091600_topology-3083/k1_bulk_writer_first_clean_summary.json` |
| clean-FIB flush summary | `bird/runs/k1-real-3083_20260618_091600_topology-3083/flush_before_k1_first_clean_summary.json` |
| start-bird verify | `bird/runs/k1-real-3083_20260618_091600_topology-3083/verify_after_start_bird.json` |
| FIB verify, skip-multipath | `bird/runs/k1-real-3083_20260618_091600_topology-3083/verify_after_fib_write.json` |
| FIB verify, multipath-first | `bird/runs/k1-real-3083_20260618_091600_topology-3083/verify_after_fib_write_first.json` |

### 阶段时间

| 阶段 | 结果 | 时间 |
|---|---:|---:|
| build | success | `176.726s` |
| up | success, 62/62 batches | `1904.256s` |
| start-bird | success, 2757/2757 brdnodes | `1271s` (`01:52:11` -> `02:13:22`) |
| verify after start-bird | success, 10 AS samples | `3.879s`, load1=`15.04` |

### K1 真实写入结果

第一轮写入使用默认 `multipath-policy=skip`：

| 指标 | 值 |
|---|---:|
| containers | `2757` |
| prefixes seen | `6,131,994` |
| ops requested/success | `5,908,151 / 5,908,151` |
| skipped multipath | `223,843` |
| dump wall | `2.345s` |
| write wall | `486.643s` |
| total wall | `488.992s` |
| kernel apply sum | `3840.851s` |
| verify after FIB | failed, `30/312` AS representatives failed |

这个结果说明：K1 对真实 BIRD route 的空 FIB 写入速度与旧 F/full writer 的历史基线接近，但它发现 3083 拓扑实际存在 `223,843` 条 multipath selected route。由于 K1 第一版不支持 ECMP，跳过 multipath 会让部分 AS 的 `ip route | wc -l` 小于 `birdc show route count` 的 network 数量，因此验证失败。

第二轮改为 `multipath-policy=first`，把每条 multipath route 近似写成第一条 nexthop：

| 指标 | 值 |
|---|---:|
| containers | `2757` |
| prefixes seen | `6,131,994` |
| ops requested/success | `6,131,994 / 6,131,994` |
| multipath-first | `223,843` |
| dump wall | `2.197s` |
| write wall | `1794.603s` |
| total wall | `1796.802s` |
| kernel apply sum | `14308.647s` |
| verify after FIB | success, `312/312` AS representatives passed |

注意：第二轮是在第一轮已经写入 `5.9M` FIB entries 后执行的 repeated `replace`，不是空 FIB 写入，因此不能直接作为 K1 空 FIB 性能基线。它的意义是证明 K1 writer 可以覆盖全部 BIRD primary networks 并通过验证；同时也说明“对已有大 FIB 重复 replace”代价很高。

第三轮先用旧 F 方法的 flush-only 清空各 brdnode 中 `proto bird metric 232` 的 route，再以 `multipath-policy=first` 做 clean-FIB K1 写入：

| 指标 | 值 |
|---|---:|
| flush wall | `338.752s` |
| flush brdnodes | `2757/2757` success |
| containers | `2757` |
| prefixes seen | `6,131,994` |
| ops requested/success | `6,131,994 / 6,131,994` |
| multipath-first | `223,843` |
| dump wall | `2.237s` |
| write wall | `498.043s` |
| total wall | `500.283s` |
| kernel apply sum | `3929.562s` |
| verify after FIB | 未执行；写入后 VM 被 reset，现场已清空 |

这个 clean-FIB 结果修正了第二轮 repeated replace 的偏差：K1 在真实 3083 拓扑上，单路径近似覆盖全部 `6.13M` selected networks 的空 FIB 写入耗时约 `500s`，与第一轮跳过 multipath 的 `488.992s` 接近。由于该轮写完后 SSH 控制面一度不可达，并随后执行了 `virsh destroy/start` 与 `reset_docker_vm.sh`，缺少 load-gated post-FIB verify。严格结论应表述为：writer summary 证明 K1 clean-FIB 写入过程成功返回 `0` 失败；FIB/RIB 正确性已经由第二轮 repeated replace 的 `312/312` 验证证明同一解析和写入路径可覆盖所有 AS，但 clean-FIB 这一轮本身没有保留现场验证。

### 重要发现

1. 3083 真实拓扑并非无 ECMP/multipath；至少 `223,843` 个 selected primary routes 有多个 nexthop。
2. K1 第一版只支持单 nexthop route op。`multipath-policy=first` 只能作为当前实验的单路径近似，不等价于真实 ECMP FIB。
3. 第一轮空 FIB bulk 写入 `5.91M` routes 用 `488.992s`，写入期间 SSH 会被压住，写后 load1 曾上升到数百，并伴随短时间大量 BIRD D-state。D-state 会自然消退，之后 load average 滞后下降。
4. 第二轮 repeated replace 全量写 `6.13M` routes 用 `1796.802s`，显著慢于第一轮，说明大 FIB 上 replace 的成本不能忽略。
5. 第三轮 clean-FIB multipath-first 全量写 `6.13M` routes 用 `500.283s`，说明 K1 的真实空 FIB 写入成本主要在约 `500s` 量级，而不是第二轮 repeated replace 的 `1796s`。
6. K1 相比 stock synthetic 的收益仍成立，但真实 BIRD 场景还需要增加 ECMP/multipath op 支持，才能做严格的真实语义对照。

### 下一步

1. K1b：扩展 `BIRD_RT_BULK` op 支持 ECMP/multipath，至少支持一条 route 携带多个 nexthop。
2. K1c：在 clean-FIB multipath-first 后不要立即 reset，等待 `load1 < 50` 后补一次 `verify_after_fib_write`，给 clean-FIB 口径补完整验证证据。
3. K2a：保留 global RTNL，研究批量 notifier 或延迟 notifier flush，避免 K2 直接 per-netns lock 触发 `fib_notifier.c` RTNL assertion。
4. K3/K4 暂不建议直接进入标准 `RTM_NEWROUTE` route-only 或 multi-netns transaction，除非先解决 notifier、nexthop、device lifetime 与 ECMP 语义。

## 2026-06-18 K2b No-Notify Per-Netns 上界实验

### 实验目的

K2 直接把 K1 中的 `rtnl_lock()` 换成 per-netns mutex 后，smoke 阶段触发 `fib_notifier.c:22` 的 `ASSERT_RTNL()`。为了确认“如果暂时忽略 route notifier/rtnetlink event，per-netns FIB 更新本身是否有性能收益”，做了一个 K2b 上界实验。

K2b 不是可交付真实语义方案，只用于判断潜在收益和隐藏依赖：

- `struct fib_config` 增加 `fc_skip_notify`。
- `BIRD_RT_BULK` path 设置 `fc_skip_notify=true`。
- `fib_table_insert()` / `fib_table_delete()` 在该 flag 下跳过 `call_fib_entry_notifiers()` 和 `rtmsg_fib()`。
- `BIRD_RT_BULK` path 使用 per-netns private `route_lock`，不拿 global RTNL。

相关 patch：

| 文件 | 说明 |
|---|---|
| `bird/kernel/patches/0003-bird-rtbulk-pernetns-no-notify-upper-bound.patch` | K2b 上界实验 patch |
| `/data/lxl/kernel-work/6.8.12-birdrtbulk-k2b.tar.gz` | K2b kernel package |
| `bird/runs/k2b_smoke_20260618_130025/` | 32 route smoke |
| `bird/runs/k2b_sanity4x10k_j4_20260618_130054/` | 4x10k sanity |
| `bird/runs/k2b_synth10x600k_j4_20260618_130108/` | 10x600k 成功样本 |
| `bird/runs/k2b_synth100x600k_j4_20260618_130138/` | 100x600k 失败样本和 Oops 证据 |

### K2b 结果

| 实验 | routers | routes/router | jobs | batch | 成功 routes | 失败 routes | wall | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| smoke | 1 | 32 | 1 | 16 | 32 | 0 | `0.00019s` writer wall | 通过 |
| sanity | 4 | 10,000 | 4 | 1024 | 40,000 | 0 | `0.249s` | 通过 |
| synthetic | 10 | 600,000 | 4 | 1024 | 6,000,000 | 0 | `16.440s` | 通过 |
| synthetic | 100 | 600,000 | 4 | 1024 | 58,200,000 已写入后失败 | 0 reported before crash | 未生成完整 summary | kernel Oops |

`10x600k_j4` 对比：

| kernel/mode | wall | throughput |
|---|---:|---:|
| K1 global RTNL, jobs=4, batch=1024 | `47.563s` 本轮复测 | `126,148 routes/s` |
| K2b per-netns no-notify, jobs=4, batch=1024 | `16.440s` | `364,963 routes/s` |

小规模结果说明：如果完全绕开 global RTNL 和 route notifier，跨 netns route insert 的性能上限确实更高。但 100x600k 失败说明这个路径不正确，不能把它作为真实方案。

### K2b Oops 证据

`100x600k_j4` 在 VM `6.8.12-birdrtbulk-k2b` 上运行时，load 和内存都正常：

```text
load average: about 3-4
MemAvailable: about 186GiB
```

但 kernel dmesg 出现 Oops：

```text
BUG: unable to handle page fault
RIP: fib_create_info+0x803/0x1080
Call Trace:
  fib_table_insert+0x80/0x670
  bird_rtbulk_apply_one_v4+0x1aa/0x1d0
  bird_rtbulk_apply_v4+0xdc/0x310
```

证据文件：

| 文件 | 内容 |
|---|---|
| `bird/runs/k2b_synth100x600k_j4_20260618_130138/dmesg_oops.txt` | Oops 栈 |
| `bird/runs/k2b_synth100x600k_j4_20260618_130138/summaries/` | 已完成的 97 个 writer summary，合计 58.2M routes |

### K2b 结论

K2b 排除了两个误判：

1. 失败不是 CPU 或内存耗尽。VM load 很低，内存充足。
2. 失败也不是 notifier 唯一问题。即使跳过 notifier，`fib_create_info()` 仍会在大规模并发下 Oops。

合理推断是：IPv4 FIB update 还有由 RTNL 隐式保护的共享状态，例如 `fib_info` hash/list、nexthop/device 引用、rules 或其他全局/跨表结构。仅靠 per-netns route mutex 保护 trie 本身不够。

因此 K2 的可行改进方向不是继续简单扩大 K2b，而是 K2c：

- 明确拆分 route trie、`fib_info`、nexthop、device lifetime、rules、notifier 的锁边界。
- 如果 `fib_info` hash 是跨 netns 共享或由 RTNL 串行保护，需要改成 per-netns 结构，或增加独立锁/RCU 生命周期管理。
- route notifier 不能简单跳过；真实方案需要 bulk notifier、延后 flush，或把 notifier 本身做成 per-netns/RCU-safe。
- 只有 K2c 在 lockdep/KASAN/KCSAN、netns delete、Docker stop/remove、route count 验证都通过后，才能进入真实 BIRD。

## 2026-06-18 K2c Fib-Info Hash Mutex 修复实验

### 修复目的

K2b 的 `100x600k_j4` 失败点是：

```text
fib_create_info -> fib_table_insert -> bird_rtbulk_apply_one_v4
```

这说明绕开 global RTNL 后，IPv4 FIB path 中仍有由 RTNL 隐式保护的共享结构。当前最直接的风险点是 `fib_info_hash` / `fib_info_laddrhash`：

- `fib_find_info()` 和 `fib_find_info_nh()` 会读取 hash/list。
- `fib_info_hash_move()` 可能扩容并释放旧 hash。
- `fib_release_info()` 会从 hash/list 删除 `fib_info`。
- K2b 允许不同 netns writer 并发进入这些路径，读写 hash 时没有 RTNL 串行保护。

K2c 的实验修复是新增 `fib_info_hash_mutex`，显式保护上述 hash 查找、扩容、插入和释放路径。它基于 K2b，仍保留 `fc_skip_notify=true` 的 no-notify synthetic 上界逻辑。

相关文件：

| 文件 | 说明 |
|---|---|
| `bird/kernel/patches/0004-bird-rtbulk-k2c-fib-info-hash-mutex.patch` | K2c 增量补丁 |
| `/data/lxl/kernel-work/6.8.12-birdrtbulk-k2c.tar.gz` | K2c kernel package |
| `bird/runs/k2c_smoke_20260618_094600/` | 32 route smoke |
| `bird/runs/k2c_sanity4x10k_j4_20260618_094613/` | 4x10k sanity |
| `bird/runs/k2c_synth10x600k_j4_20260618_094625/` | 10x600k |
| `bird/runs/k2c_synth100x600k_j4_20260618_094708/` | 100x600k |

### K2c 结果

| 实验 | routers | routes/router | jobs | batch | 成功 routes | failed routes | wall | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 32 | 1 | 16 | 32 | 0 | `0.000198s` | `161,429 routes/s` |
| sanity | 4 | 10,000 | 4 | 1024 | 40,000 | 0 | `0.245s` | `163,396 routes/s` |
| synthetic | 10 | 600,000 | 4 | 1024 | 6,000,000 | 0 | `15.969s` | `375,726 routes/s` |
| synthetic | 100 | 600,000 | 4 | 1024 | 60,000,000 | 0 | `132.815s` | `451,755 routes/s` |

`100x600k_j4` 运行后状态：

```text
kernel = 6.8.12-birdrtbulk-k2c
Docker containers = 0
temporary k2c netns = 0
dmesg grep BUG/Oops/panic/fib_create_info/bird_rtbulk = no match
loadavg after = 3.73 1.94 0.78
```

### K2c 与 K1/K2b 对比

| case | kernel/mode | wall | 说明 |
|---|---|---:|---|
| 100x600k | stock `ip -batch` | `3456.237s` | 普通 netlink route replace |
| 100x600k | K1 global RTNL bulk | `585.122s` | bulk API，仍全局 RTNL |
| 100x600k | K2b per-netns no-notify | 失败 | `58.2M` routes 后 Oops |
| 100x600k | K2c per-netns no-notify + `fib_info_hash_mutex` | `132.815s` | 达到 `<300s` 目标 |

K2c 说明了两个事实：

1. K2b 的主要崩溃点可以通过显式保护 `fib_info` hash/list 消除，至少 synthetic `100 * 600,000` 能跑完。
2. 跨 netns route update 的并行收益是真实存在的。K2c 对比 K1 约 `4.4x`，对比 stock `ip -batch` 约 `26x`。

但 K2c 仍不能直接视为真实 BIRD/main-table 方案：

- K2c 仍跳过 route notifier 和 rtnetlink event，这是为了 synthetic 上界测试。
- 还没有覆盖 nexthop object、ECMP、IPv6、device unregister、netns delete、fib rules、listener 语义。
- 还没有跑 lockdep/KASAN/KCSAN，也没有接入 Docker 容器 stop/remove 场景。

下一步如果继续推进 K2，需要把 no-notify 改为正确的 bulk notifier 或延迟 notifier，并审计 nexthop/device/rules/netns 生命周期；否则它仍然只是一个性能上界原型。

## 2026-06-18 K1 10x600k Jobs/Batch Sweep

### 实验目的

K2b 小规模显示 per-netns no-notify 有潜在收益，但大规模不正确。为了进一步确认 K1 当前瓶颈和可调参数，在稳定 K1 kernel `6.8.12-birdrtbulk` 上跑 `10 routers * 600,000 routes` sweep。

该实验不依赖 Docker，只使用临时 netns 和 `kernel_bulk_writer.py`。

### Jobs Sweep

固定参数：

- routers：`10`
- routes/router：`600,000`
- total routes：`6,000,000`
- batch：`1024`
- kernel：`6.8.12-birdrtbulk`
- run dir：`bird/runs/k1_synth10x600k_jobs_sweep_20260618_134058/`

| jobs | wall | throughput | failed routes |
|---:|---:|---:|---:|
| 1 | `49.959s` | `120,099 routes/s` | 0 |
| 2 | `42.855s` | `140,006 routes/s` | 0 |
| 4 | `47.563s` | `126,148 routes/s` | 0 |
| 8 | `48.960s` | `122,549 routes/s` | 0 |
| 12 | `54.461s` | `110,170 routes/s` | 0 |
| 16 | `57.039s` | `105,192 routes/s` | 0 |

结论：K1 在该规模下最佳 jobs 是 `2`。继续增加 writer 不会提高吞吐，反而退化。这符合 K1 仍使用 global RTNL 的设计预期：bulk API 已减少逐条 netlink 成本，但跨 writer 并发仍会在全局串行点排队。

### Batch Sweep

固定参数：

- routers：`10`
- routes/router：`600,000`
- total routes：`6,000,000`
- jobs：`2`
- kernel：`6.8.12-birdrtbulk`
- run dir：`bird/runs/k1_synth10x600k_batch_sweep_20260618_134702/`

| batch | wall | throughput | failed routes |
|---:|---:|---:|---:|
| 256 | `41.492s` | `144,608 routes/s` | 0 |
| 512 | `41.572s` | `144,326 routes/s` | 0 |
| 1024 | `41.932s` | `143,087 routes/s` | 0 |
| 2000 | `43.427s` | `138,164 routes/s` | 0 |

结论：batch 在 `256-1024` 之间差异很小，`2000` 反而稍慢。当前 K1 主要瓶颈不是 batch 太小，而是 FIB insert 和 global RTNL 语义边界。

### 对 K3/K4 的影响

K3 是把标准 `RTM_NEWROUTE/DELROUTE` route-only 化。K4 是 multi-netns bulk transaction。当前实验结论是：

- K3 不能直接做“普通 route netlink handler 不拿 RTNL”。K2/K2b 已证明 `fib_notifier` 和 `fib_create_info` 都有 RTNL 隐式依赖。
- K4 如果仍拿 global RTNL，只是把多个 netns 的 route ops 放到一个更长的全局临界区里，可能降低 netlink 往返，但不能产生跨 netns 并发，还会放大长时间持锁风险。
- K4 如果基于 per-netns lock，则必须先有正确的 K2c。否则 multi-netns transaction 只会把 K2b 的 Oops 风险扩大。

因此当前不建议继续实现 K3/K4 原型。下一轮更有价值的工作是 K2c 的锁/生命周期审计，或 K1b 的 ECMP/multipath 支持。

## 2026-06-18 3083 K1 Delta-Patched Real Run Attempt

### 实验目的

在真实 3083 规模拓扑上验证“delta event + K1 bulk writer”的完整流程：

1. 用 delta-fast image 注入 patched BIRD 和 `delta.conf`。
2. 启动拓扑。
3. 串行启动 patched BIRD，间隔 `0.1s`。
4. 等 VM `load1 < 50` 后再做 K1 full FIB write。
5. 后续 reconvergence delta route write 也计划使用 K1 writer。

### 命令与参数

运行目录：

```text
bird/runs/bird-delta-3085_20260618_152355_topology-3083/
```

关键参数：

```text
build_mode=delta-fast
fib_writer=k1-bulk
project=topology-3083
restart_jobs=1
restart_delay=0.1s
load_threshold=50
load_only_stable=true
k1_batch=2000
k1_dump_jobs=16
k1_multipath_policy=first
```

### 阶段结果

| 阶段 | 结果 | 时间 |
|---|---:|---:|
| sync | 成功 | `8.597s` |
| delta-fast build | 成功 | `170.950s` |
| docker compose up | 成功 | `3104.997s` |
| restart patched BIRD | 成功，`2757/2757` brdnode | `1331.551s` |
| wait_rib_stable | 失败/中止 | load 长期远高于 50 |
| K1 full FIB write | 未执行 | 前置 load gate 未通过 |
| reconvergence K1 delta write | 未执行 | 前置阶段未完成 |

`docker compose up` 使用 `up_jobs=4, up_batch=50`，跳过 `seedemu_internet_map`。最终 Docker 运行容器为 `3083`，其中 brdnode router 为 `2757`。

### 高负载证据

`restart_bird` 在约 2500 个容器后开始出现极高 load。`restart_bird` 完成后，runner 进入 `wait_rib_stable`，但 load 长时间无法回到 50 以下。

观测点：

| UTC 时间 | load1 | 说明 |
|---|---:|---|
| `08:34` | `337.34` | 2500 容器附近开始尖峰 |
| `08:35` | `496.04` | BIRD 大量启动中 |
| `08:39` | `1032.40` | BIRD 进程约 `2701` |
| `08:54` | `945.90` | BIRD 进程 `2757`，约 `1978` 个 D 状态 |
| `09:05` | `1207.80` | 高负载已持续约 30 分钟，仍有约 `996` 个 BIRD D 状态 |

按 `run-experient-vm` 恢复策略，load 长期超过 `250` 且持续约 30 分钟后判定本轮不可恢复，保存 summary 后执行 `virsh destroy/start` 和 `/home/lxl/bird/reset_docker_vm.sh`。

### 结论

这轮没有测到 K1 full writer 的速度；失败点在“从一开始就启用 delta-patched BIRD 记录初始 RIB 事件”的 start-bird/RIB 收敛阶段。

与此前成功的 3083 K1 real run 对比：

- 成功 run 使用 stock BIRD 初始启动，start-bird 之后 10-AS 协议验证通过，随后 K1 clean-FIB multipath-first 写入 `6,131,994` 条 route，耗时 `500.283s`。
- 本轮 delta-patched BIRD 从初始 RIB 开始记录 events，在 2757 brdnode 同时收敛时造成大量 BIRD D 状态和长期高 load，未进入 FIB write。

后续真实 delta 实验不能在初始 RIB 收敛前就打开全量 delta event。更合理的做法是：

1. 初始阶段使用 stock BIRD 或关闭 delta event 的 patched BIRD。
2. 等 RIB 稳定并完成初始 K1 full FIB write 后，再启用 delta event。
3. 启用 delta event 后必须先丢弃/mark 初始 replay，只保留故障后的 route change。
4. reconvergence 后的 delta route write 再使用 K1 command writer。

这也说明 K1 路径本身的瓶颈和 delta event 采集的瓶颈需要分开测量。

## 2026-06-18 3083 K1 Deferred-Delta Real Run

### 实验目的

上一轮 `delta-fast` 初始启动时直接启用 delta protocol，导致 start-bird/RIB 收敛阶段 load 长期超过 `250`，没有进入 K1 full writer。

本轮改成更安全的 deferred-delta 流程：

1. VM boot 到 K1 kernel `6.8.12-birdrtbulk`。
2. reset Docker，清空旧容器/network/cache。
3. 使用 `delta-fast` image，但 `restart-bird` 时设置 `DELTA_ENABLE_ON_RESTART=0`，即启动 patched BIRD 但不启用 delta protocol。
4. 等 load1 低于 `50`。
5. 用 K1 `kernel_bulk_birdc_writer.py` 做初始 full RIB-to-FIB 写入。
6. 初始 full 写入后再启用 delta protocol，准备做 reconvergence delta。

本轮脚本：

```text
bird/run_3083_k1_delta_reconvergence.sh
```

本轮目录：

```text
bird/runs/k1-delta-reconv-3083_20260618_175600/
bird/runs/bird-delta-3085_20260618_175722_topology-3083/
```

### 阶段结果

| 阶段 | 结果 | 时间 |
|---|---:|---:|
| VM K1 reset + image import | 成功 | 见 `reset_k1_vm.log` |
| delta-fast build | 成功 | `184.228s` |
| docker compose up | 成功 | `2731.090s` |
| restart patched BIRD, delta disabled | 成功，`2757/2757` brdnode | `922.878s` |
| load-only wait | 成功，load1 从 `54.46` 降到 `31.69` | `41.836s` |
| start-bird protocol validate | 成功 | `3.112s` |
| initial K1 full FIB write | 成功 | stage `509.676s`，writer wall `496.183s` |
| enable delta after initial FIB | 命令完成，但触发高 load | `245.877s` |
| verify initial FIB after enable-delta | 未完成 | VM load 已不可用 |
| reconvergence K1 delta write | 未执行 | 前置 load/verify 条件失败 |

### 初始 K1 Full Writer 结果

`kernel_bulk_birdc_writer.py` 参数：

```text
jobs=8
dump_jobs=16
batch=2000
route_action=replace
multipath_policy=first
metric=232
protocol=12
```

结果：

| 指标 | 值 |
|---|---:|
| containers | `2757` |
| dump successes/failures | `2757 / 0` |
| write successes/failures | `2757 / 0` |
| prefixes seen | `6,131,994` |
| ops requested | `6,131,994` |
| success/fail ops | `6,131,994 / 0` |
| multipath-first routes | `223,843` |
| writer wall | `496.183s` |
| dump wall | `2.185s` |
| write wall | `493.996s` |
| kernel apply seconds sum | `3902.408s` |

这证明：在 3083 真实拓扑上，K1 full RIB-to-FIB 写入路径可稳定写完 `6.13M` 条 selected route，且和前一轮 clean-FIB K1 结果一致。

### Deferred Delta 仍失败的原因

本轮关键发现是：`--defer-delta` 只能避免 start-bird 阶段的初始 delta event，但不能避免之后 `enable-delta` 时 BIRD 对当前 RIB 做一次初始导出/replay。

启用 delta 后的观测：

| 时间点 | load1 | load5 | load15 | D-state | 说明 |
|---|---:|---:|---:|---:|---|
| enable-delta 后约 1 分钟 | `1539.79` | `1225.60` | `766.94` | `1857` | 大量 BIRD 进入不可中断状态 |
| 后续约 3 分钟 | `1578.52` | `1386.57` | `919.26` | `296` | D-state 短暂下降但 load 仍极高 |
| 后续约 8 分钟 | `1507.24` | `1467.38` | `1088.25` | `1882` | 判定不可能进入 reconvergence |

因此，本轮未执行 reconvergence K1 delta write。原因不是 K1 writer 性能问题，而是当前 BIRD delta protocol 的启用语义会 replay 初始 RIB，导致和“初始收敛时开 delta”相同的负载问题。

### 恢复动作

本轮中止后执行：

```text
virsh destroy bird-rtnl-dev
virsh start bird-rtnl-dev
/home/lxl/bird/reset_docker_vm.sh
```

恢复过程中发现 libvirt `default` network 丢失，已重建标准 NAT network：

```text
name=default
bridge=virbr0
subnet=192.168.122.0/24
gateway=192.168.122.1
```

最终 VM 状态：

```text
kernel = 6.8.12-birdrtbulk
containers = 0
running = 0
images = 4
volumes = 0
custom_networks = 0
```

### 结论

真实 3083 K1 full 写入已经完成并记录。reconvergence K1 delta 还不能跑，必须先解决“delta protocol 启用时 replay 当前 RIB”的问题。

下一版真实 delta 方案需要满足以下之一：

1. BIRD delta protocol 支持 `ignore_initial_dump` / `start_after_current_rib` 语义，启用时只记录之后的 route change。
2. 常驻 agent 不依赖启用 protocol 时的全量 replay，而是在 BIRD 内部 hook route change event 并从某个 epoch 后开始记录。
3. 如果必须 replay，必须支持低速分片初始化并明确丢弃 initial dump，但这会改变实验变量，不适合作为 reconvergence delta 性能对照。

## 2026-06-19 3083 K1/K2c Full + Delta Reconvergence

### 实验目的

本轮继续验证两件事：

1. 用 K1 在 3083 真实拓扑上完成初始 full RIB-to-FIB 和故障后的 delta RIB-to-FIB。
2. 在 K1 基础上切到 K2c kernel，验证 route-only per-netns lock 原型在真实 BIRD main-table 路由上的速度和正确性。

本轮仍采用 deferred-delta 流程：

1. build/up 3083 输出目录。
2. 串行 `restart_bird`，启动 BIRD 但不启用 delta event。
3. 等 load1 低于 `50` 后，用 bulk writer 初始 full 写入 FIB。
4. 初始 FIB 验证通过后，启用 delta event，并使用 `ignore_initial_feed=true` 跳过当前 RIB 的初始 replay。
5. 停 10 个跨 AS brdnode 模拟网络波动。
6. 只按 load gate 判断 reconvergence 后，读取 BIRD delta event，折叠为 route ops，再用 bulk API 写入 main FIB。

### 本轮脚本和产物

| 项 | 路径 |
|---|---|
| K1 wrapper | `bird/run_3083_k1_delta_reconvergence.sh` |
| K2c wrapper | `bird/run_3083_k2c_delta_reconvergence.sh` |
| K1 run dir | `bird/runs/k1-delta-reconv-3083_postfibgate_20260619_112712/` |
| K1 fixed-scope rerun | `bird/runs/k1-delta-reconv-3083_fixedgate_20260619_173049/` |
| K2c run dir | `bird/runs/k2c-delta-reconv-3083_20260619_133056/` |
| K1 reconvergence artifact | `bird/runs/k1-delta-reconv-3083_postfibgate_20260619_112712/reconvergence_artifacts/k1_reconvergence_artifacts.tgz` |
| K2c reconvergence artifact | `bird/runs/k2c-delta-reconv-3083_20260619_133056/reconvergence_artifacts/k2c_reconvergence_artifacts.tgz` |

K2c wrapper 复用 K1 wrapper，所以部分日志字段仍叫 `k1-bulk` 或 `k1-command-bulk-writer`。实际内核由 wrapper 设置为 `6.8.12-birdrtbulk-k2c`。

### 初始 Full RIB-to-FIB 对比

两轮都使用同一套真实 3083 拓扑，brdnode 数 `2757`，bulk writer 参数一致：

```text
--fib-writer k1-bulk
--write-jobs 8
--k1-dump-jobs 16
--k1-batch 2000
--k1-multipath-policy first
```

| 指标 | K1 global RTNL | K2c route-only per-netns lock |
|---|---:|---:|
| kernel | `6.8.12-birdrtbulk` | `6.8.12-birdrtbulk-k2c` |
| build | `165.001s` | `177.603s` |
| docker compose up | `2840.546s` | `3480.482s` |
| restart_bird | `910.884s` | `980.416s` |
| wait load after start-bird | `0.610s`, final load1 `48.47` | `43.374s`, final load1 `42.82` |
| start-bird 10-sample validate | `2.908s`, success | `3.294s`, success |
| initial writer wall | `489.736s` | `95.537s` |
| BIRD dump wall | `2.194s` | `2.412s` |
| kernel write wall | `487.539s` | `93.123s` |
| route ops | `6,131,994` | `6,131,994` |
| failed ops | `0` | `0` |
| multipath-first routes | `221,860` | `223,843` |
| kernel apply seconds sum | `3849.479s` | `690.445s` |
| post-FIB verify | `312/312` AS reps passed | `312/312` AS reps passed |

结论：在同样的 route ops 和 writer 并发下，K2c 初始 full 写入从约 `489.736s` 降到 `95.537s`，约 `5.1x`。主要差异在内核写入阶段，不在 BIRD dump；两轮 dump 都约 `2s`。

### Reconvergence Delta 对比

故障方式：停掉 10 个分布在不同 AS 的 brdnode。两轮都先启用 delta event，并跳过 initial feed。

| 指标 | K1 global RTNL | K2c route-only per-netns lock |
|---|---:|---:|
| reconvergence summary | failed, validation failed | success |
| stop victims | `25.688s` | `27.018s` |
| wait RIB stable | `0.010s`, load-only | `0.002s`, load-only |
| prepare delta offsets | `58.086s` | `64.708s` |
| raw route events | `367,461` | `362,705` |
| collapsed route ops | `198,835` | `198,827` |
| delta dump/collapse wall | `6.031s` | `6.931s` |
| delta bulk writer wall | `260.365s` | `236.353s` |
| success_count | `141,727` | `198,827` |
| ignored_delete_miss | `57,108` | `0` |
| fail_count | `0` | `0` |
| validate | failed, stale FIB routes remained | passed, 10/10 samples RIB/FIB equal |

K2c reconvergence validation samples all passed，例如：

```text
as1269brd-r11: rib=2222 fib=2222 errors=0
as1284brd-r35: rib=2104 fib=2104 errors=0
as2903brd-r204: rib=660 fib=660 errors=0
```

K1 的失败不是内核 crash，也不是 BIRD event 折叠失败，而是 delete op 的用户态编码 bug：

- delta delete 命令为了删除 prefix，通常不带 nexthop。
- 原 writer 对无 nexthop delete 使用 `RT_SCOPE_LINK`。
- 但之前 full writer 安装的 via route 是 `RT_SCOPE_UNIVERSE`。
- `fib_table_delete()` 默认会匹配 scope；scope 不一致时返回 `-ESRCH`。
- 因此 K1 里 `57,108` 条 delete miss 被 `--ignore-delete-miss` 忽略，FIB 保留旧路由，最终验证出现 `fib > rib`。

修复已经写入用户态 writer：

```text
bird/kernel/tools/kernel_bulk_writer.py: RT_SCOPE_NOWHERE = 255
bird/kernel/tools/kernel_bulk_command_writer.py: delete op 使用 RT_SCOPE_NOWHERE
bird/kernel/tools/kernel_bulk_birdc_writer.py: route-action=del 使用 RT_SCOPE_NOWHERE
```

`fib_table_delete()` 对 `RT_SCOPE_NOWHERE` 按 wildcard 处理，可以删除 full writer 之前安装的 `UNIVERSE` scope route。K2c 真实实验在修复后执行，`ignored_delete_miss=0` 且验证通过。

### K1 Fixed-scope Rerun 结果

为确认 K1 在 delete-scope 修复后是否可以继续跑 reconvergence delta，又做了一轮 K1 fixedgate 复跑：

```text
RUN_NAME=k1-delta-reconv-3083_fixedgate_20260619_173049
kernel=6.8.12-birdrtbulk
INITIAL_WRITE_JOBS=8
DELTA_WRITE_JOBS=8
DELTA_JOBS=16
LOAD_THRESHOLD=50
```

本轮 build/up/start-bird 都成功，初始 full writer 也成功：

| 阶段/指标 | 结果 |
|---|---:|
| build | `180.644s` |
| docker compose up | `3380.977s` |
| restart_bird | `974.081s` |
| wait load after start-bird | `88.417s` |
| start-bird 10-sample validate | `3.219s`, success |
| initial writer wall | `543.164s` |
| BIRD dump wall | `2.460s` |
| kernel write wall | `540.701s` |
| route ops | `6,131,994` |
| failed ops | `0` |
| multipath-first routes | `223,843` |
| kernel apply seconds sum | `4273.490s` |

但是 full writer 返回成功后，VM 长时间不能恢复到可验证状态：

- `fetch_artifacts` 被高负载拖到 `410.795s`。
- `wait_after_initial_k1_fib` 从 `19:04:35` 开始反复 SSH/load 采样失败，持续超过 `30min`。
- 可采到的一次 load 为 `1263.02 1148.21 633.73`。
- 因超过高负载恢复阈值，本轮在 post-FIB verify 前终止并执行 VM `destroy/start/reset_docker_vm.sh`。

所以 delete-scope 修复后，K1 reconvergence delta 仍没有可用实测值。原因不是 delta writer 本身失败，而是 K1 global RTNL full 写入完成后系统恢复时间不可接受，无法进入“load < 50 后验证、启用 delta、制造故障、写 delta”的后续流程。

### 结果判断

K1：

- 初始 full RIB-to-FIB 可以完成，但在 3083 真实拓扑上需要约 `490s` 到 `543s`，并可能让 VM 在写入完成后超过 `30min` 不能恢复到 `load < 50`。
- reconvergence delta 的第一轮 K1 测试暴露了 delete scope bug；修复后复跑又卡在 post-full load gate，因此目前没有可靠的 K1 delta 性能结论。
- K1 更适合作为“持 global RTNL 的 bulk API 基线”，不适合作为 3083 规模后续 reconvergence 主方案。

K2c：

- synthetic `100x600k_j4` 已达成 `132.815s` 跑完 60M route ops。
- 真实 3083 初始 full 写入 `6.13M` route ops 用 `95.537s`，比 K1 快约 `5.1x`。
- 真实 reconvergence delta 折叠后 `198,827` route ops，bulk 写入 `236.353s`，最终抽样 RIB/FIB 一致。
- 本轮 `dmesg` 未发现 `BUG`、`Oops`、`panic`、`fib_create_info` 或 `bird_rtbulk` 相关异常。

限制仍然存在：K2c 是 no-notify upper-bound 原型。它已经能在真实 BIRD main-table 路由上工作并通过本轮抽样验证，但还不是生产语义完整方案；route notifier、rtnetlink event、nexthop object、ECMP 完整表达、IPv6、设备生命周期、fib rules、netns 删除路径仍需继续审计。

## 2026-06-19 K1 Safe-jobs 全流程成功与三方法最终对比

### K1 Safe-jobs 复跑目的

前一轮 K1 fixed-scope 使用 `INITIAL_WRITE_JOBS=8`，full writer 虽然写完了 `6,131,994` 条路由，但写完后 VM load 曾达到 `1263.02 1148.21 633.73`，超过 `30min` 不能进入 post-FIB verify。因此本轮把 K1 写入降速，只要求先跑通全流程：

```text
RUN_NAME=k1-delta-reconv-3083_safej1_20260619_194605
kernel=6.8.12-birdrtbulk
INITIAL_WRITE_JOBS=1
DELTA_WRITE_JOBS=1
K1_BATCH=256
K1_DELTA_BATCH=256
DELTA_JOBS=16
LOAD_THRESHOLD=50
```

### K1 Safe-jobs 结果

| 阶段/指标 | 结果 |
|---|---:|
| build | `165.913s` |
| docker compose up | `2929.003s` |
| restart_bird | `933.490s` |
| wait load after start-bird | `0.756s`, load1 `36.11` |
| initial full stage | `409.858s` |
| initial full writer wall | `402.388s` |
| BIRD dump wall | `2.214s` |
| kernel write wall | `400.173s` |
| full route ops | `6,131,994` |
| full failed ops | `0` |
| wait after initial FIB | `1s`, final load1 `8.12` |
| post-full FIB verify | `312/312` AS reps passed |
| enable delta | `33s`, `ignore_initial_feed=true` |
| wait after enable delta | `1s`, final load1 `16.57` |
| reconvergence total | `622.785s` |
| stop 10 victims | `28.175s` |
| wait RIB stable | `0.003s`, load-only |
| prepare delta offsets | `519.594s` |
| delta dump/collapse | `8.450s` |
| delta raw route events | `360,597` |
| delta collapsed route ops | `198,804` |
| delta K1 writer wall | `63.639s` |
| delta success/fail | `198,804 / 0` |
| delta ignored delete miss | `0` |
| delta validate | success, `10/10` samples |

关键结论：

- K1 失败点不是 full writer 不能写，而是 `jobs=8,batch=2000` 会让写完后的系统恢复时间不可接受。
- `jobs=1,batch=256` 能跑通 3083 full + delta 全流程，并且 post-full load gate 立即通过。
- K1 reconvergence 总时间 `622.785s` 看起来较长，主要不是 FIB 写入慢，而是 `prepare_delta_offsets=519.594s`。这是因为本轮为降低压力把 `DELTA_WRITE_JOBS=1`，导致 offset 准备对 `2747` 个 brdnode 顺序执行。真正 delta RIB-to-FIB 写入只有 `63.639s`。

### 三方法 RIB-to-FIB 时间对比

这里比较的是同一 3083 拓扑上的 RIB 写 main FIB。K2 使用当前可跑通的 K2c 版本；原始 K2/K2b 已在前文证明不满足正确性。

#### 初始 Full RIB-to-FIB

| 方法 | kernel / writer | jobs,batch | route ops | dump | FIB write | writer wall | stage wall | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F | stock/F host `setns-ip` | `8,-` | `6,131,994` | included | included | `577.208s` | `579.296s` | success |
| K1 safe | `6.8.12-birdrtbulk`, global RTNL bulk | `1,256` | `6,131,994` | `2.214s` | `400.173s` | `402.388s` | `409.858s` | success |
| K2c | `6.8.12-birdrtbulk-k2c`, per-netns no-notify bulk | `8,2000` | `6,131,994` | `2.412s` | `93.123s` | `95.537s` | `107.226s` | success |

Full 写入结论：

- K1 safe 比 F 快约 `577.208 / 402.388 = 1.43x`。
- K2c 比 F 快约 `577.208 / 95.537 = 6.04x`。
- K2c 比 K1 safe 快约 `402.388 / 95.537 = 4.21x`。

#### Reconvergence Delta RIB-to-FIB

| 方法 | jobs,batch | raw events | collapsed route ops | delta dump/collapse | FIB delta write | write_delta_fib stage | reconvergence total | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F | `8,-` | `365,367` | `198,830` | included | included | `225.951s` (`writer=225.089s`) | `464.938s` | success |
| K1 safe | `1,256` | `360,597` | `198,804` | `8.450s` | `63.639s` | `73.698s` | `622.785s` | success |
| K2c | `8,2000` | `362,705` | `198,827` | `6.931s` | `236.353s` | `245.045s` | `338.218s` | success |

Delta 写入结论：

- 单看真实 FIB delta write，K1 safe 是最快的：`63.639s`，约比 F 的 `225.089s` 快 `3.54x`。
- K2c 本轮 delta write `236.353s`，略慢于 F；这与 K2c 的 `jobs=8,batch=2000` 和 `fib_info_hash_mutex` 尾延迟有关，不能简单解释为 K2c 没有收益。它在 full 写入上仍显著最快。
- 看 reconvergence total，K2c 最短：`338.218s`。K1 safe 总时间被 `prepare_delta_offsets=519.594s` 拉长；如果把 offset 准备并发从 FIB 写入并发中拆出来，K1 total 还有明显优化空间。

#### F strict 控制组

另一个 strict control 在故障后直接做 full/delta command replay，用于区分 dump 和 write 成本：

| 控制组 | commands | dump | write |
|---|---:|---:|---:|
| F full post-reconvergence | `6,097,653` | `15.732s` | `2025.107s` |
| F delta post-reconvergence | `561,377` | `6.434s` | `238.320s` |

这个控制组说明：F 方法的主要瓶颈仍是 Linux FIB 写入，而不是 BIRD dump 或 event 解析。

### 当前工程判断

1. K1 已经可以跑通 3083 全流程，但需要使用保守 writer 参数，避免 full 写完后 load backlog 卡住验证。
2. K2c 已经跑通 3083 full + delta，并达成 synthetic `100x600k` 小于 `300s` 的目标；但它仍是 no-notify 原型，不是完整内核语义方案。
3. 三方法中，full 写入 K2c 优势最大；delta 写入当前 K1 safe 最快，但 K1 的整体 reconvergence 被 offset 准备串行化拖慢。
4. 下一步若继续优化，应把 reconvergence 脚本中的 offset 准备并发与 FIB writer 并发拆开，例如 `OFFSET_JOBS=16`、`DELTA_WRITE_JOBS=1`，这样保留 K1 低压写 FIB，同时避免 `prepare_delta_offsets` 顺序扫描 2747 个容器。

## 2026-06-20 K2c Mixed-batch Delta 优化

### 优化目的

上一轮 K2c 真实 3083 reconvergence 能跑通，但 delta writer 不理想：

```text
run: bird/runs/k2c-delta-reconv-3083_20260619_133056/
K2c delta writer: 236.353s
jobs,batch: 8,2000
collapsed route ops: 198,827
kernel_apply_seconds_sum: 1826.924s
```

主要问题不是 event dump，而是 writer 把 replace 和 delete 拆成更多批次并以 `jobs=8` 并发提交，`fib_info_hash_mutex` 和 route update 路径等待被放大。优化方向：

- 把 delta command 中同一容器的 replace/delete 合并为 mixed batch，减少批次数。
- 将 offset 准备并发和 FIB writer 并发拆开：`OFFSET_JOBS=16`，`DELTA_WRITE_JOBS=1`。
- 保留 K2c no-notify bulk path，避免引入 K3/K4 级别的标准 rtnetlink 改造风险。

### 代码变更

| 文件 | 变更 |
|---|---|
| `bird/kernel/tools/kernel_bulk_command_writer.py` | 新增 `apply_mixed_ops()` 和 `--mixed-batches/--no-mixed-batches`；mixed 模式将 replace/delete 合并成同一 op 序列提交给 K1/K2c bulk API |
| `bird/reconvergence_bird_delta.sh` | 新增 `OFFSET_JOBS`，使 offset 准备并发不再绑定 FIB writer 并发；新增 `K1_DELTA_MIXED_BATCHES` 开关 |
| `bird/run_3083_k1_delta_reconvergence.sh` | 传递 `OFFSET_JOBS` 与 `K1_DELTA_MIXED_BATCHES` |
| `bird/run_3083_k2c_delta_optimized.sh` | 新增 K2c mixed-batch 3083 wrapper，默认 `OFFSET_JOBS=16`、`DELTA_WRITE_JOBS=1`、`K1_DELTA_MIXED_BATCHES=1` |

语义注意：真实 reconvergence 中 `ignored_delete_miss=0`，因此 mixed-batch 路径没有靠忽略 delete miss 才通过验证。后面的 replay sweep 是在已经写入过的最终 FIB 状态上重放同一批命令，会出现幂等 delete miss；它只用于性能拐点观察，不作为新的语义正确性证据。

### 真实 3083 实验命令

```bash
RUN_NAME=k2c-delta-mixed-3083_20260620_1150 \
K1_GRUB_ENTRY_ID='1>2' RESET_VM=1 INITIAL_WRITE_JOBS=8 \
DELTA_WRITE_JOBS=1 OFFSET_JOBS=16 K1_DELTA_MIXED_BATCHES=1 \
K1_DELTA_BATCH=2000 RUN_FULL_COMPARISON=0 LOAD_THRESHOLD=50 \
MAX_WAIT_SECONDS=7200 POLL_INTERVAL=30 SSH_CONNECT_TIMEOUT=30 \
bird/run_3083_k2c_delta_optimized.sh
```

说明：VM 的 GRUB saved-entry 字符串没有稳定生效，本轮使用数字 entry `1>2` 成功启动 `6.8.12-birdrtbulk-k2c`。

证据目录：

| 项 | 路径 |
|---|---|
| local run dir | `bird/runs/k2c-delta-mixed-3083_20260620_1150/` |
| main summary | `bird/runs/k2c-delta-mixed-3083_20260620_1150/main_summary.json` |
| reconvergence summary | `bird/runs/k2c-delta-mixed-3083_20260620_1150/reconvergence/summary.json` |
| comparison summary | `bird/runs/k2c-delta-mixed-3083_20260620_1150/comparison_summary.json` |
| remote main dir | `/home/lxl/bird/runs/bird-delta-3085_20260620_114821_topology-3083` |
| remote reconvergence dir | `/home/lxl/bird/runs/k2c-delta-mixed-3083_20260620_1150_reconvergence/reconvergence` |

### 真实 3083 结果

| 阶段/指标 | 结果 |
|---|---:|
| kernel | `6.8.12-birdrtbulk-k2c` |
| total main duration | `4551.253s` |
| build | `205.432s` |
| docker compose up | `3019.109s` |
| restart_bird | `1230.731s`, serial jobs=1, delay=0.1 |
| wait load after start-bird | `0.602s`, load-only, load1 `< 50` |
| start-bird validate | success, `10/10` samples |
| initial full writer wall | `71.162s` |
| initial full stage | `81.347s` |
| initial full route ops | `6,131,994 / 6,131,994` |
| initial full failed ops | `0` |
| initial full kernel apply sum | `499.192s` |
| post-full FIB verify | success, `312/312` AS reps passed |
| enable delta | success, `34s`, `ignore_initial_feed=true` |
| reconvergence total | `157.028s` |
| stop 10 victims | `28.112s` |
| wait RIB stable | `30.003s`, load-only |
| prepare delta offsets | `32.334s` |
| delta dump/collapse | `8.596s` |
| delta raw route events | `371,214` |
| delta collapsed route ops | `198,869` |
| delta writer jobs,batch | `1,2000` |
| delta mixed batches | `true` |
| delta writer wall | `55.026s` |
| delta writer success/fail | `198,869 / 0` |
| delta ignored delete miss | `0` |
| delta kernel apply sum | `41.680s` |
| post-delta validate | success |
| post-run dmesg grep | no `BUG/Oops/panic/fib_create_info/bird_rtbulk/rtbulk/WARNING` match |

### 与上一轮 K2c 对比

| 指标 | K2c 2026-06-19 | K2c mixed 2026-06-20 | 改善 |
|---|---:|---:|---:|
| full writer wall | `95.537s` | `71.162s` | `1.34x` |
| full stage wall | `107.226s` | `81.347s` | `1.32x` |
| reconvergence total | `338.218s` | `157.028s` | `2.15x` |
| prepare delta offsets | `64.708s` | `32.334s` | `2.00x` |
| delta dump/collapse | `6.931s` | `8.596s` | slightly slower |
| delta writer wall | `236.353s` | `55.026s` | `4.30x` |
| delta kernel apply sum | `1826.924s` | `41.680s` | `43.83x` |
| delta validation | success | success | same |

解释：

- `OFFSET_JOBS=16` 直接把 offset 准备从 `64.708s` 降到 `32.334s`。
- `DELTA_WRITE_JOBS=1` 避免 8 个 writer 同时争 K2c 的 route/fib_info 路径，尾延迟明显下降。
- `--mixed-batches` 把 replace/delete 合批，批次数从上一轮拆分模式的约 `5476` 降到约 `2763`。
- CPU 不是本轮瓶颈。后续 replay 中 jobs=16 仍有约 `69%~79% idle`，但 wall time 变差，说明瓶颈主要是内核 route 更新同步等待，而不是算力耗尽。

### Replay Sweep：并发拐点

在真实 reconvergence 成功后，复用同一批 delta command 文件做 K2c mixed writer 重放：

```text
commands_dir=/home/lxl/bird/runs/k2c-delta-mixed-3083_20260620_1150_reconvergence/reconvergence/k1_delta_commands
commands=198,869
batch=2000
mixed_batches=true
```

这是幂等 replay，不是新的语义实验。由于 delete 已在真实实验中执行过，重放时会出现 `ignored_delete_miss=57,108`。

| jobs | wall | kernel_apply_seconds_sum | ignored_delete_miss | 结果 |
|---:|---:|---:|---:|---|
| 2 | `45.688s` | `72.110s` | `57,108` | success |
| 4 | `51.146s` | `173.165s` | `57,108` | success |
| 8 | `61.906s` | `440.821s` | `57,108` | success |
| 12 | `164.108s` | `1880.525s` | `57,108` | success |
| 16 | `179.387s` | `2762.181s` | `57,108` | success |

拐点判断：

- replay 中 `jobs=2` 最快，但真实语义 reconvergence 当前采用 `jobs=1`，因为它不会产生幂等 delete miss 干扰，且已经通过验证。
- `jobs=4` 开始退化，`jobs=12/16` 明显退化。
- 没有继续测试 `jobs=24/32`，因为 `jobs=16` 已明显更慢，继续加并发只会增加锁等待和 VM 风险。

### K3/K4 当前判断

本轮没有继续实现 K3/K4，因为 K2c mixed 已经成为当前最高可行性的 3083 实验路径，而 K3/K4 的风险仍没有被消除：

- K3 要改标准 `RTM_NEWROUTE/DELROUTE` 路径，让普通 rtnetlink route update 不拿 global RTNL。K2/K2b 已经证明这不是简单替换锁的问题：`fib_notifier.c` 有 `ASSERT_RTNL()`，跳过 notifier 后 `fib_create_info()` 又暴露 shared state 同步问题。
- K4 如果仍拿 global RTNL，只是把多个 namespace 的操作塞进更长的全局临界区，减少 netlink 往返但不能带来跨 namespace 并发，还可能放大长时间持锁。
- K4 如果基于 per-netns route lock，则必须先把 K2c 的 no-notify 原型补成语义完整版本：bulk notifier、rtnetlink event、nexthop/device lifetime、fib rules、VRF/lwtunnel、netns 删除、错误回滚都需要定义。

因此后续优先级建议：

1. 继续把 K2c mixed 固化为 controlled-experiment 方案，跑更多规模曲线。
2. 若要迈向真实内核语义，先做 K2d：保留 per-netns/bulk route update 的性能收益，同时补 bulk notifier 或延迟 notifier flush。
3. 等 K2d 能通过 smoke、synthetic、3083 real full/delta 后，再考虑 K3/K4。
