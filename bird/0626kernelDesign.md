# 2026-06-26 Kernel RTNL/FIB Design Notes

本文记录 K1、K2c、K3c 三种内核方案的设计、实验结果，以及“修改内核作为论文贡献是否站得住”的判断。结论先放前面：

- K1/K2c/K3c 的核心价值不是“发布一个要永久维护的 forked Linux kernel”，而是证明当前 Linux route update 缺少适合大规模 network namespace 仿真的 bulk/per-netns FIB update primitive。
- K3c 是目前最强的性能证据：3083 规模真实初始 full FIB 写入 `6,131,994` 条 route，6 月 22 日 writer wall `10.552s`、kernel write `7.867s`；7 月 1 日复跑 writer wall `10.226s`、kernel write `7.668s`。
- eBPF 不是 K1/K2c/K3c 的等价替代。eBPF 可以做观测，或做绕开 Linux FIB 的 BPF-map 数据面；但不能直接替代 `fib_table_insert()` / RTNL / `fib_info` 的内核写路径。
- 如果写 NSDI，应该把内核修改定位为受控原型和设计证据，并明确生产化路线：上游化 route bulk API、DKMS/kernel module 版本、或 eBPF-based dataplane 版本。

## 0. 方法简写和论文叙事摘要

### 0.1 方法简写

后续文档和论文中建议统一使用下面的简称，避免把不同语义的实验混在一起：

| 简写 | 名称 | Linux main FIB 语义 | 是否改内核 | 核心目的 | 3083 关键结果 |
|---|---|---|---|---|---|
| Stock | 容器内 BIRD kernel protocol 直接写 FIB | 是 | 否 | 原始问题基线 | 大量容器同时抢 global RTNL，数据面收敛慢或卡住 |
| F-full | host 侧集中 `setns + ip -batch` 全量写 FIB | 是 | 否 | 控制并发，避免每个容器各自抢锁 | 初始 full `520-579s`，`6.13M` routes |
| F-delta | patched BIRD event + host writer，只写变化路由 | 是 | 否，改 BIRD | 减少必须写入 FIB 的 route ops | 故障后 `561k` ops，FIB write `238.320s` |
| K1 | `BIRD_RT_BULK` global-RTNL bulk API | 是 | 是 | 批量提交 route ops，减少 netlink/lock 往返 | full `402.388s`，delta `63.639s` |
| K2c | K1 + per-netns route lock + no-notify + `fib_info_hash_mutex` | 是，受控 no-notify 原型 | 是 | 验证 global RTNL 是否是跨 netns 瓶颈 | full `71.162s`，delta `55.026s` |
| K3c | K2c + cached `fib_info` fast path | 是，受控 no-notify 原型 | 是 | 避免每条 route 重复 `fib_create_info()` | full `10.552s` / rerun `10.226s`，delta `6.791s` / rerun `6.386s` |
| BPF-DP | TC/BPF LPM map dataplane | 否 | 否，加载 BPF 程序 | 完全绕开 Linux FIB 写入和 RTNL | full `228.919s`，delta `118.626s` |

早期 D/E/G 和 table-252 原型主要用于验证“能否把 route event 变成外部可观测 delta”。它们不再作为主线方案，因为 table-252 仍然不是 Linux main FIB 真实收敛，且后续 F-delta、K1/K2c/K3c、BPF-DP 已经分别覆盖了“真实 FIB 写入”和“绕开 FIB 的数据面”两条路线。

文档中的 `restart_bird` 是实验 runner 的历史阶段名，含义是“在 router 容器内启动或重启 BIRD 进程并应用 patched BIRD/delta 配置”，不是重启 Docker 容器、VM 或宿主机。

### 0.2 发现问题后的尝试路线

我们最初观察到，SeedEMU 在 3000 级以上 router 容器同时收敛时，控制面 BIRD RIB 本身并不是主要瓶颈；真正拖慢数据面的是大量容器几乎同时向 Linux main FIB 执行 route add/replace/delete。虽然这些 router 位于不同 network namespace，它们最终仍会在宿主机内核中竞争 global RTNL 以及 FIB 共享结构，导致 route update 被串行化，并在高并发下出现长尾和系统 load 积压。

第一步，我们把无序的容器内写入改成 host 侧集中写入。F-full 通过 `setns + ip -batch` 控制 writer 并发，能让 3083 规模初始 full 写入完成，但 `6.13M` 条 route 仍需要 `520-579s`；故障后如果重新 full 写入，dump BIRD RIB 只需 `15.732s`，真正 FIB 写入却需要 `2025.107s`。这说明问题不是“拿不到 RIB”，而是 Linux route update primitive 本身不适合这种大规模 netns 批量更新。随后 F-delta 修改 BIRD 输出 selected-route change event，把故障后的写入量降到 `561k` 条，写入时间降到 `238.320s`。这证明减少 route ops 是有效的，但没有改变单条 route 的内核写入成本。

第二步，我们做了一组内核原型来拆解瓶颈。K1 增加受限 bulk route API，在一次 global RTNL 临界区里处理多条 route，证明 batching 本身有价值，但它仍不能跨 netns 并行。K2c 把受控 bulk path 的同步域降到 per-netns route lock，并显式保护 `fib_info` hash/list，3083 full 从 K1 的 `402.388s` 降到 `71.162s`，证明 global RTNL 对独立 netns route update 是结构性瓶颈。K3c 进一步发现 `fib_create_info()` 的 per-route 重复创建/查找成为新的主成本，于是对共享 nexthop/device/metric/table 的 route 复用 cached `fib_info`，将 3083 full writer 降到约 `10s`，delta writer 降到约 `6-7s`。这组 ablation 说明，性能提升不是简单调参，而是来自三个缺失 primitive：bulk route update、namespace-aware route serialization、batched/cached FIB object construction。

第三步，我们验证了不要求 Linux main FIB 语义时的替代路线。BPF-DP 把 selected routes 写入 BPF LPM map，由 TC 程序转发包，避免写 Linux FIB，因此完全绕开 RTNL。它在 3083 规模上能保持 packet-level reachability，优化后 full 为 `228.919s`、delta 为 `118.626s`。但 zero-change delta 仍需 `94.261s`，说明 BPF-DP 当前瓶颈转移为全量 `birdc show route primary` 扫描；同时它不满足 `ip route | wc` 这类 main-FIB 验证语义。因此，BPF-DP 是部署友好的仿真数据面路线，而 K1/K2c/K3c 是证明 Linux FIB 缺少高效 bulk/per-netns update primitive 的内核路线。

### 0.3 K1/K2c/K3c 改内核位置、改动规模和编译成本

| 方法 | 主要内核改动位置 | 改动规模 | 改动大吗 | 重新编译/打包耗时 |
|---|---|---:|---|---:|
| K1 | 新增 `include/uapi/linux/bird_rtbulk.h`、新增 `net/ipv4/fib_bulk.c`、修改 `net/ipv4/Kconfig` 和 `net/ipv4/Makefile` | patch `352` 行，约 `284` 行新增代码 | 中等。主要是新增私有 generic-netlink API 和新文件，不改标准 `RTM_NEWROUTE` 路径 | 首次/冷构建到 tar 包约 `385s`；后续类似增量通常几分钟 |
| K2c | 在 K1 基础上修改 `net/ipv4/fib_bulk.c` 引入 per-netns mutex；修改 `include/net/ip_fib.h`、`net/ipv4/fib_trie.c` 跳过 notifier；修改 `net/ipv4/fib_semantics.c` 加 `fib_info_hash_mutex` | K2/K2b/K2c 增量 patch 合计约 `353` 行；核心新增几十到百行级别 | 中等偏高。改到 FIB notifier 和 `fib_info` 共享结构，语义风险明显高于 K1 | K2/K2c 构建到 tar 包约 `127-220s`；K2c 实测约 `2-4min` |
| K3c | 在 `net/ipv4/fib_bulk.c` 增加 grouped apply；在 `net/ipv4/fib_trie.c` 增加 `fib_table_insert_info()`；在 `include/net/ip_fib.h` 增加声明 | 没有保留为单独 patch 文件；核心触碰 3 个内核文件，百行级别，并配套修改用户态 writer 分组 | 中等偏高。仍限定在私有 bulk path，但新增 FIB 内部 fast path，需审计 refcount/notifier 语义 | K3c 构建到 tar 包约 `199s`；K3b/K3c 实测约 `3-4min` |

编译口径：上述时间来自 `/data/lxl/kernel-work/build-birdrtbulk*.log` 和对应 `6.8.12-birdrtbulk*.tar.gz` 时间戳。在这台 30 CPU 实验宿主机上，完整 `make -j30/-j64 bzImage modules` 加 `modules_install` staging 和 tar 打包通常是 `2-7min`。部署到 VM 还需要 `rsync`、`update-initramfs`、`update-grub` 和一次 VM 重启；这部分通常再增加几分钟，取决于 VM I/O 和 SSH 状态。

论文中建议这样表述改动规模：

```text
Our kernel changes are deliberately narrow. K1 adds a private bulk route API
without changing the standard rtnetlink route path. K2c and K3c modify only the
controlled bulk path and the minimum IPv4 FIB helpers needed to quantify the
missing primitive. They are research prototypes, not a replacement for the
general Linux route-update semantics.
```

## 1. 背景

在 SeedEMU 大规模拓扑中，如果每个 router 容器里的 BIRD 同时通过 kernel protocol 写 Linux main FIB，会触发大量 `RTM_NEWROUTE` / `RTM_DELROUTE`。这些写入最终进入 Linux FIB/RTNL 路径，多个 network namespace 仍会竞争宿主机同一个 global RTNL 及 FIB 共享结构。

前期 stock/F 方法实验说明：

| 场景 | route ops | 时间 | 结论 |
|---|---:|---:|---|
| 3083 初始 full F writer | `6,131,994` | `520-579s` | host writer 能控并发，但仍受 FIB/RTNL 写入成本限制 |
| 3083 故障后 full 对照 | `6,097,653` | `2025.107s` | full dump 只 `15.732s`，真正慢在 FIB write |
| 3083 故障后 delta | `561,377` | `238.320s` | delta 的主要收益是减少 route ops |
| 100 router * 600k synthetic stock/F | `60,000,000` | `552.530s` | 不是 CPU 打满，而是 FIB/RTNL 路径成本 |

因此内核方向关注两件事：

1. 同一 netns 内减少每条 route 的 netlink/RTNL/fib 创建开销。
2. 不同 netns 之间尽量避免抢同一个 global RTNL。

## 2. K1：Global RTNL Bulk Route API

### 2.1 设计

K1 新增私有 generic-netlink family：`BIRD_RT_BULK`。用户态 writer 进入目标容器 netns 后，一次提交一组 IPv4 route ops，内核在一次 global RTNL 临界区里循环执行 route add/replace/delete。

```text
kernel_bulk_*_writer.py
  -> setns(target router netns)
  -> generic netlink BIRD_RT_BULK/APPLY_V4
  -> net/ipv4/fib_bulk.c
  -> rtnl_lock()
  -> loop fib_table_insert/delete()
  -> rtnl_unlock()
```

K1 解决：

- 每条 route 一个 netlink message。
- 每条 route 一个 ACK/解析/调度开销。
- 每条 route 一次 lock/unlock。

K1 不解决：

- 不同 netns 之间仍共享 global RTNL。
- 高并发 writer 仍会排队抢 global RTNL。

### 2.2 实现位置

| 文件 | 作用 |
|---|---|
| `bird/kernel/patches/0001-bird-rtbulk-global-rtnl.patch` | 新增 UAPI 和 `net/ipv4/fib_bulk.c` |
| `bird/kernel/tools/kernel_bulk_writer.py` | synthetic bulk writer |
| `bird/kernel/tools/kernel_bulk_birdc_writer.py` | full `birdc show route primary` -> bulk |
| `bird/kernel/tools/kernel_bulk_command_writer.py` | delta command files -> bulk |
| `bird/run_3083_k1_delta_reconvergence.sh` | K1 3083 full + delta runner |

### 2.3 实验结果

| 实验 | route ops | 参数 | 时间 | 结果 |
|---|---:|---|---:|---|
| synthetic `100x600k_j4` | `60,000,000` | jobs=4,batch=1024 | `585.122s` | success |
| 3083 initial full | `6,131,994` | jobs=1,batch=256 | writer `402.388s` | success |
| 3083 reconvergence delta | `198,804` | jobs=1,batch=256 | writer `63.639s` | success |

K1 的经验：

- K1 是可靠的 global-RTNL bulk baseline。
- K1 能减少单 netns 内重复 netlink/lock 成本，但不提供跨 netns 并行。
- 3083 jobs=8 曾导致 VM load 长时间不可恢复；K1 需要保守并发。

## 3. K2c：Per-netns Route Lock + No-notify Upper Bound

### 3.1 设计演进

K2 的目标是把 K1 的 global RTNL 替换成每个 network namespace 一个 route mutex：

```text
BIRD_RT_BULK/APPLY_V4
  -> mutex_lock(per-netns route_lock)
  -> loop fib_table_insert/delete()
  -> mutex_unlock(per-netns route_lock)
```

直接这样做会失败，因为 `fib_table_insert()` / `fib_table_delete()` 的 notifier 路径有 `ASSERT_RTNL()`。K2b 为了测性能上限，加入 `fc_skip_notify`，跳过 route notifier 和 rtnetlink multicast。K2b 又暴露出 `fib_create_info()` / `fib_info` hash/list 共享结构仍隐式依赖 RTNL。K2c 因此增加 `fib_info_hash_mutex`，显式保护跨 netns 共享的 fib_info hash/list。

K2c 路径：

```text
user writer
  -> setns(target router netns)
  -> BIRD_RT_BULK/APPLY_V4
kernel
  -> per-netns route_lock
  -> skip route notifier/rtnetlink multicast
  -> fib_info_hash_mutex for shared fib_info structures
  -> fib_table_insert/delete
```

### 3.2 实现位置

| 文件 | 作用 |
|---|---|
| `bird/kernel/patches/0002-bird-rtbulk-pernetns-lock.patch` | per-netns route mutex |
| `bird/kernel/patches/0003-bird-rtbulk-pernetns-no-notify-upper-bound.patch` | `fc_skip_notify` |
| `bird/kernel/patches/0004-bird-rtbulk-k2c-fib-info-hash-mutex.patch` | `fib_info_hash_mutex` |
| `bird/run_3083_k2c_delta_optimized.sh` | K2c mixed-batch 3083 runner |

### 3.3 实验结果

| 实验 | route ops | 参数 | 时间 | 结果 |
|---|---:|---|---:|---|
| synthetic `100x600k_j4` | `60,000,000` | jobs=4,batch=1024 | `132.815s` | success |
| 3083 initial full | `6,131,994` | jobs=8 | writer `71.162s` | success |
| 3083 reconvergence delta | `198,869` | mixed batch, delta jobs=1 | writer `55.026s` | success |

真实 3083 K2c mixed 关键阶段：

| 阶段 | 时间 |
|---|---:|
| build | `205.432s` |
| docker compose up | `3019.109s` |
| restart_bird | `1230.731s` |
| initial full writer | `71.162s` |
| reconvergence total | `157.028s` |
| stop 10 victims | `28.112s` |
| prepare delta offsets | `32.334s` |
| delta dump/collapse | `8.596s` |
| delta writer | `55.026s` |

K2c 的经验：

- K2c 明显证明了跨 netns global RTNL 是核心瓶颈之一。
- mixed replace/delete batch 对 sparse delta 非常重要。
- K2c 高并发 delta 会放大 `fib_info_hash_mutex` 等共享结构等待，因此真实 delta 用 jobs=1 更稳。
- K2c 仍是 no-notify upper-bound，不是生产语义完整方案。

## 4. K3c：Cached fib_info Fast Path

### 4.1 设计

K3b 先验证了 multi-attr bulk request，但收益有限，说明主要瓶颈不再是 generic-netlink request 次数，而是每条 route 都要重复执行 `fib_create_info()`。

K3c 的核心是：同一批 route 如果共享 nexthop/device/metric/proto/table，只是 prefix 不同，那么只创建一次 `struct fib_info`，然后对多条 prefix 复用它。

```text
bird_rtbulk_apply_many_v4()
  -> 检查一个 OPS attr 内所有 route 是否共享 fib_info key
  -> fib_create_info(base_cfg) 一次
  -> 对每条 prefix 调 fib_table_insert_info(..., cached_fi)
  -> 每条 alias 持有 cached fib_info ref
```

用户态 writer 配合：

- full writer 默认 `--group-fib-info`。
- delta command writer 默认 `--group-fib-info`。
- 每个 attr 内尽量放共享 nexthop/device/metric/table 的 routes。

### 4.2 实现位置

| 文件 | 作用 |
|---|---|
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_bulk.c` | `bird_rtbulk_apply_many_v4()` |
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/net/ipv4/fib_trie.c` | `fib_table_insert_info()` |
| `/data/lxl/kernel-work/linux-source-6.8.0-124.124/include/net/ip_fib.h` | 声明 `fib_table_insert_info()` |
| `bird/kernel/tools/kernel_bulk_birdc_writer.py` | full writer 分组 |
| `bird/kernel/tools/kernel_bulk_command_writer.py` | delta writer 分组 |
| `bird/run_3083_k3c_delta_experiment.sh` | K3c 3083 runner |

### 4.3 Synthetic 实验结果

| 实验 | route ops | 参数 | 时间 | 吞吐 |
|---|---:|---|---:|---:|
| smoke | `1,000` | jobs=1 | `0.00165s` | - |
| `10x600k_j4_b16000` | `6,000,000` | jobs=4,batch=16000 | `7.025s` | `854k routes/s` |
| `100x600k_j4_b16000` | `60,000,000` | jobs=4,batch=16000 | `56.095s` | `1.070M routes/s` |
| `100x600k_j8_b16000` | `60,000,000` | jobs=8,batch=16000 | `30.567s` | `1.963M routes/s` |
| `100x600k_j16_b16000` | `60,000,000` | jobs=16,batch=16000 | `18.098s` | `3.315M routes/s` |
| `100x600k_j24_b16000` | `60,000,000` | jobs=24,batch=16000 | `13.608s` | `4.409M routes/s` |
| `100x600k_j32_b16000` | `60,000,000` | jobs=32,batch=16000 | `11.621s` | `5.163M routes/s` |

Synthetic 对比：

| 方法 | case | 时间 | 说明 |
|---|---|---:|---|
| K1 | `100x600k_j4` | `585.122s` | global RTNL bulk |
| K2c | `100x600k_j4` | `132.815s` | per-netns no-notify + fib_info hash mutex |
| K3c | `100x600k_j4` | `56.095s` | cached fib_info |
| K3c | `100x600k_j32` | `11.621s` | 当前 synthetic 最优 |

### 4.4 真实 3083 实验结果

运行目录：

```text
bird/runs/k3c-delta-reconv-3083_20260622_163400
```

| 阶段 | 时间 / 结果 |
|---|---:|
| kernel | `6.8.12-birdrtbulk-k3c` |
| build | `178.328s` |
| docker compose up | `3078.891s` |
| serial restart_bird | `931.366s` |
| wait load after start-bird | `62.016s` |
| initial full writer wall | `10.552s` |
| initial full dump | `2.683s` |
| initial full kernel write | `7.867s` |
| initial route ops | `6,131,994 / 6,131,994` |
| post-full FIB verify | `312/312` AS samples passed |
| enable delta | `37s` |
| reconvergence total | `74.980s` |
| stop 10 brdnodes | `27.136s` |
| wait load/RIB | `0.002s` |
| prepare offset marks | `31.638s` |
| delta dump/collapse | `6.589s` |
| delta writer | `6.791s` |
| delta raw events | `353,663` |
| delta collapsed ops | `198,860` |
| delta result | `198,860 / 198,860`, `0` fail |

K3c delta writer replay sweep：

| jobs | replay writer time |
|---:|---:|
| 4 | `7.583s` |
| 8 | `6.555s` |
| 16 | `6.061s` |
| 24 | `6.182s` |
| 32 | `6.229s` |

K3c 的经验：

- 初始 full 的 kernel write `7.867s`，说明 K3c 对真实 BIRD route 也有效。
- delta writer `6.791s` 并不比 full 慢；此前“delta 看起来慢”主要来自固定流程开销：遍历 2747 个 netns 读 event file、生成 commands、写 offset mark、重复 install/stop-agent。
- K3c 当前是最高性能 controlled experiment，但仍继承 K2c 的 no-notify 限制。

## 5. 安全性与维护成本

### 5.1 直接 fork kernel 的风险

如果论文把方案表述成“用户必须长期运行我们 fork 的 Linux kernel”，reviewer 很可能会攻击：

- 维护成本高：每次 kernel 安全更新都要 rebase、重新编译、重新测试。
- 安全风险高：绕过 upstream review，可能破坏 route notifier、rtnetlink multicast、device lifecycle、netns teardown 等隐含语义。
- 可部署性弱：云环境和生产机器通常不允许随意替换 host kernel。
- 功能覆盖窄：当前只覆盖 IPv4 main table、简单 nexthop、`proto bird metric 232` 的受控实验 route。

这些攻击是成立的。我们不能把 K2c/K3c 写成 production-ready kernel。

### 5.2 当前原型的安全边界

我们可以把安全边界说清楚：

- 运行环境是隔离 VM，用于大规模网络仿真实验，不是多租户生产 host。
- 私有 `BIRD_RT_BULK` generic-netlink family 需要管理员能力；它不是暴露给 unprivileged tenant 的接口。
- writer 只对受控 container netns 写 main table 中 `proto bird metric 232` 路由。
- 实验不依赖 route multicast/notifier 给其它生产组件同步状态。
- 实验结束会 destroy/start VM 并 reset Docker，避免残留状态污染下一轮。

这能说明“实验安全”，但不能说明“生产安全”。论文里必须区分这两者。

### 5.3 为什么 per-netns RTNL/route lock 本身是有意义的贡献

这里需要把论证说得更硬一些：我们的目标不是“为了快而随便改锁”，而是指出 Linux 当前 route update 的同步粒度和大规模 netns 仿真负载之间存在结构性错配。

Linux network namespace 已经把大部分路由状态隔离在 `struct net` 下面：每个 namespace 有自己的 FIB table、rule、device view、邻居表和协议实例。对真实机器来说，一个 host 上通常不会同时有几千个 router namespace 在短时间内批量写 FIB；因此 global RTNL 的简单性是合理的。但是在 SeedEMU 这类大规模 Internet emulation 中，1000-5000 个容器 router 的 route update 大多发生在不同 netns 的不同 FIB table 上。此时把这些互不相干的 FIB mutation 全部串到同一个 global RTNL 上，是过度保守的同步。

所以 per-netns RTNL/route lock 的贡献点是：

- **隔离域和同步域重新对齐**：既然 netns 是路由状态的隔离单位，普通 route update 的锁粒度也应该至少能降到 netns 级别，而不是整个 host 级别。
- **把无关临界区拆开**：`ns A` 的 IPv4 route insert 与 `ns B` 的 IPv4 route insert 不应该互相等待，除非它们触碰共享对象，例如全局 `fib_info` hash、netdevice lifecycle、notifier/multicast 等。
- **保留复杂路径的保守性**：我们不主张马上把整个 RTNL 全量拆掉。更合理的设计是：simple route bulk path 使用 per-netns route lock；复杂 route、netdevice 变更、VRF/lwtunnel/nexthop object、需要完整 notifier 的路径继续走 global RTNL 或更强的同步。
- **给上游 API 留出形状**：最终可以抽象成 `RTM_NEWROUTE_BULK`、`RTM_DELROUTE_BULK` 或一个受限 generic-netlink route bulk API。API 明确声明只覆盖受限 route class；不满足条件时 fallback 到现有 rtnetlink path。
- **不是一次性 hack**：K1/K2c/K3c 从 global bulk、per-netns serialization 到 cached `fib_info`，逐层量化瓶颈。这说明我们发现的是一个可泛化的 kernel primitive，而不是只针对某个脚本的局部优化。

从实验结果看，这个 primitive 的意义很强：K1 证明 bulk API 可以把大量 netlink/lock 往返合并；K2c 证明跨 netns global RTNL 是核心瓶颈；K3c 进一步证明 FIB 内部 `fib_info` 创建/复用也需要批处理语义。3083 规模下，full FIB writer 从 K1 的 `402.388s` 降到 K2c 的 `71.162s`，再到 K3c 的约 `10s`。这个量级不是普通工程调参，而是 OS 同步和 API 粒度错误带来的结构性差异。

更精确的论文表述应是：

```text
We do not split the entire RTNL subsystem. We introduce a restricted per-netns
route-update serialization domain for bulk updates whose objects are scoped to
one network namespace. Existing global RTNL semantics remain the fallback for
complex routes and device lifecycle operations.
```

这句话能避免 reviewer 把我们攻击成“你们粗暴拆了 RTNL，破坏 Linux 网络栈语义”。我们的真实主张是：route update 需要一个 namespace-aware bulk primitive。

### 5.4 更合理的论文表述

建议在 NSDI 论文中这样表述：

1. **系统问题贡献**：大规模 Internet emulation 中，成千上万个 network namespace 同时写 Linux FIB，会触发 global RTNL 和 FIB 共享结构瓶颈；这个瓶颈不是 CPU，不是 BIRD RIB dump，而是内核 route update primitive 不适配该场景。
2. **用户态贡献**：BIRD delta event + host writer 把无序并发写者变成受控 writer，并把故障后写入量从 full routes 降到 collapsed delta routes。
3. **内核原型贡献**：K1/K2c/K3c 是一组逐步深入的内核原型，用来回答“需要什么样的 kernel primitive 才能支撑大规模仿真”：bulk route API、per-netns route serialization、cached nexthop/fib_info。
4. **可扩展性贡献**：per-netns route lock 不是只服务 SeedEMU。任何把大量独立 network namespace 当作 router、tenant、service mesh node、testbed node 的系统，都会遇到相同 global RTNL serialization。我们的 primitive 可以作为这些系统的共同底座。
5. **部署路线贡献**：生产化不要求 fork kernel；可以把原型收敛为一个 upstreamable route bulk API，或实现为可选 kernel module/DKMS；若不要求 Linux FIB 语义，则可以走 eBPF map-based dataplane。

不要说：

```text
Our system requires replacing Linux with a forked kernel.
```

应说：

```text
We use a small Linux kernel prototype to quantify the missing kernel primitive.
The prototype is not required for correctness of the emulator; it exposes the
performance headroom and motivates an upstreamable bulk/per-netns FIB update API.
```

## 6. 相关顶会工作如何处理“改内核”

### 6.1 改内核/内核模块在顶会中不是禁忌

顶会系统论文接受过多种 kernel-level 设计，只要它们回答的是清晰的系统问题，并且给出足够的正确性、隔离、可部署性讨论。很多系统论文并不是“拿现有 Linux 完全不动”，而是在发现已有 OS abstraction 不适合新负载后，提出新的 kernel primitive、kernel module、kernel bypass boundary，甚至重构 kernel/user 边界。

| 论文 | 会议 | 做法 | 对我们的启发 |
|---|---|---|---|
| [Slim: OS Kernel Support for a Low-Overhead Container Overlay Network](https://www.usenix.org/conference/nsdi19/presentation/zhuo) | NSDI 2019 | 为 container overlay network 增加 OS kernel support | 和我们最接近：目标也是容器网络性能，贡献不是“应用层调参”，而是指出现有 kernel datapath abstraction 不合适 |
| [MegaPipe: A New Programming Interface for Scalable Network I/O](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/han) | OSDI 2012 | 提出新的 scalable network I/O API，核心包括 batching、partitioning、lightweight socket | 我们的 route bulk API 与它的思想一致：当 syscall/API 粒度错误时，需要新的 batched kernel interface |
| [Dune: Safe User-level Access to Privileged CPU Features](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/belay) | OSDI 2012 | 用 loadable kernel module 让进程安全访问特权 CPU feature | 顶会接受“小内核模块 + 用户态库”作为新 primitive；这支持我们未来把 K3d 收敛成 module/DKMS 的部署路径 |
| [Caladan: Mitigating Interference at Microsecond Timescales](https://www.usenix.org/conference/osdi20/presentation/fried) | OSDI 2020 | 引入 Linux kernel module KSCHED，加速调度和监控操作 | 当标准 Linux 调度/同步路径太慢时，内核模块化 fast path 是合理系统设计 |
| [ghOSt: Fast & Flexible User-Space Delegation of Linux Scheduling](https://research.google/pubs/ghost-fast-and-flexible-user-space-delegation-of-linux-scheduling/) | SOSP 2021 | 改造 Linux scheduling framework，将 policy 委托到 user space | 复杂 policy 放用户态，内核只提供安全、高效、可验证的机制；这正是我们的长期方向：BIRD/delta policy 在用户态，内核提供 bulk/per-netns FIB mechanism |
| [Arrakis: The Operating System is the Control Plane](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/peter_simon) | OSDI 2014 | 重新划分 kernel control plane 与 application data plane 边界 | 顶会接受对 OS/kernel 边界的结构性重思考。我们的工作规模更小：只重划 route update 的同步边界 |

因此，“改内核”本身不是论文硬伤。真正的判断标准是：

1. **是否存在必须进入内核才能解决的瓶颈**：RTNL/FIB update 的锁和 `fib_info` 生命周期都在内核，用户态只能绕开或排队，不能改变同步域。
2. **是否有最小化改动边界**：我们不是全量替换 Linux FIB；只是增加受限 route bulk fast path，并为 simple route 提供 per-netns serialization。
3. **是否量化了 primitive 的收益**：K1/K2c/K3c 形成 ablation study，能说明每个 primitive 分别解决什么瓶颈。
4. **是否有生产化路径**：上游化受限 bulk API、DKMS/module、或 eBPF dataplane 替代，而不是要求用户永远运行一个不可维护的 fork。
5. **是否诚实承认语义限制**：当前 no-notify fast path 是 controlled prototype；后续 K3d 要补 batched/delayed notifier 和复杂 route fallback。

换句话说，reviewer 可以攻击“语义不完整”或“部署故事弱”，但不应该能轻易攻击“改内核没有价值”。我们的反驳是：如果问题本质是 kernel synchronization primitive 缺失，那么内核原型正是系统论文应该给出的证据。

### 6.1.1 与每次重编内核相关的论文说法

论文中可以主动承认：研究原型需要重新编译内核，这是评估 missing primitive 的方式，不是最终交付形态。更好的表述是：

```text
The prototype is implemented as a Linux kernel patch to expose and evaluate the
missing route-update primitive. The design can be upstreamed as a restricted
bulk route API, packaged as a kernel module where feasible, or bypassed by a
BPF-map dataplane when Linux-FIB semantics are unnecessary.
```

这会把“重新编译内核”的弱点转成“受控原型用于量化 OS primitive”的研究方法。顶会并不排斥这种方法；Slim、MegaPipe、Dune、Caladan、ghOSt 都是在不同层面说明：当现有 OS abstraction 阻碍新负载时，设计新的 kernel mechanism 是可接受的贡献。关键是我们不能把 prototype 说成 deployment story 的全部。

### 6.2 eBPF 是强部署故事，但不是万能替代

近年 NSDI/OSDI 明显偏好 eBPF 的安全可扩展性：

| 论文 | 会议 | 做法 | 对我们的启发 |
|---|---|---|---|
| [Electrode: Accelerating Distributed Protocols with eBPF](https://www.usenix.org/conference/nsdi23/presentation/zhou) | NSDI 2023 | 用 eBPF 将分布式协议 frequent path 下沉到内核，避免 user/kernel crossing | 若逻辑可以表达为 packet path 上的 bounded program，eBPF 是非常强的部署故事 |
| [DINT: Fast In-Kernel Distributed Transactions with eBPF](https://www.usenix.org/conference/nsdi24/presentation/zhou-yang) | NSDI 2024 | 用 eBPF 在内核中执行交易 frequent path，无 kernel modification/module | reviewer 会喜欢“无需改内核”的可部署性，但这要求问题能适配 eBPF hook/helper |
| [XRP: In-Kernel Storage Functions with eBPF](https://www.usenix.org/conference/osdi22/presentation/zhong) | OSDI 2022 | 在 storage path 中用 eBPF 安全执行应用定义函数 | eBPF 适合把应用逻辑下沉到已有 hook，而不是任意修改内核内部同步语义 |
| [XDP: Fast Programmable Packet Processing in the OS Kernel](https://dl.acm.org/doi/10.1145/3281411.3281443) | CoNEXT 2018 | eBPF/XDP 在 driver context 中做高速 packet processing | eBPF 很适合实现替代数据面，例如 BPF LPM map forwarding |

但是，eBPF 当前不能等价替代 K1/K2c/K3c：

- BPF helper/kfunc 不是“任意内核函数调用”。Linux BPF 文档说明 helper ABI 稳定，但 tracing 程序读取内核内部结构会受 kernel 变化影响；kfunc 也没有硬稳定保证。参考 [BPF Design Q&A](https://docs.kernel.org/bpf/bpf_design_QA.html) 和 [BPF kfunc docs](https://docs.kernel.org/bpf/kfuncs.html)。
- `bpf_fib_lookup()` 是 lookup helper，不是 insert/delete Linux FIB。参考 [`bpf_fib_lookup`](https://docs.ebpf.io/linux/helper-function/bpf_fib_lookup/)。
- `bpf_redirect_neigh()` / `bpf_redirect()` 是 packet redirect helper，不会更新 Linux main FIB。参考 [`bpf_redirect_neigh`](https://docs.ebpf.io/linux/helper-function/bpf_redirect_neigh/)。
- K2c/K3c 修改的是 route update 的锁语义、notifier 语义、`fib_info` 创建/复用语义；这些不在现有 eBPF hook/helper 能安全表达的范围内。

所以 eBPF 可作为另一条路线：

```text
BIRD selected-route delta
  -> host/controller
  -> update BPF LPM trie / nexthop maps
  -> TC/XDP/LWT program does forwarding
  -> no Linux FIB write, no RTNL contention
```

这条路线适合“仿真数据面转发”，但它不再满足“Linux main FIB 中真实存在这些 routes”。验证方式也要从 `ip route | wc` 改成 packet-level reachability 和 BPF map consistency。

## 7. 面向 NSDI 的推荐调整

### 7.1 不把 K3c 当最终系统，而当“kernel primitive study”

论文贡献可以这样组织：

1. **Characterization**：证明大规模 container router 的数据面收敛瓶颈来自 Linux RTNL/FIB update，而非控制面 RIB 或 CPU。
2. **User-space design**：BIRD delta event + host writer，减少 route ops 并控制写入并发。
3. **Kernel primitive study**：K1/K2c/K3c 逐步回答三个问题：
   - bulk route API 能减少多少单 netns 开销？
   - per-netns route serialization 能带来多少跨 netns 并行？
   - cached `fib_info` 能减少多少 per-route FIB 创建成本？
4. **Deployment discussion**：生产部署可选三条路径：
   - upstreamable `RTM_NEWROUTE_BULK` / generic netlink route bulk API；
   - DKMS/kernel module 形式，减少 fork kernel 维护成本；
   - eBPF map dataplane，完全避免写 Linux FIB。

这样 reviewer 不容易把贡献攻击成“只是 fork 了 kernel”。我们的核心是发现并量化缺失 primitive，并提供可复现实验。

### 7.2 必须诚实承认的限制

论文中应明确列出：

- K2c/K3c 当前 no-notify，因此不适合直接上游为通用 route path。
- 当前只覆盖 IPv4/simple nexthop/main table/controlled metric。
- 生产语义需要补齐 delayed notifier、rtnetlink multicast、device lifetime、netns teardown、IPv6、ECMP/nexthop object、VRF/lwtunnel。
- 实验运行在隔离 VM，不面向多租户 host。

### 7.3 最推荐的下一步

如果要增强论文说服力，我建议做两个补充实现：

1. **K3d / K2d semantic patch**：在 K3c fast path 后做 batched/delayed notifier，而不是彻底 no-notify。哪怕只覆盖 IPv4 simple route，也能显著缓解“语义不完整”的攻击。
2. **BPF dataplane prototype**：实现最小 TC/LWT BPF LPM map 转发，只支持 IPv4/simple nexthop。它不替代 K3c，但能证明“如果不要求 Linux FIB 语义，我们可以不改内核、不抢 RTNL”。

最终论文叙事可以是：

```text
User-space delta reduces the number of route updates.
K3c quantifies the performance headroom of a missing Linux FIB bulk primitive.
An eBPF dataplane is a deployable alternative when Linux-FIB semantics are unnecessary.
```

这比单独说“我们改了 Linux kernel”更容易通过 review。

## 8. 2026-07-01 K3c 3083 复现实验与安全影响探针

### 8.1 实验环境与命令

本轮复现实验运行在 VM `bird-rtnl-dev`，内核版本为 `6.8.12-birdrtbulk-k3c`。实验前对 VM 做了断电式恢复：`virsh destroy/start`，随后清理 Docker/containerd 残留；由于 Docker registry 拉取 base image 卡住，最后复用 host 上已有的 `handsonsecurity/seedemu-base:2.0`、`handsonsecurity/seedemu-router:2.0`、`ubuntu:24.04`，通过 `docker save | docker load` 导入 VM。

核心命令：

```bash
RUN_NAME=k3c-rerun-3083_$(date +%Y%m%d_%H%M%S) \
RESET_VM=0 \
PROJECT_NAME=topology-3083 \
OUTPUT_DIR=/data/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output \
LOAD_THRESHOLD=50 \
MAX_WAIT_SECONDS=7200 \
POLL_INTERVAL=30 \
RESTART_JOBS=1 \
RESTART_DELAY=0.1 \
BUILD_MODE=delta-fast \
INITIAL_WRITE_JOBS=16 \
K1_DUMP_JOBS=16 \
K1_BATCH=16000 \
K1_COMMAND_TIMEOUT=900 \
K1_MULTIPATH_POLICY=first \
DELTA_JOBS=16 \
DOWN_COUNT=10 \
DELTA_WRITE_JOBS=8 \
OFFSET_JOBS=16 \
K1_DELTA_BATCH=16000 \
K1_OPS_PER_ATTR=2000 \
RUN_FULL_COMPARISON=0 \
bird/run_3083_k3c_delta_experiment.sh
```

本地结果目录：

```text
/data/lxl/seed-emulator/topology/bird/runs/k3c-rerun-3083_20260701_114246
```

### 8.2 3083 全流程结果

| 阶段 | 结果 | 时间/规模 |
|---|---:|---:|
| total | success | `4190.193s` |
| sync | success | `12.836s` |
| build | success | `191.845s` |
| up | success | `2997.117s` |
| restart_bird | success | `919.520s`，串行，间隔 `0.1s` |
| wait_rib_stable | success | `42.620s`，以 load1 `< 50` 为门控 |
| validate_bird | success | `3.085s` |
| initial K3c full FIB stage | success | `20.252s` |
| initial FIB verify | success | `312/312` AS representatives passed，`112.834s` |

initial full FIB writer 的细项：

| 指标 | 数值 |
|---|---:|
| brdnode containers | `2757` |
| route ops | `6,131,994 / 6,131,994` |
| failed ops | `0` |
| writer wall | `10.226s` |
| birdc dump wall | `2.554s` |
| kernel write wall | `7.668s` |
| kernel apply sum | `66.105s` |
| batches/groups | `114,169 / 114,169` |

reconvergence delta 的细项：

| 阶段/指标 | 数值 |
|---|---:|
| total reconvergence | `41.922s` |
| stop 10 victims | `27.050s` |
| wait load/RIB | `0.002s` |
| delta dump/collapse | `5.507s` |
| delta write stage | `13.649s` |
| K3c delta writer wall | `6.386s` |
| raw route events | `359,102` |
| collapsed route ops | `198,832` |
| successful ops | `198,832` |
| failed ops | `0` |
| delta kernel apply sum | `3.515s` |

结论：这轮 3083 复现实验证明 K3c 仍然稳定复现了前面观察到的数量级收益。initial full 的真实 kernel write 只有 `7.668s`，故障后的 delta kernel writer wall 为 `6.386s`，且没有 route op 失败。

### 8.3 K3c 安全/兼容性探针

新增 host 侧探针脚本：

```text
bird/kernel/tools/k3c_safety_probe_vm.py
```

探针命令：

```bash
python3 bird/kernel/tools/k3c_safety_probe_vm.py \
  --vm 192.168.122.60 \
  --vm-user lxl \
  --project-name topology-3083 \
  --run-dir /data/lxl/seed-emulator/topology/bird/runs/k3c-rerun-3083_20260701_114246 \
  --routes 10000
```

结果目录：

```text
/data/lxl/seed-emulator/topology/bird/runs/k3c-rerun-3083_20260701_114246/k3c_safety_probe
```

探针总结果为 success。检查项如下：

| 检查项 | 结果 | 说明 |
|---|---:|---|
| VM identity | pass | `6.8.12-birdrtbulk-k3c` |
| standard rtnetlink temp netns | pass | `ip route replace/get/del` 正常，root netns 无泄漏 |
| K3c bulk temp netns | pass | 临时 netns 写入 `10,000` 条 blackhole route，再删除 `10,000` 条，删除后为 `0`，root netns 无泄漏 |
| unprivileged caller | pass | `nobody` 调用 bulk API 返回 `EPERM` |
| Docker netns lifecycle | pass | 临时容器创建、inspect、remove 后无残留 |
| severe kernel log scan | pass | 探针窗口内无 `BUG/Oops/panic/WARNING/lockdep/KASAN/refcount/use-after-free/general protection/rtnl_mutex/bird_rtbulk/fib_create_info` 命中 |
| route monitor notification | pass with finding | 标准 route 操作产生 `2` 条 `ip monitor route` 事件；K3c bulk 产生 `0` 条 |
| lifecycle warnings | recorded | 观察到 `6` 条 `unregister_netdevice: waiting ... Usage count = ...` |

K3c bulk 临时 netns 的性能也说明 fast path 本身没有异常退化：

| 操作 | routes | wall | kernel apply |
|---|---:|---:|---:|
| replace blackhole | `10,000` | `0.089s` | `0.076s` |
| delete blackhole | `10,000` | `0.155s` | `0.144s` |

### 8.4 已证明的影响边界

这轮 probe 支持以下结论：

- **标准 rtnetlink 路径没有被破坏**：普通 `ip route replace/get/del` 在临时 netns 中正常工作。
- **K3c bulk 的 netns 隔离成立**：临时 netns 中写入和删除的 route 没有泄漏到 root netns。
- **权限边界成立**：非特权用户无法调用 bulk route API，返回 `Operation not permitted`。
- **Docker 基本生命周期成立**：在 3083 topology 运行期间，仍可创建和删除临时 Docker 容器。
- **严重内核日志未出现**：探针窗口内没有 Oops、panic、lockdep、KASAN、refcount、use-after-free 等强错误信号。

### 8.5 仍需承认的语义/安全风险

这轮 probe 也明确暴露了 K3c 当前不能直接声称 production-ready：

- **no-notify 是真实功能差异**：K3c bulk 不触发 `ip monitor route` 可见的 route 事件。依赖 rtnetlink multicast、route notifier、`ip monitor route`、FRR/daemon route-watch 的组件不会看到 bulk fast path 的 route changes。
- **netdevice teardown 需要继续审计**：实验现场出现 `unregister_netdevice: waiting ... Usage count = ...`。这不是 Oops/panic，但说明大规模 Docker/netns teardown 下存在引用释放延迟；K3d/上游化设计必须审计 device lifetime、dst/fib_info 引用和 notifier 顺序。
- **当前覆盖面仍受限**：probe 只验证 IPv4/simple route/main table/blackhole route/controlled BIRD route；尚未覆盖 IPv6、ECMP、nexthop object、VRF、lwtunnel、policy rule、route exception 等复杂路径。
- **安全证明只适用于受控 VM**：当前实验环境是单用户 VM，不是多租户 production host。论文中应表述为“controlled kernel prototype”，不能声称已经满足通用 Linux networking 安全语义。

因此，K3c 的正确定位仍然是：它证明了 missing kernel primitive 的性能上限和可行性；要成为可上游或可生产部署的方案，需要补齐 batched/delayed notifier、复杂 route fallback、device/netns 生命周期审计和更完整的权限/回归测试。

## 9. 2026-07-05 K2b Unsafe Per-netns RTNL/Route Lock 实验

### 9.1 实验目的

这轮实验的目标是更接近“每个 network namespace 独立拥有 route update lock”的最初设想，观察如果不再让不同 netns 的 bulk route update 争抢同一个 global RTNL，会出现哪些性能收益和语义风险。

这里使用的内核是：

```text
6.8.12-birdrtbulk-k2b
```

它不是把 Linux 全部 RTNL 子系统完整拆成 per-netns lock，而是在我们自定义的 BIRD bulk route fast path 上使用 per-netns route lock，并跳过 route notifier/rtnetlink multicast。它对应的 patch 栈是：

```text
0001-bird-rtbulk-global-rtnl.patch
0002-bird-rtbulk-pernetns-lock.patch
0003-bird-rtbulk-pernetns-no-notify-upper-bound.patch
```

它刻意不包含 K2c 的：

```text
0004-bird-rtbulk-k2c-fib-info-hash-mutex.patch
```

因此 K2b 是一个 unsafe upper-bound：它用于回答“如果 per-netns 并行大幅放开，性能上限和明显语义问题是什么”，不能当作 production-ready 方案。

### 9.2 Smoke 实验

本地结果目录：

```text
/data/lxl/seed-emulator/topology/bird/runs/k2b-unsafe_20260705_113031
```

Synthetic 结果：

| case | jobs | route type | routes | wall | throughput | failed |
|---|---:|---|---:|---:|---:|---:|
| 100 router x 600k | 4 | dev lo | `60,000,000` | `132.614s` | `452,441 routes/s` | `0` |
| 100 router x 600k | 32 | dev lo | `60,000,000` | `146.761s` | `408,828 routes/s` | `0` |
| 100 router x 600k | 32 | blackhole | `60,000,000` | `159.883s` | `375,273 routes/s` | `0` |

这些 smoke run 没有在 dmesg 中命中 `BUG/Oops/panic/WARNING/refcount/use-after-free/general protection/fib_create_info/fib_info/list corruption` 等关键词。但这只能说明这些输入没有触发确定性崩溃，不能证明 K2b 安全。

No-notify 语义差异已经被直接验证：

| 操作 | `ip monitor route` 可见事件 |
|---|---:|
| 标准 `ip route add blackhole 198.18.0.1/32` | `1` |
| K2b bulk 写入 10 条 blackhole route | `0` |

这说明 K2b fast path 绕过 notifier/rtnetlink multicast 后，依赖 route event 的用户态组件无法感知 route changes。这是“为什么 Linux 不能简单把这条路径无语义补偿地拆开”的直接证据。

### 9.3 3083 规模真实 SeedEMU 实验

实验命令使用已有 3083 output，内核为 `6.8.12-birdrtbulk-k2b`，BIRD 启动串行、间隔 `0.1s`，初始 full FIB 写入使用 bulk writer `jobs=32`，reconvergence delta 写入使用 `jobs=16`。

本地结果目录：

```text
/data/lxl/seed-emulator/topology/bird/runs/k2b-unsafe-3083_20260705_114753
```

整体结果：

| 阶段 | 结果 | 时间/规模 |
|---|---:|---:|
| total | success | `4497.156s` |
| build | success | `197.175s` |
| docker up | success | `3283.344s` |
| start-bird | success | `965.006s`，串行，间隔 `0.1s` |
| wait RIB stable | success | `1.143s`，load-only，final load1 `48.05` |
| initial full FIB writer | success | `19.783s` writer wall |
| initial FIB verify | success | `312/312` AS representatives passed，`122.488s` |
| reconvergence | success | `48.474s` |

Initial full FIB writer 细项：

| 指标 | 数值 |
|---|---:|
| brdnode containers | `2757` |
| route ops | `6,131,994` |
| failed ops | `0` |
| writer wall | `19.783s` |
| birdc dump wall | `2.392s` |
| kernel write wall | `17.389s` |
| kernel apply sum | `489.455s` |
| batches/groups | `117,449 / 114,169` |

Reconvergence delta 细项：

| 指标 | 数值 |
|---|---:|
| total reconvergence | `48.474s` |
| stop 10 victims | `29.259s` |
| wait load/RIB | `0.002s` |
| delta dump/collapse | `6.750s` |
| raw route events | `366,195` |
| collapsed route ops | `198,866` |
| delta writer wall | `9.379s` |
| successful ops | `198,866` |
| failed ops | `0` |
| delta kernel apply sum | `42.905s` |

最终 VM 状态：

```text
kernel=6.8.12-birdrtbulk-k2b
containers=3073
bird_procs=2747
dmesg risk keyword scan: empty
```

### 9.4 结论

这轮实验给出两个结论：

1. **性能上限非常高**：K2b 在 3083 规模下把 `6.13M` 条真实 BIRD selected route 写入 main FIB，writer wall 只有 `19.783s`；故障后 `198,866` 条 delta ops 的 writer wall 为 `9.379s`。这说明跨 netns 的 global RTNL serialization 确实是大规模仿真中的结构性瓶颈。
2. **不能简单声明“per-netns RTNL 已经安全”**：K2b 的性能来自受控 bulk path、per-netns route lock 和 no-notify。它没有完整处理 route notifier/rtnetlink multicast，也没有 K2c 的 `fib_info_hash_mutex`。本轮没有复现 `fib_info` 崩溃，但 absence of crash 不是安全证明。

因此，K2b 更适合作为论文中的 upper-bound / stress experiment：它证明“如果内核有 per-netns/bulk FIB primitive，性能可以到什么程度”；同时也暴露出需要补齐的语义边界：`fib_info` 共享结构、route notifier、netdevice lifecycle、复杂 route 类型和用户态 route monitor 可见性。真正可上游或可长期复用的方向应当是 K2c/K3c/K3d 这种显式保护共享结构、并逐步补齐 batched notifier/fallback 的设计，而不是直接把整个 RTNL 粗暴改成 per-netns lock。

## 10. 2026-07-05 K2 粗暴拆 RTNL 失败复现

### 10.1 实验目的

这轮实验专门复现“粗暴全拆 RTNL”的失败模式，而不是追求性能上限。测试内核是：

```text
6.8.12-birdrtbulk-k2
```

它只包含：

```text
0001-bird-rtbulk-global-rtnl.patch
0002-bird-rtbulk-pernetns-lock.patch
```

也就是说，BIRD_RT_BULK fast path 不再持有 global RTNL，而是按 network namespace 使用私有 route lock；但它没有 K2b 的 no-notify 规避，也没有 K2c 的 `fib_info_hash_mutex`。这正好用于回答：如果只把 route update 并发放开，而不补齐 Linux FIB 共享语义，会出现什么问题。

本地结果目录：

```text
/data/lxl/seed-emulator/topology/bird/runs/k2_brutal_repro_20260705
```

### 10.2 问题一：1 条 route 就触发 RTNL assertion

最小 smoke：

```text
1 netns
1 dev lo route
BIRD_RT_BULK replace
```

结果目录：

```text
/data/lxl/seed-emulator/topology/bird/runs/k2_brutal_repro_20260705/k2_brutal_smoke_20260705_060716
```

writer 本身返回成功：

```text
success_count=1
fail_count=0
kernel_apply_seconds=0.000628
```

但 dmesg 立即出现：

```text
RTNL: assertion failed at net/ipv4/fib_notifier.c (22)
WARNING: CPU: 8 PID: 20465 at net/ipv4/fib_notifier.c:22 call_fib4_notifiers+0x89/0x90
Call Trace:
  fib_table_insert+0x501/0x630
  bird_rtbulk_apply_one_v4+0x1a3/0x1c0
  bird_rtbulk_apply_v4+0xdc/0x310
```

这说明 `fib_table_insert()` 后面的 route notifier 路径仍然假设调用者持有 RTNL。粗暴拆锁后，route 可以写进去，但 notifier 语义已经违反，内核用 `ASSERT_RTNL()` 明确报错。

这也是 K2b 为什么要跳过 notifier 才能作为 upper-bound 跑起来。但跳过 notifier 会让 `ip monitor route`、rtnetlink multicast 订阅者和依赖 route event 的用户态组件看不到变化，因此它不是完整语义。

### 10.3 问题二：不补 fib_info 会触发 fib_create_info Oops

为了强制触发 `fib_info` 共享结构竞争，本轮给 synthetic writer 增加了默认关闭的参数：

```text
--metric-step
```

默认 `--metric-step 0` 保持旧行为；测试时设置 `--metric-step 1`，让每条 route 使用不同 metric，从而减少 fib_info 复用，迫使内核大量并发创建和链接 `fib_info`。

小规模 `8 x 50k` unique-metric route 没有触发崩溃：

```text
routes=400,000
jobs=8
success=400,000
failed=0
wall=1.038s
dmesg risk scan=empty
```

加大到 `32 x 200k` 后复现崩溃：

```text
routes target=6,400,000
jobs=32
batch=1000
metric-step=1
exit_code=123
```

结果目录：

```text
/data/lxl/seed-emulator/topology/bird/runs/k2_brutal_repro_20260705/k2_brutal_fibinfo_32x200k_mstep1_20260705_061522
```

用户态表现：

```text
Segmentation fault  python3 ... kernel_bulk_writer.py ...
Killed              python3 ... kernel_bulk_writer.py ...
```

内核 dmesg：

```text
general protection fault, probably for non-canonical address 0xdead000000000152
RIP: 0010:fib_create_info+0x803/0x1080
Call Trace:
  fib_table_insert+0x85/0x630
  bird_rtbulk_apply_one_v4+0x1a3/0x1c0
  bird_rtbulk_apply_v4+0xdc/0x310

BUG: unable to handle page fault for address: 000000000c51dd32
Oops: 0000 [#2] PREEMPT SMP NOPTI
RIP: 0010:fib_create_info+0x803/0x1080

general protection fault, probably for non-canonical address 0xf63318c92eea02e2
RIP: 0010:fib_create_info+0x803/0x1080
```

寄存器中出现 `0xdead...` 地址，符合被释放/污染链表节点被并发遍历的典型形态。结合栈：

```text
fib_create_info -> fib_table_insert -> bird_rtbulk_apply_one_v4
```

可以定性说明：`fib_info_hash`、`fib_info_laddrhash`、`fib_info_cnt`、`fib_find_info()`、`fib_info_hash_move()`、`fib_release_info()` 等结构和路径并不是 per-netns 独立语义。它们在 upstream 中依赖 RTNL 进行更大范围串行化。只用 per-netns route lock 并不能保护这些全局共享结构。

这直接解释了 K2c 中 `fib_info_hash_mutex` 的必要性。K2c 不是单纯性能优化，而是补齐 K2/K2b 粗暴拆锁后暴露出的共享 `fib_info` 一致性问题。

### 10.4 对照压力结果

在不使用 `--metric-step` 的普通 synthetic 输入下，K2 即使 `100 x 600k` 也没有复现 `fib_info` 崩溃：

```text
routes=60,000,000
jobs=32
batch=1000
success=60,000,000
failed=0
wall=151.976s
dmesg new risk scan=empty
```

原因是这些 route 使用相同 nexthop/metric，大量 route 复用少量 `fib_info`，没有充分刺激 `fib_info` hash/list 的并发创建和 resize。这个对照说明，是否复现崩溃不仅取决于 route 数量，还取决于 route key 是否导致大量唯一 `fib_info`。

### 10.5 结论

这轮实验把“为什么 Linux 没有简单实现 per-netns RTNL”这个问题具体化了：

1. **notifier 语义不是 per-netns route lock 能自动覆盖的。** `fib_notifier.c` 中的 `ASSERT_RTNL()` 在 1 条 route 写入时就触发，说明 route event 发布路径仍依赖 global RTNL 的调用约束。
2. **`fib_info` 不是纯 per-netns 私有对象。** 当并发 route update 产生大量唯一 `fib_info` 时，未补锁的 K2 会在 `fib_create_info()` 中 Oops，证明 hash/list/refcount 生命周期仍需额外同步。
3. **K2b 的 no-notify 只能作为性能上限实验。** 它可以绕开第一类 ASSERT，但牺牲 route monitor 可见性；如果再不补 `fib_info`，在 unique fib_info 压力下仍有结构性风险。
4. **K2c/K3c 的价值更明确。** 它们不是“为了性能随意改内核”，而是在实验性拆分 RTNL 后，逐步把真实内核语义补回来：共享 `fib_info` 串行化、batched route apply、必要时恢复/批量化 notifier，以及处理 netdevice 生命周期。

因此，论文中可以把 K2 作为反例/失败复现：粗暴 per-netns RTNL 会马上遇到 notifier 和 `fib_info` 两类问题；把 K2c/K3c 作为设计贡献：它们识别并修复这些必须保留的全局语义，同时保留跨 namespace FIB 写入并发收益。
