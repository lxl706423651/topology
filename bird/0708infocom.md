# 0708 INFOCOM 写作草稿：Deferred FIB Update 与 RTNL/FIB 优化路线

本文整理从 BIRD kernel protocol 延迟写入，到 F、K1、K2c、K3c 的方法演进。写作目标是把工程实验转化为论文叙事：问题不是单纯“启动 BIRD 慢”，而是 Linux route update primitive 在大规模 network namespace 仿真中缺少批量化、namespace-aware 并行化和 FIB 对象复用能力。

## 1. 动机：仅依赖 BIRD Kernel Protocol 延迟写入仍不够

最直接的 Deferred Kernel Update 思路，是修改或配置 BIRD 的 kernel protocol：先让控制面协议完成 RIB 收敛，再按固定间隔逐个容器切换 kernel protocol，把 RIB 导出到 Linux main FIB。这个方向可以避免所有 BIRD 实例在控制面尚未稳定时同时写 FIB，从而缓解最初观察到的 RTNL lock storm。

但是，这种方法仍然有两个结构性缺陷。

第一，不同 router 的 RIB/FIB 规模高度不均匀。若使用统一串行间隔，route 数较少的容器会让系统空等，route 数较多的容器可能在下一批触发时还没有完成 FIB 安装。后者会重新制造多个 writer 同时进入内核 route update 路径的情况，导致 RTNL lock 和 FIB 内部结构再次竞争。

第二，BIRD kernel protocol 的写入粒度仍然是 route-level netlink update。即使我们把容器启动顺序调得更温和，每条 `RTM_NEWROUTE` / `RTM_DELROUTE` 仍然需要经过 Linux rtnetlink/FIB 更新路径。换言之，固定间隔只能控制“谁先写”，不能改变“每条 route 如何被内核处理”。

因此，我们将 Deferred Kernel Update 拆成两个问题：

1. 用户态如何避免无序写者风暴。
2. 内核如何更高效地批量处理来自大量独立 namespace 的 route updates。

## 2. F 方法：用户态集中调度 FIB 写入

F 方法保留 stock Linux kernel，不修改内核。它的核心思想是把容器内 BIRD kernel protocol 的即时写入，替换为 host/controller 侧的集中调度：

```text
BIRD RIB converges
  -> host enters router netns
  -> birdc show route primary
  -> parse selected routes and nexthops
  -> generate ip route replace/del commands
  -> ip -force -batch - writes main FIB
```

相比固定间隔切换 BIRD kernel protocol，F 方法有三个优点。

第一，写入量可观测。controller 可以先读取每个容器的 selected routes，知道每个 namespace 需要安装多少 route，而不是盲目按容器数分配时间。

第二，写入速率可控。我们可以控制 writer 并发数、每个 `ip -batch` 的 batch size、batch sleep 等参数，让系统以稳定速率执行 `RTM_NEWROUTE` 和 `RTM_DELROUTE`。

第三，故障后可以使用 delta。修改 BIRD 后，controller 可以读取 selected-route change events，只写故障后真正变化的 route ops，而不是每次都全量重写 main FIB。

F 方法显著缓解了“所有容器同时写 FIB”的写者风暴。但它仍然无法消除每条 route 进入内核时的 rtnetlink/FIB 成本。3083 规模下，故障后的 full `birdc show route primary` dump 只需要 `15.732s`，真正慢的是 `6.10M` route ops 的 FIB 写入，需要 `2025.107s`。这说明瓶颈已经从“如何拿到 RIB”转移到“Linux 如何执行大批量 FIB updates”。

## 3. K1：全局 RTNL 下的内核批量 Route API

K1 在 F 框架基础上修改内核，引入一个受控的 `BIRD_RT_BULK` generic-netlink API。用户态不再为每条 route 单独发送标准 `RTM_NEWROUTE` / `RTM_DELROUTE`，而是把一批 route ops 一次提交给内核。内核在处理一个 batch 时只获取一次 RTNL lock，然后在同一个临界区内串行调用 FIB insert/delete。

```text
user batch
  -> BIRD_RT_BULK request
  -> rtnl_lock()
  -> for op in batch: fib_table_insert/delete
  -> rtnl_unlock()
```

K1 减少了三类开销：

1. 用户态到内核态的 netlink 往返和消息解析开销。
2. 每条 route 重复获取/释放 RTNL lock 的开销。
3. 高频 netlink ACK、调度切换和 lock handoff 带来的尾延迟。

但是，K1 的同步域仍然是 global RTNL。不同 network namespace 的 route updates 仍然被同一个全局锁串行化。因此 K1 可以证明 batching 有价值，但无法证明跨 namespace 的并行性。

## 4. K2c：Per-NS Route Lock + No-Notify + fib_info Hash Mutex

从仿真语义看，不同 router 容器位于不同 network namespace，它们的 main FIB 应该是彼此独立的。因此我们进一步尝试在 K1 的 bulk API 上拆分同步域：把受控 bulk path 的 route update lock 从 global RTNL 降到 per-network-namespace route lock。

粗暴拆锁会立即暴露两个问题。

第一，route notifier 仍然要求 RTNL。K2 smoke test 中，即使只通过 bulk path 写入 1 条 route，也会触发：

```text
RTNL: assertion failed at net/ipv4/fib_notifier.c (22)
WARNING ... call_fib4_notifiers
fib_table_insert -> bird_rtbulk_apply_one_v4
```

这说明 `fib_table_insert()` 后续的 route notifier/rtnetlink multicast 路径仍然假设调用者持有 global RTNL。

第二，`fib_info` 不是纯 per-netns 私有对象。我们用 `--metric-step 1` 构造大量唯一 `fib_info`，在 `32 x 200k` 并发压力下复现：

```text
general protection fault ... 0xdead000000000152
RIP: fib_create_info+0x803/0x1080
fib_create_info -> fib_table_insert -> bird_rtbulk_apply_one_v4
```

这说明 `fib_info_hash`、`fib_info_laddrhash`、`fib_info_cnt`、`fib_find_info()`、`fib_info_hash_move()` 和 `fib_release_info()` 等共享结构在 upstream 中依赖 RTNL 串行化。只给每个 netns 一个独立 route lock，不足以保护这些跨 netns 共享的 FIB 对象。

K2c 因此采用三部分设计：

| 组件 | 作用 | 为什么需要 |
|---|---|---|
| per-NS route lock | 不同 netns 的 bulk route update 可以并行进入 FIB 更新路径 | 解除 global RTNL 对独立 namespace 的过度串行化 |
| no-notify | bulk fast path 暂时跳过 route notifier/rtnetlink multicast | notifier 仍要求 RTNL；不跳过会触发 `ASSERT_RTNL()` |
| `fib_info_hash_mutex` | 串行保护 `fib_info` hash/list/refcount 相关路径 | `fib_info` 是共享结构，粗暴 per-NS lock 会导致 `fib_create_info()` Oops |

K2c 的本质是：把真正 per-namespace 的 route table update 并行化，同时保留对跨 namespace 共享 FIB metadata 的串行保护。它不是完整 production 语义，因为 no-notify 会让 `ip monitor route` 和依赖 rtnetlink multicast 的用户态组件看不到 bulk route changes；但作为受控仿真 fast path，它明确证明 global RTNL 是跨 namespace route update 的结构性瓶颈。

## 5. K3c：Cached fib_info Fast Path

K2c 之后，新的瓶颈变成每条 route 反复执行 `fib_create_info()` 和相关查找/链接。真实 SeedEMU route 中，许多 route 共享相同 nexthop、device、metric、protocol 和 table，只有 prefix 不同。K3c 利用这一点，把同一批共享 FIB key 的 route ops 分组：

```text
group routes by nexthop/device/metric/proto/table
  -> fib_create_info(base_cfg) once
  -> fib_table_insert_info(prefix_i, cached_fi) for each route
```

这样做避免了每条 route 重复创建/查找相同 `fib_info` 的成本。3083 规模下，K3c 把 initial full FIB writer 从 K2c 的 `71.162s` 进一步降到 `10.226s`，其中真实 kernel write 只有 `7.668s`。

K3c 也暴露了新的工程边界。第一，它仍沿用 K2c 的 no-notify，因此 route monitor 可见性不是完整 Linux 语义。第二，cached `fib_info` fast path 对引用计数和 netdevice lifetime 更敏感；实验中观察过 `unregister_netdevice: waiting ... Usage count = ...` 这类 teardown warning。后续 fibref 修复通过在 cached `fib_info` 被 alias 持有前补齐引用计数来降低该风险。论文中应将 K3c 定位为 controlled kernel prototype：它证明 missing primitive 的性能上限，但要成为 production/upstream 方案，还需要补齐 batched notifier、复杂 route fallback、netdevice/netns 生命周期审计和更完整回归测试。

## 6. 3083 规模实验结果

实验对象为 3083 output，实际 compose services 为 `3085`，brdnode containers 约 `2757`。initial full FIB 写入规模为 `6,131,994` route ops。reconvergence 实验停止 10 个 brdnode 后，delta route ops 约为 `198k-561k`，具体数量取决于 FIB writer 和 delta event 版本。

### 6.1 Initial Full FIB 写入

| 方法 | 内核/路径 | route ops | writer wall time | dump time | kernel/write time | 结果 |
|---|---|---:|---:|---:|---:|---|
| Stock BIRD kernel protocol | 容器内 BIRD 直接写 FIB | n/a | 未稳定完成 | n/a | n/a | 大规模同时写入导致 RTNL storm，未得到可比较成功 run |
| F-full | stock kernel, host `setns + ip -batch` | `6,131,994` | `518.714s` | included | included | 成功 |
| K1 | global RTNL bulk API | `6,131,994` | `402.388s` | `2.214s` | `400.173s` | 成功 |
| K2c | per-NS route lock + no-notify + `fib_info_hash_mutex` | `6,131,994` | `71.162s` | `2.235s` | `68.922s` | 成功 |
| K3c | K2c + cached `fib_info` | `6,131,994` | `10.226s` | `2.554s` | `7.668s` | 成功 |

以 F-full 为用户态 baseline，initial full writer speedup 为：

| 方法 | speedup vs. F-full |
|---|---:|
| K1 | `1.29x` |
| K2c | `7.29x` |
| K3c | `50.72x` |

### 6.2 故障后的 Reconvergence FIB 更新

下表的 writer wall time 只统计 route ops 写入 FIB 的时间；端到端时间还包括停止 victim、等待 RIB/load 稳定、delta discovery/collapse 和验证。

| 方法 | route discovery | delta/full route ops | FIB writer wall time | reconv pipeline / total | 结果 |
|---|---|---:|---:|---:|---|
| Stock BIRD kernel protocol | BIRD 自行触发 kernel protocol | n/a | 未稳定完成 | n/a | 大规模下不可稳定对照 |
| F-delta | BIRD delta event + `ip -batch` | `561,377` | `238.320s` | `245s` 级别，dump `6.434s` + write `238.320s` | 成功 |
| F-full post-fault 对照 | full `birdc show route primary` + `ip -batch` | `6,097,653` | `2025.107s` | dump `15.732s` + write `2025.107s` | 成功但很慢 |
| K1 delta | BIRD delta event + global RTNL bulk | `198,804` | `63.639s` | `622.785s` total，主要被 `prepare_delta_offsets=519.594s` 拉长 | 成功 |
| K2c delta | BIRD delta event + per-NS bulk | `198,869` | `55.026s` | `157.028s` total | 成功 |
| K3c delta | BIRD delta event + cached `fib_info` bulk | `198,832` | `6.386s` | `41.922s` total | 成功 |

以 F-delta 的 FIB writer 为 baseline，delta writer speedup 为：

| 方法 | speedup vs. F-delta writer |
|---|---:|
| K1 | `3.74x` |
| K2c | `4.33x` |
| K3c | `37.32x` |

## 7. 论文层面的结论

这组实验支持三个层次的论点。

第一，Deferred Kernel Update 必须从“时间调度”升级为“工作量感知调度”。固定间隔切换 BIRD kernel protocol 无法适配不同容器 RIB/FIB 规模差异；F 方法通过先读取 selected routes，再集中生成和调度 FIB ops，避免了无序写者风暴。

第二，用户态调度只能缓解 RTNL lock storm，不能消除 Linux route update primitive 的低效。F-full 在 3083 initial full 中仍需要约 `519s`；故障后如果 full 重写，则 FIB write 需要 `2025s`，而 full RIB dump 仅 `15.7s`。这说明瓶颈不是 BIRD RIB 可见性，而是内核 FIB update API 和同步粒度。

第三，内核原型的 ablation 证明了缺失 primitive 的具体形态。K1 说明 bulk route API 可以减少 netlink/lock 往返；K2c 说明 global RTNL 对独立 netns route updates 是过度串行化；K3c 说明 `fib_info` 创建/查找也需要批处理和复用。K3c 将 3083 initial full FIB writer 从 F-full 的 `518.714s` 降到 `10.226s`，将故障后的 delta writer 从 `238.320s` 降到 `6.386s`。这不是简单参数调优，而是 API 粒度和内核同步域不匹配导致的数量级差异。

因此，论文中可以将这条路线表述为：

```text
We first decouple control-plane convergence from data-plane FIB installation.
Then, we show that user-space scheduling alone is insufficient because Linux
still exposes route updates as fine-grained rtnetlink operations serialized by
global kernel state. Our kernel prototypes progressively isolate the missing
primitives: bulk route submission, namespace-aware route serialization, and
batched FIB object construction.
```

需要同时承认边界：K2c/K3c 当前是 controlled prototypes，不是通用 Linux networking 替代品。特别是 no-notify 会影响 route monitor 可见性，复杂 route 类型、IPv6、ECMP、nexthop object、VRF 和 netdevice teardown 仍需进一步审计。其论文价值在于证明大规模 network namespace 仿真需要新的 kernel FIB update primitive，并量化该 primitive 可以带来的数量级收益。

## 8. 数据来源

主要数据来自以下本地实验目录：

| 数据 | 目录 |
|---|---|
| F 3083 strict delta/full | `bird/runs/strict_delta_full_3083_20260617_090018` |
| F 多规模实验 | `bird/runs/F_reconf_multiscale_20260614_094101/scale_3083` |
| K1 3083 | `bird/runs/k1-delta-reconv-3083_safej1_20260619_194605` |
| K2c 3083 mixed | `bird/runs/k2c-delta-mixed-3083_20260620_1150` |
| K3c 3083 rerun | `bird/runs/k3c-rerun-3083_20260701_114246` |
| K2 粗暴拆锁失败复现 | `bird/runs/k2_brutal_repro_20260705` |
| K2b unsafe upper-bound | `bird/runs/k2b-unsafe-3083_20260705_114753` |

## 9. 性能收益与语义损失的 Tradeoff

上述方法的目标不是把 Linux networking 改造成一个完全通用的新网络栈，而是在大规模 network namespace 仿真场景中，解决“成千上万个容器同时把 RIB 写入 FIB 时争抢全局 RTNL lock”的核心瓶颈。因此，每一种方法都引入了一定的功能代价。这些代价需要在论文中明确说明；它们是追求大规模仿真性能时有意识接受的 tradeoff，而不是被忽略的副作用。

### 9.1 F 方法的代价

F 方法不再让每个容器内的 BIRD kernel protocol 自发、持续地把 selected routes 写入 Linux FIB，而是由外部 controller 在 RIB 稳定后统一读取 `birdc show route primary` 或 delta event，再生成 `ip route replace/delete` batch 写入 FIB。

这带来两个直接代价。第一，F 方法把 data-plane installation 从 BIRD 内部 kernel protocol 中拆出来，因此它不再完全复刻 BIRD kernel protocol 的所有细节语义，例如 kernel protocol 的周期性 scan、import/export filter 交互、异常重试逻辑以及复杂 route 类型处理。第二，F 方法需要一个外部调度器维护“什么时候读取 RIB、什么时候写 FIB、故障后如何获取 delta”的流程，系统复杂度高于直接 reload BIRD kernel protocol。

但 F 方法仍然使用标准 `ip route`/rtnetlink 路径写入 Linux main FIB，因此最终 FIB 是真实存在于每个 namespace 中的。`ping`、`iperf`、`traceroute` 等依赖 Linux FIB 的数据面工具仍然按普通 Linux 路由语义工作。它牺牲的是 BIRD kernel protocol 的自动安装过程，而不是仿真的核心转发语义。

### 9.2 K1 的代价

K1 在内核中增加 bulk route API，使一个 batch 中的多条 route ops 可以在一次内核入口和一次 RTNL 持锁区间内完成。它减少了 netlink syscall、消息解析和重复 lock/unlock 的开销，但仍然使用全局 RTNL 串行化所有 namespace 的 route update。

K1 的语义改动相对保守，主要代价是新增了一条非标准内核接口，用户态 writer 必须使用我们定制的 bulk path 才能获得收益。它没有从根本上改变 RTNL 的全局串行化语义，因此性能提升有限；但也正因为它仍然保留 global RTNL，兼容性和安全风险最低。

### 9.3 K2c 的代价

K2c 是第一个真正改变同步域的方法。它将独立 namespace 中的 route update 从全局 RTNL 中拆出来，用 per-namespace route lock 并行化写入；同时通过 `fib_info_hash_mutex` 保护跨 namespace 共享的 `fib_info` metadata，并通过 no-notify 绕开仍然假设调用者持有 global RTNL 的 route notifier path。

K2c 的主要代价正来自 no-notify。通过 K2c bulk path 写入的路由不会向 rtnetlink multicast listener 逐条发送 route notification。因此，依赖 route event 的工具或 daemon 会失去实时可见性，例如：

- `ip monitor route` 无法观察到 bulk path 产生的每一条 `RTM_NEWROUTE` / `RTM_DELROUTE` event；
- 依赖 rtnetlink route notification 的外部控制器、监控系统或网络管理 daemon 不能仅靠订阅事件来同步状态；
- 某些内核 notifier 或 out-of-tree module 如果假设 route update 必然经过标准 notifier path，也不会看到这些变化。

但这并不破坏我们的核心仿真语义。SeedEMU 大规模互联网仿真的关键要求是：控制面 RIB 收敛后，Linux main FIB 中存在正确的 selected routes，并且数据包按照这些 routes 转发。K2c 写入后的路由仍然可以通过 `ip route` 查询，仍然参与 Linux FIB lookup，仍然驱动真实 packet forwarding。失效的是“事件观测语义”，不是“最终转发语义”。因此，K2c 更适合受控仿真实验环境，而不是直接替代通用 Linux 主机的动态网络管理路径。

### 9.4 K3c 的代价

K3c 在 K2c 的基础上继续优化 `fib_info` 创建和查找开销，对具有相同 nexthop、metric 和 protocol 属性的一组 routes 复用 cached `fib_info`。这使 initial full FIB 写入从 K2c 的几十秒进一步降到十秒级，但也引入了更强的适用条件。

K3c 目前主要面向受控的 IPv4 unicast 仿真路由批量写入。对于 IPv6、ECMP、nexthop object、VRF、复杂 rule/table 组合、route metrics 扩展字段、netdevice 生命周期交错等更一般的 Linux networking 场景，还需要继续审计。早期 K3c 原型也曾暴露出 `fib_info` 引用计数问题：cached `fib_info` 被多个 FIB alias 共享时，必须为每个 alias 正确维护 `fib_treeref`，否则在 route delete 或 namespace teardown 时会触发 refcount underflow、use-after-free 或 netdevice 引用泄漏。后续 K3c-fibref 通过在每次 `fib_table_insert_info()` 前显式增加 `fib_treeref` 修复了该问题。

因此，K3c 的定位不是“完整替换 Linux FIB 子系统”，而是证明在大规模仿真中，`fib_info` 构造本身也是 FIB 收敛瓶颈的一部分。它展示了如果内核提供面向批处理的 FIB object construction primitive，可以获得数量级性能收益。

### 9.5 为什么这些损失是可接受的

我们的仿真目标不是验证 Linux 所有网络管理 API 的事件一致性，而是构造大规模 AS/router 拓扑，在真实 namespace、真实 Linux FIB 和真实 packet forwarding 上运行控制面与数据面实验。对这个目标而言，最核心的 correctness condition 是：

```text
after control-plane convergence, selected routes are installed in the target
namespace FIB, and packets are forwarded according to those routes.
```

F、K1、K2c 和 K3c 都保留了这个核心条件。它们牺牲或弱化的是 route update 的观测方式、通知路径、通用内核兼容性和部分复杂 route 类型支持。换句话说，它们把优化边界明确限定在“受控大规模仿真中的 FIB installation path”，而不是试图无条件替换 Linux networking 的全部语义。

这也是本文修改内核的合理性所在：RTNL lock storm 是由 Linux 当前 fine-grained rtnetlink route update primitive 与大规模 namespace 仿真 workload 不匹配造成的。用户态调度可以缓解争抢，但无法消除每条 route update 仍要进入全局同步路径的根本问题。内核原型通过逐步引入 bulk submission、namespace-aware route serialization 和 batched FIB object construction，明确展示了缺失 primitive 的形态及其性能上限。代价是部分事件通知和通用网络管理语义被弱化；收益是在不放弃 Linux FIB 和真实 packet forwarding 的前提下，使数百万级 route installation 从数百秒甚至数千秒降到秒级到十秒级。

因此，在论文中应将这些方法表述为面向 large-scale network emulation 的 specialized kernel support，而不是 general-purpose Linux networking replacement。这样的 tradeoff 与仿真系统的设计目标一致：优先保证可扩展、可复现、可验证的数据面收敛，同时清晰标注不被支持或被弱化的非核心语义。
