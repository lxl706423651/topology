# 2026-06-18 内核 RTNL/FIB 优化实验设计

生成时间：2026-06-18

目标：开始设计两条内核方向的实验路线：

1. **route-only / per-netns RTNL lock**：让不同 network namespace 的 route update 尽量不要争同一个全局 RTNL。
2. **bulk route API**：让用户态一次提交一批 route ops，内核一次解析和一次持锁处理多条路由，减少 `ip -batch` 当前仍然逐条 netlink message 的开销。

当前宿主机和 VM kernel：

- host：`6.8.0-124-generic`
- VM：`6.8.0-124-generic`
- 当前机器只有 headers，后续真正 patch 前需要准备对应 Ubuntu kernel source。

## 背景结论

已有实验说明：

- 3083 full 对照中，`birdc show route primary` dump 只需 `15.732s`，而 FIB 写入需要 `2025.107s`。
- 3083 full writer jobs sweep 中，`jobs=4` 能完成 `6,097,653` 条写入，但 `jobs=8` 反而超过 `4000s` 未完成，并出现多个 `ip -batch` D-state。
- 100 个 netns 每个写 `600,000` 条 route 的合成实验中，即使绕过 BIRD/Docker，只写极简 `dev lo` route，总计 `60M` route ops 也需要 `3456.237s`。

因此当前瓶颈不是 RIB dump，也不是 CPU 算力，而是 Linux FIB/RTNL 更新路径的并发能力和单 namespace 大 FIB 表写入成本。

## Linux Route Update 路径假设

需要在 Ubuntu 6.8 source 中最终确认，但按 Linux 6.8 常见路径，IPv4 route add/replace 大致是：

```text
user space
  ip route replace ...
    -> netlink RTM_NEWROUTE
      -> net/core/rtnetlink.c
        -> rtnetlink_rcv_msg()
        -> rtnl_lock()
        -> inet_rtm_newroute()
          -> net/ipv4/fib_frontend.c
            -> fib_table_insert()
              -> net/ipv4/fib_trie.c
```

关键问题：

- 传统 `RTM_NEWROUTE`/`RTM_DELROUTE` 处理依赖 RTNL 作为串行化点。
- `/usr/src/linux-headers-6.8.0-124/include/net/netns/ipv4.h` 中 `fib_seq` 注释仍写着 `protected by rtnl_mutex`。
- nexthop namespace 状态中也有字段标注 `protected by rtnl_mutex`。
- 这说明不能简单把全局 `rtnl_mutex` 替换成每 netns 一个锁，否则可能破坏 netdevice、nexthop、rules、notifier、VRF、lwtunnel 等路径的隐含同步关系。

所以本设计不建议第一步直接全局改 `rtnl_lock()`。更现实路线是：

1. 先做可观测性，量化 RTNL/FIB 临界区耗时。
2. 先做 **bulk route API with global RTNL**，降低每条 route 的 netlink/ACK/锁开销。
3. 再在 bulk API 内做 **route-only per-netns lock**，只让受控路径绕开全局 RTNL。
4. 最后才评估是否让标准 `RTM_NEWROUTE` 也走 per-netns route lock。

## 总体路线

| 阶段 | 目标 | 风险 | 是否改现有 RTM_NEWROUTE |
|---|---|---:|---|
| K0 | 加观测和基线，不改变行为 | 低 | 否 |
| K1 | 新增 bulk route API，但内部仍拿全局 RTNL | 中 | 否 |
| K2 | bulk route API 改为 route-only per-netns lock | 高 | 否 |
| K3 | 标准 RTM_NEWROUTE/DELROUTE route-only 化 | 很高 | 是 |
| K4 | multi-netns bulk transaction | 很高 | 否，新增 API |

建议优先级：

```text
K0 -> K1 -> K2 -> 小规模验证 -> 3083 复现实验 -> 再讨论 K3/K4
```

## K0：可观测性与基线

### 目的

在改内核前，先能回答：

- 每条 route update 在 RTNL 内耗时多少？
- `fib_table_insert()`、`fib_table_delete()` 耗时多少？
- `jobs=4` 到 `jobs=8` 时，等待锁的时间是不是明显上升？
- 是否存在长时间持锁的 route、nexthop、device notifier 路径？

### 实现方式

优先使用不改内核的观测：

- `lockstat` / `perf lock`
- ftrace function graph
- kprobe/eBPF 观测：
  - `rtnl_lock`
  - `rtnl_unlock`
  - `rtnetlink_rcv_msg`
  - `inet_rtm_newroute`
  - `inet_rtm_delroute`
  - `fib_table_insert`
  - `fib_table_delete`

如果函数符号不可直接探测，再加最小内核 patch：

- 在 route add/delete handler 入口打 tracepoint。
- 在 FIB insert/delete 前后打 tracepoint。
- 在 RTNL lock acquire/release 周围记录时间。

### 输出指标

每次实验输出：

- route ops 总数。
- RTNL wait time p50/p95/p99/max。
- RTNL hold time p50/p95/p99/max。
- FIB insert/delete time p50/p95/p99/max。
- D-state `ip -batch` 进程数量。
- CPU user/system/softirq/idle。
- `jobs=1/2/4/8` 的吞吐曲线。

## K1：新增 Bulk Route API，先保留全局 RTNL

### 设计目标

减少当前 `ip -batch` 的逐条 netlink message 成本。

当前 `ip -batch` 只是减少了用户态进程启动和命令解析开销，但每条 route 仍基本对应一个 `RTM_NEWROUTE` netlink message。内核仍逐条解析、逐条执行、逐条 ACK。

K1 希望变成：

```text
user writer
  -> one bulk request with N route ops
kernel
  -> parse once
  -> rtnl_lock once
  -> loop N route ops
  -> rtnl_unlock once
  -> return per-route errno array
```

注意：K1 仍使用全局 RTNL，所以它不解决跨 netns 真并行，但能减少同一 namespace 内反复 lock/unlock、netlink 解析和 ACK 往返。

### API 形式选择

有三种选择：

| API | 优点 | 缺点 | 建议 |
|---|---|---|---|
| 新增 `RTM_NEWROUTES_BULK` | 语义贴近 rtnetlink | 需要扩展 uapi rtnetlink，和现有 handler/ACK 语义耦合深 | 不作为第一版 |
| Generic Netlink family | 独立、实验友好、易开关 | 仍受 netlink message size 限制 | 推荐第一版 |
| char device `/dev/rtbulk` | 可以用 writev/ioctl 传大块二进制 | 不像网络 API，需要额外设备生命周期 | 可作为性能极限版 |

建议第一版采用 **Generic Netlink family**，名字例如：

```text
family: BIRD_RT_BULK
cmd:    BIRD_RT_BULK_APPLY_V4
```

### 用户态调用方式

为避免跨 netns 参数设计复杂，第一版保持和当前 host writer 一样：

1. host writer 打开目标容器 netns。
2. `setns()` 进入目标 netns。
3. 打开 generic netlink socket。
4. 发送该 netns 内的 bulk route ops。
5. 回到原 netns，处理下一个容器。

这意味着 K1 不需要让内核一次请求跨多个 netns，也不需要传 netns id。

### 请求格式

第一版只支持 IPv4，且不支持 IPv6/ECMP/multipath。

建议二进制数组，避免每条 route 都用 nested netlink attr，降低解析开销：

```c
struct bird_rtbulk_req_v1 {
    __u16 version;        /* 1 */
    __u16 flags;          /* e.g. ATOMIC=0, NOTIFY=0 */
    __u32 count;
    __u32 default_table;  /* e.g. RT_TABLE_MAIN */
    __u32 default_proto;  /* e.g. 232 */
    __u32 default_metric; /* e.g. 232 */
    struct bird_rtbulk_op_v1 ops[];
};

struct bird_rtbulk_op_v1 {
    __u8  op;             /* ADD, REPLACE, DEL */
    __u8  dst_len;        /* 0..32 */
    __u8  scope;
    __u8  type;           /* RTN_UNICAST, RTN_BLACKHOLE */
    __be32 dst;
    __be32 gateway;       /* 0 if no gateway */
    __s32 ifindex;        /* 0 if unused */
    __u32 table;          /* 0 means default_table */
    __u32 metric;         /* UINT_MAX means default_metric */
    __u32 flags;          /* ONLINK, etc. */
};
```

响应：

```c
struct bird_rtbulk_reply_v1 {
    __u16 version;
    __u16 flags;
    __u32 count;
    __u32 success_count;
    __u32 fail_count;
    struct bird_rtbulk_result_v1 results[];
};

struct bird_rtbulk_result_v1 {
    __u32 index;
    __s32 errno_value;
};
```

如果全部成功，可以只返回 summary，不返回每条 result。

### Kernel 内部处理

候选文件：

- `net/ipv4/fib_bulk.c`：新增 bulk route 实现。
- `include/uapi/linux/bird_rtbulk.h`：新增实验 uapi。
- `net/ipv4/Kconfig`：加 `CONFIG_BIRD_RT_BULK`.
- `net/ipv4/Makefile`：编译 `fib_bulk.o`.

处理流程：

```text
bird_rtbulk_apply_v4()
  -> parse request
  -> validate count <= max_batch
  -> allocate result array
  -> rtnl_lock()
  -> for each op:
       build fib_config
       validate table/proto/metric/dst/gw/ifindex
       call fib_table_insert/delete path
       record errno
       if time budget exceeded: cond_resched point after unlock/relock or stop
  -> rtnl_unlock()
  -> return summary/results
```

### 限制

第一版限制：

- IPv4 only。
- 单 netns only。
- 单请求在当前 netns 生效。
- 不支持 ECMP/multipath。
- 不支持 IPv6。
- 不支持 nexthop object (`RTA_NH_ID`)。
- 先支持 `RTN_UNICAST` 和 `RTN_BLACKHOLE`。
- 支持 main table 和指定 table，但实验默认 main table。
- 默认 partial success，不做事务回滚。

### Batch 大小

建议矩阵：

| batch size | 目的 |
|---:|---|
| 256 | 保守起点 |
| 1024 | 低风险性能点 |
| 4096 | 主要实验点 |
| 16384 | 压测点，观察长时间持锁副作用 |

不能无限增大 batch，因为持锁太久会影响其他网络操作。第一版应设置：

- `max_batch=4096` 默认。
- 通过 sysctl 或 module param 临时调大。
- 每个 bulk 请求记录 hold time。

### 用户态 Writer 修改

新增一个独立 writer，不直接改现有 F 脚本：

- `bird/kernel_bulk_writer.c` 或 `bird/kernel_bulk_writer.py + C helper`
- 推荐 C helper：
  - 支持 `setns`
  - 读取 command file 或 stdin
  - 转换成 `bird_rtbulk_op_v1`
  - 发送 generic netlink bulk request
  - 输出 JSON summary

再让 `delta_fib_host_writer.py` 增加可选模式：

```bash
--write-mode kernel-bulk
--kernel-bulk-batch 4096
```

但第一轮验证可以完全独立，不集成进现有主流程。

### K1 验证实验

对比：

| 实验 | kernel | writer | 目标 |
|---|---|---|---|
| baseline | stock | `ip -batch` | 当前基准 |
| K1 | patched | bulk API + global RTNL | 验证减少单 netns lock/netlink 开销 |

测试规模：

1. `1 netns * 600k routes`
2. `100 netns * 600k routes, jobs=1/2/4`
3. 3083 full command-files，route ops `6.10M`

预期：

- 单 netns 600k 应明显快于 stock `40s`。
- 100x600k 的 `jobs=1` 应提升明显。
- `jobs=4/8` 是否提升，取决于 RTNL 是否仍为主瓶颈。

## K2：Bulk API 内 Route-only Per-netns Lock

### 设计目标

让不同 netns 的 bulk route update 可以真正并行，不再全部排队等全局 RTNL。

K2 不修改标准 `RTM_NEWROUTE`，只让新增的 bulk API 使用 route-only per-netns lock。这是为了把风险限制在我们自己控制的实验 API 内。

### 锁设计

新增锁：

```c
struct netns_ipv4 {
    ...
    struct mutex fib_route_mutex; /* experiment: protects route-only bulk updates */
};
```

或放在 `struct net`：

```c
struct net {
    ...
    struct mutex route_cfg_mutex;
};
```

建议第一版放 `struct netns_ipv4`，因为只做 IPv4。

K2 bulk path：

```text
bird_rtbulk_apply_v4()
  -> parse request
  -> mutex_lock(&net->ipv4.fib_route_mutex)
  -> loop fib insert/delete
  -> mutex_unlock(&net->ipv4.fib_route_mutex)
```

标准 netdevice/link 变更仍使用全局 `rtnl_mutex`。

### 锁顺序

必须固定锁顺序：

```text
global rtnl_mutex -> per-netns fib_route_mutex
```

规则：

- route-only bulk API 不拿 global RTNL，只拿当前 netns 的 `fib_route_mutex`。
- 如果某个路径已经拿了 global RTNL，并需要修改该 netns FIB，则必须再拿 `fib_route_mutex`。
- 禁止在持有 `fib_route_mutex` 时再尝试拿 global RTNL。
- multi-netns 操作必须按 netns id 排序加锁，第一阶段不做 multi-netns。

### 最大风险

全局 RTNL 当前不仅保护 FIB 表，还间接保护：

- netdevice 生命周期。
- ifindex 查找和 device unregister。
- nexthop object。
- fib rules。
- route notifier。
- lwtunnel state。
- VRF。
- neighbor/link 相关状态。

如果 bulk path 不拿 global RTNL，必须确保 route ops 中引用的 device/gateway 在执行期间安全。

### 第一版限制

为了降低风险，K2 第一版只支持最简单场景：

- IPv4。
- `RT_TABLE_MAIN`。
- 不支持 ECMP/multipath。
- 不支持 nexthop object。
- 不支持 IPv6。
- 不支持 VRF。
- 不支持 lwtunnel。
- 不支持 route append。
- route op 只允许：
  - `replace blackhole <prefix>`
  - `replace <prefix> dev lo`
  - 可选：`replace <prefix> via <gateway> dev <ifindex>`，但需要 `dev_hold()`/RCU 审计通过后再打开。

建议先用 `dev lo`/blackhole 合成基准验证 per-netns lock 是否能带来跨 netns 并行，然后再支持真实 BIRD nexthop。

### Device 引用策略

对于 `dev lo`：

- 当前 netns 内 `lo` 生命周期和 netns 生命周期绑定，风险最低。

对于一般 ifindex：

候选策略：

1. 在 RCU read lock 下查找 device，并 `dev_hold()`。
2. bulk op 完成后 `dev_put()`。
3. device unregister 路径必须和 `fib_route_mutex` 协调，避免 route 插入引用正在注销的 device。

如果这一步风险过大，K2 真实 BIRD nexthop 支持要推迟。

### K2 验证实验

先跑合成，不碰真实拓扑：

| 实验 | 规模 | jobs | 预期 |
|---|---|---:|---|
| stock | 100 netns * 600k | 4 | 3456s 基准 |
| K1 global bulk | 100 netns * 600k | 4 | 减少单 writer 开销 |
| K2 per-netns bulk | 100 netns * 600k | 4 | 真正跨 netns 并行，应明显优于 K1 |
| K2 per-netns bulk | 100 netns * 600k | 8/16 | 找新拐点 |

如果 K2 在 `dev lo` 路由上有效，再进入真实 BIRD：

| 实验 | 规模 | 目标 |
|---|---|---|
| 1897 strict | 小规模安全验证 |
| 3083 initial full | 对比 `520-579s` |
| 3083 post-fault full | 对比 `2025s` |
| 3083 delta | 对比 `238s` |

### 成功标准

K2 只有满足以下条件才算可继续：

- lockdep 无告警。
- KASAN/KCSAN 无明显错误。
- netns 删除无 hang。
- Docker 容器 stop/remove 正常。
- route count 抽样一致。
- `ip route get` 抽样正确。
- 并发提升不是靠牺牲 correctness 换来的。

## K3：标准 RTM_NEWROUTE/DELROUTE Route-only 化

K3 是高风险阶段，不建议第一轮做。

目标是让普通 `ip route replace` 也可以绕过全局 RTNL，针对 route-only 操作走 per-netns route lock。

可能修改：

- `net/core/rtnetlink.c`
  - 在 `rtnetlink_rcv_msg()` 或 handler dispatch 层识别 `RTM_NEWROUTE`/`RTM_DELROUTE`。
  - 对 route-only handler 使用 per-netns lock，而不是全局 `rtnl_mutex`。
- `net/ipv4/fib_frontend.c`
  - IPv4 route add/delete handler 明确不依赖全局 RTNL，或改成显式拿 route lock。
- `net/ipv6/route.c`
  - 第一阶段不做 IPv6。

风险：

- 很多 handler 假设自己在 RTNL 下运行。
- 即使 message 是 `RTM_NEWROUTE`，内部也可能触发 notifier、nexthop、device 引用路径。
- 上游 rtnetlink dispatch 不是只服务 route，还服务 link/address/rule/qdisc 等多类对象，局部改错容易造成隐式竞态。

建议只有在 K1/K2 证明收益很大且真实拓扑正确后，再考虑 K3。

## K4：Multi-netns Bulk Transaction

这是用户最初提出的“调用一次锁，让多个 router 一起写入 FIB”的内核化版本。

第一阶段不建议做，原因：

- 单请求跨多个 netns，需要传 netns id/fd。
- 内核要持有多个 `struct net` 引用。
- 多个 netns 锁必须排序，避免死锁。
- 如果仍拿全局 RTNL，一次持锁时间会极长。
- 如果使用 per-netns lock，则需要处理部分 netns 成功、部分失败。

设计雏形：

```c
struct bird_rtbulk_multi_req_v1 {
    __u32 netns_count;
    struct bird_rtbulk_netns_batch batches[];
};

struct bird_rtbulk_netns_batch {
    __s32 netns_fd_index;
    __u32 count;
    struct bird_rtbulk_op_v1 ops[];
};
```

锁顺序：

1. 用户态传入 netns fd 数组。
2. 内核转换为 `struct net *`。
3. 按 netns inode/cookie 排序。
4. 固定顺序加锁。
5. 每个 netns 内批量写入。
6. 反序解锁。

该方案适合研究验证，不作为第一阶段目标。

## 内核源码准备方案

需要准备 Ubuntu 6.8.0-124 对应 source：

```bash
mkdir -p /data/lxl/kernel-work
cd /data/lxl/kernel-work
# 方案一：apt source linux-image-unsigned-$(uname -r)
# 方案二：git clone Ubuntu kernel source 并 checkout 对应 tag
```

构建原则：

- 不覆盖当前可启动 kernel。
- 自定义 local version，例如 `6.8.0-124-birdrtbulk`.
- VM 安装新 kernel 后保留原 `6.8.0-124-generic` 作为 GRUB fallback。
- 每轮 kernel 实验前保存：
  - `/boot/config-$(uname -r)`
  - `/proc/cmdline`
  - `lsmod`
  - Docker/VM reset 状态。

建议目录：

```text
bird/kernel/
  0618ExperimentDesign.md          # 本文档
  patches/
    0001-bird-rtbulk-tracepoints.patch
    0002-bird-rtbulk-global-rtnl.patch
    0003-bird-rtbulk-pernetns-lock.patch
  tools/
    kernel_bulk_writer.c
    parse_f_writer_commands.py
  tests/
    synthetic_1x600k.sh
    synthetic_100x600k.sh
    strict_3083_bulk.sh
```

## 用户态工具设计

新增工具：`kernel_bulk_writer`

输入格式：

1. 直接读取现有 command files：
   - `ip route replace ...`
2. 或读取 compact JSONL：
   - `{ "op": "replace", "dst": "10.0.0.0/24", "via": "...", "dev": "...", "metric": 232 }`

建议第一版支持 command files，便于复用已有 F/full/delta 输出。

命令示例：

```bash
kernel_bulk_writer \
  --netns-pid 12345 \
  --input commands/as1269.txt \
  --batch 4096 \
  --metric 232 \
  --json-summary summary.json
```

输出 summary：

```json
{
  "mode": "kernel-bulk",
  "netns_pid": 12345,
  "batch_size": 4096,
  "commands": 600000,
  "batches": 147,
  "wall_seconds": 12.345,
  "successes": 600000,
  "failures": 0,
  "kernel": "6.8.0-124-birdrtbulk",
  "lock_mode": "global-rtnl"
}
```

## 实验矩阵

### 合成基准

| kernel | writer | lock mode | routers | routes/router | jobs | 指标 |
|---|---|---|---:|---:|---:|---|
| stock | `ip -batch` | global RTNL | 1 | 600k | 1 | baseline 40s |
| K1 | bulk | global RTNL | 1 | 600k | 1 | 单 netns bulk 收益 |
| K1 | bulk | global RTNL | 100 | 600k | 4 | bulk 但仍全局锁 |
| K2 | bulk | per-netns route lock | 100 | 600k | 4 | 跨 netns 并行收益 |
| K2 | bulk | per-netns route lock | 100 | 600k | 8/16 | 新拐点 |

### 真实 3083

| kernel | writer | 场景 | route ops | 当前基线 |
|---|---|---|---:|---:|
| stock | F full `ip-batch` | initial full | 6.13M | 520-579s |
| K1 | bulk global RTNL | initial full | 6.13M | 待测 |
| K2 | bulk per-netns | initial full | 6.13M | 待测 |
| stock | full post-fault `jobs=2` | full control | 6.10M | 2025s |
| stock | full post-fault `jobs=4` | command replay | 6.10M | 2511s |
| K1/K2 | bulk | full post-fault | 6.10M | 待测 |
| stock | delta | post-fault delta | 561k | 238s |
| K1/K2 | bulk | post-fault delta | 561k | 待测 |

## 正确性验证

每轮内核实验必须验证：

- `ip route show table main | wc -l`
- `birdc show route primary count`
- 10 个不同 AS 抽样：
  - FIB route count >= RIB selected route count 或与实验定义一致。
  - `ip route get` 对随机目的地址返回预期 nexthop/dev。
- Docker stop/remove 网络 namespace 不 hang。
- VM 可正常 reboot。
- `dmesg` 无 lockdep/KASAN/KCSAN warning。
- `/proc/slabinfo` 和 memory usage 无异常增长。

## 风险与回滚

主要风险：

- 死锁：global RTNL 与 per-netns route lock 顺序错误。
- use-after-free：route op 引用正在 unregister 的 net_device。
- route notifier 缺失或乱序。
- netns teardown 卡死。
- Docker 网络清理失败。
- 自定义 API 写入 FIB 正确，但破坏标准 `ip route`/Docker 行为。

回滚策略：

1. 每次安装自定义 kernel 前确认 GRUB 有 stock kernel。
2. VM 使用 libvirt，可用：
   - `virsh destroy bird-rtnl-dev`
   - `virsh start bird-rtnl-dev`
3. 如果自定义 kernel 无法启动，从 GRUB 选择 stock kernel。
4. 如果 Docker runtime 被破坏，启动 stock kernel 后执行：
   - `/home/lxl/bird/reset_docker_vm.sh`

## 建议的下一步

第一轮不要直接做 per-netns RTNL。建议按下面顺序：

1. 准备 Ubuntu `6.8.0-124-generic` 对应 kernel source。
2. 做 K0 tracepoint/观测补丁，跑 `jobs=4/8`，确认 RTNL wait/hold 分布。
3. 做 K1 bulk API with global RTNL。
4. 写 `kernel_bulk_writer`，先跑 `1x600k` 和 `100x600k` 合成实验。
5. 如果 K1 对单 netns 明显提速，再做 K2 per-netns route lock。
6. K2 先只支持 `dev lo` 或 blackhole route，证明跨 netns 并行。
7. 再扩展到真实 BIRD nexthop，跑 1897，再跑 3083。

阶段性预期：

- K1 能降低单 writer 单 netns 开销，但不能解决跨 netns 全局 RTNL 竞争。
- K2 如果正确实现，应在 `100 netns * 600k` 的 `jobs=4/8` 上明显优于 K1/stock。
- 即使 K2 成功，delta route 仍然有价值，因为减少 route ops 永远比全量写更稳。
