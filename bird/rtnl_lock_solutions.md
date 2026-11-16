# BIRD/RTNL 大规模路由写入拥塞解决方案

## 背景

在大规模仿真互联网中，如果 10000 个 router 容器同时由 BIRD 向 Linux FIB 写入路由，每个容器内的 BIRD 都会通过 netlink 发送 `RTM_NEWROUTE`/`RTM_DELROUTE` 请求。Linux 路由配置路径需要进入 RTNL 临界区，因此大量容器会在宿主机内核上争抢 RTNL 锁。

观察到的现象是：

- 10000 个 router 每个写入少量路由时，容易因为写者数量太多形成 RTNL 锁拥塞。
- 10 个 router 每个写入大量路由时，反而不容易出现同等级锁风暴，因为并发争锁主体数量少。
- 问题的核心不是单条 FIB 插入慢，而是大量独立 BIRD 进程同时触发内核路由配置事务。

以 BIRD 2.17.2 为例，Linux kernel protocol 的关键路径在源码 `sysdep/linux/netlink.c`：

- `nl_send_route()` 为单条路由构造 netlink 消息。
- 消息类型是 `RTM_NEWROUTE` 或 `RTM_DELROUTE`。
- 默认带 `NLM_F_ACK`，随后 `nl_exchange()` 等待内核 ACK。
- 默认行为是逐条路由写入，而不是一次 netlink 事务提交多条路由。

因此，单纯使用当前 apt/PPA 安装的 BIRD，不能避免多个容器同时抢 RTNL。

## 目标

目标不是让控制面停止收敛，而是减少数据面 FIB 写入阶段对 RTNL 的瞬时冲击：

- 降低同时写 FIB 的 BIRD 进程数量。
- 降低每个进程的瞬时 netlink 请求速率。
- 尽可能合并或批量化路由写入。
- 如果需要更彻底的方案，再改 Linux 内核的 RTNL 或 FIB 写入接口。

## 方案总览

| 方案 | 修改对象 | 效果 | 风险 | 推荐程度 |
| --- | --- | --- | --- | --- |
| 分阶段开启 kernel export | BIRD 配置和 orchestrator | 快速缓解锁风暴 | 低 | 强烈推荐 |
| BIRD kernel protocol 加节流队列 | BIRD 源码 | 限制每个实例写 FIB 速率 | 中 | 推荐 |
| BIRD netlink 多消息批量发送 | BIRD 源码 | 减少 syscall/ACK 往返 | 中高 | 可实验 |
| 宿主机集中式 FIB writer | 外部控制器 + BIRD 配置 | 把 10000 个写者变成少数写者 | 中 | 强烈推荐 |
| 内核新增 bulk route API | Linux 内核 | 一个 netns 内一次锁写多条路由 | 高 | 实验方向 |
| 内核 per-netns RTNL | Linux 内核 | 从根上降低跨 netns 锁竞争 | 很高 | 长期研究方向 |
| 减少必须写入 FIB 的路由 | 仿真模型和导出策略 | 直接减少内核写入规模 | 低 | 强烈推荐 |

## 方向一：改 BIRD

### 方案 A：分阶段开启 kernel export

这是最先应该做的方案，不需要立刻改 BIRD 源码。

思路：

1. BIRD 启动初期只跑控制面协议，例如 OSPF/BGP/iBGP/eBGP。
2. `protocol kernel` 初期不要 `export all`，只导出必要路由，或者直接 `export none`。
3. 等控制面 RIB 稳定后，由宿主机 orchestrator 按批次切换部分容器的 `kernel.conf`。
4. 每批执行 `birdc configure && birdc reload kernel`。
5. 根据系统 load、RTNL 延迟、netlink 操作速率决定下一批是否继续。

推荐配置阶段：

```bird
protocol kernel {
    persist;
    scan time 60000;
    ipv4 {
        import none;
        export filter {
            if source = RTS_DEVICE then accept;
            if source = RTS_OSPF then accept;
            reject;
        };
    };
}
```

控制面稳定后再切换到：

```bird
protocol kernel {
    merge paths on;
    persist;
    scan time 60000;
    ipv4 {
        import none;
        export all;
    };
}
```

实现重点：

- 批次大小从 10/50/100 个容器开始压测，不要直接 10000 个容器同时 reload。
- 每个容器 reload 前加入随机 jitter，例如 0 到 10 秒。
- 按 AS 或拓扑区域分组，避免同一时刻大量相邻路由器一起写 FIB。
- 用 eBPF 或 trace 脚本统计 `RTM_NEWROUTE`/`RTM_DELROUTE` 的速率。
- 用 `ip route show table main | wc -l` 或 `birdc show route count` 验证每批完成情况。

优点：

- 实现最快。
- 不需要维护 BIRD fork。
- 对现有 SeedEMU 脚本改动小。

缺点：

- 不能减少最终总写入次数。
- 收敛完成时间仍受批次节奏影响。

### 方案 B：在 BIRD kernel protocol 中加入导出节流队列

这是改 BIRD 的首选源码方案。

当前 BIRD 的 kernel protocol 在路由变化时会调用 `krt_replace_rte()`，进一步调用 Linux 下的 `nl_send_route()`。可以把这里改成两阶段：

1. 路由变化时不立即写 netlink。
2. 将 `add/replace/delete` 操作放入 per-kernel-protocol 队列。
3. 由 timer 周期性 flush 队列。
4. 每次 flush 限定最多处理 N 条路由。
5. flush 周期和批大小通过 BIRD 配置项控制。

建议新增配置项：

```bird
protocol kernel {
    ipv4 {
        import none;
        export all;
    };

    netlink export batch 256;
    netlink export interval 20 ms;
    netlink export jitter 500 ms;
}
```

内部行为：

- `netlink export batch`：每轮最多写多少条路由。
- `netlink export interval`：两轮写入之间的最小间隔。
- `netlink export jitter`：BIRD 启动后首次 flush 的随机延迟，避免所有容器同相位写入。

实现位置建议：

- `sysdep/unix/krt.c`：把 `krt_replace_rte()` 调用点改为 enqueue。
- `sysdep/linux/netlink.c`：保留实际 `nl_send_route()`，作为 flush 阶段的底层发送函数。
- `sysdep/unix/krt.h`：扩展 `struct krt_proto`，添加 pending queue、timer、配置字段。
- `sysdep/linux/netlink.Y` 或相关配置语法文件：添加新配置项解析。

关键细节：

- 同一个 prefix 的多次变更要合并，避免队列中出现 add/delete/add 的无效抖动。
- delete 和 replace 的顺序必须保守处理。
- flush 失败时要保留或重试，不能直接把 sync_map 标成已同步。
- BIRD 退出或 `birdc down` 时，要决定是否立即 flush 或丢弃 pending 操作。
- `persist` 模式下要保持现有语义。

优点：

- 保留每个容器 BIRD 自治。
- 可直接降低 RTNL 瞬时并发。
- 对内核无侵入。

缺点：

- 仍然是每个容器写自己的 netns。
- 不能实现多个容器共享一次 RTNL。
- 需要维护 BIRD fork 和配置兼容性。

### 方案 C：BIRD netlink 多消息批量发送

思路是把多条 `nlmsghdr` 拼到同一个 netlink buffer，通过一次 `sendmsg()` 发给内核。

预期收益：

- 减少 syscall 次数。
- 减少用户态到内核态切换。
- 可以减少等待 ACK 的往返开销。

限制：

- Linux 内核仍然会按 netlink message 逐条处理路由。
- RTNL 是否能减少加锁次数取决于内核处理路径，不能假设一定从 N 次锁变成 1 次锁。
- ACK 和错误处理复杂，必须能定位批次中哪条路由失败。

建议实现方式：

1. 先实现不带批量 ACK 的实验版，只用于可丢弃/可重试的仿真环境。
2. 再实现按 sequence number 收集 ACK 的正式版。
3. 如果批次失败，自动降级为单条重放，便于定位错误路由。

这个方案适合作为节流队列之后的优化，不建议作为第一个改动。

### 方案 D：宿主机集中式 FIB writer

这是比深改 BIRD 更适合仿真平台的方案。

思路：

1. 容器内 BIRD 只负责控制面和 RIB 计算。
2. `protocol kernel` 默认不导出全量 BGP 路由。
3. 宿主机进程从每个 BIRD 读取待安装路由。
4. 宿主机进程通过 `setns()` 进入目标容器 netns。
5. 由一个或少量 writer 按队列写入每个 netns 的 FIB。

路由来源可以选：

- `birdc show route export kernel`。
- BIRD MRT dump。
- BIRD pipe/table 导出。
- 修改 BIRD 添加本地 socket，把 kernel export 操作发送给宿主机 writer。

写入方式可以选：

- `pyroute2` 进入 netns 写路由。
- C/C++ writer 使用 `setns()` + raw netlink。
- Rust/Go netlink 库配合 netns 切换。

优点：

- 把 10000 个并发写者变成 1 个或少数几个受控写者。
- 不需要改内核。
- 更容易做全局调度、限速、优先级和观测。

缺点：

- 要实现 BIRD RIB 到 host writer 的可靠同步。
- 需要处理容器生命周期、netns 句柄、失败重试。
- 如果 writer 单进程过慢，需要按 netns 分片成少量 writer。

这是推荐的工程路线：先集中调度，再决定是否需要改 BIRD 或内核。

## 方向二：改 Linux 内核

### 方案 E：新增 bulk route netlink API

这个方案解决“一个 netns 内一次锁写多条路由”。

思路：

1. 新增一个内核接口，例如 `RTM_NEWROUTES_BULK` 或 generic netlink family。
2. 用户态一次提交多条 route 描述。
3. 内核解析整个批次。
4. 对同一个 netns 获取一次 RTNL。
5. 在锁内循环执行 FIB insert/delete/replace。
6. 返回每条路由的执行结果。

用户态可以由两类程序使用：

- 改过的 BIRD。
- 宿主机集中式 FIB writer。

设计重点：

- 批次最大大小必须有限制，避免长时间持锁。
- 需要 per-route 错误码，不能只返回批次成功/失败。
- replace/delete/add 的顺序要明确。
- 失败是否回滚要明确；仿真环境可以接受部分成功，但生产语义更复杂。
- IPv4 和 IPv6 可以分阶段实现，先做 IPv4 主路径。

优点：

- 真正减少同一 netns 内的 RTNL 加锁次数。
- 可以配合 host writer，把写入性能提升明显。

缺点：

- 不是上游标准 API。
- 需要维护自定义内核和用户态工具。
- 不能天然解决多个 netns 之间的全局 RTNL 争用，除非同时改锁模型。

### 方案 F：per-netns RTNL 或分片 RTNL

这是最彻底但风险最高的方向。

目标是把全局 RTNL 拆成：

- 每个 netns 一个 RTNL 锁；或
- 一组 netns 共享一个 RTNL 分片锁；或
- 对 route-only 操作使用更细粒度锁，对 netdevice/link 操作保留全局锁。

理论收益：

- 不同容器/netns 的 FIB 写入可以并行。
- 10000 个路由器不再全部排在同一个全局 RTNL 后面。

主要困难：

- Linux 网络栈里大量路径默认假设 RTNL 是全局串行化点。
- route、nexthop、netdevice、neighbor、notifier chain、xfrm、lwtunnel 等路径可能隐含依赖全局 RTNL。
- 锁顺序设计错误会引入死锁。
- 跨 netns 操作、设备迁移、veth pair、bridge、VRF 等场景很容易踩到边界条件。

更现实的实验路线：

1. 不要一开始全局替换 `rtnl_mutex`。
2. 先只针对 IPv4 route add/delete 的仿真路径做局部优化。
3. 保留 link/device 变更使用全局 RTNL。
4. route-only 路径尝试 per-netns 锁。
5. 用 lockdep、KASAN、KCSAN 和大规模 SeedEMU 压测验证。

优点：

- 从根本上解决跨容器 RTNL 争用。

缺点：

- 工程量大，风险高。
- 需要维护自定义内核。
- 调试成本远高于 BIRD/host writer 方案。

### 方案 G：多个 netns 一次锁批量写入

用户提出的“调用一次 RTNL 锁，实现多个路由器一起把路由写入 FIB”可以作为自定义内核实验，但不建议作为第一阶段。

原因：

- 每个 router 容器对应不同 netns。
- 单个 netlink socket 通常在当前进程所在 netns 上下文处理请求。
- 一次请求跨多个 netns 写 FIB，需要用户态传入目标 netns 标识，内核查找并持有多个 `struct net` 引用。
- 如果仍使用全局 RTNL，一次持锁时间会非常长，可能让其他网络操作全部阻塞。
- 如果使用 per-netns 锁，又需要设计多个 netns 锁的排序规则，避免死锁。

可行但高风险的实验设计：

1. 自定义 char device 或 generic netlink family。
2. 请求中包含多组 `{netns_id, route_batch}`。
3. 内核按 netns id 排序，固定顺序加锁。
4. 每个 netns 内批量写入。
5. 返回 `{netns_id, route_index, errno}` 结果数组。

这个方案适合研究型验证，不适合作为当前工程优先方案。

## 当前实验代码中的 D/E/F/G 实现

当前仓库把 D/E/F/G 做成了两类可执行实验：

1. `bird/rtnl_solution_lab.py` 中的 table-252 原型实验，用合成 blackhole route 写入隔离路由表，主要用于快速验证调度方式和 RTNL 写入形态。
2. `bird/rtnl_solution_lab.py --real-bird` 中的真实 main-FIB 实验，从每个容器内 BIRD 的真实 RIB 读取 selected routes，然后写入默认 main FIB。

真实实验的公共流程如下：

1. 通过 Docker compose project label 找到当前项目的 brdnode 容器，只处理名称匹配 `^as\d+brd-r` 的路由器容器。
2. 读取容器 init PID，用 `nsenter -t <pid> -n -m` 进入目标容器 network namespace。
3. 对每个目标容器执行 `birdc show route primary`，解析 BIRD 当前 selected IPv4 routes。
4. 将每条 BIRD route 转成 Linux `ip route` batch line：
   - 单下一跳：`route replace <prefix> via <nexthop> dev <dev> proto bird metric 32`
   - 直连 dev：`route replace <prefix> dev <dev> proto bird metric 32`
   - ECMP：`route replace <prefix> proto bird metric 32 nexthop ...`
5. 通过 `ip -force -batch -` 写入目标 netns 的 main FIB。
6. 写完后抽样执行 `ip route | wc -l` 和 `birdc show route count`，验证 FIB route 行数是否满足实验设定的 RIB/FIB 条件。

相关核心函数：

- `docker_router_containers()`：发现运行中的 brdnode 容器和 PID。
- `real_bird_route_batch()`：执行 `birdc show route primary` 并生成 route batch。
- `real_bird_write_batch()`：在目标 netns 内执行 `ip -force -batch -`。
- `run_real_parallel()`：并行执行 dump+write。
- `run_real_transaction()`：先 dump 全部容器，再按 PID 排序后写入。
- `validate_real_solution()`：load gate + FIB/RIB 抽样验证。

### 当前 D：集中式受控 host writer

代码入口：

```bash
python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution D \
  --real-bird
```

实现函数：

- `real_solution_d()`
- `run_real_parallel(..., workers=args.d_workers)`

具体实现：

1. 可选执行 `cleanup_main_proto_bird()`，清理旧的 `proto bird` route。
2. 宿主机 Python 进程维护一个全局任务队列。
3. 默认 `--d-workers 1`，即同一时刻只有一个 host worker 进入某个容器 netns 写 FIB。
4. 每个任务对一个容器执行完整的 `birdc show route primary -> 解析 -> ip -batch 写入`。
5. 所有容器串行或低并发地完成，因此把原来上千个 BIRD 并发写者压缩成极少数受控 host writer。

实验含义：

- D 验证“集中调度/限流”是否可以减少 RTNL 锁风暴。
- 它不追求最快，而是追求最可控。
- D 的吞吐受 `--d-workers` 控制；默认 1 个 writer 时 RTNL 并发最低。

当前局限：

- 每个容器仍需要全量 dump RIB。
- 默认每个容器全量写 selected routes，不做增量 diff。
- 单 worker 模式下总时间可能较长。

### 当前 E：单 netns bulk batch 写入

代码入口：

```bash
python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution E \
  --real-bird
```

实现函数：

- `real_solution_e()`
- `run_real_parallel(..., workers=args.e_workers)`

具体实现：

1. 对每个容器读取 BIRD selected routes。
2. 将该容器内所有 route 合并成一个 `ip -force -batch -` 输入。
3. 每个容器只启动一次 `ip -batch`，而不是每条 route 启动一次 `ip route replace`。
4. 默认 `--e-workers 1`，所以也是受控 host writer，但每个 netns 内是批量提交。

实验含义：

- E 验证“每个 netns 内 bulk 写入”能否减少进程创建、shell 调用和用户态/内核态往返。
- 它不是内核级 bulk API；Linux 内核仍会逐条处理 batch 中的 route message。
- 它是用户态可立即实现的 bulk analogue。

当前局限：

- `ip -batch` 不能保证内核只获取一次 RTNL。
- route 级错误通过 `ip` 输出定位，批次错误处理比单条命令复杂。
- 仍然是全量写入，除非配合增量 diff 脚本。

### 当前 F：分片 netns writer

代码入口：

```bash
python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution F \
  --real-bird \
  --f-shards 8
```

实现函数：

- `real_solution_f()`
- `run_real_parallel(..., workers=args.f_shards)`

具体实现：

1. 宿主机启动 `--f-shards` 个并发 worker，当前常用值为 8。
2. 每个 worker 从全局任务队列中取一个 brdnode 容器。
3. 对该容器执行 `birdc show route primary`，生成这个容器自己的 route batch。
4. 进入该容器 netns，执行 `ip -force -batch -` 写 main FIB。
5. 多个容器的 FIB 写入可以同时进行，但并发数被限制在 8，而不是让 1600/1900 个 BIRD 同时写。

实验含义：

- F 是当前真实 main-FIB 实验中最快的通用全量写方案。
- 它模拟“少量分片 writer 受控地进入多个 netns 写 FIB”。
- 相比 D/E，F 提高吞吐；相比原始 BIRD 同时 export，它控制了并发上限。

配套脚本：

```bash
./bird/run_f_real_main_fib_vm.sh writeFIB_F
./bird/reconfergence_F.sh
```

`run_f_real_main_fib_vm.sh writeFIB_F` 的语义：

- 不 build/up。
- 不重新 start BIRD。
- 不等待 `birdc show route primary count` 稳定。
- 不重写 BIRD 配置。
- 默认 `--no-pre-cleanup --no-cleanup-after`，即在已有基础路由上补写 FIB 并保留现场。

`reconfergence_F.sh` 的语义：

- 先 stop 10 个不同 AS 的 brdnode，模拟网络波动。
- 等待固定时间让 BIRD 自己做 RIB 重收敛。
- 默认 `PRE_CLEANUP=1`，先 flush stale `proto bird` route。
- 再用 F 对当前仍运行的 brdnode 全量重写 main FIB。
- 记录 `f_duration_seconds`、`f_route_ops`、抽样验证结果。

当前局限：

- 全量 F 每次仍会写入约百万级 route。
- 故障后如果只需要撤销/替换少量路由，全量 F 会做大量无变化写入。
- 因此新增了增量版本 `reconfergence_delta_F.sh`。

### 当前 F-delta：基于 RIB snapshot diff 的增量 F

代码入口：

```bash
./bird/reconfergence_delta_F.sh
```

实现位置：

- `bird/reconfergence_delta_F.sh`
- 内嵌 Python 中的 `dump_all()`、`dump_fib_all()`、`diff_one()`、`write_delta()`

具体实现：

1. 故障前对运行中的 brdnode 做 baseline snapshot：
   - 每个容器执行 `birdc show route primary`
   - 解析成 `{container_name -> {prefix -> route_line}}`
2. stop 10 个不同 AS 的 brdnode。
3. 等待 `POST_DOWN_SLEEP_SECONDS`，让 BIRD 自行进行 RIB 重收敛。
4. 故障后再次 dump 当前运行 brdnode 的 BIRD RIB。
5. 故障后 dump 当前 main FIB 中的 `proto bird` route：
   - 每个容器执行 `ip -o route show proto bird`
   - 解析出当前 FIB 中真实存在的 prefix 集合
   - 用于过滤 RIB diff 中已经不存在的 delete 操作
6. 对每个容器做 prefix 级 diff：
   - `added`：当前 RIB 有、baseline 没有，生成 `ip route replace`
   - `changed`：prefix 存在但 nexthop/route line 变化，生成 `ip route replace`
   - `deleted`：baseline 有、当前 RIB 没有，并且当前 FIB 仍存在该 prefix，才生成 `ip route del ... proto bird metric 32`
   - `unchanged`：不写
7. 用 `F_SHARDS=8` 并行执行 delta batch。
   - 默认 `WRITE_MODE=setns-ip`
   - 远端 Python 默认以 `sudo -n env ... python3 -` 运行
   - worker 进程用 `setns()` 直接进入目标 netns，再调用 `ip -force -batch -`
   - 这避免了每个容器写入时反复启动 `sudo nsenter`
8. 对 deleted route，如果 `ip route del` 返回 `RTNETLINK answers: No such process`，按成功处理，因为目标状态就是该 route 不存在。

实验含义：

- F-delta 验证“从 RIB 变化中计算 FIB delta”能否减少 RTNL 写入条数。
- 它比全量 F 更接近规范的数据面更新语义。
- 当前实现仍需要故障后全量 dump RIB 来计算 diff；真正生产级方案应该由 BIRD 直接输出 route change event。

已验证观测：

- 旧版 delta：`bird/runs/reconfergence-delta-F_20260611_113725_topology-291e63c1/summary.json`
  - `baseline_dump=547.191193s`
  - `current_dump=547.311924s`
  - `delta_write=335.358637s`
  - `route_ops=59826`
  - 慢点主要来自重复 `sudo nsenter` 和每容器固定启动成本。
- 优化后 delta：`bird/runs/reconfergence-delta-F_20260611_135340_topology-291e63c1/summary.json`
  - `baseline_dump=15.765287s`
  - `current_dump=15.691294s`
  - `fib_dump=3.858498s`
  - `diff=0.577724s`
  - `delta_write=4.439854s`
  - `route_ops=35834`
  - `delete_filtered=999`
  - `write_mode=setns-ip`
  - FIB/RIB 抽样 `10/10` 通过。
- 全量 F 基线：`bird/runs/reconfergence-F_20260611_092737_topology-291e63c1/summary.json`
  - `F duration=98.123879s`
  - `route_ops=2056716`
- 因此当前 delta 写入阶段已经小于全量 FIB 重写时间，且写入条数也明显减少。

仍需注意：

- 当前 F-delta 仍依赖故障前/故障后两次 RIB snapshot。
- 如果要进一步降低总重收敛时间，应让 BIRD 直接输出 route change event，或维护常驻 host-side RIB/FIB cache，避免每次故障后再全量 `birdc show route primary`。

### F-event：常驻 agent 直接收集 route change event

新增实验入口：

```bash
./bird/reconfergence_event_F.sh
```

这轮实验先验证了 BIRD CLI 是否能直接作为 selected-route 事件源：

- `birdc monitor all`：BIRD 2.17.2 中不可用。
- `birdc debug all { routes, events }` + `echo all`：能看到普通 debug log，但实测没有稳定、可解析的 prefix-level selected route delta。
- `birdc dump events`：只输出内部 event 指针，不包含 prefix/nexthop。
- `mrt dump`/`mrtdump`：不是 master table selected-route delta，不能替代 route change event。

因此当前可运行原型采用 staging kernel table 作为事件桥：

```text
master4
  -> protocol pipe delta252_pipe
  -> BIRD table t_delta252
  -> protocol kernel delta252
  -> Linux kernel table 252
  -> ip -ts monitor route
  -> agent 解析 table 252 netlink route events
  -> 生成 main FIB 的 ip route batch
```

实现位置：

- `bird/reconfergence_event_F.sh`
- 内嵌 Python 中的 `configure_staging()`、`start_monitors()`、`parse_monitor_log()`、`write_main()`

关键行为：

1. 对 watcher brdnode 临时写入 `/etc/bird/conf/delta252.conf`。
2. `birdc configure` 后，BIRD 通过 `delta252_pipe` 把 `master4` selected routes 复制到 `t_delta252`。
3. `protocol kernel delta252` 把 `t_delta252` export 到 Linux table 252。
4. 常驻 agent 在每个 watcher netns 中运行 `ip -ts monitor route`。
5. 发生 route add/delete/change 时，agent 直接从 netlink event 中解析 prefix/nexthop。
6. agent 把 `table 252` 事件转换成 main-FIB batch：
   - add/change -> `route replace ... proto bird metric 32`
   - delete -> `route del <prefix> proto bird metric 32`
7. 默认 `WRITE_MAIN=0` 只生成 batch；设置 `WRITE_MAIN=1` 才写 main FIB。

已验证实验：

- selftest：`bird/runs/event-delta-F_20260611_165427_topology-291e63c1/summary.json`
  - 1 个 watcher
  - `prefix_events=2001`
  - `collapsed_ops=675`
  - parse 耗时 `0.017795s`
- failure：`bird/runs/event-delta-F_20260611_165456_topology-291e63c1/summary.json`
  - 20 个 watcher
  - 停止 3 个非 watcher brdnode
  - 没有做故障后全量 `birdc show route primary`
  - `prefix_events=3072`
  - `collapsed_ops=2724`
  - parse 耗时 `0.026229s`

这个方案的意义：

- 证明“常驻 agent 直接拿 delta route”是可行的。
- delta 获取成本从全量 RIB dump 变为 netlink event parse，解析耗时是毫秒级。
- 生成的 batch 已经是 main-FIB 可执行格式，不包含 `table 252`。

限制：

- 当前原型仍让 BIRD 先写 staging kernel table 252，所以它不是完全无 RTNL。
- 它适合验证事件式 delta 获取链路；若要彻底解决 RTNL 争抢，最终应改 BIRD 源码，在 selected route change 处直接输出事件到 Unix socket/共享内存/文件，由 host writer 统一写 main FIB。

### 最终 F-event：直接改 BIRD 源码输出 selected-route delta

源码目录：

```bash
bird/bird-2.17.2-delta/
```

核心实现：

- 新增 `proto/delta/` 协议模块。
- 在 `protocol.h` 中注册 `PROTOCOL_DELTA` 和 `proto_delta`。
- 在 `configure.ac` 的 `all_protocols` 中加入 `delta`。
- `protocol delta` 通过 `P->rt_notify = delta_rt_notify` 接入 BIRD core 的路由通知路径。
- `delta_postconfig()` 默认把 channel 设置为 `RA_MERGED`，因此收到的是合并后的 selected route，
  更接近原 kernel protocol `merge paths on` 的导出语义。
- `delta_rt_notify()` 收到 `new/old` 后，只把最终 delta 写入事件文件，不写 kernel FIB。

配置示例：

```bird
protocol delta delta_main {
    event "/run/bird/delta-events.tsv";
    metric 32;
    merge paths on limit 16;
    ipv4 {
        table master4;
        export all;
    };
}
```

事件格式：

```text
ts_us  R  proto  table  replace|delete  prefix  dest  metric  nh_count  [gw iface flags weight]...
```

写入 main FIB：

```bash
python3 bird/delta_fib_agent.py \
  --event-file /run/bird/delta-events.tsv \
  --follow \
  --from-end \
  --stats-file /run/bird/delta-fib-agent.json
```

agent 的行为：

- 解析 BIRD 事件文件。
- 将同一 prefix 在短 flush 窗口内的多次变化 collapse 成最后一次。
- 对 IPv4/IPv6 分别调用 `ip -4/-6 -force -batch -`。
- `replace` 转换为 `route replace ... proto bird metric 32`。
- `delete` 转换为 `route del ... proto bird metric 32`。

VM/容器集成脚本：

```bash
./bird/install_bird_delta_vm.sh install
./bird/install_bird_delta_vm.sh restart-bird
./bird/install_bird_delta_vm.sh write-initial
./bird/install_bird_delta_vm.sh start-agent
```

推荐流程：

1. build/up 容器。
2. `install` 分发 patched BIRD 和 agent，并写入 `delta.conf`。
3. `restart-bird` 使用 patched BIRD 启动控制面，并清空旧事件文件。
4. 等 RIB 收敛后执行 `write-initial` 初始化 main FIB。
5. 执行 `start-agent`，后续网络故障只消费 BIRD 的实时 delta event。

与 table-252 原型相比：

- 不再有 `pipe -> kernel table 252 -> ip monitor` 链路。
- BIRD 不再为了生成事件而写 staging FIB，因此少一层 RTNL 争抢。
- 故障后不需要全量 `birdc show route primary` dump。
- 后续主要成本变成事件解析、prefix collapse 和 main-FIB batch 写入。

### 当前 G：dump 后按 netns/PID 顺序事务化写入

代码入口：

```bash
python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution G \
  --real-bird
```

实现函数：

- `real_solution_g()`
- `run_real_transaction()`

具体实现：

1. 第一阶段只 dump，不写：
   - 用 `--real-dump-workers` 并发对所有 brdnode 执行 `birdc show route primary`
   - 生成每个容器的 route batch
2. 将 dump 结果按目标容器 init PID 升序排序。
3. 第二阶段写入：
   - 按排序后的目标 netns 顺序提交写入任务
   - 默认 `--g-shards 1`，即单 writer 顺序写
   - 可调大 `--g-shards` 做有序分片实验
4. 写完后执行 load gate 和 FIB/RIB 抽样验证。

实验含义：

- G 模拟“先构建全局写入计划，再按固定 netns 顺序提交”的事务化 writer。
- 这个顺序对未来内核多 netns 锁实验有价值：如果需要一次操作多个 netns，必须有确定的锁顺序避免死锁。
- 当前 G 并没有真正把多个 netns 放进一个内核事务；它是在用户态按 PID 顺序调度。

当前局限：

- 不是内核级 multi-netns transaction。
- 默认单 writer 时速度接近 D/E，而不是 F。
- 优势主要是可预测顺序和便于研究锁顺序，不是绝对吞吐。

## 推荐实施路线

### 第一阶段：不改内核，快速缓解

1. 初始 `kernel.conf` 只导出 device/OSPF 或 `export none`。
2. 控制面收敛后由宿主机 orchestrator 分批 reload kernel protocol。
3. 加入全局令牌桶，限制同时处于 kernel reload 的容器数量。
4. 按 AS/区域分组，加入随机 jitter。
5. 用 eBPF 统计 RTM_NEWROUTE/RTM_DELROUTE 速率和 RTNL 等待时间。

目标：

- 验证锁风暴是否主要来自同时 reload。
- 找到每批容器数量和 route 写入速率上限。

### 第二阶段：实现 host FIB writer

1. BIRD 容器内不直接 export 全量 BGP 到 kernel。
2. 宿主机收集 BIRD export routes。
3. 单 writer 先实现正确性。
4. 再实现多 writer 分片，但总并发保持很小。
5. 记录每个 netns 的写入耗时、失败路由、重试次数。

目标：

- 把 10000 个竞争 RTNL 的 BIRD 进程变为 1 到 16 个受控 writer。

### 第三阶段：改 BIRD

1. 给 kernel protocol 加 pending queue 和 timer flush。
2. 加配置项控制 batch/interval/jitter。
3. 支持同 prefix 操作合并。
4. 支持失败重试和单条降级。

目标：

- 即使继续让容器内 BIRD 写 FIB，也能避免所有容器同一时刻猛烈写入。

### 第四阶段：内核实验

优先顺序：

1. 实现单 netns route bulk API。
2. 配合 host FIB writer 使用。
3. 如仍不够，再研究 route-only per-netns RTNL。
4. 最后才考虑一次请求跨多个 netns 写入。

## 实验指标

每个方案都应记录：

- 总控制面收敛时间。
- 开始写 FIB 到完成写 FIB 的时间。
- 每秒 `RTM_NEWROUTE`/`RTM_DELROUTE` 数量。
- RTNL 等待时间分布。
- 每个容器最终 FIB 路由数量。
- `birdc show protocols` 状态。
- `birdc show route count`。
- 系统 load、CPU iowait、上下文切换。
- BIRD 进程内存和 CPU。
- netlink 错误，例如 `ENOBUFS`、`ESRCH`、`EEXIST`、`EINVAL`。

## 当前建议

短期工程建议：

1. 继续强化现有 `start_bird_kernel.py` 这类分阶段脚本。
2. 增加全局批次控制和 jitter。
3. 实现 host FIB writer 原型。

中期源码建议：

1. fork BIRD 2.17.2。
2. 在 kernel protocol 加 export queue。
3. 再评估 netlink 多消息批量发送。

长期研究建议：

1. 自定义内核 bulk route API。
2. route-only per-netns RTNL 实验。
3. 对比自定义内核与 host FIB writer 的收益和维护成本。
