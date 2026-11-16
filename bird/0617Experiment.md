# 2026-06-17 F 方案与 Delta FIB 写入实验汇总

生成时间：2026-06-17

本文整理当前已经完成的 F 方案相关实验，包括：

- F 方法初始全量写入 FIB 的流程和多规模结果。
- BIRD delta route event 方案在网络波动后的增量写入结果。
- full `birdc show route primary` 对照实验结果。
- full writer 并发数实验。
- 100 个 router 每个写入 600,000 条 route 的合成 FIB 写入实验。

## 术语

- `start-bird` / `restart_bird`：启动容器内 BIRD，让 BIRD 按已有配置跑控制面并形成 RIB。这个阶段不让 BIRD kernel protocol 全量导出所有 route 到 main FIB。
- F 方法初始全量写入：等 RIB 稳定后，宿主机 writer 进入各个 router namespace，执行：
  - `birdc show route primary`
  - 解析 selected route 和 nexthop
  - 生成 `ip route replace ... proto <metric>`
  - 用 `ip -force -batch -` 写入目标 namespace 的 main FIB
- delta route 写入：使用修改后的 BIRD 记录 route change event。网络波动后，等 RIB 稳定，再读取 fault 之后的 delta event，折叠同一 prefix 的多次变化，只把变化后的 route ops 写入 FIB。
- full 对照写入：网络波动后不使用 delta event，直接对每个运行中的 brdnode 执行 full `birdc show route primary`，生成全量 route ops 后写入 FIB。
- `jobs`：宿主机 writer 并发数。`jobs=4` 表示最多同时 4 个 writer 各自进入不同 namespace 写 FIB，不是 4 个 writer 共同写同一个 namespace 的 FIB。

## 实验流程

### F 初始全量写入流程

对应脚本：

- `bird/run_bird_delta_experiment.py`
- 多规模封装：`bird/run_scale_f_reconf_experiments.py`

流程：

1. 在 VM 上 build/up SeedEMU 输出的 Docker 网络。
2. 启动 BIRD，先让 RIB 收敛。
3. 等待稳定门：
   - 早期多规模实验使用 `load_threshold=50` 和 route count 稳定检查。
   - 6 月 17 日严格实验使用 `load_threshold=40`、`load_only=true`，并保持 `restart_bird` 串行、`restart_delay=0.1s`。
4. 用 F 方法从 BIRD RIB 读取 selected routes，全量写入 Linux main FIB。
5. 抽样验证 RIB/FIB route count。

### 网络波动后的 delta 写入流程

对应脚本：

- `bird/run_strict_delta_full_experiment.py`
- `bird/delta_fib_host_writer.py`

严格实验流程：

1. 初始阶段启动 BIRD，RIB 稳定后用 F 方法做一次全量 FIB 同步。
2. 启用 BIRD delta event 记录，并记录 checkpoint。
3. 停止 10 个不同 AS 的 brdnode，模拟网络波动。
4. 等 RIB 稳定。
5. 读取 checkpoint 之后的 BIRD delta event。
6. 折叠 event，生成最终需要执行的 route ops。
7. 用 `ip -batch` 写入 main FIB。

### full 对照流程

严格对照实验使用单独 VM arm，不复用 delta arm，避免 fault 前预知故障：

1. 同样启动 BIRD、等 RIB 稳定、用 F 方法做初始全量 FIB 同步。
2. 停止同一组 10 个 brdnode。
3. 等 RIB 稳定。
4. 不读取 delta event，直接全量执行 `birdc show route primary`。
5. 先单独统计 full dump 时间，再单独统计 command-files 写 FIB 时间。

这个设计用于区分：

- 卡在 `birdc show route primary` dump。
- 还是卡在 `ip -batch` 写 FIB。

## 多规模 F 初始全量与 Delta Reconvergence 结果

实验目录：

- `bird/runs/F_reconf_multiscale_20260614_094101`

参数摘要：

- build mode：`delta-fast`
- 初始 F 写入：`source=birdc-primary`，`route_action=replace`
- 初始 F writer：`host-setns`
- 初始 F `jobs=8`
- reconvergence delta writer：`jobs=8`
- start-bird：当时使用 `restart_jobs=2`
- 稳定门：`load_threshold=50`，`stable_seconds=120`

| scale | services | build(s) | up(s) | start-bird(s) | RIB stable wait(s) | 初始 F route ops | 初始 F 写入(s) | 故障后 delta raw events | delta route ops | delta 写入(s) | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2599 | 2601 | 125.355 | 2284.493 | 350.413 | 672.252 | 4,142,556 | 287.089 | 237,428 | 128,477 | 88.079 | 成功 |
| 3083 | 3085 | 182.336 | 3160.299 | 337.945 | 300.207 | 6,131,994 | 579.296 | 365,367 | 198,830 | 225.951 | 成功 |
| 4192 | 4194 | 223.800 | 6464.404 | 710.797 | 未完成 | n/a | n/a | n/a | n/a | n/a | start-bird 后等待稳定时 VM 高负载/SSH timeout |
| 4954 | 4956 | 281.132 | 10150.515 | 1680.201 | 未完成 | n/a | n/a | n/a | n/a | n/a | start-bird 后等待稳定时 VM 高负载/SSH timeout |

观察：

- 2599 和 3083 能完成初始全量 FIB 写入和故障后的 delta 写入。
- 4192 和 4954 并不是 FIB 写入失败，而是在 `start-bird` 完成后等待 RIB/load 稳定时 VM 进入高负载、SSH 不稳定，因此没有进入初始 F 写入阶段。
- 3083 的初始 F 写入 route ops 约为 2599 的 `1.48x`，但写入时间约为 `2.02x`，说明大规模时单条 route 的平均写入成本也在上升。

## 6 月 17 日严格 Delta-vs-Full 对照实验

实验目录：

- 1897：`bird/runs/strict_delta_full_1897_20260617_085002`
- 3083：`bird/runs/strict_delta_full_3083_20260617_090018`

关键参数：

- `restart_bird` 串行：`jobs=1`
- `restart_delay=0.1s`
- RIB 稳定门：`load_threshold=40`
- 初始 F 写入：`jobs=8`
- delta event dump：`jobs=8`
- delta write：`jobs=4`
- full dump：`jobs=4`
- full write：`jobs=2`
- delta write batch：`batch_lines=1000`，`batch_sleep_ms=2`
- full write batch：`batch_lines=1000`，`batch_sleep_ms=5`

### 初始启动与 F 全量同步

| scale | actual services | arm | build(s) | up(s) | start-bird(s) | RIB wait(s) | 初始 F route ops | 初始 F 写入(s) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|

| 3083 | 3085 | delta | 185.276 | 3213.121 | 963.640 | 88.493 | 6,131,994 | 549.996 |
| 3083 | 3085 | full | 183.643 | 3353.283 | 993.744 | 88.341 | 6,131,994 | 520.676 |

说明：

- 1897 这个 run 的实际 output 很小，只有 `216` 个 compose services，不是 1897 个 brdnode；因此它的 FIB 写入时间不能和 3083 线性比较。
- 3083 严格实验中，串行 `start-bird` 约 `16min`，初始 F 全量写入约 `8.7-9.2min`。
- 3083 初始 F route ops 是 `6.13M`，是后续性能分析里的核心规模。

### 网络波动后的 delta 与 full 对照

| scale | actual services | 方法 | discovery/dump(s) | raw events / route ops | collapsed/write ops | FIB 写入(s) | writer jobs | 失败数 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 3083 | 3085 | delta event | 6.434 | 748,028 | 561,377 | 238.320 | 4 | 0 |
| 3083 | 3085 | full birdc-primary | 15.732 | 6,097,653 | 6,097,653 | 2025.107 | 2 | 0 |

核心结论：

- 3083 下，full `birdc show route primary` dump 只用了 `15.732s`，不是瓶颈。
- 3083 full 对照真正慢在 FIB 写入：`2025.107s`。
- 3083 delta event 把写入量从 `6,097,653` 降到 `561,377`，减少约 `10.9x`。
- 3083 delta FIB 写入时间 `238.320s`，相比 full 对照 `2025.107s` 快约 `8.5x`。
- delta 的收益主要来自减少 FIB route ops，而不是让单条 route 写入更快。

## Full Writer 并发数实验

实验目录：

- `bird/runs/full_write_jobs_sweep_3083_20260617_160611`

这轮实验在 3083 的同一份 post-fault full command files 上重放，目的是观察 `ip -batch` writer 并发的拐点。

准备阶段：

| 阶段 | 数值 |
|---|---:|
| actual services | 3085 |
| start-bird | 971.070s |
| 初始 F route ops | 6,131,994 |
| 初始 F 写入 jobs=8 | 538.511s |
| full dump route ops | 6,097,653 |
| full dump jobs=4 | 18.634s |

重放结果：

| writer jobs | route ops | FIB 写入(s) | 结果 | 观察 |
|---:|---:|---:|---|---|
| 4 | 6,097,653 | 2511.169 | 成功 | VM load 约 10-18，CPU 仍有大量 idle，0 batch errors |
| 8 | 6,097,653 | >4000 | 手动终止 | 多个 `ip -4 -force -batch -` 长时间 D-state，SSH/proc 枚举尾延迟明显 |
| 12/16/24/32 | n/a | n/a | 未继续 | `jobs=8` 已经越过负优化拐点 |

解释：

- `jobs=4` 的含义是最多 4 个 writer 同时进入 4 个不同 namespace 写 FIB。
- 它不是让内核真正并行修改同一个 FIB，也不能绕过 Linux RTNL/FIB 更新路径中的共享锁和临界区。
- 当 jobs 从 4 增加到 8 时，吞吐没有提高，反而出现 D-state 和更差 wall time。
- 这说明瓶颈不是 CPU 算力；更像是 RTNL/FIB 更新路径的并发竞争。

## 100 个 Router 每个写入 600,000 条 Route 的合成实验

实验目录：

- `bird/runs/synthetic_fib_bench_100x600k_20260617`

脚本：

- `bird/synthetic_fib_bench.py`

实验方法：

1. 在 VM 上创建临时 Linux netns：`fibbench-r000` 到 `fibbench-r099`。
2. 每个 netns 设置 `lo up`。
3. 每个 netns 写入 `600,000` 条 `/32` route：
   - `route replace 10.x.y.z/32 dev lo proto 232 metric 232`
4. 用 `ip -n <ns> -4 -force -batch -` 一次大 batch 写入。
5. 使用 `jobs=4`，即同时最多 4 个 netns 写入。
6. 实验结束后删除所有临时 netns。

结果：

| 场景 | routers/netns | 每 router route 数 | 总 route ops | jobs | 写入时间(s) | 吞吐(routes/s) | 单 router 平均写入(s) | CPU avg | max load1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 校准 | 1 | 600,000 | 600,000 | 1 | 40.052 | 14,980.5 | 39.609 | 17.3% | 6.50 |
| 主实验 | 100 | 600,000 | 60,000,000 | 4 | 3456.237 | 17,359.9 | 138.196 | 25.8% | 21.25 |

观察：

- 100 个 router 每个 600k route，总量是 `60M`，写入时间是 `57m36s`。
- 单个 netns 写 600k 约 `40s`，但 4 个大 FIB writer 并发时，单 router 平均写入时间变成约 `138s`。
- 这个合成实验绕过了 BIRD、Docker exec、真实 nexthop 解析、命令文件分散等开销，仍然超过 `2500s`，说明总 route ops 和大 FIB 表写入成本本身已经足够重。
- 写入期间 CPU 没有打满，load1 也不高；这和 full writer jobs sweep 一致，说明主要瓶颈不是 CPU，而是内核 FIB/RTNL 更新路径和单 namespace 大 FIB 表成本。

## 综合分析

### 1. F 方法解决的是“瞬时写者风暴”，不是减少总写入量

原始问题是大量 BIRD 实例同时通过 kernel protocol 写 FIB，形成 RTNL 锁争抢。F 方法把写入改成宿主机集中调度：

- BIRD 先完成 RIB 收敛。
- 宿主机在稳定后读取 RIB。
- 用有限 writer 数写入 FIB。

这避免了“所有容器同时写 FIB”的最坏情况，但如果最终仍然要写 `6M`、`60M` route ops，写入时间仍然受内核 FIB 更新路径限制。

### 2. delta route event 的主要价值是减少 FIB ops

3083 严格实验中：

- full 对照：`6,097,653` route ops，FIB 写入 `2025.107s`。
- delta：`561,377` route ops，FIB 写入 `238.320s`。

route ops 减少约 `10.9x`，写入时间减少约 `8.5x`。这说明 delta 方案方向是正确的：与其试图把 full 写入并发无限调高，不如减少必须写入的 route ops。

### 3. full dump 不是 3083 的主要瓶颈

3083 full 对照中：

- full `birdc show route primary` dump：`15.732s`
- full FIB write：`2025.107s`

所以慢的不是从 BIRD RIB 获取 route，而是写 Linux main FIB。

### 4. writer jobs 存在拐点

`jobs=4` 和 `jobs=8` 的实验说明：

- `jobs=4` 可以稳定完成，但耗时长。
- `jobs=8` 在 3083 full command-files 重放中明显负优化，超过 `4000s` 仍未完成，并出现 D-state。

因此，`jobs` 不是越大越好。`jobs=4` 的意义是用户态同时驱动 4 个 namespace 写 FIB，但内核中的 RTNL/FIB 路径仍会竞争。高并发会把竞争放大，导致 CPU idle 但吞吐下降。

### 5. 少 router、多 route 不一定更快

100 个 router 每个 600k route 的实验说明：

- router 数减少可以降低跨容器调度和连接开销。
- 但每个 namespace 的 FIB 表变大后，单 namespace 写入成本明显上升。
- `60M` route ops 即使是极简 `dev lo` route，也需要 `3456s`。

所以性能不能只看 router 数，还必须同时看：

- 总 route ops。
- 每个 namespace 的 FIB 表规模。
- writer jobs。
- route 类型和 nexthop 复杂度。
- 是否使用 delta 减少写入。

## 当前结论

1. 当前 F 方法已经把失控的“每个容器都自己抢 RTNL”改成了受控的宿主机集中写入。
2. F 方法全量写入在 3083 规模约 `520-579s`，route ops 约 `6.13M`。
3. 网络波动后，delta route 写入在 3083 规模约 `238s`，route ops 约 `561k`。
4. 网络波动后 full 对照在 3083 规模中，dump 只需 `15.7s`，但 FIB 写入需要 `2025s`。
5. full writer 并发不能盲目增加；`jobs=8` 在 3083 full 重放中已经触发明显负优化。
6. 100x600k 合成实验进一步说明：即使 router 数少，只要总 route ops 很大、每个 namespace FIB 很大，写入仍然会很慢。

## 后续建议

优先级建议：

1. 继续完善 BIRD delta event 方案，减少 route ops。
2. 保持 full writer jobs 在实测稳定区间，当前 3083 full 写入不建议超过 `jobs=4`。
3. 对 delta writer 也做 `jobs=2/4/6/8` 小范围拐点测试，避免 delta 规模增大后踩到同类 RTNL/FIB 拐点。
4. 如果必须继续提升 full 写入能力，优先考虑多 VM 分片，让不同 VM 拥有不同 kernel/RTNL，而不是在单 kernel 内盲目增加 writer。
5. 内核方向可以继续研究 per-netns RTNL 或 bulk route API，但这是长期工程，不应作为当前实验提速的第一依赖。

## 主要 Artifact

- 多规模 F/reconvergence：`bird/runs/F_reconf_multiscale_20260614_094101`
- 1897 strict delta/full：`bird/runs/strict_delta_full_1897_20260617_085002`
- 3083 strict delta/full：`bird/runs/strict_delta_full_3083_20260617_090018`
- 3083 full writer jobs sweep：`bird/runs/full_write_jobs_sweep_3083_20260617_160611`
- 100x600k synthetic FIB bench：`bird/runs/synthetic_fib_bench_100x600k_20260617`
- 详细调查：`bird/fib_write_anomaly_investigation.md`
