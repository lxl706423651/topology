# FIB 全量写入异常耗时调查

生成时间：2026-06-17

## 背景

这次调查针对两个看起来不合理的现象：

1. 3083 规模下，最开始 `start-bird` 后执行 F 方法全量写入 FIB 只用了约 `520s`，但故障收敛后再做 full 对照全量写入用了 `2025s`。
2. 1897 到 3083 的 full FIB 写入时间差距极大，1897 约 `1s`，3083 约 `2025s`，看起来不符合直觉。

本报告使用现有严格实验结果做复盘和离线复算：

- 1897：`bird/runs/strict_delta_full_1897_20260617_085002`
- 3083：`bird/runs/strict_delta_full_3083_20260617_090018`

## 结论摘要

这两个现象的主因不是故障后内核 FIB 写入突然变慢，也不是 `birdc show route primary` dump 变慢，而是实验参数和实际拓扑规模没有做到直观可比。

核心结论：

1. `3083 initial full = 520s` 和 `3083 post-fault full = 2025s` 的差异，几乎完全由 writer 并发数不同解释：
   - initial full：`jobs=8`，`batch_lines=0`，`batch_sleep_ms=0`
   - post-fault full：`jobs=2`，`batch_lines=1000`，`batch_sleep_ms=5`
   - 两者总写入工作量几乎相同，`write_seconds.sum` 分别约 `4043s` 和 `4036s`
   - 用 post-fault full 的逐容器 `write_seconds` 复算：如果用 `jobs=8`，理论 wall time 约 `504s`，和 initial full 的 `519s` 基本一致
2. 1897 很快，是因为这个 run 实际非常小：
   - 1897 output 只有 `216` 个 compose service，其中约 `118` 个 brdnode
   - 3083 output 有 `3085` 个 compose service，其中约 `2757` 个 brdnode
   - 1897 post-fault full 只写 `17,412` 条 route ops
   - 3083 post-fault full 写 `6,097,653` 条 route ops，是 1897 的 `350x`
3. 3083 的真正瓶颈是 FIB 写入，不是 RIB dump：
   - full `birdc-primary` dump：`15.732s`
   - full FIB write：`2025.107s`
   - delta dump：`6.434s`
   - delta FIB write：`238.320s`

## 关键数据

### 3083：initial full vs post-fault full

| 阶段 | jobs | batch_lines | batch_sleep_ms | route ops | wall time | write_seconds.sum | read/dump sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| initial full FIB | 8 | 0 | 0 | 6,131,994 | 518.714s | 4,043.084s | 58.664s |
| post-fault full dump | 4 | 0 | 0 | 6,097,653 | 15.732s | 0s | 39.190s |
| post-fault full FIB write | 2 | 1000 | 5 | 6,097,653 | 2025.107s | 4,035.586s | 2.993s |

解释：

- initial full 和 post-fault full 的 route ops 数量几乎一样：`6.13M` vs `6.10M`
- 两者 FIB 写入总工作量也几乎一样：`4043s` vs `4036s`
- wall time 不一样，是因为 initial 用 `8` 个 writer 并发，而 post-fault full 对照只用 `2` 个 writer 并发

### 用 post-fault full details 做并发复算

使用 `full/full_write/command-files_write_details.json` 中每个容器的 `write_seconds`，按 greedy scheduling 复算不同并发下的理论 wall time：

| jobs | 复算 wall time |
|---:|---:|
| 1 | 4035.586s |
| 2 | 2017.796s |
| 4 | 1008.909s |
| 8 | 504.466s |
| 16 | 252.239s |

实测 post-fault full write：`2025.107s`，和 `jobs=2` 复算值 `2017.796s` 非常接近。

因此 `2025s` 不是故障后 FIB 写入退化，而是 `2` 并发设置下的正常结果。如果把 post-fault full write 改成 `8` 并发，预计会接近 `500s`，和 initial full 基本一致。

### batch sleep 不是主因

post-fault full write 使用：

- `batch_lines=1000`
- `batch_sleep_ms=5`
- 总 chunks：`8112`
- sleep events：`5365`
- sleep 预算：约 `26.825s`

这只占 `2025s` 的很小一部分。主因仍是 `jobs=2`。

## 1897 为什么特别快

1897 和 3083 的实际拓扑规模不是按名字线性增长。

| run | compose services | brdnode services | post-fault full brdnodes | post-fault full route ops |
|---|---:|---:|---:|---:|
| 1897 | 216 | 118 | 108 | 17,412 |
| 3083 | 3085 | 2757 | 2747 | 6,097,653 |

比例：

- brdnode 数：`2747 / 108 = 25.4x`
- total route ops：`6,097,653 / 17,412 = 350.2x`
- 每个 brdnode 平均 route ops：约 `2219.7 / 161.2 = 13.8x`

所以 3083 不是比 1897 多 1.6 倍，而是在这个实验输出里：

- 容器级 brdnode 数多了约 `25x`
- 每个 brdnode 的路由量又多了约 `14x`
- 总写入 route ops 多了约 `350x`

### 单条 route 写入成本也变高

post-fault full write 的逐容器统计：

| run | route ops | write_seconds.sum | 平均每条 route 写入时间 |
|---|---:|---:|---:|
| 1897 | 17,412 | 1.110s | 63.8 us/op |
| 3083 | 6,097,653 | 4,035.586s | 661.8 us/op |

3083 的单条 route 写入成本约为 1897 的 `10.4x`。

这说明大规模时不只是 route ops 数量变多，单个 namespace 里的 FIB 表也更大，`ip route replace` 写入的内核处理成本、`ip -batch` 调用成本、netlink 应答处理等也明显放大。

## 真正瓶颈在哪里

本轮严格实验中，瓶颈非常明确：

1. `birdc show route primary` dump 不是瓶颈。
   - 3083 full dump 只用了 `15.732s`
   - 即使 full dump 生成了 `6,097,653` 条命令，也远小于 FIB write 时间
2. full FIB write 是瓶颈。
   - 3083 post-fault full write 用了 `2025.107s`
   - 主要时间来自每个容器执行 `ip -force -batch -` 写 main FIB
3. 本轮没有证据说明 post-fault full write 比 initial full write 在内核层面退化。
   - initial full 的 `write_seconds.sum = 4043.084s`
   - post-fault full 的 `write_seconds.sum = 4035.586s`
   - 总写入工作量几乎一致
   - wall time 差异由并发数解释

## 对 RTNL 拥塞判断的影响

如果 RTNL 锁在本轮 full writer 中已经成为严格全局串行瓶颈，那么 `jobs=8` 不应该比 `jobs=2` 快接近 4 倍。

但当前数据是：

- `jobs=8` initial full：约 `519s`
- `jobs=2` post-fault full：约 `2025s`
- post-fault details 复算 `jobs=8`：约 `504s`

这说明在当前 writer 参数和 3083 规模下，主要瓶颈表现为“可并行的逐 namespace FIB 写入工作量”，而不是完全不可并行的 RTNL 全局串行锁等待。

这不代表更高并发或更大规模下不会出现 RTNL 拥塞，只是说明本轮 `2025s` 这个异常现象不能归因于 RTNL 退化。

## 追加实测：post-fault full writer jobs sweep

追加实验时间：2026-06-17 下午

为验证“把 post-fault full write 改成更高 writer 并发是否会更快”，在同一份 3083 post-fault `birdc-primary` command files 上做了重放实验：

- run dir：`bird/runs/full_write_jobs_sweep_3083_20260617_160611`
- remote run dir：`/home/lxl/bird/runs/bird-delta-3085_20260617_160621_topology-3083-sweep`
- command files：`13,735` 个文件，约 `480M`
- full dump：`18.634s`，生成 `6,097,653` 条 FIB 写入命令
- 写入参数：`batch_lines=1000`，`batch_sleep_ms=5`

结果：

| jobs | 结果 | wall time | route ops | 观察 |
|---:|---|---:|---:|---|
| 4 | 成功 | `2511.169s` | `6,097,653` | VM load 约 `10-18`，CPU idle 多数在 `65-74%`，0 batch errors |
| 8 | 手动终止 | `>4000s` 未完成 | 未生成 summary | 多个 `ip -4 -force -batch -` 长时间处于 `D` 状态，SSH/proc 枚举尾延迟明显，load 约 `17-23` |
| 12 | 未执行 | - | - | `jobs=8` 已经劣于 `jobs=4`，继续放大没有意义 |
| 16 | 未执行 | - | - | 同上 |
| 24/32 | 未执行 | - | - | 同上 |

这次追加实测修正了前面“按逐容器 `write_seconds` 做 greedy scheduling 复算”的乐观判断。那个复算默认每个容器的写入成本不随并发变化，但真实运行中，writer 并发提高后，多个 `ip -batch` 同时进入内核 FIB/RTNL 路径，会增加锁竞争和 D-state 等待，导致单个容器的 `write_seconds` 不再保持不变。

因此，当前 3083 command-files 重放实验中，实测可用结论是：

- `jobs=4` 是本轮成功且温和的点，但耗时较长：`2511s`
- `jobs=8` 已经越过拐点：没有缩短 wall time，反而超过 `jobs=4` 后仍未完成，并显著拉长 SSH/proc 交互尾延迟
- 触发阈值不是 CPU 或内存耗尽；CPU 仍大量 idle，内存也充足
- 更像是 RTNL/FIB 更新路径的并发竞争：`ip -batch` 进程出现 D-state，load 只升到十几二十几，但吞吐下降

对后续实验的含义：

1. 不应继续盲目把 full writer jobs 调到 `12/16/24/32`。在 3083 的 full main-FIB replace workload 下，`8` 已经是负优化。
2. full FIB 对照如果要稳定完成，应先使用 `jobs=4` 或更保守参数。
3. 真正提升方向仍然是减少 FIB route ops 数量，也就是 delta/event 方案；单靠提高 full writer 并发会遇到 RTNL/FIB 竞争拐点。
4. 如果后续要继续研究拐点，可以在更短的样本上做 `jobs=1/2/4/6/8`，并采集 `ip -batch` D-state 数、`max_ssh_seconds`、`system/softirq CPU`，而不是直接全量跑 `12/16/24/32`。

## 追加实测：100 个 router 每个写入 600,000 条

追加实验时间：2026-06-17 晚上

为验证“如果 router 数少，但每个 router 路由量很大，是否会明显快于 2500s”，增加了一个不依赖 BIRD/Docker 的合成 FIB 写入基准：

- 脚本：`bird/synthetic_fib_bench.py`
- VM 脚本路径：`/home/lxl/bird/synthetic_fib_bench.py`
- 本地 summary：`bird/runs/synthetic_fib_bench_100x600k_20260617/summary_jobs4.json`
- 方法：创建 `fibbench-r000` 到 `fibbench-r099` 共 100 个临时 netns，每个 netns 执行一次大 `ip -n <ns> -4 -force -batch -`
- 每个 netns 写入 `600,000` 条 `/32` route，总计 `60,000,000` 条 route 命令
- route 形式：`route replace 10.x.y.z/32 dev lo proto 232 metric 232`
- writer 并发：`jobs=4`
- cleanup：实验结束后删除 100 个临时 netns，删除成功

结果：

| 项目 | 数值 |
|---|---:|
| routers/netns | `100` |
| routes per router | `600,000` |
| total route commands | `60,000,000` |
| jobs | `4` |
| write wall time | `3456.237s` |
| write wall time | `57m 36s` |
| aggregate throughput | `17,359.9 routes/s` |
| successful routers | `100/100` |
| failed routers | `0` |
| per-router write time min | `132.377s` |
| per-router write time avg | `138.196s` |
| per-router write time max | `143.749s` |
| max load1 during write | `21.25` |

校准结果：

| 场景 | wall time | throughput |
|---|---:|---:|
| 1 个 netns 写 600,000 条，jobs=1 | `40.052s` | `14,980.5 routes/s` |
| 100 个 netns 各写 600,000 条，jobs=4 | `3456.237s` | `17,359.9 routes/s` |

这说明：

1. `100 * 600,000 = 60M` 的总写入量本身很大，不会自然低于 `2500s`。
2. 单个空 netns 写 600k 约 `40s`，但 4 个大 FIB 写入同时跑时，每个 netns 的写入时间变成约 `138s`，单个 writer 的效率下降约 `3.5x`。
3. 这轮合成测试已经绕过了 BIRD dump、Docker exec、容器内脚本分发、复杂 nexthop 解析等开销；仍然需要 `3456s`，说明大头在内核 FIB 更新路径和大 FIB 表自身的插入/replace 成本。
4. 该实验是在当前 VM 仍保留上一轮 3083 拓扑容器的背景下运行的，因此它不是“空机极限基准”。但它反映了当前实验 VM 的实战背景负载，并且负载监测显示写入期间 load1 最高只有 `21.25`，没有出现 3083 jobs=8 command-files 重放时那种明显 D-state 堵塞。
5. 删除 100 个 netns 后，VM load 出现过短暂高峰，随后从 `800+` 降到 `235`、再降到 `99`；`fibbench` netns 和进程均已清理。

与 3083 full command-files 重放的关系：

- 3083 full command-files 重放：约 `6.10M` route ops，`jobs=4` 用 `2511s`
- 合成 100x600k：`60M` route ops，`jobs=4` 用 `3456s`

合成测试单条 route 的平均吞吐明显更好，因为它是极简 `dev lo` 路由、每个 netns 一次大 batch，不需要 `birdc`、Docker、分散命令文件和真实 nexthop 处理。但总 route 数是 3083 full 对照的约 `9.8x`，所以总耗时仍然超过 `2500s`。

因此，“100 个 router 每个 600k”并不是天然更快。router 数减少能减少跨容器/跨 namespace 调度开销，但每个 namespace 的 FIB 规模变大后，单 namespace 写入成本会显著上升；最终耗时主要由总 route ops、每个 namespace 的 FIB 表规模、以及 `ip -batch` 并发下的内核 FIB/RTNL 路径竞争共同决定。

## 对 delta 方案的实际意义

3083 post-fault 对比：

| 方法 | dump/discovery | FIB write | route ops |
|---|---:|---:|---:|
| delta event | 6.434s | 238.320s | 561,377 |
| full birdc-primary | 15.732s | 2025.107s | 6,097,653 |

delta 的收益主要来自减少 FIB 写入数量：

- full 需要写 `6.10M` 条
- delta 只写 `0.56M` 条
- route ops 减少约 `10.9x`
- write wall time 减少约 `8.5x`

即使把 full write 改成与 initial 相同的 `jobs=8`，预计 full write 约 `500s`，仍然比 delta write `238s` 慢约 2 倍。

## 发现的问题

### 问题 1：full 对照参数不公平

当前严格实验里：

- initial full sync 用 `--initial-write-jobs 8`
- post-fault full 对照用 `--full-write-jobs 2`

因此不能直接拿 `520s` 和 `2025s` 比较来判断故障后 full 写入退化。

更公平的对照应该至少增加两组：

1. post-fault full write with `jobs=8`, `batch_lines=0`, `batch_sleep_ms=0`
2. initial full write with `jobs=2`, `batch_lines=1000`, `batch_sleep_ms=5`

这样才能隔离：

- 并发数影响
- batch chunk/sleep 影响
- 故障前后内核状态影响

### 问题 2：1897 不是大规模代表

1897 run 实际只有 `216` 个 service 和 `17k` 级 full route ops。这个输出与 3083 的 `3085` 个 service、`6.1M` full route ops 不是线性关系。

后续报告不要只写“1897/3083 规模”，应该同时写：

- compose service 数
- brdnode 数
- full route ops 数
- delta route ops 数

否则很容易误读性能曲线。

### 问题 3：load-only RIB gate 对小规模可能过弱

1897 initial full sync 的 route ops 只有：

- delta arm initial：`5,140`
- full arm initial：`6,541`

但 post-fault full dump 有 `17,412` route ops。

这可能意味着 1897 小规模下 `load < 40` 很快满足，但 RIB route 数还没完全稳定。虽然这不影响“1897 为什么快”的结论，因为 post-fault full 也只有 `17k` ops，但后续若要严谨比较 RIB 完整性，应在小规模也加入 route count 稳定门或抽样 route count 验证。

## 建议的下一轮实验

为了彻底确认参数影响，建议下一轮在 3083 上只跑 FIB writer 参数矩阵，不必重跑 build/up：

1. 同一份 post-fault full command files，分别跑：
   - `jobs=2, batch_lines=1000, sleep=5`
   - `jobs=4, batch_lines=1000, sleep=5`
   - `jobs=8, batch_lines=1000, sleep=5`
   - `jobs=8, batch_lines=0, sleep=0`
2. 每组记录：
   - wall time
   - write_seconds.sum
   - batch_errors
   - VM load
   - 是否触发 SSH timeout
3. 若 `jobs=8` 稳定跑到约 `500s`，则可以确认这次 `2025s` 完全是对照参数造成的。

不过基于现有 details 的复算结果，这个结论已经有较强证据。

## 复现/核验命令

核心数据抽取：

```bash
python3 - <<'PY'
import json, os, math, heapq

def greedy(times, jobs):
    heap = [0.0] * jobs
    for t in sorted(times, reverse=True):
        x = heapq.heappop(heap)
        heapq.heappush(heap, x + t)
    return max(heap)

r = 'bird/runs/strict_delta_full_3083_20260617_090018'
d = json.load(open(os.path.join(r, 'full/full_write/command-files_write_details.json')))
times = [x.get('write_seconds') or 0.0 for x in d]
print('write_sum', sum(times))
for jobs in [1, 2, 4, 8, 16]:
    print(jobs, greedy(times, jobs))
PY
```

实际结果：

```text
write_sum 4035.586s
jobs=1  4035.586s
jobs=2  2017.796s
jobs=4  1008.909s
jobs=8   504.466s
jobs=16  252.239s
```

这就是 `2025s` 的核心解释。
