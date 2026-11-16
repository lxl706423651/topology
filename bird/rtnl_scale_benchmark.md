# RTNL Scale Benchmark

`rtnl_scale_benchmark.py` 用 Linux network namespace 模拟大量 router 节点，
用于比较 D/E/F/G 四个方案在大规模 FIB 写入场景下的速度。

默认目标规模：

```bash
5000 nodes x 10000 routes/node = 50000000 route writes/solution
```

脚本写入的是 table `252` 中的 synthetic blackhole `/32` 路由，不改 main table。

## 方法映射

| 方法 | 默认执行方式 |
| --- | --- |
| D | 1 个集中式 host writer 顺序进入各 netns 写入。 |
| E | 每个 netns 使用一次 `ip -batch` 批量提交 10000 条路由，默认 8 workers。 |
| F | 按 netns 分片并发写入，默认 30 shards/workers，模拟 per-netns/sharded RTNL 方向。 |
| G | 按 netns 序号排序后执行多 netns 有序事务，默认 1 worker。 |

这些是当前 Linux 内核上的可执行对照实验。F/G 的 summary 里同时记录
`ideal_all_netns_independent_seconds` 和 `actual_over_ideal_ratio`，用于估算如果内核
真的实现 per-netns/sharded RTNL 或 multi-netns one-lock bulk writer 后的空间。

## 小规模校准

```bash
ssh lxl@192.168.122.60
python3 /home/lxl/bird/rtnl_scale_benchmark.py \
  --nodes 20 \
  --routes-per-node 1000 \
  --solutions D,E,F,G \
  --cleanup-netns
```

## 5000x10000 正式比较

```bash
ssh lxl@192.168.122.60
python3 /home/lxl/bird/rtnl_scale_benchmark.py \
  --nodes 5000 \
  --routes-per-node 10000 \
  --solutions D,E,F,G \
  --d-workers 1 \
  --e-workers 8 \
  --f-shards 30 \
  --g-workers 1 \
  --progress-interval 25 \
  --load-sample-interval 5
```

每次运行会生成：

```bash
/home/lxl/bird/runs/rtnl-scale_5000x10000_<timestamp>/summary.json
```

每个方法的 per-node 结果会写入同目录下的 JSONL，例如：

```bash
D_D-centralized-host-fib-writer.jsonl
E_E-per-netns-bulk-route-batch.jsonl
F_F-sharded-per-netns-rtnl-proxy.jsonl
G_G-sorted-multi-netns-transaction.jsonl
```

宿主机拉回结果：

```bash
mkdir -p bird/runs
rsync -a \
  lxl@192.168.122.60:/home/lxl/bird/runs/rtnl-scale_5000x10000_<timestamp>/ \
  bird/runs/rtnl-scale_5000x10000_<timestamp>/
```
