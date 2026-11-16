# RTNL Solution Lab

`rtnl_solution_lab.py` 是 D/E/F/G 四个 RTNL 方案的可执行实验原型。脚本应该在
VM 内运行，它按 Docker Compose project label 找到正在运行的 SeedEMU router
容器，进入目标容器 network namespace，并向独立路由表 `252` 写入 synthetic
blackhole `/32` 路由。

这些路由不会写入 main table，实验结束默认会清理 table `252`。

## 方案映射

| 方案 | 脚本模式 | 实现含义 |
| --- | --- | --- |
| D | `--solution D` | 宿主机集中式 FIB writer 原型。比较大量并发 writer 和一个受控 host writer。 |
| E | `--solution E` | bulk route API 原型。比较单条 `ip route replace` 和一次 `ip -batch` 批量提交。 |
| F | `--solution F` | per-netns/sharded RTNL 原型。比较串行写入和按 shard 并发写入，并估算理想 per-netns RTNL 耗时。 |
| G | `--solution G` | 多 netns 有序事务原型。按目标 netns PID 排序后执行批量写入，模拟固定锁顺序。 |

F 当前不是内核补丁，而是在现有内核上执行可观测的 sharding 对照实验，用来量化
“如果 route-only RTNL 能按 netns/shard 并行”可能获得的收益。

## 常用命令

先确认 VM 中已有运行中的 topology：

```bash
ssh lxl@192.168.122.60
docker ps --filter label=com.docker.compose.project=topology-291e63c1
```

只清理实验路由表：

```bash
python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --limit-containers 4 \
  --routes-per-netns 2 \
  --cleanup-only
```

必须验证的 D/F：

```bash
python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution D \
  --limit-containers 8 \
  --routes-per-netns 4 \
  --d-workers 1 \
  --d-baseline-workers 8

python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution F \
  --limit-containers 8 \
  --routes-per-netns 4 \
  --f-shards 4
```

全套 D/E/F/G 小样本：

```bash
python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution all \
  --limit-containers 16 \
  --routes-per-netns 8 \
  --d-workers 1 \
  --d-baseline-workers 16 \
  --f-shards 4 \
  --g-shards 1
```

每次运行都会生成：

```bash
/home/lxl/bird/runs/rtnl-solutions_<timestamp>/summary.json
```

宿主机可以拉回本地：

```bash
mkdir -p bird/runs
rsync -a \
  lxl@192.168.122.60:/home/lxl/bird/runs/rtnl-solutions_<timestamp>/ \
  bird/runs/rtnl-solutions_<timestamp>/
```

## 本轮验证结果

本轮全套小样本 summary 已保存到：

```bash
bird/runs/rtnl-solutions_20260609_134922/summary.json
```

关键结果：

| 方案 | 耗时/结果 | 失败数 |
| --- | --- | --- |
| D baseline | 3.493172s | 0 |
| D controlled | 4.896590s | 0 |
| E per-route | 2.537247s | 0 |
| E bulk-batch | 0.313913s, speedup 8.082644x | 0 |
| F serial | 4.913803s | 0 |
| F sharded | 3.443218s, actual/ideal 0.280582 | 0 |
| G ordered transaction | 4.910360s | 0 |

VM 负载在本轮小样本期间保持低位，没有触发 destroy/start/reset 策略。
