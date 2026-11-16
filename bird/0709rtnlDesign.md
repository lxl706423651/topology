# 0709 RTNL/FIB 方法流程与参数梳理

本文档基于本机包目录：

`/home/lxl/seed-emulator/topology/bird/packaged/rtnl_fib_methods_repro_20260701_133858/`

梳理日期：2026-07-09。这里只描述包内脚本的实际控制流与默认参数；IDC 上运行时通常还会通过外层 prompt 覆盖 `VM_HOST`、`VM_USER`、`PROJECT_NAME`、`OUTPUT_DIR`、`RESET_VM` 等变量。

## 总览

四个方法的主入口如下：

| 方法 | 主入口 | reconvergence 入口 | 目标 kernel/假设 |
|---|---|---|---|
| F | `F/run_f_real_main_fib_vm.sh` | `F/reconfergence_delta_F.sh` | stock kernel；脚本本身不校验 kernel |
| K1 | `K1/run_3083_k1_delta_reconvergence.sh` | `K1/reconvergence_bird_delta.sh` | K1 bulk kernel；包内 prebuilt 为 `6.8.12-birdrtbulk-k1-global-rtnl.tar.gz` |
| K2c | `K2c/run_3083_k2c_delta_optimized.sh` | `K2c/reconvergence_bird_delta.sh` | `6.8.12-birdrtbulk-k2c` |
| K3c | `K3c/run_3083_k3c_delta_experiment.sh` | `K3c/reconvergence_bird_delta.sh` | wrapper 默认 `6.8.12-birdrtbulk-k3c`；包内另有修复版 `6.8.12-birdrtbulk-k3c-fibref.tar.gz` |

关键差异：

- F 的 `start-bird` 是 `run_f_real_main_fib_vm.sh` 内联 Python 实现，不走 `start-clean-bird`。
- K 系列的主 runner `run_bird_delta_experiment.py` 会先执行 `probe-bird-state`，然后把 summary 里的阶段记为 `restart_bird`：
  - 如果探测为 `bird_not_started`，实际 installer phase 是 `start-clean-bird`。
  - 如果探测为 `bird_running` 或 `unknown`，实际 installer phase 是 `restart-bird`。
- K1 目录的 `start-clean-bird` 仍包含较多清理与 ready check；K2c/K3c 目录的 `start-clean-bird` 已经简化为只执行 `bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf`。
- K2c/K3c wrapper 默认开启 start/restart load gate：串行启动时每 50 个 brdnode 检查一次 `load1 < 50`。K1 目录当前版本没有这个 start/restart load gate。
- K 系列 wrapper 的初始 FIB 写入都通过 `--fib-writer k1-bulk`，即使用 `kernel_bulk_birdc_writer.py` 路径。K2c/K3c 是在不同内核实现下复用同一个 bulk writer 接口。

## 通用 build/up 逻辑

`F/run.py`、`K1/run.py`、`K2c/run.py`、`K3c/run.py` 的基础默认值基本一致：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `DEFAULT_BUILD_MODE` | `fast` | 普通 fast build 模式 |
| `DEFAULT_BUILD_BATCH` | `50` | build batch |
| `DEFAULT_BUILD_JOBS` | `16` | build 并发 |
| `DEFAULT_UP_BATCH` | `50` | compose up batch |
| `DEFAULT_UP_JOBS` | `4` | compose up 并发 |

K3c 的 `run.py` 额外从环境变量读取 `UP_BATCH`、`UP_JOBS` 作为默认值；K1/K2c/F 的 `run.py` 默认值是静态值。

K 系列的当前主流程通常不直接使用 `run_all.py` 的 start-bird 逻辑。`run_bird_delta_experiment.py` 会把 `run_all.py` 同步到远端 run 目录，但 build/up/start/write 的主控制流由 `run_bird_delta_experiment.py` 和 `install_bird_delta_vm.sh` 完成。

## F 方法

### 主流程

主入口：`F/run_f_real_main_fib_vm.sh`

`all` 只包含：

1. `run`
2. `start-bird`
3. `writeFIB_F`

reconvergence delta 需要单独调用 `F/reconfergence_delta_F.sh run`。

完整实验流程通常是：

1. `status`：查看远端 Docker/load/BIRD 概况。
2. `run`：同步 `OUTPUT_DIR` 到远端，build image，并 `docker compose up`。
3. `start-bird`：在 brdnode 容器中启动 BIRD。
4. `writeFIB_F`：用 F sharded writer 写 main FIB。
5. `reconfergence_delta_F.sh run`：停止部分 brdnode，dump 前后 RIB，diff 后写 delta。

### F start-bird 具体行为

F 的 `start-bird` 不叫 `start-clean-bird`，也不走 K 系列 installer。它会：

1. 找出当前 compose project 中运行的 `as\d+brd-r*` 容器。
2. 对每个容器先执行：
   - `birdc show status >/dev/null 2>&1`
3. 如果已经成功，则记录 `already_running=true`，不再启动。
4. 如果未运行，则执行：
   - `rm -f /run/bird/bird.pid /run/bird/bird.ctl; bird -p -c /etc/bird/bird.conf && bird`
5. 使用 `ThreadPoolExecutor(max_workers=START_BIRD_JOBS)` 并发启动。
6. 结果写入远端和本地 `runs/start-bird_<timestamp>_<project>/summary.json`。

这个阶段不会改写 `kernel.conf`，也不会执行 `birdc down` 或 `pkill bird`。

### F 主要参数

`run_f_real_main_fib_vm.sh` 默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `VM_HOST` | `192.168.122.60` | 远端主机 |
| `VM_USER` | `lxl` | 远端用户 |
| `VM_DOMAIN` | `bird-rtnl-dev` | 旧 VM reset 路径用 |
| `PROJECT_NAME` | `topology-291e63c1` | Docker compose project |
| `OUTPUT_DIR` | `${TOPOLOGY_DIR}/output` | topology output |
| `VM_RUN_ROOT` | `/home/lxl/bird/runs` | 远端 run 目录 |
| `LOCAL_RUN_ROOT` | `${SCRIPT_DIR}/runs` | 本地 run 目录 |
| `BUILD_JOBS` | `16` | build 并发 |
| `BUILD_BATCH` | `50` | build batch |
| `UP_JOBS` | `8` | wrapper 默认 up 并发 |
| `UP_BATCH` | `50` | up batch |
| `START_BIRD_JOBS` | `16` | start-bird 并发 |
| `START_BIRD_TIMEOUT` | `180` | 单容器 start 超时 |
| `START_BIRD_DELAY` | `0` | 单任务后延迟 |
| `F_SHARDS` | `8` | F writer shard 数 |
| `REAL_DUMP_WORKERS` | `16` | real BIRD dump workers |
| `CLEANUP_WORKERS` | `16` | cleanup workers |
| `COMMAND_TIMEOUT` | `180` | 单命令超时 |
| `SUCCESS_LOAD_THRESHOLD` | `50` | 写后成功 load 阈值 |
| `SUCCESS_STABLE_SECONDS` | `120` | 写后 load 稳定秒数 |
| `SUCCESS_LOAD_TIMEOUT` | `7200` | 写后 load 等待超时 |
| `SAMPLE_COUNT` | `10` | 验证采样数 |
| `LIMIT_CONTAINERS` | `0` | 0 表示不限制 |

`writeFIB_F` 实际调用远端：

```bash
sudo -n python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project "$PROJECT_NAME" \
  --solution F \
  --real-bird \
  --no-real-prepare-bird \
  --no-pre-cleanup \
  --work-dir "$remote_dir" \
  --timeout "$COMMAND_TIMEOUT" \
  --no-pre-write-rib-stable \
  --success-load-threshold "$SUCCESS_LOAD_THRESHOLD" \
  --success-stable-seconds "$SUCCESS_STABLE_SECONDS" \
  --success-load-timeout "$SUCCESS_LOAD_TIMEOUT" \
  --sample-count "$SAMPLE_COUNT" \
  --f-shards "$F_SHARDS" \
  --real-dump-workers "$REAL_DUMP_WORKERS" \
  --cleanup-workers "$CLEANUP_WORKERS" \
  --no-cleanup-after
```

### F reconvergence 子阶段

入口：`F/reconfergence_delta_F.sh run`

默认参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `DOWN_COUNT` | `10` | 停止 brdnode 数 |
| `STOP_TIMEOUT` | `1` | `docker stop -t` 超时 |
| `SELECT_STRATEGY` | `spread` | 尽量跨 AS 选择 victim |
| `POST_DOWN_SLEEP_SECONDS` | `30` | stop 后等待 |
| `DUMP_WORKERS` | `32` | BIRD RIB dump workers |
| `FIB_DUMP_WORKERS` | `${DUMP_WORKERS}` | FIB dump workers |
| `F_SHARDS` | `8` | F writer shard 数 |
| `COMMAND_TIMEOUT` | `180` | 单命令超时 |
| `SAMPLE_COUNT` | `10` | 验证采样数 |
| `FIB_RIB_MIN_DELTA` | `0` | 最小 delta 约束 |
| `LOAD_SAMPLE_INTERVAL` | `0.2` | load 采样间隔 |
| `FIB_AWARE_DELETE` | `1` | 删除时参考当前 FIB |
| `WRITE_MODE` | `setns-ip` | 写 FIB 方式 |
| `REMOTE_ROOT_PYTHON` | `1` | 用 `sudo -n env` 跑远端 Python |

主要步骤：

1. dump 当前 BIRD RIB。
2. 选择并停止 `DOWN_COUNT` 个 brdnode。
3. sleep `POST_DOWN_SLEEP_SECONDS`。
4. dump 收敛后的 BIRD RIB。
5. 可选 dump 当前 FIB，用于 delete 过滤。
6. 对比前后 RIB，生成 add/change/delete delta。
7. 用 F 路径写入 main FIB。
8. 写 summary，并把远端结果 rsync 回本地。

## K 系列公共主流程

K1/K2c/K3c 的主 wrapper 最终都会进入各自目录下的 `run_3083_k1_delta_reconvergence.sh` 风格脚本，再调用 `run_bird_delta_experiment.py` 进行初始全量 FIB 写入，之后调用 `reconvergence_bird_delta.sh` 做 delta reconvergence。

### 初始阶段主 runner

入口：`K*/run_bird_delta_experiment.py`

当前 wrapper 传入的核心参数：

```bash
--build-mode "$BUILD_MODE"
--defer-delta
--skip-enable-delta-after-initial
--skip-start-agents
--skip-final-validate
--restart-jobs "$RESTART_JOBS"
--restart-delay "$RESTART_DELAY"
--load-only-stable
--load-threshold "$LOAD_THRESHOLD"
--stable-seconds 0
--max-wait-seconds "$MAX_WAIT_SECONDS"
--poll-interval "$POLL_INTERVAL"
--delta-jobs "$DELTA_JOBS"
--write-jobs "$INITIAL_WRITE_JOBS"
--fib-writer k1-bulk
--k1-dump-jobs "$K1_DUMP_JOBS"
--k1-batch "$K1_BATCH"
--k1-command-timeout "$K1_COMMAND_TIMEOUT"
--k1-multipath-policy "$K1_MULTIPATH_POLICY"
```

K2c/K3c 目录版本额外传入：

```bash
--restart-load-gate-every "$RESTART_LOAD_GATE_EVERY"
--restart-load-threshold "$RESTART_LOAD_THRESHOLD"
--restart-load-poll-interval "$RESTART_LOAD_POLL_INTERVAL"
--restart-load-max-wait "$RESTART_LOAD_MAX_WAIT"
```

`run_bird_delta_experiment.py` 的实际 stage 顺序：

1. `sync`
   - 创建远端 run 目录。
   - 同步 `OUTPUT_DIR`。
   - 同步 `run.py`、`run_all.py`、`install_bird_delta_vm.sh`、`delta_fib_agent.py`、`delta_fib_host_writer.py`、`verify_bird_fib_vm.py`、`build_delta_fast_images.py`、`1buildFastDockerImages.py`、kernel writer tools、patched BIRD binary。
2. `build`
   - `build-mode=delta-fast` 时调用 `build_delta_fast_images.py`，把 patched BIRD 和 agent baked into image。
   - `build-mode=fast` 时调用远端 `run.py --build-mode fast --no-up`。
   - `build-mode=reuse` 时跳过 build。
3. `up`
   - 调用远端 `run.py --no-build`。
   - 如果环境里有 `UP_BATCH`/`UP_JOBS`，会传给远端 `run.py`。
4. `probe_bird_state`
   - 通过 `install_bird_delta_vm.sh probe-bird-state` 抽样最多 20 个 brdnode。
   - `birdc -s /run/bird/bird.ctl show status` 成功、`birdc show status` 成功或 `pgrep -x bird` 命中，都算 `bird_running`。
5. `restart_bird`
   - summary stage 名固定为 `restart_bird`。
   - 如果 probe 结果是 `bird_not_started`，实际 phase 是 `start-clean-bird`。
   - 否则实际 phase 是 `restart-bird`。
6. `wait_rib_stable`
   - wrapper 使用 `--load-only-stable --stable-seconds 0`，所以这里只等 `load1 < LOAD_THRESHOLD`，不再要求 route count 连续稳定。
7. `validate_bird`
   - 抽样检查 BIRD status、`show route primary count`。
   - 因为初始阶段使用 `--defer-delta`，这里不要求 `delta_main` 已经 up。
8. `write_initial_fib`
   - wrapper 使用 `--fib-writer k1-bulk`。
   - 实际调用 `kernel_bulk_birdc_writer.py` 先从 BIRD dump route，再通过内核 bulk 接口写 main FIB。
9. `enable_delta`
   - wrapper 传了 `--skip-enable-delta-after-initial`，所以这里跳过。
10. `start_agents`
   - wrapper 传了 `--skip-start-agents`，所以这里跳过。
11. `validate`
   - wrapper 传了 `--skip-final-validate`，所以这里跳过。
12. `fetch_artifacts`
   - 把远端 summary/artifacts 拉回本地 run 目录。

初始阶段结束后，wrapper 还会继续：

1. `wait_after_initial_k1_fib`：等 VM load 低于阈值。
2. `verify_initial_fib`：调用远端 `verify_bird_fib_vm.py --stage fib-write`。
3. `enable_delta_after_initial_verify`：调用 `install_bird_delta_vm.sh enable-delta`，设置 `DELTA_IGNORE_INITIAL_FEED=1`。
4. `wait_after_enable_delta`：再次等 load。
5. `reconvergence_bird_delta.sh run`。
6. `summarize`：汇总 `main_summary.json`、`verify_initial_fib.json`、`reconvergence/summary.json` 等为 comparison summary。

### K 系列 start-clean-bird 与 restart-bird

K1 目录：

- `start-clean-bird` 不是单纯一条 `bird`：
  - `mkdir -p /run/bird`
  - 停掉旧 `delta_fib_agent`
  - 清空 `/run/bird/delta-events.tsv`
  - 删除 `/run/bird/bird.pid`、`/run/bird/bird.ctl`
  - 根据 `DELTA_ENABLE_ON_RESTART` 调整 `/etc/bird/conf/delta.conf`
  - `bird -p -c /etc/bird/bird.conf`
  - `bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf`
  - 最多 50 次、每次 0.2 秒执行 `birdc -s /run/bird/bird.ctl show status` ready check
- `restart-bird` 在上述基础上还会先：
  - `birdc down`
  - `pkill bird`
  - 等最多 50 次，每次 0.2 秒
  - 必要时 `pkill -9 bird`
- K1 installer 当前没有每 N 个容器一次的 restart load gate。

K2c/K3c 目录：

- `start-clean-bird` 已简化为：

```bash
docker exec "$cid" sh -lc 'set -e; bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf'
```

- `restart-bird` 仍是完整重启路径：
  - 停旧 agent
  - 清空 delta events
  - `birdc down`
  - `pkill bird`
  - 必要时 `pkill -9 bird`
  - 清理 pid/socket
  - 调整 delta conf
  - `bird -p -c /etc/bird/bird.conf`
  - `bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf`
  - ready check
- K2c/K3c installer 有 `restart_load_gate()`。只有串行启动，即 `DELTA_JOBS=1`/wrapper 的 `RESTART_JOBS=1` 时才会进入串行循环并生效。wrapper 默认每 50 个容器检查一次 load。

### K 系列 reconvergence 子阶段

入口：`K*/reconvergence_bird_delta.sh run`

默认参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `DOWN_COUNT` | `10` | 停止 brdnode 数 |
| `STOP_TIMEOUT` | `1` | `docker stop -t` 超时 |
| `SELECT_STRATEGY` | `spread` | victim 选择策略 |
| `METRIC` | `232` | BIRD/FIB metric |
| `LOAD_THRESHOLD` | `50` | load 阈值 |
| `LOAD_ONLY_STABLE` | `0` | 默认不只看 load；wrapper 覆盖为 `1` |
| `STABLE_SECONDS` | `120` | 默认 RIB 稳定秒数；wrapper 覆盖为 `0` |
| `MAX_WAIT_SECONDS` | `1800` | 默认稳定等待超时；wrapper 覆盖为 `7200` |
| `POLL_INTERVAL` | `10` | 默认轮询间隔；wrapper 覆盖为 `30` |
| `STABILITY_SAMPLES` | `20` | RIB 稳定采样容器数 |
| `VALIDATE_SAMPLES` | `10` | 验证采样容器数 |
| `INSTALL_JOBS` | `16` | agent install 并发 |
| `WRITE_JOBS` | `8` | delta dump/write 前半段并发 |
| `OFFSET_JOBS` | `${WRITE_JOBS}` | offset 准备并发 |
| `FULL_WRITE_JOBS` | `${WRITE_JOBS}` | full comparison 写入并发 |
| `RUN_FULL_COMPARISON` | `0` | 是否跑 full comparison |
| `COMMAND_TIMEOUT` | `180` | 普通命令超时 |
| `FULL_COMMAND_TIMEOUT` | `1800` | full comparison 超时 |
| `WRITE_BACKEND` | `host-setns` | wrapper 覆盖为 `k1-bulk` |
| `SKIP_AGENT_INSTALL` | `0` | wrapper 覆盖为 `1` |
| `USE_OFFSET_MARK` | `1` | wrapper 覆盖为 `0` |
| `K1_DELTA_JOBS` | `${WRITE_JOBS}` | bulk delta writer jobs |
| `K1_DELTA_BATCH` | `2000` | bulk delta batch |
| `K1_OPS_PER_ATTR` | `2000` | 单 attr ops |
| `K1_MULTIPATH_POLICY` | `first` | multipath 处理 |
| `K1_DELTA_MIXED_BATCHES` | `0` | 是否混合 replace/delete batch |

wrapper 运行 reconvergence 时实际覆盖：

```bash
WRITE_BACKEND=k1-bulk
SKIP_AGENT_INSTALL=1
USE_OFFSET_MARK=0
LOAD_ONLY_STABLE=1
STABLE_SECONDS=0
MAX_WAIT_SECONDS="$MAX_WAIT_SECONDS"
POLL_INTERVAL="$POLL_INTERVAL"
WRITE_JOBS="$DELTA_WRITE_JOBS"
OFFSET_JOBS="$OFFSET_JOBS"
K1_DELTA_JOBS="$DELTA_WRITE_JOBS"
K1_DELTA_BATCH="$K1_DELTA_BATCH"
K1_DELTA_MIXED_BATCHES="$K1_DELTA_MIXED_BATCHES"
```

`WRITE_BACKEND=k1-bulk` 的 reconvergence 写入路径：

1. 停止 selected brdnodes。
2. 等 load/RIB 稳定。wrapper 当前只要求 `load1 < LOAD_THRESHOLD`。
3. 通过 `delta_fib_host_writer.py --source event --dump-only` 从每个 brdnode 的 `/run/bird/delta-events.tsv` 读取 delta event，collapse 为每容器 command 文件。
4. 通过 `kernel_bulk_command_writer.py` 读取 commands dir，用内核 bulk 接口写入 main FIB。
5. 检查 `write_failures == 0`、`fail_count == 0`。
6. 可选执行 full comparison。
7. 写 `reconvergence/summary.json`。

## K1 方法

### 入口与 kernel

入口：`K1/run_3083_k1_delta_reconvergence.sh`

包内 prebuilt：

`K1/kernel/prebuilt/6.8.12-birdrtbulk-k1-global-rtnl.tar.gz`

注意：`K1/run_3083_k1_delta_reconvergence.sh` 本身不显式设置 `K1_KERNEL`，且 `run_3083_k1_bulk_real.sh` 的老默认值是 `6.8.12-birdrtbulk`。如果在 IDC 裸机上跑，应按实际安装的 K1 kernel 显式设置：

```bash
K1_KERNEL=6.8.12-birdrtbulk-k1-global-rtnl
RESET_VM=0
```

### K1 wrapper 默认参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `VM_HOST` | `192.168.122.60` | 默认远端 |
| `VM_USER` | `lxl` | 默认用户 |
| `PROJECT_NAME` | `topology-3083` | 默认 project |
| `OUTPUT_DIR` | `.../scale_3083/output` | 默认 3083 output |
| `RUN_NAME` | `k1-delta-reconv-3083_<timestamp>` | run name |
| `RESET_VM` | `1` | 会走旧 VM reset 流程；IDC 应覆盖为 `0` |
| `LOAD_THRESHOLD` | `50` | load 阈值 |
| `MAX_WAIT_SECONDS` | `7200` | load 等待上限 |
| `POLL_INTERVAL` | `30` | load 轮询 |
| `RESTART_JOBS` | `1` | start/restart 串行 |
| `RESTART_DELAY` | `0.1` | 每容器间隔 |
| `BUILD_MODE` | `delta-fast` | 初始 build 模式 |
| `SKIP_BUILD_UP` | `0` | 是否跳过 build/up |
| `FORCE_INSTALL` | `0` | delta-fast 下是否强制 install |
| `RESUME_AFTER_INITIAL` | `0` | 是否复用已有 initial result |
| `INITIAL_WRITE_JOBS` | `8` | 初始全量写并发 |
| `K1_DUMP_JOBS` | `16` | 初始 BIRD dump 并发 |
| `K1_BATCH` | `2000` | 初始 bulk batch |
| `K1_COMMAND_TIMEOUT` | `240` | 初始 bulk 命令超时 |
| `K1_MULTIPATH_POLICY` | `first` | multipath 处理 |
| `DELTA_JOBS` | `16` | install/enable 等 delta 并发 |
| `DOWN_COUNT` | `10` | reconvergence 停止节点数 |
| `DELTA_WRITE_JOBS` | `8` | delta 写并发 |
| `OFFSET_JOBS` | `${DELTA_WRITE_JOBS}` | offset 并发 |
| `K1_DELTA_BATCH` | `2000` | delta bulk batch |
| `K1_DELTA_MIXED_BATCHES` | `0` | 不混合 batch |
| `RUN_FULL_COMPARISON` | `0` | 默认不跑 full comparison |
| `FULL_WRITE_JOBS` | `8` | full comparison jobs |
| `FULL_COMMAND_TIMEOUT` | `1800` | full comparison 超时 |
| `SKIP_AGENT_INSTALL` | `1` | reconvergence 不重复安装 agent |
| `USE_OFFSET_MARK` | `0` | reconvergence 不用旧 offset mark 路径 |
| `SSH_CONNECT_TIMEOUT` | `30` | SSH timeout |

### K1 全流程

1. 创建 `RUN_DIR`。
2. 如果 `RESET_VM=1`，调用 `run_3083_k1_bulk_real.sh prepare-k1-reset`。
3. `run_initial_full_k1`：
   - `run_bird_delta_experiment.py`
   - build/up
   - probe BIRD
   - `start-clean-bird` 或 `restart-bird`
   - wait load stable
   - validate BIRD
   - `write_initial_fib` with `k1-bulk`
4. `wait_after_initial_k1_fib`
5. `verify_initial_fib`
6. `enable_delta_after_initial_verify`
7. `wait_after_enable_delta`
8. `run_reconvergence_k1_delta`
9. `summarize`

## K2c 方法

### 入口与 kernel

推荐入口：`K2c/run_3083_k2c_delta_optimized.sh`

该入口先设置：

```bash
K2C_KERNEL=6.8.12-birdrtbulk-k2c
RUN_NAME=k2c-delta-mixed-3083_<timestamp>
K1_DELTA_MIXED_BATCHES=1
OFFSET_JOBS=16
DELTA_WRITE_JOBS=1
K1_DELTA_BATCH=2000
```

然后进入：

```bash
K2c/run_3083_k2c_delta_reconvergence.sh
```

后者设置：

```bash
K1_KERNEL="${K2C_KERNEL:-6.8.12-birdrtbulk-k2c}"
RUN_NAME=k2c-delta-reconv-3083_<timestamp>
```

再进入 K2c 目录里的 `run_3083_k1_delta_reconvergence.sh`。

### K2c wrapper 默认参数

K2c 继承 K1 风格参数，但 K2c 目录的 base wrapper 比 K1 多了 start/restart load gate：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `RESTART_LOAD_GATE_EVERY` | `50` | 每 50 个 brdnode gate 一次 |
| `RESTART_LOAD_THRESHOLD` | `${LOAD_THRESHOLD}` | 默认 50 |
| `RESTART_LOAD_POLL_INTERVAL` | `15` | gate 轮询 |
| `RESTART_LOAD_MAX_WAIT` | `${MAX_WAIT_SECONDS}` | 默认 7200 |

`run_3083_k2c_delta_optimized.sh` 的优化覆盖：

| 参数 | 覆盖值 | 目的 |
|---|---:|---|
| `K1_DELTA_MIXED_BATCHES` | `1` | 稀疏 reconvergence delta 中混合 replace/delete，减少小 batch 次数 |
| `OFFSET_JOBS` | `16` | offset 准备是 Docker exec 密集型，与 writer jobs 解耦 |
| `DELTA_WRITE_JOBS` | `1` | 避免 sparse delta 写阶段在 `fib_info_hash_mutex` 上竞争 |
| `K1_DELTA_BATCH` | `2000` | K2c delta batch |

其他默认值与 K1 base wrapper 类似：

- `INITIAL_WRITE_JOBS=8`
- `K1_DUMP_JOBS=16`
- `K1_BATCH=2000`
- `K1_COMMAND_TIMEOUT=240`
- `LOAD_THRESHOLD=50`
- `MAX_WAIT_SECONDS=7200`
- `RESTART_JOBS=1`
- `RESTART_DELAY=0.1`
- `BUILD_MODE=delta-fast`
- `DOWN_COUNT=10`
- `RUN_FULL_COMPARISON=0`

### K2c 全流程

1. 确认或外部保证已启动到 `6.8.12-birdrtbulk-k2c`。
2. build/up delta-fast image。
3. `probe-bird-state`。
4. 如果 BIRD 未启动，执行 K2c 简化版 `start-clean-bird`。
5. 如果 BIRD 已启动或状态未知，执行完整 `restart-bird`。
6. start/restart 串行过程中每 50 个容器进入 load gate。
7. 等 `load1 < LOAD_THRESHOLD`。
8. 初始全量写：`k1-bulk`，batch 2000。
9. 初始 FIB verify。
10. enable delta，设置 `DELTA_IGNORE_INITIAL_FEED=1`。
11. reconvergence：stop 10 个 brdnode，dump delta event commands，使用 `kernel_bulk_command_writer.py` 写 delta。
12. summarize。

## K3c 方法

### 入口与 kernel

入口：`K3c/run_3083_k3c_delta_experiment.sh`

默认：

```bash
K3C_KERNEL=6.8.12-birdrtbulk-k3c
K1_KERNEL="${K3C_KERNEL}"
RESET_VM=0
```

包内 prebuilt 有两个：

- `K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c.tar.gz`
- `K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz`

如果要跑修复版 K3c，需要显式覆盖：

```bash
K3C_KERNEL=6.8.12-birdrtbulk-k3c-fibref
K1_KERNEL=6.8.12-birdrtbulk-k3c-fibref
```

脚本会通过 SSH 读取远端 `uname -r`。当 `RESET_VM=0` 且当前 kernel 不等于 `K3C_KERNEL` 时，直接退出，不会继续实验。

### K3c wrapper 默认参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `VM_HOST` | `192.168.122.60` | 默认远端 |
| `VM_USER` | `lxl` | 默认用户 |
| `PROJECT_NAME` | `topology-3083` | 默认 project |
| `OUTPUT_DIR` | `.../scale_3083/output` | 默认 output |
| `RUN_NAME` | `k3c-delta-reconv-3083_<timestamp>` | run name |
| `RESET_VM` | `0` | 不主动 reset/reboot |
| `LOAD_THRESHOLD` | `50` | load 阈值 |
| `MAX_WAIT_SECONDS` | `7200` | 等待上限 |
| `POLL_INTERVAL` | `30` | 轮询间隔 |
| `RESTART_JOBS` | `1` | start/restart 串行 |
| `RESTART_DELAY` | `0.1` | 每容器间隔 |
| `RESTART_LOAD_GATE_EVERY` | `50` | 继承 base wrapper，start/restart gate |
| `RESTART_LOAD_THRESHOLD` | `${LOAD_THRESHOLD}` | 默认 50 |
| `RESTART_LOAD_POLL_INTERVAL` | `15` | gate 轮询 |
| `RESTART_LOAD_MAX_WAIT` | `${MAX_WAIT_SECONDS}` | 默认 7200 |
| `BUILD_MODE` | `delta-fast` | build mode |
| `SKIP_BUILD_UP` | `0` | 是否跳过 build/up |
| `FORCE_INSTALL` | `0` | 是否强制 install |
| `INITIAL_WRITE_JOBS` | `16` | 初始全量写并发 |
| `K1_DUMP_JOBS` | `16` | 初始 dump 并发 |
| `K1_BATCH` | `16000` | 初始 K3c bulk batch |
| `K1_COMMAND_TIMEOUT` | `900` | 初始写超时 |
| `K1_MULTIPATH_POLICY` | `first` | multipath 处理 |
| `DELTA_JOBS` | `16` | enable/install 并发 |
| `DOWN_COUNT` | `10` | reconvergence 停止节点数 |
| `DELTA_WRITE_JOBS` | `8` | delta writer jobs |
| `OFFSET_JOBS` | `16` | offset/dump 准备并发 |
| `K1_DELTA_JOBS` | `${DELTA_WRITE_JOBS}` | delta bulk jobs |
| `K1_DELTA_BATCH` | `16000` | delta bulk batch |
| `K1_OPS_PER_ATTR` | `2000` | 单 attr ops |
| `K1_DELTA_MIXED_BATCHES` | `0` | 不混合 replace/delete |
| `RUN_FULL_COMPARISON` | `0` | 默认不跑 full comparison |
| `FULL_WRITE_JOBS` | `8` | full comparison jobs |
| `FULL_COMMAND_TIMEOUT` | `1800` | full comparison timeout |
| `SSH_CONNECT_TIMEOUT` | `30` | SSH timeout |

### K3c 全流程

1. SSH 到目标机检查 `uname -r`，必须等于 `K3C_KERNEL`，否则退出。
2. 进入 K3c 目录里的 `run_3083_k1_delta_reconvergence.sh`。
3. build/up delta-fast image。
4. `probe-bird-state`。
5. 如果 BIRD 未启动，执行简化版 `start-clean-bird`：
   - `bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf`
6. 如果 BIRD 已启动或状态未知，执行完整 `restart-bird`。
7. start/restart 串行过程中每 50 个容器 load gate。
8. 等 `load1 < 50`。
9. validate BIRD。
10. 初始全量写：`k1-bulk`，`INITIAL_WRITE_JOBS=16`，`K1_BATCH=16000`。
11. 初始 FIB verify。
12. enable delta with `DELTA_IGNORE_INITIAL_FEED=1`。
13. reconvergence：stop 10 个 brdnode，delta dump commands，`kernel_bulk_command_writer.py` 写 delta，`K1_DELTA_BATCH=16000`。
14. summarize。

## 输出文件

F：

- `F/runs/start-bird_<timestamp>_<project>/summary.json`
- `F/runs/realfib-F_<timestamp>_<project>/summary.json`
- `F/runs/reconfergence-delta-F_<timestamp>_<project>/summary.json`

K 系列 wrapper 期望/生成：

- `K*/runs/<RUN_NAME>/main_summary.json`
- `K*/runs/<RUN_NAME>/verify_initial_fib.json`
- `K*/runs/<RUN_NAME>/enable_delta_after_verify.json`
- `K*/runs/<RUN_NAME>/reconvergence/summary.json`
- `K*/runs/<RUN_NAME>/comparison_summary.json`
- `K*/runs/<RUN_NAME>/comparison_summary.pretty.json`

`main_summary.json` 的关键 stage：

- `sync`
- `build`
- `up`
- `probe_bird_state`
- `restart_bird`，其中字段 `phase` 会说明实际是 `start-clean-bird` 还是 `restart-bird`
- `wait_rib_stable`
- `validate_bird`
- `write_initial_fib`
- `enable_delta`，wrapper 通常跳过
- `start_agents`，wrapper 通常跳过
- `validate`，wrapper 通常跳过
- `fetch_artifacts`

## IDC/裸机运行时需要显式覆盖的点

包内默认值更像早期 libvirt VM 环境。IDC 裸机实验通常应显式覆盖：

```bash
export VM_HOST=127.0.0.1
export VM_USER=seed
unset VM_DOMAIN
export RESET_VM=0
export UP_BATCH=50
export UP_JOBS=4
export COMPOSE_PARALLEL_LIMIT=16
```

并按具体规模设置：

```bash
export OUTPUT_DIR=/home/seed/lxl_topology/autocoder_test/output
export PROJECT_NAME=<method-and-scale-specific-project>
```

K3c 修复版需要额外覆盖：

```bash
export K3C_KERNEL=6.8.12-birdrtbulk-k3c-fibref
export K1_KERNEL=6.8.12-birdrtbulk-k3c-fibref
```

K1 推荐显式覆盖：

```bash
export K1_KERNEL=6.8.12-birdrtbulk-k1-global-rtnl
```

不要依赖 K1 旧 wrapper 的 `RESET_VM=1` 或 `run_3083_k1_bulk_real.sh` 里的旧默认 kernel 名。

## 对 start-bird/start-clean-bird 的结论

1. F 的 `start-bird` 当前不是 `start-clean-bird`，而是独立内联逻辑：先 ready check，已运行则跳过，未运行才清 pid/socket 并 `bird -p && bird`。
2. K 系列 summary 中看到的 `restart_bird` 不一定是真的 restart；要看 `main_summary.json` 中 `restart_bird.phase`：
   - `start-clean-bird`：probe 判断 BIRD 未运行。
   - `restart-bird`：probe 判断 BIRD 已运行或状态未知。
3. K2c/K3c 的 `start-clean-bird` 已经基本等价于“只运行 bird”，没有 `birdc down`、`pkill bird`、ready check。
4. K1 目录的 `start-clean-bird` 仍较重，会改 delta conf、清 event/socket、执行 `bird -p` 和 ready check。
5. K2c/K3c 有 start/restart load gate；K1 当前目录版本没有。若实验中 K3c/K2c start 阶段明显更慢，需要同时看：
   - 是否实际走了 `restart-bird` 而不是 `start-clean-bird`。
   - `RESTART_LOAD_GATE_EVERY=50` 是否频繁等待 load。
   - `main_summary.json` 的 `restart_bird.phase`、`duration_seconds` 和 installer log 中的 `load gate count=...`。
