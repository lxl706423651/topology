# 0705 AMD Local Kernel Experiment Prompt

下面这段 prompt 用于交接给**另一台 IDC 机器上的 agent**。IDC agent 必须先从 IDC SSH 到当前本机 AMD 实验宿主机，再在 AMD 本机上执行 SeedEMU/BIRD RTNL/FIB 实验。不要在 IDC 机器本地执行实验命令。

默认 AMD SSH 目标：

```text
lxl@10.202.236.88
```

如果用户给了新的 SSH 地址，以用户给的为准。

---

## 你的任务

你现在要从 IDC 机器 SSH 到 AMD 本机，按 3083 规模跑通四种方法：

- F: stock kernel baseline，不改内核。
- K1: `6.8.12-birdrtbulk-k1-global-rtnl`。
- K2c: `6.8.12-birdrtbulk-k2c`。
- K3c fixed: `6.8.12-birdrtbulk-k3c-fibref`，这是修补了 K3c `fib_info` tree ref / netdevice leak 问题的方法，不要误用旧的 `6.8.12-birdrtbulk-k3c`。

目标是所有方法都能在 AMD 本机跑通 3083 全流程，并记录耗时、ops、fail count、Docker 状态和内核日志风险信号。

---

## 最高安全约束

- AMD 本机是裸机实验宿主，不是 VM。不要执行 `virsh destroy/start`，不要运行任何 VM reset 阶段。
- 所有实验脚本都必须设置 `RESET_VM=0`。特别是 K1/K2c 底层脚本默认可能是 `RESET_VM=1`，必须显式覆盖。
- 新内核必须 side-by-side 安装。不能删除旧 kernel、旧 initrd、旧 `/lib/modules`。
- 不要执行 `apt autoremove --purge`、`rm -rf /boot/*`、`update-initramfs -d` 或删除旧 `/lib/modules/<kernel>`。
- 切换 K1/K2c/K3c-fibref 内核会导致 SSH 断开。必须先安装并汇报，等待用户明确同意维护窗口 reboot 后才能切换。
- `/home/lxl/seed-emulator/topology/reset_docker.sh` 会删除 `/var/lib/docker`、`/var/lib/containerd`、`/etc/docker` 并重建 SeedEMU 镜像。它是 Docker 初始化脚本，不是轻量 project cleanup。只有确认 AMD 本机 Docker 没有需要保留的非实验数据时才运行。
- Docker cleanup 优先只清当前实验 `PROJECT_NAME` 的 container/network。不要删除源码、数据集、内核 tarball、run 结果。
- start-bird 阶段在 fresh up 后应该是 clean start：最多清 stale pid/socket、`bird -p` 校验，然后运行 `bird`。不要在 clean start 情况下对每个 brdnode 执行 `birdc down`、`pkill bird`、`pkill -9 bird`。如果脚本 summary 显示实际 phase 是 `restart-bird`，先停下检查为什么 BIRD 已经在跑。

---

## 登录和环境盘点

先从 IDC SSH 到 AMD，本轮后续命令都在 AMD 上执行：

```bash
ssh -tt -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  lxl@10.202.236.88
```

登录后先执行：

```bash
hostname
uname -a
lsb_release -a 2>/dev/null || cat /etc/os-release
df -h
free -h
nproc
docker --version
docker compose version
systemctl is-active ssh sshd 2>/dev/null || true
virsh --version || true
```

当前 AMD 预期状态：

```text
host: ubuntu
stock kernel: 6.8.0-124-generic
memory: about 1 TiB
docker: Docker 29.0.0 / Compose v2.40.3
3083 output: /home/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output
```

确认本机可以 SSH 到自己，因为这些 runner 仍按“host 通过 SSH 控制实验机器”的模式写成：

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  lxl@127.0.0.1 'echo localhost-ssh-ok'
```

如果 self-SSH 不通，先修 SSH，不要把脚本改成会绕过安全检查的 reset/reboot 流程。

---

## 通用变量

```bash
export BIRD=/home/lxl/seed-emulator/topology/bird
export PKG=/home/lxl/seed-emulator/topology/bird/packaged/rtnl_fib_methods_repro_20260701_133858
export OUTPUT_DIR=/home/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output
export VM_HOST=127.0.0.1
export VM_USER=lxl
unset VM_DOMAIN
export RESET_VM=0
export UP_BATCH=50
export UP_JOBS=4
export COMPOSE_PARALLEL_LIMIT=16
export RESTART_JOBS=1
export RESTART_DELAY=0.1
export RESTART_LOAD_GATE_EVERY=50
export RESTART_LOAD_THRESHOLD=50
export RESTART_LOAD_POLL_INTERVAL=15
export RESTART_LOAD_MAX_WAIT=7200
```

确认 3083 output 存在：

```bash
test -f "$OUTPUT_DIR/docker-compose.yml" || { echo "missing docker-compose.yml"; exit 1; }
python3 - <<'PY'
from pathlib import Path
import yaml
p = Path("/home/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output/docker-compose.yml")
d = yaml.safe_load(p.read_text())
print("services", len(d.get("services", {})))
print("networks", len(d.get("networks", {})))
PY
```

---

## Docker 初始化和 project cleanup

如果 AMD Docker 需要从干净状态开始，并且确认可以清掉本机 Docker 数据：

```bash
/home/lxl/seed-emulator/topology/reset_docker.sh
docker info
docker images | grep -E 'handsonsecurity/seedemu-(base|router)' || true
```

如果只是清理某次实验，优先用 project-scoped cleanup：

```bash
PROJECT_NAME=<project-name>
docker ps -a --filter label=com.docker.compose.project="$PROJECT_NAME" -q | xargs -r -n 100 docker rm -f
docker network ls --filter label=com.docker.compose.project="$PROJECT_NAME" -q | xargs -r -n 100 docker network rm
docker ps -a --filter label=com.docker.compose.project="$PROJECT_NAME" -q | wc -l
docker network ls --filter label=com.docker.compose.project="$PROJECT_NAME" -q | wc -l
```

切换内核前必须安全 down 当前 project，不能带着上一个实验的容器和网络直接 reboot。

---

## start-bird / restart-bird 要求

F baseline 的 `start-bird` 已经是轻量路径：

```bash
birdc show status >/dev/null 2>&1 || \
  sh -lc 'rm -f /run/bird/bird.pid /run/bird/bird.ctl; bird -p -c /etc/bird/bird.conf && bird'
```

K1/K2c/K3c 的 runner 历史字段名叫 `restart_bird`，但实际会先 `probe-bird-state`：

- `bird_not_started` -> `start-clean-bird`
- `bird_running` 或状态不确定 -> `restart-bird`

fresh Docker up 后应该进入 `start-clean-bird`。该路径只做：

```bash
mkdir -p /run/bird
: > /run/bird/delta-events.tsv
rm -f /run/bird/bird.pid /run/bird/bird.ctl
bird -p -c /etc/bird/bird.conf
bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf
birdc -s /run/bird/bird.ctl show status
```

运行后必须检查：

```bash
jq '.stages.restart_bird.phase, .stages.restart_bird.start_mode, .stages.restart_bird.probed_state' runs/<run-name>/main_summary.json
```

3083 fresh run 期望：

```text
"start-clean-bird"
"clean-start"
"bird_not_started"
```

如果看到 `"restart-bird"`，不要继续后续 FIB 写入；先检查是否复用了旧 project、BIRD 是否已经启动、是否需要 project cleanup 后重新 up。

---

## F Stock-Kernel Baseline

仅当 `uname -r` 是 stock generic kernel 时跑 F，例如：

```bash
uname -r
```

期望当前为：

```text
6.8.0-124-generic
```

执行：

```bash
cd "$PKG/F"
export VM_HOST=127.0.0.1
export VM_USER=lxl
unset VM_DOMAIN
export RESET_VM=0
export OUTPUT_DIR=/home/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output
export PROJECT_NAME=rtnl-amd-f-3083

./run_f_real_main_fib_vm.sh status
UP_BATCH=50 UP_JOBS=4 COMPOSE_PARALLEL_LIMIT=16 ./run_f_real_main_fib_vm.sh run
START_BIRD_JOBS=1 START_BIRD_DELAY=0.1 ./run_f_real_main_fib_vm.sh start-bird
F_SHARDS=8 REAL_DUMP_WORKERS=16 LIMIT_CONTAINERS=0 ./run_f_real_main_fib_vm.sh writeFIB_F
DOWN_COUNT=10 DUMP_WORKERS=32 FIB_DUMP_WORKERS=32 F_SHARDS=8 ./reconfergence_delta_F.sh run
```

结束后收集并清理 project：

```bash
uname -r
cat /proc/loadavg
docker ps --format '{{.Names}} {{.Status}}' | head
journalctl -k --since "30 min ago" --no-pager | egrep -i 'BUG:|Oops|panic|WARNING:|lockdep|KASAN|refcount|use-after-free|general protection|unregister_netdevice|bird_rtbulk|fib_create_info' || true
```

---

## side-by-side 安装内核

安装前记录旧内核和 GRUB 状态：

```bash
echo "old_kernel=$(uname -r)" | tee ~/old_kernel_before_amd_rtnl_experiment.txt
ls -lh /boot/vmlinuz-* /boot/initrd.img-* | tail -n 40
grep -n "menuentry 'Ubuntu, with Linux" /boot/grub/grub.cfg | head -n 100
```

安装模板：

```bash
cd "$PKG"
KERNEL=<kernel-release>
TARBALL=<tarball-path>
sha256sum "$TARBALL"
sudo tar --no-overwrite-dir -C / -xzf "$TARBALL"
sudo update-initramfs -c -k "$KERNEL"
sudo update-grub
ls -lh "/boot/vmlinuz-$KERNEL" "/boot/initrd.img-$KERNEL" 2>/dev/null
test -d "/lib/modules/$KERNEL" && echo modules-present
grep -n "menuentry 'Ubuntu, with Linux" /boot/grub/grub.cfg | head -n 120
```

Kernels:

```text
K1   6.8.12-birdrtbulk-k1-global-rtnl
     $PKG/K1/kernel/prebuilt/6.8.12-birdrtbulk-k1-global-rtnl.tar.gz
     sha256 980aa1c4c14fafe249dac782814308b44d7a17cb117eb99d70a9620245af0cda

K2c  6.8.12-birdrtbulk-k2c
     $PKG/K2c/kernel/prebuilt/6.8.12-birdrtbulk-k2c.tar.gz
     sha256 d9fab6107ca3b6239119138111dedd28d5636c1b841aa2655475b0e4b2c0e68d

K3c fixed
     6.8.12-birdrtbulk-k3c-fibref
     $PKG/K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c-fibref.tar.gz
     sha256 605cde762d438ced00e8d15c777659a0bbda44c49b3eb6af0494f53017f4ffe5
```

安装后停下汇报。只有用户明确确认 reboot 后才切内核。建议使用 one-shot `grub-reboot`，不要永久改默认启动项：

```bash
KERNEL=<kernel-release>
ENTRY="$(sudo awk -F"'" -v k="$KERNEL" '/menuentry / && $2 ~ ("Ubuntu, with Linux " k) {print $2; exit}' /boot/grub/grub.cfg)"
echo "ENTRY=$ENTRY"
test -n "$ENTRY" || { echo "could not resolve grub entry"; exit 1; }
sudo grub-reboot "$ENTRY"
REBOOT_START="$(date -Is)"
echo "reboot_start=$REBOOT_START"
sudo reboot
```

SSH 会断开。IDC agent 等待 3-5 分钟后重连；最多尝试 20 分钟。如果 20 分钟仍 SSH 不通，停止并告诉用户需要人工介入。

重连后先 sleep 120 秒，再执行：

```bash
hostname
uname -r
cat /proc/loadavg
free -h
docker info --format 'Containers={{.Containers}} Running={{.ContainersRunning}} Images={{.Images}} NFd={{.NFd}} NGoroutines={{.NGoroutines}}'
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lxl@127.0.0.1 'echo localhost-ssh-ok'
```

---

## K1 3083

必须已经启动到：

```text
6.8.12-birdrtbulk-k1-global-rtnl
```

执行：

```bash
test "$(uname -r)" = "6.8.12-birdrtbulk-k1-global-rtnl" || { echo "wrong kernel"; exit 1; }
cd "$PKG/K1"

export VM_HOST=127.0.0.1
export VM_USER=lxl
unset VM_DOMAIN
export RESET_VM=0
export OUTPUT_DIR=/home/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output
export PROJECT_NAME=rtnl-amd-k1-3083
export K1_KERNEL=6.8.12-birdrtbulk-k1-global-rtnl
export UP_BATCH=50 UP_JOBS=4 COMPOSE_PARALLEL_LIMIT=16
export RESTART_JOBS=1 RESTART_DELAY=0.1
export RESTART_LOAD_GATE_EVERY=50 RESTART_LOAD_THRESHOLD=50 RESTART_LOAD_POLL_INTERVAL=15 RESTART_LOAD_MAX_WAIT=7200
export INITIAL_WRITE_JOBS=1 K1_DUMP_JOBS=16 K1_BATCH=256
export DELTA_WRITE_JOBS=1 OFFSET_JOBS=16 K1_DELTA_BATCH=256
export RUN_FULL_COMPARISON=0

./run_3083_k1_delta_reconvergence.sh
```

检查 start phase：

```bash
jq '.stages.restart_bird.phase, .stages.restart_bird.start_mode, .stages.restart_bird.probed_state' runs/<run-name>/main_summary.json
```

---

## K2c 3083

必须已经启动到：

```text
6.8.12-birdrtbulk-k2c
```

执行：

```bash
test "$(uname -r)" = "6.8.12-birdrtbulk-k2c" || { echo "wrong kernel"; exit 1; }
cd "$PKG/K2c"

export VM_HOST=127.0.0.1
export VM_USER=lxl
unset VM_DOMAIN
export RESET_VM=0
export OUTPUT_DIR=/home/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output
export PROJECT_NAME=rtnl-amd-k2c-3083
export K2C_KERNEL=6.8.12-birdrtbulk-k2c
export K1_KERNEL=6.8.12-birdrtbulk-k2c
export UP_BATCH=50 UP_JOBS=4 COMPOSE_PARALLEL_LIMIT=16
export RESTART_JOBS=1 RESTART_DELAY=0.1
export RESTART_LOAD_GATE_EVERY=50 RESTART_LOAD_THRESHOLD=50 RESTART_LOAD_POLL_INTERVAL=15 RESTART_LOAD_MAX_WAIT=7200
export RUN_FULL_COMPARISON=0

./run_3083_k2c_delta_optimized.sh
```

---

## K3c fixed 3083

必须已经启动到 fixed kernel：

```text
6.8.12-birdrtbulk-k3c-fibref
```

不要跑旧的 `6.8.12-birdrtbulk-k3c`。

执行：

```bash
test "$(uname -r)" = "6.8.12-birdrtbulk-k3c-fibref" || { echo "wrong kernel"; exit 1; }
cd "$PKG/K3c"

export VM_HOST=127.0.0.1
export VM_USER=lxl
unset VM_DOMAIN
export RESET_VM=0
export OUTPUT_DIR=/home/lxl/seed-emulator/topology/bird/runs/F_reconf_multiscale_20260614_094101/scale_3083/output
export PROJECT_NAME=rtnl-amd-k3c-fibref-3083
export K3C_KERNEL=6.8.12-birdrtbulk-k3c-fibref
export K1_KERNEL=6.8.12-birdrtbulk-k3c-fibref
export UP_BATCH=50 UP_JOBS=4 COMPOSE_PARALLEL_LIMIT=16
export RESTART_JOBS=1 RESTART_DELAY=0.1
export RESTART_LOAD_GATE_EVERY=50 RESTART_LOAD_THRESHOLD=50 RESTART_LOAD_POLL_INTERVAL=15 RESTART_LOAD_MAX_WAIT=7200
export INITIAL_WRITE_JOBS=16 K1_DUMP_JOBS=16 K1_BATCH=16000 K1_OPS_PER_ATTR=2000
export DELTA_WRITE_JOBS=8 OFFSET_JOBS=16 K1_DELTA_BATCH=16000
export RUN_FULL_COMPARISON=0

./run_3083_k3c_delta_experiment.sh
```

检查 K3c fixed 重点风险：

```bash
journalctl -k --since "60 min ago" --no-pager | egrep -i 'refcount|use-after-free|unregister_netdevice|bird_rtbulk|fib_create_info|BUG:|Oops|panic|WARNING:' || true
```

不应再出现旧 K3c 的 `refcount_t: underflow`、`use-after-free`、`unregister_netdevice: waiting ...`。

---

## 每轮结束必须收集

```bash
uname -r
cat /proc/loadavg
free -h
docker ps --format '{{.Names}} {{.Status}}' | head
docker info --format 'Containers={{.Containers}} Running={{.ContainersRunning}} Images={{.Images}} NFd={{.NFd}} NGoroutines={{.NGoroutines}}'
journalctl -k --since "30 min ago" --no-pager | egrep -i 'BUG:|Oops|panic|WARNING:|lockdep|KASAN|refcount|use-after-free|general protection|unregister_netdevice|bird_rtbulk|fib_create_info' || true
```

汇报每种方法：

- kernel version
- `PROJECT_NAME`
- build/up/start-bird 或 `restart_bird` 实际 phase/RIB stable/full FIB/reconvergence delta 时间
- full route ops、delta raw events、delta collapsed ops、fail count
- Docker 状态
- kernel log 是否有风险信号
- summary 文件路径

K1/K2c/K3c 重点 summary：

```text
runs/<run-name>/main_summary.json
runs/<run-name>/verify_initial_fib.json
runs/<run-name>/reconvergence/summary.json
runs/<run-name>/comparison_summary.json
```

F 重点 summary 在：

```text
$PKG/F/runs/
```

---

## 如果 Docker up 卡在 network create

- 不要全局 `docker system prune`。
- 先记录：

```bash
docker info --format 'Containers={{.Containers}} Running={{.ContainersRunning}} Images={{.Images}} NFd={{.NFd}} NGoroutines={{.NGoroutines}}'
docker ps -a | wc -l
docker network ls | wc -l
ip -o link show type bridge | awk -F': ' '{print $2}' | grep -E '^br-[0-9a-f]{12}$' | wc -l
cat /proc/loadavg
```

- 如果是某个 compose network gateway 和容器 fixed IP 冲突，优先定位具体 network，像 IDC 7001 的 IX1 问题那样只重建该 network，不要全局 reset。
- 只有确认 Docker 没有需要保留的数据，且用户同意 Docker 初始化时，才运行 `/home/lxl/seed-emulator/topology/reset_docker.sh`。

---

## 最终输出格式

请最终给出一张对比表：

```text
method | kernel | project | build/up | start phase | start-bird/restart_bird | RIB wait | full FIB wall/ops/fail | reconvergence wall | delta dump | delta writer wall/ops/fail | kernel risk | summary path
```

并明确回答：

- 四种方法是否都跑通 3083。
- AMD 本机内存是否足够。
- K3c fixed 是否还出现 netdevice/refcount/fib_create_info 相关风险日志。
- start-bird 是否走了 clean start；如果没有，解释原因并不要把 heavy restart 路径当作正常结果。
