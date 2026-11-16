# 0701 IDC Reproduction Prompt

下面这段 prompt 用于交接给**另一台 IDC 机器上的 agent**。它不是给当前 `/data/lxl/seed-emulator/topology` 这台机器继续执行的。目标是在 IDC 机器上安全复现 F、K1、K2c、K3c 四种 Linux main FIB 写入实验，并且在涉及重新编译/安装内核时保证原内核、文件和机器状态可回退。

---

你现在接手的是 SeedEMU/BIRD RTNL/FIB 收敛实验。你应当已经运行在 IDC 机器上，或已经通过 SSH 登录到 IDC 机器。请严格按以下约束执行。

非常重要：

- 不要假设当前机器就是原实验宿主机。
- 不要沿用原实验机器的默认 VM IP `192.168.122.60`、默认 VM 名 `bird-rtnl-dev` 或默认路径，除非你在 IDC 现场确认它们确实存在。
- 所有 `VM_HOST`、`VM_USER`、`VM_DOMAIN`、`OUTPUT_DIR`、`PROJECT_NAME` 都必须按 IDC 机器上的实际环境重新设置。
- IDC 机器就是裸机实验环境：不要轻易关机、重启，也不要执行 `virsh destroy/start`。
- 可以 reset Docker，但只能清理实验相关容器、Docker 网络、Docker volume/cache 和 Docker/containerd runtime；不能破坏源码、数据集、内核和系统文件。
- 裸机模式下默认设置 `RESET_VM=0`。不要运行会重启机器或切换 VM 的 reset 阶段。

## 一、最高优先级安全约束

1. **不得销毁原先可启动内核。**
   - 任何新内核都必须 side-by-side 安装，不能覆盖、删除、清理当前能正常启动的原内核。
   - 不允许执行 `apt autoremove --purge`、`rm -rf /boot/*old*`、`update-initramfs -d -k <old-kernel>`、删除 `/lib/modules/<old-kernel>` 等会破坏 fallback 的操作。
   - 安装新内核后必须确认 GRUB 菜单里仍然能看到原内核和新内核。

2. **不得破坏文件系统和实验源码。**
   - 不要在未知目录执行 `rm -rf`。
   - IDC 是裸机实验环境，因此清理 Docker 等同于清理本机 Docker。只允许在确认实验容器/网络/volume 范围后执行 Docker reset；不要误删非实验容器和非实验数据。
   - 对需要改的脚本/配置，先复制备份或在独立工作目录修改。

3. **新内核出现问题时必须能切回原内核。**
   - 在安装/切换内核前记录：
     ```bash
     uname -r
     ls -lh /boot/vmlinuz-* /boot/initrd.img-* | tail -n 20
     grep -n "menuentry 'Ubuntu, with Linux" /boot/grub/grub.cfg | head -n 50
     ```
   - 设置新内核为默认前，先保存当前 GRUB 默认项。
   - IDC 是裸机，不要在未确认维护窗口时自行重启切换内核。需要切换 K1/K2c/K3c 内核时，先安装 side-by-side 内核并汇报，等待用户确认可以重启后再操作。
   - 如果新内核启动后 SSH 失联、Docker 异常或出现内核 Oops/panic，应通过 IDC 控制台/IPMI/带外管理切回旧内核，不要继续强行跑实验。

4. **不要把研究原型当生产内核。**
   - K2c/K3c 是 no-notify controlled prototype。它们可能不触发 `ip monitor route`/rtnetlink multicast route event。
   - 每次实验后记录 `dmesg`/`journalctl -k` 中是否有 Oops、panic、lockdep、refcount、unregister_netdevice waiting 等信号。

## 二、当前实验背景

问题：大规模 SeedEMU Internet 仿真中，很多 router 容器里的 BIRD 同时把 RIB selected routes 写入 Linux main FIB，会争抢全局 RTNL lock，导致数据面收敛很慢甚至长时间不收敛。

已有方法：

- **F**：不改内核。host 侧集中 `setns + ip -batch` 写 Linux main FIB。初始 full 写入可控，但 3000 规模仍较慢；reconvergence 走 delta route。
- **K1**：新增私有 `BIRD_RT_BULK` generic-netlink API，在一次 global RTNL 临界区中批量处理 route ops。
- **K2c**：K1 + per-netns route lock + no-notify + `fib_info_hash_mutex`。证明跨 namespace 的 global RTNL 是核心瓶颈。
- **K3c**：K2c + cached `fib_info` fast path。对共享 nexthop/device/metric/table 的 routes 复用 `fib_info`，当前性能最好。

关键结果参考：

- F initial full：`6.13M` routes 大约 `520-579s`。
- F reconvergence delta：约 `561k` ops，FIB write `238.320s`。
- K1 3083 initial full：`402.388s`；reconvergence delta writer `63.639s`。
- K2c 3083 initial full：`71.162s`；reconvergence delta writer `55.026s`。
- K3c 3083 initial full：6 月 22 日 writer `10.552s`、kernel write `7.867s`；7 月 1 日复跑 writer `10.226s`、kernel write `7.668s`。
- K3c 3083 reconvergence delta：6 月 22 日 `6.791s`；7 月 1 日复跑 `6.386s`，`198,832/198,832` ops 成功。

注意：文档里的 `restart_bird` 是历史阶段名，表示在 router 容器内启动/重启 BIRD 进程，不是重启 VM、Docker 或容器。

## 三、已传输到 IDC 的包

源机器已将复现实验包传到 IDC 机器的用户目录：

```text
seed@10.202.191.39:~/rtnl_packages/rtnl_fib_methods_repro_20260701_133858.tar.gz
```

checksum：

```text
bfc6868980c6e1233241bb923943545efff2e38104f03fe06b96b9ed263037a9
```

远端校验曾通过：

```text
rtnl_fib_methods_repro_20260701_133858.tar.gz: OK
```

请在 IDC 机器本地先执行：

```bash
cd ~/rtnl_packages
sha256sum -c rtnl_fib_methods_repro_20260701_133858.tar.gz.sha256
tar -xzf rtnl_fib_methods_repro_20260701_133858.tar.gz
cd rtnl_fib_methods_repro_20260701_133858
sed -n '1,220p' README.md
```

包结构：

```text
README.md
0626kernelDesign.md
1buildFastDockerImages.py
F/
K1/
K2c/
K3c/
```

其中：

- `README.md`：复现实验执行说明，含分阶段命令。
- `0626kernelDesign.md`：方法对比、实验结果、内核风险和论文叙事。
- `F/`：F 方法脚本。
- `K1/`：K1 脚本、patch、prebuilt kernel、patched BIRD。
- `K2c/`：K2c 脚本、patch、prebuilt kernel、patched BIRD。
- `K3c/`：K3c 脚本、prebuilt kernel、K3c overlay、patched BIRD。

## 四、需要你在 IDC 上做的事

### 1. 先做环境盘点

请先在 IDC 机器上记录：

```bash
hostname
uname -a
lsb_release -a 2>/dev/null || cat /etc/os-release
df -h
free -h
nproc
docker --version
docker compose version
virsh --version || true
```

确认实验是在 IDC 裸机上进行。该机器不能轻易关机或重启，因此：

- 不要执行 `virsh destroy/start`。
- 不要设置 `RESET_VM=1`。
- 不要运行脚本里的 `reset` 阶段，除非你已经确认它只 reset Docker，不会重启机器。
- 可以执行 Docker reset，但要限制在 Docker/containerd、容器、Docker 网络、Docker cache 范围内。
- 如需切换 K1/K2c/K3c 内核，先准备 side-by-side 内核，再汇报需要一次维护窗口重启，由用户确认后再操作。

裸机本地模式建议设置：

```bash
export VM_HOST=127.0.0.1
export VM_USER=$(whoami)
unset VM_DOMAIN
export RESET_VM=0
```

这些脚本历史上按“host 通过 SSH 控制 VM”的方式写成，因此裸机本地模式需要确认本机能免密 SSH 到自己：

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${VM_USER}@127.0.0.1 'echo localhost-ssh-ok'
```

如果本机没有 ssh server 或免密 SSH，请先配置本机 SSH，或者修改脚本为本地执行模式后再跑；不要为了绕过 SSH 直接改成危险的 reset/reboot 流程。

### 2. 安装或准备 Docker

需要 Docker、Docker Compose、buildx。若已有环境，先不要破坏；只记录版本。

SeedEMU base image 可以直接使用裸机上的 Docker 镜像，或让脚本 build。优先复用已有镜像以减少网络依赖：

```bash
docker images | grep -E 'handsonsecurity/seedemu-(base|router)|ubuntu'
```

### 3. 准备 compiled SeedEMU output

所有方法都通过 `OUTPUT_DIR` 切换拓扑规模。不要改脚本。

示例：

```bash
export OUTPUT_DIR=/absolute/path/to/compiled/output
export PROJECT_NAME=topology-3083
```

`OUTPUT_DIR` 必须包含：

```text
docker-compose.yml
各节点 Dockerfile/build context
```

如果要换规模：

```bash
export OUTPUT_DIR=/absolute/path/to/another/output
export PROJECT_NAME=topology-1897
```

必须用不同 `PROJECT_NAME`，避免和旧 Docker Compose project 冲突。

### 4. 先跑 F 方法作为 stock-kernel 基线

F 不要求改内核。建议先跑 F，确认 Docker/SeedEMU/BIRD 基本流程正常。

```bash
cd ~/rtnl_packages/rtnl_fib_methods_repro_20260701_133858/F

export VM_HOST=127.0.0.1
export VM_USER=$(whoami)
unset VM_DOMAIN
export RESET_VM=0
export OUTPUT_DIR=/absolute/path/to/output
export PROJECT_NAME=topology-smoke

./run_f_real_main_fib_vm.sh status
```

裸机上不要直接跑 `./run_f_real_main_fib_vm.sh reset`，因为旧脚本的 reset 阶段可能包含 `virsh destroy/start`。先使用 Docker-only reset 脚本或手动 reset Docker。

Docker-only reset 示例：

```bash
sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
sudo pkill -9 -f 'containerd-shim|dockerd|containerd|runc' 2>/dev/null || true
sudo systemctl start containerd docker
docker ps -aq | xargs -r docker rm -f
docker network ls --filter type=custom -q | xargs -r docker network rm
docker volume ls -q | xargs -r docker volume rm
```

如果已有安全的 `reset_docker_vm.sh`，确认它不会关机、重启或调用 `virsh` 后可运行：

```bash
./reset_docker_vm.sh
```

然后按 `run -> start-bird -> writeFIB_F -> reconfergence_delta_F.sh` 分阶段跑。

F 分阶段参考：

```bash
./run_f_real_main_fib_vm.sh run
START_BIRD_JOBS=1 START_BIRD_DELAY=0.1 ./run_f_real_main_fib_vm.sh start-bird
F_SHARDS=8 REAL_DUMP_WORKERS=16 LIMIT_CONTAINERS=0 ./run_f_real_main_fib_vm.sh writeFIB_F
DOWN_COUNT=10 DUMP_WORKERS=32 FIB_DUMP_WORKERS=32 F_SHARDS=8 ./reconfergence_delta_F.sh run
```

### 5. 安装 K1/K2c/K3c 内核时必须 side-by-side

包内有 prebuilt kernel tar：

```text
K1/kernel/prebuilt/6.8.12-birdrtbulk-k1-global-rtnl.tar.gz
K2c/kernel/prebuilt/6.8.12-birdrtbulk-k2c.tar.gz
K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c.tar.gz
```

安装示例：

```bash
KERNEL=6.8.12-birdrtbulk-k3c
TARBALL=K3c/kernel/prebuilt/6.8.12-birdrtbulk-k3c.tar.gz

sudo tar -C / -xzf "$TARBALL"
sudo update-initramfs -c -k "$KERNEL"
sudo update-grub
```

安装后必须确认：

```bash
ls -d /lib/modules/*
ls -lh /boot/vmlinuz-* /boot/initrd.img-* | tail -n 30
grep -n "menuentry 'Ubuntu, with Linux" /boot/grub/grub.cfg | head -n 80
```

原内核必须仍在列表里。

切换内核前记录当前内核：

```bash
echo "old_kernel=$(uname -r)" | tee ~/old_kernel_before_rtnl_experiment.txt
```

不要删除旧内核，不要覆盖旧 initrd。

IDC 裸机内核切换要求：

- 这份 prompt 不能自行重启 IDC 裸机，也不能执行任何等价于断电、强制重启或自动改 GRUB 后立即 reboot 的动作。
- 如果需要从 stock kernel 切换到 K1/K2c/K3c，先完成 side-by-side 安装和 GRUB 检查，然后把“需要一次维护窗口重启”的结论汇报给用户，等待用户明确确认后再切换。
- 用户确认重启后，也必须保留旧内核为 GRUB 可选项；不要把旧内核移除或覆盖。

### 6. K1/K2c/K3c 实验顺序建议

推荐顺序：

1. F baseline。
2. K1。
3. K2c。
4. K3c。

不要一开始就跑最大规模。先用小规模 output 做 smoke test，再跑 1897/3083。

注意：下面的 K1/K2c/K3c 命令默认 IDC 裸机已经**由用户确认并安全启动到对应目标内核**。命令中的 `RESET_VM=0` 是必须项，表示脚本只跑实验流程，不负责重启、destroy/start 或切换内核。如果当前 `uname -r` 不是对应内核，先停止并汇报，不要强行跑。

K1 一键：

```bash
cd K1
RESET_VM=0 \
OUTPUT_DIR="${OUTPUT_DIR}" \
PROJECT_NAME="${PROJECT_NAME}" \
INITIAL_WRITE_JOBS=1 \
K1_DUMP_JOBS=16 \
K1_BATCH=256 \
DELTA_WRITE_JOBS=1 \
OFFSET_JOBS=16 \
K1_DELTA_BATCH=256 \
RUN_FULL_COMPARISON=0 \
./run_3083_k1_delta_reconvergence.sh
```

K2c 一键：

```bash
cd K2c
RESET_VM=0 \
OUTPUT_DIR="${OUTPUT_DIR}" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_FULL_COMPARISON=0 \
./run_3083_k2c_delta_optimized.sh
```

K3c 一键：

```bash
cd K3c
RESET_VM=0 \
OUTPUT_DIR="${OUTPUT_DIR}" \
PROJECT_NAME="${PROJECT_NAME}" \
INITIAL_WRITE_JOBS=16 \
K1_DUMP_JOBS=16 \
K1_BATCH=16000 \
K1_OPS_PER_ATTR=2000 \
DELTA_WRITE_JOBS=8 \
OFFSET_JOBS=16 \
K1_DELTA_BATCH=16000 \
RUN_FULL_COMPARISON=0 \
./run_3083_k3c_delta_experiment.sh
```

如果裸机已经安全启动到 K3c 并且 Docker 状态已准备好，可用：

```bash
RESET_VM=0 ./run_3083_k3c_delta_experiment.sh
```

### 7. 每轮必须保存的结果

每轮实验结束后保存：

```text
runs/<run-name>/main_summary.json
runs/<run-name>/verify_initial_fib.json
runs/<run-name>/reconvergence/summary.json
runs/<run-name>/comparison_summary.json
```

并记录：

```bash
uname -r
cat /proc/loadavg
docker ps --format '{{.Names}} {{.Status}}' | head
journalctl -k --since "30 min ago" --no-pager | egrep -i 'BUG:|Oops|panic|WARNING:|lockdep|KASAN|refcount|use-after-free|general protection|unregister_netdevice|bird_rtbulk|fib_create_info' || true
```

### 8. 判断成功条件

初始 full 成功：

- build/up/start-bird 成功。
- load gate 通过。
- full FIB writer ops success，fail count 为 0。
- post-FIB verify 通过。

reconvergence 成功：

- 按 `DOWN_COUNT` 停掉若干 brdnode。
- RIB/load 稳定后，通过 delta route event/diff 计算变化路由。
- delta FIB writer fail count 为 0。
- validate 通过。

### 9. 如果出现问题

如果 SSH 卡死、load average 长时间极高、内核日志有 Oops/panic：

1. 不要继续加大规模。
2. 不要删除旧内核。
3. 不要随便关机、断电或强制重启裸机；先保存能保存的日志。
4. 如果确实需要切回旧内核，通过 IDC 控制台、IPMI/带外管理或现场 GRUB 菜单切回原内核，并确保这是用户确认过的维护动作。
5. 保存失败日志和 summary。
6. 报告当前阶段、命令、内核版本、load、journalctl 片段。

### 10. 你需要最终汇报

请最终给出：

- 当前机器环境。
- 使用的 output 规模和 `PROJECT_NAME`。
- 每个方法是否跑通。
- 每个方法的 build/up/start-bird/full FIB/reconvergence delta 时间。
- full FIB route ops、delta route ops、fail count。
- 内核日志是否有风险信号。
- 当前默认内核和 fallback 原内核是否仍可用。
- 所有 summary 文件路径。

再次强调：**这台 IDC 是裸机实验环境，不能轻易关机、断电、重启或执行 VM destroy/start；可以 reset Docker，但只能在确认范围后清理实验相关 Docker 状态。重新编译或安装新内核时，绝不能销毁原先可启动内核；必须 side-by-side 安装，保证出现问题时能切回旧内核。**

---
