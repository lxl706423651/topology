# BIRD/RTNL 实验目录

这个目录用于后续实现 BIRD/RTNL 锁拥塞相关实验，并配套一台专用 VM：

- VM 名称：`bird-rtnl-dev`
- VM IP：`192.168.122.60`
- VM 用户：`lxl`
- VM 内 runner：`/home/lxl/bird/run.py`
- VM 内全流程 runner：`/home/lxl/bird/run_all.py`
- 本仓库 build/up runner：[run.py](./run.py)
- 本仓库 build/up/start-bird/start-kernel runner：[run_all.py](./run_all.py)
- VM Docker reset 脚本：[reset_docker_vm.sh](./reset_docker_vm.sh)
- RTNL D/E/F/G 可执行实验原型：[rtnl_solution_lab.py](./rtnl_solution_lab.py)，说明文档：[rtnl_solution_lab.md](./rtnl_solution_lab.md)
- 5000x10000 RTNL scale benchmark：[rtnl_scale_benchmark.py](./rtnl_scale_benchmark.py)，说明文档：[rtnl_scale_benchmark.md](./rtnl_scale_benchmark.md)

## SSH 进入 VM

当前 DHCP 地址是：

```bash
ssh lxl@192.168.122.60
```

如果 VM 重启后 IP 发生变化，在宿主机上查询：

```bash
virsh domifaddr bird-rtnl-dev --source lease
```

VM 生命周期命令：

```bash
virsh start bird-rtnl-dev
virsh shutdown bird-rtnl-dev
virsh dominfo bird-rtnl-dev
```

## run.py 的执行边界

`run.py` 只负责 Docker build/up，不负责启动 BIRD。它有两种运行方式。

本地模式：

```bash
python3 bird/run.py /path/to/output
```

这种模式在哪里运行，就操作哪里的 Docker daemon 和哪里的文件系统。例如在宿主机运行会使用宿主机 Docker，在 VM 里运行会使用 VM Docker。

VM 自动同步模式：

```bash
python3 bird/run.py output/ --vm 192.168.122.60
```

这种模式会在宿主机侧执行以下动作：

1. 解析本地 `output/`。
2. 把当前 `bird/run.py` 同步到 VM 的 `/home/lxl/bird/run.py`。
3. 在 VM 上创建一次性运行目录，并保存当次 runner 脚本和 manifest：

   ```bash
   /home/lxl/bird/runs/topology-<services数>_<YYYYmmdd_HHMMSS>_<project>/
   ```

4. 把本地 `output/` 同步到该运行目录下：

   ```bash
   /home/lxl/bird/runs/topology-<services数>_<YYYYmmdd_HHMMSS>_<project>/output/
   ```

5. 通过 SSH 执行运行目录中的脚本副本：

   ```bash
   /home/lxl/bird/runs/topology-<services数>_<YYYYmmdd_HHMMSS>_<project>/run.py \
     /home/lxl/bird/runs/topology-<services数>_<YYYYmmdd_HHMMSS>_<project>/output/
   ```

6. 最终在 VM 的 Docker daemon 中 build images 并启动容器。

也就是说，使用 `--vm` 后，它会自动同步 output 到 VM，并在 VM 上 build/up。

常用 VM 模式命令：

```bash
python3 bird/run.py output/ --vm 192.168.122.60
python3 bird/run.py output/ --vm 192.168.122.60 --down-first
python3 bird/run.py output/ --vm 192.168.122.60 --build-jobs 16 --build-batch 50
```

可选参数：

```bash
python3 bird/run.py output/ \
  --vm 192.168.122.60 \
  --vm-user lxl \
  --vm-output-root /home/lxl/bird/runs
```

`--vm-output-root` 用来控制 VM 运行记录根目录。默认是 `/home/lxl/bird/runs`。
每次 `--vm` 运行都会在该根目录下创建一个新的 `topology-<services数>_<时间戳>_<project>/`
目录，避免覆盖上一次运行的日志和中间文件。

### 手动同步方式

如果目标是在 VM 中启动仿真网络，必须先让 `output` 目录出现在 VM 内，例如用 `rsync` 同步：

```bash
ssh lxl@192.168.122.60 'mkdir -p /home/lxl/seed-emulator/examples/basic/A00_simple_as'

rsync -a --delete \
  /home/lxl/seed-emulator/examples/basic/A00_simple_as/output/ \
  lxl@192.168.122.60:/home/lxl/seed-emulator/examples/basic/A00_simple_as/output/
```

然后 SSH 进入 VM，或者从宿主机通过 SSH 远程执行 VM 内脚本：

```bash
ssh lxl@192.168.122.60 \
  '/home/lxl/bird/run.py /home/lxl/seed-emulator/examples/basic/A00_simple_as/output'
```

等价地，进入 VM 后运行：

```bash
/home/lxl/bird/run.py /home/lxl/seed-emulator/examples/basic/A00_simple_as/output
```

有了 `--vm` 后，通常不需要手动执行这一组 `rsync + ssh` 命令；直接用 `python3 bird/run.py output/ --vm 192.168.122.60` 即可。

## run.py 做了什么

给定一个 SeedEMU Docker compiler 产生的 `output` 目录后，`run.py` 会：

1. 检查 `output/docker-compose.yml` 是否存在。
2. 通过 `docker compose config --services` 获取服务列表。
3. 按照 `buildTest1.sh` 的逻辑，先 warmup build 第一个 service。
4. 继续按照 `buildTest1.sh` 的逻辑，以默认 `BATCH_SIZE=50`、`PARALLEL_JOBS=16` 分批并发执行 `docker compose build`。
5. 按照 `dockerUp.sh` 的逻辑，以默认 `BATCH_SIZE=50`、`PARALLEL_JOBS=4` 分批并发执行 `docker compose up -d`。
6. 把 Python runner 主日志写到 `<output>/run_logs/`，并把兼容 shell 脚本的 build/up 日志写到 `<output>` 父目录下的 `logs/dockerBuild/`。

本实验中，runner 默认在 build/up 阶段跳过 `seedemu_internet_map`。SeedEMU
生成的该服务带有固定 `container_name: seedemu_internet_map`，旧实验即使已经
退出但没有 `docker compose down`/`docker rm`，也会占用这个全局名字并导致新
实验 `docker compose up` 失败。该容器不参与路由器数据面收敛实验。

在 VM 模式下，实际远端目录结构类似：

```bash
/home/lxl/bird/runs/topology-1899_20260609_081500_topology-291e63c1/
├── output/
│   └── run_logs/
├── logs/dockerBuild/
├── run.py
└── run_manifest.json
```

常用命令：

```bash
/home/lxl/bird/run.py /path/to/output
/home/lxl/bird/run.py /path/to/output --down-first
/home/lxl/bird/run.py /path/to/output --build-jobs 16 --build-batch 50
python3 bird/run.py /path/to/output --vm 192.168.122.60
```

当前 `run.py` 保留了自动 Compose project name，避免不同 output 目录在同一个 Docker daemon 上相互覆盖。`--pull` 参数会被接受，但在 `buildTest1.sh` 兼容 build 流程中不会传给 `docker compose build`。

`--down-first` 会先对同一个 Compose project 执行：

```bash
docker compose down --remove-orphans
```

## run_all.py 全流程

`run_all.py` 是把 `bird/run.py` 的 build/up 流程、`start_bird0318.py`
的 BIRD 启动流程、`start_bird_kernel0318.py` 的 kernel 协议切换流程融合后的入口。默认阶段是：

1. build：默认使用 fast archive 流程，只构建少量公共 base image，然后把每个节点的 `start.sh`、`bird.conf`、`kernel.conf` 等 Dockerfile `COPY` 文件直接打成小 layer 并 `docker load`；如需回退旧流程可加 `--build-mode compose`。
2. up：沿用 `dockerUp.sh` 兼容流程。
3. start-bird：按 Compose project label 找到本次启动的容器，进入容器 namespace 后执行 `bird`。
4. start-kernel：只处理匹配 `^as(\d+)brd-r` 的路由器容器，覆盖 `/etc/bird/conf/kernel.conf` 为显式命名的 `protocol kernel kernel1` 并 `export all`，先执行 `bird -p -c /etc/bird/bird.conf` 检查配置语法，再执行 `birdc configure` 应用配置。
5. summary：持续写入 `<output>/summary.json`，统计 `build`、`up`、`start_bird`、`start_kernel` 四个阶段的开始时间、结束时间、耗时、成功状态和关键计数。

宿主机一条命令同步到 VM 并远程执行：

```bash
python3 bird/run_all.py output/ --vm 192.168.122.60
```

该命令会把本地 `bird/run_all.py` 同步到 VM 的 `/home/lxl/bird/run_all.py`，
在 VM 上创建一次性运行目录，并把本地 `output/` 同步到：

```bash
/home/lxl/bird/runs/topology-<services数>_<YYYYmmdd_HHMMSS>_<project>/output/
```

然后在 VM 上执行该运行目录里的 `run_all.py` 副本。远端生成的 `summary.json`
会同步回本地 `<output>/summary.json`；其中 `vm_host` 字段记录宿主机侧的 sync
和 remote_run 阶段。

常用命令：

```bash
python3 bird/run_all.py output/ --vm 192.168.122.60
python3 bird/run_all.py output/ --vm 192.168.122.60 --down-first
python3 bird/run_all.py output/ --vm 192.168.122.60 --build-mode compose
python3 bird/run_all.py output/ --vm 192.168.122.60 --no-start-bird
python3 bird/run_all.py output/ --vm 192.168.122.60 --no-start-kernel
python3 bird/run_all.py output/ --vm 192.168.122.60 --summary-path output/summary.json
```

VM 模式下，`run_all.py` 的远端目录结构类似：

```bash
/home/lxl/bird/runs/topology-1899_20260609_081500_topology-291e63c1/
├── output/
│   └── run_logs/
├── logs/dockerBuild/
├── run_all.py
├── run_manifest.json
├── summary.json
└── container_pids.json
```

查看 VM 上这次运行的容器时，先 SSH 到 VM，再按 Compose project label 查：

```bash
ssh lxl@192.168.122.60
docker ps --filter label=com.docker.compose.project=topology-291e63c1
```

如果只想做非破坏性检查：

```bash
python3 bird/run_all.py output/ --no-build --no-up --no-start-bird --no-start-kernel
```

`start-bird` 和 `start-kernel` 阶段需要 `nsenter` 权限。脚本默认使用 `--nsenter-sudo auto`：
如果不是 root，会尝试 `sudo -n nsenter`。VM 中的 `lxl` 用户需要具备免密 sudo，
否则该阶段会失败并写入 `summary.json`。

`start-kernel` 默认使用 `--kernel-load-threshold 50.0`，每个路由器更新后等待
1 分钟 load average 低于该阈值再继续。可以用 `--kernel-load-check-interval`
和 `--kernel-per-router-delay` 调整等待间隔。

## VM Docker Reset

`bird/reset_docker_vm.sh` 是给 `bird-rtnl-dev` 使用的危险恢复脚本，会清空 VM
里的 Docker 对象、BuildKit/cache、Docker/containerd runtime 数据，重建
`/etc/docker/daemon.json`，启动 Docker 后验证 containers/images/volumes/custom
networks 都已经归零，并默认从 `/home/lxl/seed-emulator` 重建 `seedemu-base`
和 `seedemu-router` 镜像：

```bash
ssh lxl@192.168.122.60
/home/lxl/bird/reset_docker_vm.sh
```

无参数执行等价于：

```bash
/home/lxl/bird/reset_docker_vm.sh --yes --rebuild-seedemu-images --seedemu-root /home/lxl/seed-emulator
```

如果 VM 内 SeedEMU 仓库不在默认路径，可以指定：

```bash
/home/lxl/bird/reset_docker_vm.sh --seedemu-root /path/to/seed-emulator
```

如果只想初始化 Docker 而不重建 SeedEMU 基础镜像：

```bash
/home/lxl/bird/reset_docker_vm.sh --skip-rebuild-seedemu-images
```

## Real F Main-FIB 三阶段流程

当前推荐的 F 方案入口是宿主机侧脚本：

```bash
./bird/run_f_real_main_fib_vm.sh
```

它把完整流程拆成三个明确阶段，避免把 build/up、start-bird、写 FIB 混在一起：

```bash
./bird/run_f_real_main_fib_vm.sh run
./bird/run_f_real_main_fib_vm.sh start-bird
./bird/run_f_real_main_fib_vm.sh writeFIB_F
```

也可以连续执行三段：

```bash
./bird/run_f_real_main_fib_vm.sh all
```

### 0. 初始化 VM/Docker

如果 VM 负载过高、Docker 状态不干净，或者要从头开始一轮实验，先在宿主机运行：

```bash
./bird/run_f_real_main_fib_vm.sh reset
```

这个命令等价于：

```bash
virsh destroy bird-rtnl-dev || true
virsh start bird-rtnl-dev
ssh lxl@192.168.122.60 '/home/lxl/bird/reset_docker_vm.sh --skip-rebuild-seedemu-images'
docker save handsonsecurity/seedemu-base:2.0 handsonsecurity/seedemu-router:2.0 | \
  ssh lxl@192.168.122.60 'sudo -n docker load'
```

它会清空 VM 内 Docker 容器、网络、volume、BuildKit/cache 和 containerd runtime
数据。默认情况下，如果宿主机已有 `handsonsecurity/seedemu-base:2.0` 和
`handsonsecurity/seedemu-router:2.0`，脚本会跳过 VM 内部重建，直接从宿主机
导入这两个基础镜像并补打 `latest` 标签，避免 VM 侧重新拉取 `ubuntu:24.04`。
如果宿主机没有这两个镜像，则回退为在 VM 内执行 `/home/lxl/bird/reset_docker_vm.sh`
并重建基础镜像。

可以用环境变量强制控制这个行为：

```bash
IMPORT_SEEDEMU_IMAGES_FROM_HOST=1 ./bird/run_f_real_main_fib_vm.sh reset
IMPORT_SEEDEMU_IMAGES_FROM_HOST=0 ./bird/run_f_real_main_fib_vm.sh reset
```

### 1. run：build/up

```bash
./bird/run_f_real_main_fib_vm.sh run
```

该阶段只负责：

1. 把宿主机 `output/` 同步到 VM。
2. 在 VM 上 build images。
3. 在 VM 上执行 docker compose up。

它调用的是：

```bash
python3 bird/run.py output/ --vm 192.168.122.60 \
  --project-name topology-291e63c1 \
  --build-jobs 16 \
  --up-jobs 8
```

该阶段不会启动 BIRD，也不会写 FIB。

### 2. start-bird：按编译好的 BIRD 配置启动 BIRD

```bash
./bird/run_f_real_main_fib_vm.sh start-bird
```

该阶段只做一件事：在 VM 中按 Compose project label 找到所有 `brdnode`
容器，然后对每个容器执行 BIRD 启动。

重要：这个阶段不会覆盖 `/etc/bird/conf/kernel.conf`，也不会把 kernel protocol
改成 `export none`。它使用的是 SeedEMU 编译输出、Docker image 里已经写好的
BIRD 配置；因此这一步就是正常的控制面启动和 RIB 收敛过程。按当前实验设计，
此时容器里的 kernel protocol 应该只负责导出 OSPF/直连等基础路由，而不是导出
完整 BGP 路由。

VM 内每个容器的启动逻辑等价于：

```bash
birdc show status >/dev/null 2>&1 || \
  sh -lc 'rm -f /run/bird/bird.pid /run/bird/bird.ctl; bird -p -c /etc/bird/bird.conf && bird'
```

该阶段会在本地和 VM 内分别留下 summary，例如：

```bash
bird/runs/start-bird_<timestamp>_topology-291e63c1/summary.json
/home/lxl/bird/runs/start-bird_<timestamp>_topology-291e63c1/summary.json
```

### 3. writeFIB_F：用 F 写 main FIB

```bash
./bird/run_f_real_main_fib_vm.sh writeFIB_F
```

该阶段不会 build/up，也不会重新 start-bird，也不会等待 `birdc show route primary count`
连续稳定。它假定 `start-bird` 阶段已经完成，直接调用 VM 内：

```bash
sudo -n python3 /home/lxl/bird/rtnl_solution_lab.py \
  --project topology-291e63c1 \
  --solution F \
  --real-bird \
  --no-real-prepare-bird \
  --no-pre-cleanup \
  --no-pre-write-rib-stable \
  --f-shards 8 \
  --no-cleanup-after
```

核心流程：

1. 用 F 方法，即 8 个 shard 并行 writer：
   - `birdc show route primary`
   - 解析每条 selected route 的 nexthop
   - 生成 `ip route replace ... proto bird metric 32`
   - 用 `ip -force -batch -` 写入 main FIB
2. 做 10 个不同 AS 的 FIB/RIB 抽样验证。
3. 因为使用 `--no-cleanup-after`，最后不会清理写入的 FIB，现场会保留给手动验证。

这里使用 `--no-pre-cleanup` 是为了保留 `start-bird` 阶段由原始 kernel protocol
已经写入的 OSPF/直连基础路由。也就是说，`writeFIB_F` 是在已有基础路由之上补写
BIRD RIB 中的 selected routes，而不是先 flush 掉 `proto bird` 再重建。

如果要手动验证：

```bash
ssh lxl@192.168.122.60

docker exec as1269brd-r10-1.10.4.245 sh -lc '
  printf "ip_route="; ip route | wc -l
  printf "proto_bird="; ip route show proto bird | wc -l
  birdc show route primary count
'
```

如果只想看 VM 当前状态：

```bash
./bird/run_f_real_main_fib_vm.sh status
```

常用环境变量：

```bash
VM_HOST=192.168.122.60
VM_USER=lxl
PROJECT_NAME=topology-291e63c1
OUTPUT_DIR=/home/lxl/seed-emulator/topology/output
F_SHARDS=8
```

例如指定另一个 output：

```bash
OUTPUT_DIR=/path/to/output ./bird/run_f_real_main_fib_vm.sh run
```

## Reconfergence F 故障后手动刷新 FIB

`bird/reconfergence_F.sh` 用于模拟已经完成数据面 FIB 收敛后的网络波动：按 AS 范围均匀选取并让
10 个不同 AS 的 brdnode 容器 down 掉，然后等待一个固定时间，再手动执行 F 方法刷新
main FIB 并记录耗时。

默认命令：

```bash
./bird/reconfergence_F.sh
```

默认行为：

1. 从当前 `topology-291e63c1` 项目中选择 10 个不同 AS 的运行中 brdnode 容器。
2. 执行 `docker stop -t 1`，模拟这些边界路由器突然 down 掉。
3. 等待 `POST_DOWN_SLEEP_SECONDS=30` 秒，让 BIRD 自行进行 RIB 重收敛。
4. 调用 `rtnl_solution_lab.py --solution F --real-bird --no-real-prepare-bird --no-pre-write-rib-stable`。
5. 默认 `PRE_CLEANUP=1`，先 flush stale `proto bird` 路由，再用当前 BIRD RIB 通过 F 写回 main FIB。
6. 做 10 个不同 AS 的 FIB/RIB 抽样验证，并且 `--no-cleanup-after` 保留现场。

默认开启 `PRE_CLEANUP=1` 是因为故障后旧 FIB 可能存在已经撤销的 stale route；
`ip route replace` 只能新增/替换当前 RIB 中存在的路由，不能删除旧路由。普通
`run_f_real_main_fib_vm.sh writeFIB_F` 仍然使用 `--no-pre-cleanup`，适合无故障的
补写/保持现场流程。

重收敛脚本会给验证加 `--fib-rib-min-delta 0`，即允许
`ip route | wc -l == birdc show route count networks`。这是因为故障刷新默认
先 flush stale `proto bird`，再按当前 RIB 重写；等号通常表示没有额外 stale route。
普通 `writeFIB_F` 仍使用默认 `--fib-rib-min-delta 1`，保持原来的严格 `>` 验证。

常用参数：

```bash
DOWN_COUNT=10 ./bird/reconfergence_F.sh
POST_DOWN_SLEEP_SECONDS=60 ./bird/reconfergence_F.sh
PRE_CLEANUP=0 ./bird/reconfergence_F.sh
F_SHARDS=8 ./bird/reconfergence_F.sh
```

查看 VM 当前状态：

```bash
./bird/reconfergence_F.sh status
```

每次运行会生成本地 summary：

```bash
bird/runs/reconfergence-F_<timestamp>_topology-291e63c1/summary.json
```

其中关键字段包括：

- `victims`：本次被 stop 的 10 个不同 AS brdnode 容器。
- `fib_command_wall_seconds`：手动 F 命令从开始到结束的墙钟时间。
- `event_to_fib_done_seconds`：容器 stop 完成到 F 写入与验证结束的总时间。
- `f_duration_seconds`：F sharded writer 自身报告的写入耗时。
- `f_route_ops`：F 本次写入的路由条数。

## Reconfergence Delta F 增量刷新 FIB

`bird/reconfergence_delta_F.sh` 用于验证“只写变化路由”的方案。它不再故障后全量
重写所有 BIRD selected routes，而是先维护一份故障前 RIB 快照，故障后再次 dump
当前 RIB，然后对比出 added/changed/deleted，只把 delta 写入 main FIB。当前实现默认
使用 root 远端 Python 进程执行，写入阶段通过 `setns()` 直接进入目标 netns，避免每个
容器反复启动 `sudo nsenter`。

默认命令：

```bash
./bird/reconfergence_delta_F.sh
```

默认流程：

1. 对当前运行中的 brdnode 执行 baseline dump：`birdc show route primary`。
2. 从不同 AS 均匀选择 10 个 brdnode，执行 `docker stop -t 1`。
3. 等待 `POST_DOWN_SLEEP_SECONDS=30` 秒，让 BIRD 自行重收敛。
4. 对故障后的运行中 brdnode 再次 dump 当前 RIB。
5. 执行当前 FIB dump：`ip -o route show proto bird`。
6. 对比 baseline/current：
   - added：生成 `ip route replace`
   - changed：生成 `ip route replace`
   - deleted：只对当前 FIB 中真实存在的 prefix 生成 `ip route del ... proto bird metric 32`
   - unchanged：不写
7. 用 `F_SHARDS=8` 个 root worker 并行写 delta batch，默认 `WRITE_MODE=setns-ip`。
8. 做 10 个不同 AS 的 FIB/RIB 抽样验证，并保留现场。

对于 deleted 路由，如果 `ip route del` 返回 `RTNETLINK answers: No such process`，
脚本会按成功处理，因为目标状态就是该路由不存在。这通常发生在上一轮 FIB 刷新已经
提前删除了某些 stale route，或者当前 FIB 本来就没有该条 `proto bird` 路由。

常用参数：

```bash
DOWN_COUNT=10 ./bird/reconfergence_delta_F.sh
POST_DOWN_SLEEP_SECONDS=60 ./bird/reconfergence_delta_F.sh
DUMP_WORKERS=32 F_SHARDS=8 ./bird/reconfergence_delta_F.sh
FIB_AWARE_DELETE=1 WRITE_MODE=setns-ip ./bird/reconfergence_delta_F.sh
```

查看 VM 当前状态：

```bash
./bird/reconfergence_delta_F.sh status
```

每次运行会生成本地 summary：

```bash
bird/runs/reconfergence-delta-F_<timestamp>_topology-291e63c1/summary.json
```

对比全量 F 时，重点看：

- `stages.baseline_dump.duration_seconds`：故障前快照维护成本。
- `stages.current_dump.duration_seconds`：故障后读取当前 RIB 的时间。
- `stages.fib_dump.duration_seconds`：故障后读取当前 `proto bird` FIB 的时间，用于过滤无效 delete。
- `stages.diff.route_ops`：真正需要写入 FIB 的 delta 条数。
- `stages.diff.delete_filtered`：FIB-aware delete 过滤掉的无效删除条数。
- `stages.delta_write.duration_seconds`：实际 RTNL/FIB 增量写入耗时。
- `stages.delta_write.write_mode`：实际写入模式，优化后应为 `setns-ip`。
- `timing.post_fault_manual_refresh_seconds`：故障后 current dump + diff + delta write 的总时间。

已验证的 1600 级结果：

- 全量 F 基线：`bird/runs/reconfergence-F_20260611_092737_topology-291e63c1/summary.json`，
  `stages.F.duration_seconds=98.123879s`，写入 `2056716` 条。
- 优化后 delta-F：`bird/runs/reconfergence-delta-F_20260611_135340_topology-291e63c1/summary.json`，
  `stages.delta_write.duration_seconds=4.439854s`，`WRITE_MODE=setns-ip`，
  写入 `35834` 条，`delete_filtered=999`，10 个 FIB/RIB 抽样全部通过。

## Event Delta F 事件式 delta 源实验

`bird/reconfergence_event_F.sh` 用于验证“不在故障后全量执行
`birdc show route primary`，而是由常驻 agent 收集 route change event”的方向。

先做过的 BIRD CLI 探测结论：

- `birdc monitor all` 在当前 BIRD 2.17.2 中不可用。
- `birdc debug all { routes, events }` + `echo all` 可以看到普通日志，但实测不能稳定输出
  prefix-level selected route delta。
- `birdc dump events` 只输出 BIRD 内部事件指针，不包含可解析 prefix/nexthop。
- `mrt dump` 是即时全表 dump；`mrtdump` 偏协议消息/状态，也不是 master table selected-route delta。

当前可运行的事件式原型采用 staging kernel table：

```text
BIRD master4
  -> protocol pipe delta252_pipe
  -> BIRD table t_delta252
  -> protocol kernel delta252
  -> Linux kernel table 252
  -> ip -ts monitor route
  -> event agent 解析 table 252 路由事件
  -> 生成 main-FIB ip route batch
```

默认命令：

```bash
./bird/reconfergence_event_F.sh
```

常用参数：

```bash
LIMIT_CONTAINERS=20 EVENT_MODE=failure DOWN_COUNT=3 WATCH_SECONDS=180 ./bird/reconfergence_event_F.sh
LIMIT_CONTAINERS=1 EVENT_MODE=selftest WATCH_SECONDS=12 ./bird/reconfergence_event_F.sh
WRITE_MAIN=1 ./bird/reconfergence_event_F.sh
```

参数含义：

- `EVENT_MODE=selftest`：不制造网络故障，只对 watcher 执行
  `birdc disable delta252_pipe; birdc enable delta252_pipe`，验证事件链路。
- `EVENT_MODE=failure`：停止 `DOWN_COUNT` 个非 watcher brdnode，观察真实控制面变化事件。
- `WRITE_MAIN=0`：只生成 batch，不写 main FIB，适合验证事件源。
- `WRITE_MAIN=1`：把生成的 collapsed batch 写入 main FIB。
- `CLEANUP_STAGING=1`：实验结束后删除 `delta252.conf` 并 flush table 252。

已验证结果：

- selftest：`bird/runs/event-delta-F_20260611_165427_topology-291e63c1/summary.json`
  - 1 个 watcher
  - `prefix_events=2001`
  - `collapsed_ops=675`
  - parse 耗时 `0.017795s`
- failure：`bird/runs/event-delta-F_20260611_165456_topology-291e63c1/summary.json`
  - 20 个 watcher
  - 停止 3 个非 watcher brdnode
  - 未执行故障后全量 `birdc show route primary`
  - `prefix_events=3072`
  - `collapsed_ops=2724`
  - parse 耗时 `0.026229s`

重要限制：

- 这个方案避免了“故障后全量扫描 BIRD RIB”，但并不是完全无 RTNL：BIRD 会把变化先写入
  staging kernel table 252，agent 再从 netlink monitor 读取事件。
- 如果最终目标是完全绕开 BIRD kernel protocol 的 RTNL 写入，仍需要改 BIRD，在 route
  selected/change 位置直接把 delta 输出到 Unix socket、共享内存或文件，再由 host writer
  统一批量写 main FIB。

## BIRD 源码 Delta F 最终方案

源码位置：

```bash
bird/bird-2.17.2-delta/
```

新增协议：

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

这个协议挂在 BIRD 的 `rt_notify()` selected-route 通知路径上。BIRD 的 core 在
`RA_MERGED` 模式下只要 `master4` 的最终 selected route 发生 add/update/delete，
就会写一行 `/run/bird/delta-events.tsv`，不经过 kernel table 252，也不调用
BIRD kernel protocol 写 FIB。

事件格式：

```text
ts_us  R  proto  table  replace|delete  prefix  dest  metric  nh_count  [gw iface flags weight]...
```

用户态 writer：

```bash
python3 bird/delta_fib_agent.py \
  --event-file /run/bird/delta-events.tsv \
  --follow \
  --from-end \
  --stats-file /run/bird/delta-fib-agent.json
```

`delta_fib_agent.py` 会把短时间窗口内同一 prefix 的多次变化折叠成最后一次操作，
然后用 `ip -4/-6 -force -batch -` 写 main FIB。它必须运行在目标路由器容器或目标
netns 中。

VM 上把 patched BIRD 分发到正在运行的 brdnode 容器：

```bash
./bird/install_bird_delta_vm.sh install
./bird/install_bird_delta_vm.sh restart-bird
./bird/install_bird_delta_vm.sh write-initial
./bird/install_bird_delta_vm.sh start-agent
./bird/install_bird_delta_vm.sh status
```

推荐阶段语义：

1. `install`：把 `bird-2.17.2-delta/bird`、`birdc`、`delta_fib_agent.py` 复制到容器，
   并写入 `/etc/bird/conf/delta.conf`。
2. `restart-bird`：清空 `/run/bird/delta-events.tsv`，用 patched BIRD 重启协议栈。
3. 等 RIB 收敛后执行 `write-initial`：重放当前事件文件，初始化 main FIB。
4. 执行 `start-agent`：agent 从事件文件末尾开始跟随，只处理后续网络波动产生的 delta。

与 `reconfergence_event_F.sh` 的区别：

- 旧原型：`master4 -> pipe -> table252 -> kernel table252 -> ip monitor`，仍然会额外争抢 RTNL。
- 最终源码方案：`master4 selected-route delta -> event file -> ip batch writer`，不需要 staging kernel table。
- 后续故障重收敛时不需要全量 `birdc show route primary` dump，也不需要 table 252 cleanup。

## 已验证示例

A00 示例已经同步到 VM 并成功运行：

```bash
/home/lxl/bird/run.py /home/lxl/seed-emulator/examples/basic/A00_simple_as/output
```

验证结果：

- A00 网络已在 VM Docker 中启动。
- 8 个长期运行容器处于 `running`。
- 跨 AS host ping 测试 0% 丢包。
- BIRD BGP session 为 `Established`。
- `seedemu_internet_map` 在 VM 的 `8080` 端口可访问。

更多 VM 细节见：[vm/README.md](./vm/README.md)。

docker exec -it 588f93cecfbc /bin/bash

结果摘要：

   方案     写入条数    写入耗时            吞吐    验证
  ━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━
   D       2,037,133     159.22s    12,794 ops/s    10/10 通过
  ──────  ───────────  ──────────  ──────────────  ────────────
   E       2,004,749     156.32s    12,824 ops/s    10/10 通过
  ──────  ───────────  ──────────  ──────────────  ────────────
   F       2,070,070      96.52s    21,448 ops/s    10/10 通过
  ──────  ───────────  ──────────  ──────────────  ────────────
   G       2,070,134     156.39s    13,237 ops/s    10/10 通过

  对比之前 Scheme A：A 的 kernel export 阶段是 1256.752s，并且最终 load gate 没过。
  按写入阶段算，D/E/G 大约是 A 的 7.9x-8.0x，F 大约是 13.0x。
