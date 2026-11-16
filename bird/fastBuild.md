# Fast Build 说明

本文档说明 `bird/run_all.py` 当前 build 阶段的 fast archive 方法、旧的
`docker compose build` 方法、BuildKit cache 实际优化的位置，以及为什么当前
方法能明显缩短大规模 SeedEMU 镜像构建时间。

## 文件角色

- `bird/run_all.py`：全流程入口。默认 `--build-mode fast`，build 完后继续
  `docker compose up`、start-bird、start-kernel。
- `1buildFastDockerImages.py`：fast build helper。负责解析 Compose/Dockerfile、
  构造 Docker archive，并供 `bird/run_all.py` 动态加载。
- `output/docker-compose.yml`：SeedEMU Docker compiler 生成的 Compose 文件，
  是 build/up 的主要输入。
- `output/<service>/Dockerfile`：每个节点镜像的 Dockerfile。fast build 通过
  这里的 `COPY` 指令判断哪些文件要写入镜像。
- `output/.fastDockerBuild/`：fast build 中间产物目录，包含 `images_all.tar`、
  `summary.json`、`build_images.sh` 和 base layer 缓存。
- `logs/dockerBuild/dockerBuild_*.log`：build 阶段日志。

## 先给 Docker 小白的结论

Docker 镜像可以先理解成“很多层文件系统叠在一起”。下面一层是操作系统和软件，
上面一层可以只是几个配置文件。启动容器时，Docker 把这些层按顺序叠起来，看到的
就是完整文件系统。

旧方式让 Docker 对每个节点都完整走一遍 `docker build`。即使公共层已经命中 cache，
Docker 仍然要重复调度 builder、解析 Dockerfile、检查 build context、处理 `COPY`、
生成 image config、登记 tag。节点有 1900 个，这套流程就重复接近 1900 次。

当前 fast 方式利用了 SeedEMU 输出的规律：这些路由器镜像底层几乎一样，差异主要是
`bird.conf`、`kernel.conf`、`start.sh`、`ifinfo.txt`、`interface_setup` 等小配置。
所以脚本只让 Docker 真正构建少量公共底座，然后自己把每个节点的小配置文件写成
Docker layer，最后一次性生成 `images_all.tar` 并 `docker load`。

一句话：

```text
旧方式：让 Docker 做 1900 次 build，cache 只能加速每次 build 里的部分步骤。
新方式：让 Docker 只做少量公共 build，1900 个节点镜像由脚本按 Docker 镜像格式拼出来。
```

## 最新一轮实测数字

远端最新一轮检查路径：

```text
/home/lxl/bird/runs/topology-1899_20260610_231351_topology-291e63c1/
```

这轮 `output/.fastDockerBuild/summary.json` 显示：

```text
Compose services discovered: 1899
active build/up services:     1898
skipped service:              seedemu_internet_map
fast archive images:          1897
normal docker build images:   1
common fast base images:      2
images_all.tar:               638371840 bytes, ls 显示约 609M
archive generation time:      23.87s
run_all build stage total:    94.749s
```

`run_all` 的 build 阶段总耗时比 23.87s 长，是因为它还包含 `docker load -i
images_all.tar`。这一轮日志里实际只调用了 3 次 `docker build`：

```text
docker build -t 072d7840db6a44d06e5c5fd4d7ea8a7d ...
docker build -t seedemu-fast-base:4e733d528bc6a1982b1bef6d ...
docker build -t seedemu-fast-base:2c709ccf4ecf19dd2694b07e ...
docker load -i output/.fastDockerBuild/images_all.tar
```

对应旧批量构建的一轮远端记录：

```text
/home/lxl/bird/runs/topology-1899_20260609_174643_topology-291e63c1/
```

该轮 build 阶段是 warmup + 38 个 batch、每组 50 个服务、4 组并发，共 1898 个服务，
耗时 `1746.683s`。日志里 `load build definition from Dockerfile` 出现了 1975 次，
说明大量服务仍然分别进入了 Docker/BuildKit 的 Dockerfile 解析和 build pipeline。

按这两轮同规模记录粗略比较：

```text
旧批量 build 阶段：1746.683s，约 29.1 分钟
新 fast build 阶段：  94.749s，约  1.6 分钟
阶段总耗时约快 18 倍

只看 fast archive 生成：23.87s
相当于把真正“生成 1897 个节点镜像描述”的部分压到半分钟以内
```

## 当前入口

默认使用 fast build：

```bash
python3 bird/run_all.py output/
python3 bird/run_all.py output/ --vm 192.168.122.60
```

如需回退旧的 Compose batch build：

```bash
python3 bird/run_all.py output/ --build-mode compose
python3 bird/run_all.py output/ --vm 192.168.122.60 --build-mode compose
```

## 旧流程：docker compose build

旧流程对应 `bird/run_all.py --build-mode compose`，逻辑接近原来的
`buildTest1.sh`。

### 阶段 0：准备

1. 读取 `output/docker-compose.yml`。
2. 执行 `docker compose config --services` 获取全部服务名。
3. 跳过 `seedemu_internet_map`，避免固定 `container_name` 冲突。
4. 设置：

   ```bash
   DOCKER_BUILDKIT=1
   COMPOSE_DOCKER_CLI_BUILD=1
   ```

### 阶段 1：warmup

1. 取第一个 build 服务。
2. 执行：

   ```bash
   docker compose -f output/docker-compose.yml -p <project> build <first-service>
   ```

3. 这一步的目的不是只构建第一个节点，而是先让 Docker/BuildKit 建好或确认可用的
   cache，例如基础镜像、公共 `RUN` 层、apt 安装层等。

### 阶段 2：批量构建

1. 把服务按 `--build-batch` 分组，默认每组 50 个。
2. 用 `--build-jobs` 并发跑多组，默认 4 组并发。
3. 每组执行：

   ```bash
   docker compose -f output/docker-compose.yml -p <project> build service1 service2 ...
   ```

### 每个镜像内部 Docker builder 做的事

即使 cache 命中，每个服务镜像仍要进入一次 Docker builder。对于 1900 个服务，
这部分会重复 1900 次：

1. Compose 把 service 的 build 配置交给 Docker。
2. Docker/BuildKit 读取该 service 的 build context。
3. 读取并解析该 service 的 Dockerfile。
4. 处理 `FROM`，确认基础镜像。
5. 对每条 Dockerfile 指令计算 cache key。
6. 判断 `RUN apt-get ...`、`RUN mkdir ...`、`RUN touch ...` 等是否命中 cache。
7. 处理 `COPY`，读取当前节点目录里的文件并计算内容摘要。
8. 生成或复用 `COPY` 对应的 layer。
9. 处理 `RUN chmod +x ...`。
10. 生成 image config。
11. 给镜像打 tag，例如 `<project>-brdnode_...`。
12. 把该镜像登记到本机 Docker image store。

## BuildKit cache 优化的是哪一段

BuildKit cache 主要优化的是 Dockerfile 中“内容完全相同”的层。

典型可命中的部分：

```dockerfile
FROM 072d7840db6a44d06e5c5fd4d7ea8a7d
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends iperf3
RUN mkdir -p /usr/share/doc/bird2/examples/
RUN touch /usr/share/doc/bird2/examples/bird.conf
```

如果很多节点 Dockerfile 前半段完全相同，那么第一个镜像构建后，后续镜像的这些
`RUN` 层可以命中 cache，不需要再次真的执行 apt、mkdir、touch。

BuildKit cache 不能完全省掉的部分：

- 不能省掉 1900 次 Docker builder 调度。
- 不能省掉 1900 次 Dockerfile 解析。
- 不能省掉 1900 次 build context 访问。
- 不能省掉 1900 次 `COPY` 文件内容检查。
- 不能省掉 1900 次 image config 生成和 tag 登记。
- 不能把 1900 个 service 自动合并成一次批量 image archive 写入。

所以旧流程即使命中 cache，本质仍然是：

```text
少量公共层被 cache 加速
+ 1900 次完整 docker build 流程
```

## 当前流程：fast archive build

当前默认流程对应 `bird/run_all.py --build-mode fast`，核心逻辑在
`1buildFastDockerImages.py`。

它不再对每个节点执行一次 `docker build`。它利用 SeedEMU 镜像高度规律化的特点：

```text
绝大多数层相同
每个节点主要只差 start.sh、bird.conf、kernel.conf、ifinfo.txt、interface_setup 等配置文件
```

### 阶段 0：准备

1. `bird/run_all.py` 读取 Compose 服务列表。
2. 跳过 `seedemu_internet_map`。
3. 设置：

   ```bash
   DOCKER_BUILDKIT=1
   COMPOSE_DOCKER_CLI_BUILD=1
   COMPOSE_PROJECT_NAME=<project>
   ```

4. 动态加载 `1buildFastDockerImages.py`。
5. 工作目录固定为：

   ```text
   output/.fastDockerBuild/
   ```

### 阶段 1：解析 Compose 和 Dockerfile

`1buildFastDockerImages.py` 做这些事：

1. 读取 `output/docker-compose.yml`。
2. 找到每个 service 的 build context。
3. 推导每个 service 最终的 image tag。因为设置了 `COMPOSE_PROJECT_NAME`，
   tag 会和后续 `docker compose up -p <project>` 期望的一致。
4. 读取每个 `output/<service>/Dockerfile`。
5. 把 Dockerfile 分成两段：

   ```text
   COPY 之前：公共 base 部分
   COPY 之后：节点自己的配置文件层
   ```

### 阶段 2：只构建少量公共 base image

对于 `COPY` 之前完全相同的 Dockerfile 前缀，fast build 只构建一次公共 base。

例如 border router 和 route server 可能对应两类公共前缀，因此实际只需要构建少量
base image：

```text
seedemu-fast-base:<hash-A>
seedemu-fast-base:<hash-B>
```

这一步仍然使用 Docker builder，因为公共前缀里可能有真实 `RUN` 指令，例如 apt
安装、目录创建等。

### 阶段 3：直接生成每个节点的小配置 layer

对于每个节点，不再执行 `docker build`。脚本直接读取 Dockerfile 里的 `COPY`
声明。

例如 Dockerfile 中有：

```dockerfile
COPY 082b96ec819c95ae773daebde675ef80 /start.sh
COPY 2b0ae038330eccd43095538618caee7d /etc/bird/bird.conf
COPY 8e9a384ab59c7626a3ffab63886cc30f /etc/bird/conf/kernel.conf
COPY e01e36443f9f72c6204189260d0bd276 /ifinfo.txt
COPY d3d51fdf7f4bad30dc5db560a01ce629 /interface_setup
```

fast build 会：

1. 从该节点 build context 读取 `082b...` 文件内容。
2. 在 tar layer 中写入路径 `/start.sh`。
3. 从该节点 build context 读取 `2b0...` 文件内容。
4. 在 tar layer 中写入路径 `/etc/bird/bird.conf`。
5. 同理写入 `/etc/bird/conf/kernel.conf`、`/ifinfo.txt`、`/interface_setup`。
6. 如果 Dockerfile 后面有 `RUN chmod +x /start.sh`，则在 tar metadata 里设置
   `/start.sh` 的可执行权限。
7. 如果 Dockerfile 有 `CMD ["/start.sh"]`，则写入 image config。

这里的“替换关系”不需要另写配置文件声明。Dockerfile 的 `COPY 源 目标` 就是声明。

### 阶段 4：生成 Docker image archive

脚本直接写 Docker 可识别的 archive：

```text
output/.fastDockerBuild/images_all.tar
```

archive 内部核心是：

```text
base layers
per-node config layers
per-image config JSON
manifest.json
```

对每个节点镜像，`manifest.json` 里会声明：

```text
<project>-brdnode_xxx =
    公共 base layers
  + 该节点自己的配置 layer
```

换句话说，1900 个镜像不是通过 1900 次 `docker build` 构建出来的，而是通过 Docker
镜像格式“拼装”出来的。

### 阶段 5：一次 docker load

最后执行：

```bash
docker load -i output/.fastDockerBuild/images_all.tar
```

Docker 读取 archive，一次性导入所有 image tag。之后 `docker compose up` 使用这些
tag，就不需要再 build。

### 代码实现对应关系

`1buildFastDockerImages.py` 的实现可以按下面几步理解：

1. `discoverBuildEntries()` 读取 `docker-compose.yml`，找到每个 service 的 build
   context、Dockerfile 路径和最终 image tag。
2. `createDockerfilePlan()` 解析 Dockerfile，把 `COPY` 之前的指令记为公共
   `prefix`，把 `COPY <src> <dest>` 记为节点私有文件清单，并记录 `RUN chmod ...`
   和 `CMD [...]`。
3. `buildCommonBaseImage()` 对相同 `prefix` 算 hash。相同 hash 只构建一次
   `seedemu-fast-base:<hash>`，这就是“公共底座”。
4. `extractImageArchive()` 用 `docker save` 把公共底座导出到
   `cache/common_base_archives/<hash>/`，后面写总 archive 时直接复用这些 base
   layers。
5. `createRuntimeLayer()` 逐个节点读取 build context 里的配置文件，把它们按
   Dockerfile 的 `COPY` 目标路径写进一个 tar layer。如果 Dockerfile 里有
   `RUN chmod +x /start.sh`，这里直接在 tar metadata 中把权限改成可执行。
6. `createImageConfig()` 在公共底座的 image config 上追加刚生成的配置 layer
   diff id，并写入 `CMD`。
7. `writeDockerArchive()` 写出 Docker 可 `load` 的 `images_all.tar`，里面包含公共
   base layers、每个节点的配置 layer、每个镜像的 config JSON，以及最终
   `manifest.json`。

因此 `output/.fastDockerBuild/` 里的几个文件含义是：

```text
images_all.tar
    最终给 docker load 使用的批量镜像包。

summary.json
    本次 fast build 的结果摘要，例如 imageCount、baseCount、archiveBytes、
    durationSeconds。

build_images.sh
    从 Compose 解析出来的 image tag 到 build context 的映射，主要用于排查和复用。

cache/common_base_contexts/<hash>/Dockerfile
    自动生成的公共底座 Dockerfile。本轮只有两个：一个带 iperf3，一个不带 iperf3。

cache/common_base_archives/<hash>/
    公共底座 docker save 之后展开的 layer/config 缓存。
```

## 新旧流程对照

旧流程：

```text
docker compose config --services
warmup build 1 个服务
把 1900 个服务分 batch
并发执行 docker compose build
    每个服务进入 Docker builder
    每个服务解析 Dockerfile
    每个服务判断 cache
    每个服务处理 COPY
    每个服务生成 image config
    每个服务登记 tag
```

当前 fast 流程：

```text
docker compose config --services
解析 docker-compose.yml
解析所有 Dockerfile
按 COPY 前的公共前缀分组
只 docker build 少量公共 base image
为每个节点直接生成一个小配置 layer
写所有 image config 和 manifest.json
一次 docker load 导入全部 image tag
```

更短地说：

```text
旧流程 = 1900 次 docker build
新流程 = 少量 docker build + 1900 个小 tar layer + 1 次 docker load
```

结合最新一轮和旧批量构建记录，可以把差异看成：

| 项目 | 旧 Compose batch build | 当前 fast archive build |
| --- | --- | --- |
| 面向 Docker 的 build 次数 | 接近每个 service 一次 | 1 个 base-only 镜像 + 2 个公共 base |
| 节点私有配置处理方式 | 每个节点由 Docker builder 执行 `COPY` | 脚本直接写入 per-node tar layer |
| 公共 `RUN` 层 | 依赖 BuildKit cache 复用 | 只在公共 base 中真实构建一次 |
| image config/tag | 每次 `docker build` 逐个提交 | 脚本批量写 config 和 `manifest.json` |
| 导入 Docker image store | 每个 build 逐个登记 | `docker load` 一次性导入 `images_all.tar` |
| 同规模记录耗时 | 1746.683s | 94.749s，其中 archive 生成 23.87s |

## 为什么快很多

主要加速来自“绕过 1900 次 Docker builder 流程”，不是单纯依赖 cache。

### 省掉的重复开销

fast build 省掉了每个节点都要重复做的这些工作：

- Compose 对每个服务发起 build。
- Docker builder 对每个服务启动 build pipeline。
- BuildKit 对每个 Dockerfile 逐条算 cache。
- Docker 对每个 build context 做常规 build 处理。
- Docker 为每个节点执行 `COPY` 指令的完整 builder 流程。
- Docker 为每个节点逐个提交和登记 image。

fast build 保留了真正必要的工作：

- 公共 base 的真实构建。
- 每个节点不同配置文件的读取。
- 每个节点小 layer 的生成。
- 每个 image config 和 tag 的生成。
- 最后导入到 Docker image store。

这就是为什么速度提升明显：不是把 1900 个镜像的内容都省略了，而是把“对 1900 个
高度相似镜像重复运行 Docker build machinery”的部分省掉了。配置文件仍然会被读取，
节点 layer 仍然会生成，镜像 tag 仍然存在；只是这些动作由一个顺序写 tar/archive 的
脚本完成，避免了 Docker builder 1900 次启动、解析、cache 判断和提交镜像的固定成本。

### 为什么 SeedEMU 特别适合

SeedEMU 编译出来的路由器镜像高度相似。大多数节点 Dockerfile 差异只在这些文件：

- `/start.sh`
- `/etc/bird/bird.conf`
- `/etc/bird/conf/kernel.conf`
- `/ifinfo.txt`
- `/interface_setup`
- `/seedemu_sniffer`
- `/seedemu_worker`

因此 Docker 镜像可以自然表示为：

```text
镜像 = 公共 base layers + 节点自己的配置 layer + image config
```

fast build 正是直接按这个结构写 archive。

### dummy image 和 fast build 的区别

dummy image 思路已经复用了公共基础层：

```text
build dummy image 1 次
其他节点 FROM dummy
```

但后续仍然是：

```text
docker build node image 1900 次
```

fast build 更进一步：

```text
build common base 少量几次
不再对每个节点调用 docker build
直接拼出 1900 个节点镜像
```

所以 dummy image 解决的是“公共层不要重复执行”，fast build 解决的是“不要重复进入
1900 次 Docker builder”。

## 限制和等价性边界

fast build 依赖当前 SeedEMU Dockerfile 的规律：

- 支持简单 `COPY <src> <dest>`。
- 支持 `COPY` 之后的简单 `RUN chmod ...`。
- 支持 `CMD [...]`。
- base-only 服务会回退到普通 `docker build`。

如果 Dockerfile 在 `COPY` 之后出现复杂构建逻辑，例如：

```dockerfile
RUN apt-get install ...
ADD ...
COPY ["a", "b", "/dest/"]
RUN sed -i ...
```

fast build 会报错或需要扩展。这样做是故意的：宁可失败，也不要生成和
`docker compose build` 不等价的镜像。

## 输出和排查

fast build 成功后可检查：

```text
output/.fastDockerBuild/images_all.tar
output/.fastDockerBuild/summary.json
output/.fastDockerBuild/build_images.sh
logs/dockerBuild/dockerBuild_*.log
```

如果怀疑 fast build 和 Compose build 不一致，可临时回退：

```bash
python3 bird/run_all.py output/ --build-mode compose
```

如果只想验证 fast build 的 archive 生成逻辑，可直接运行 helper：

```bash
python3 1buildFastDockerImages.py output/docker-compose.yml --skip-load
```


1. discoverBuildEntries()：读取 compose，找到要构建哪些镜像

  输入：

  output/docker-compose.yml

  它做的事：

  读取所有 services
  跳过 seedemu_internet_map
  找到每个 service 的 build context
  找到每个 service 的 Dockerfile
  推导最终 image tag

  你获得了一个 entries 列表，里面每个元素类似：

  service: brdnode_1801_r12
  image: topology-291e63c1-brdnode_1801_r12
  context: output/brdnode_1801_r12
  dockerfile: output/brdnode_1801_r12/Dockerfile

  这一轮结果大概是：

  发现 services: 1899
  跳过: seedemu_internet_map
  参与 build/up: 1898
  fast image: 1897
  normal image: 1

  ———

  2. writeBuildImagesFile()：写调试用映射文件

  产出：

  output/.fastDockerBuild/build_images.sh

  内容类似：

  seedemu_build_and_push topology-291e63c1-brdnode_1801_r12 brdnode_1801_r12
  seedemu_build_and_push topology-291e63c1-brdnode_1801_r13 brdnode_1801_r13
  ...

  这个文件不是最终镜像本体，主要是“image tag -> build context”的清单，方便排查。

  ———

  3. parseDockerfile()：解析每个节点 Dockerfile

  输入：

  output/brdnode_xxx/Dockerfile

  它把 Dockerfile 拆成指令列表，比如：

  FROM ...
  ARG ...
  RUN ...
  COPY ...
  RUN chmod ...
  CMD ...

  这里还没有构建，只是读文本并解析。

  ———

  4. createDockerfilePlan()：把 Dockerfile 分成公共部分和节点私有部分

  对你的例子，它会分成：

  公共 prefix：

  FROM 072d7840db6a44d06e5c5fd4d7ea8a7d
  ARG DEBIAN_FRONTEND=noninteractive
  RUN apt-get update && apt-get install -y --no-install-recommends iperf3
  RUN mkdir -p /usr/share/doc/bird2/examples/
  RUN touch /usr/share/doc/bird2/examples/bird.conf

  节点私有 copies：

  COPY ... /start.sh
  COPY ... /seedemu_sniffer
  COPY ... /seedemu_worker
  COPY ... /etc/bird/bird.conf
  COPY ... /ifinfo.txt
  COPY ... /interface_setup
  COPY ... /etc/bird/conf/kernel.conf

  节点私有 metadata：

  executable_targets:
    /start.sh
    /seedemu_sniffer
    /seedemu_worker

  cmd:
    ["/start.sh"]

  这一轮实际只有 2 类公共 prefix：

  1643 个节点: 带 iperf3 的 prefix
  254 个节点: 不带 iperf3 的 prefix

  ———

  5. buildImageEntry()：普通构建 base-only 镜像

  有一个 service 没有 COPY，是 dummy/base-only 镜像：

  072d7840db6a44d06e5c5fd4d7ea8a7d

  它走普通 Docker build：

  docker build -t 072d7840db6a44d06e5c5fd4d7ea8a7d ...

  获得：

  本机 Docker image store 里有了 072d7840db6a44d06e5c5fd4d7ea8a7d

  ———

  6. buildCommonBaseImage()：把公共 prefix 构建成公共 base image

  对于每一种公共 prefix，脚本生成一个临时 Dockerfile：

  产出：

  output/.fastDockerBuild/cache/common_base_contexts/4e733d528bc6a1982b1bef6d/Dockerfile
  output/.fastDockerBuild/cache/common_base_contexts/2c709ccf4ecf19dd2694b07e/Dockerfile

  然后执行：

  docker build -t seedemu-fast-base:4e733d528bc6a1982b1bef6d ...
  docker build -t seedemu-fast-base:2c709ccf4ecf19dd2694b07e ...

  获得：

  两个公共 base image
  seedemu-fast-base:4e733d528bc6a1982b1bef6d
  seedemu-fast-base:2c709ccf4ecf19dd2694b07e

  ———

  7. extractImageArchive() + loadBaseArchive()：把公共 base image 保存成可复用 layers

  它对两个公共 base image 做：

  docker save seedemu-fast-base:<hash>

  然后展开到：

  output/.fastDockerBuild/cache/common_base_archives/4e733d528bc6a1982b1bef6d/
  output/.fastDockerBuild/cache/common_base_archives/2c709ccf4ecf19dd2694b07e/

  获得：

  base image 的 config JSON
  base image 的 layers 列表
  base image 的 layer 文件

  这些会被后面的 1897 个镜像复用。

  ———

  8. createRuntimeLayer()：为每个节点生成自己的小 layer

  这是你问的“后续 COPY 内容变成什么文件”。

  它不是单独落成一个普通文件在目录里，而是先在内存里生成一个 tar layer。这个 tar 里面放这个节点自己的文件：

  /start.sh
  /seedemu_sniffer
  /seedemu_worker
  /etc/bird/bird.conf
  /ifinfo.txt
  /interface_setup
  /etc/bird/conf/kernel.conf

  同时处理：

  RUN chmod +x /start.sh
  RUN chmod +x /seedemu_sniffer
  RUN chmod +x /seedemu_worker

  方式不是运行 shell，而是在 tar metadata 里把这些文件 mode 写成可执行。

  然后这个 layer 被写进总包：

  output/.fastDockerBuild/images_all.tar
    blobs/sha256/<node_layer_digest>

  所以节点 layer 的“文件”就是：

  images_all.tar 里面的 blobs/sha256/<digest>

  它的内容本身是一个 tar。

  ———

  9. createImageConfig()：为每个节点生成 image config

  这个 config 记录这个镜像的元数据：

  Cmd = ["/start.sh"]
  rootfs.diff_ids = base diff_ids + 当前节点 layer diff_id
  history += "SeedEMU runtime configuration layer"

  它也写进：

  images_all.tar
    blobs/sha256/<config_digest>

  ———

  10. writeDockerArchive()：写最终批量镜像包

  这个函数最终产出：

  output/.fastDockerBuild/images_all.tar

  里面核心结构是：

  manifest.json
  blobs/sha256/<base-layer-1>
  blobs/sha256/<base-layer-2>
  blobs/sha256/<node-layer-1>
  blobs/sha256/<node-layer-2>
  blobs/sha256/<image-config-1>
  blobs/sha256/<image-config-2>
  ...

  manifest.json 说明每个 image tag 怎么组成：

  topology-291e63c1-brdnode_1801_r12:latest =
    config: blobs/sha256/<config>
    layers:
      base layer 1
      base layer 2
      ...
      brdnode_1801_r12 自己的小配置 layer

  ———

  11. createFastArchive() 最后写 summary

  产出：

  output/.fastDockerBuild/summary.json

  这一轮里面记录了：

  status: PASS
  imageCount: 1897
  normalImageCount: 1
  baseCount: 2
  archiveBytes: 638371840
  durationSeconds: 23.87

  ———

  12. docker load -i images_all.tar：导入 Docker

  最后不是再构建，而是把已经写好的 archive 注册进 Docker：

  docker load -i output/.fastDockerBuild/images_all.tar

  获得：

  Docker image store 里出现 1897 个节点镜像 tag

  例如：

  topology-291e63c1-brdnode_1801_r12:latest
  topology-291e63c1-brdnode_1801_r13:latest
  ...

  之后 docker compose up 看到这些 tag 已经存在，就不需要 docker build 了。

  所以完整一句话就是：

  createFastArchive() 负责把公共 base layers、每节点小配置 layer、每节点 image config、
  manifest.json 写成 images_all.tar；docker load 负责把这个 tar 导入 Docker，让 Docker
  认识里面的所有完整镜像。