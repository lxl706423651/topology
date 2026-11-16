# 不同规模的 RR Internet 实验

本例从 `topology/input/real_topology_<规模>.txt` 读取网络拓扑，使用同目录的
`assignment.pkl` 映射 AS/IX 编号与地址。`large_internrt.py` 在 `seedpy310`
环境中构建每个 AS 的环形内部网络、一个 Route Reflector、OSPF/iBGP 和 IX
上的 eBGP 对等关系，再编译为 Docker Compose 和节点镜像构建目录。

分阶段启动所需的 BIRD/kernel 行为由下面的 **SeedEMU 手动准备** 提供。
`large_internrt.py` 直接使用当前环境中的 SeedEMU 完成 render/compile，
不自动改写框架源码或节点的 BIRD 启动命令、kernel 配置。

运行顺序为：

```text
topology/input/{assignment.pkl, real_topology_<规模>.txt}
       ↓ large_internrt.py
output/{docker-compose.yml, 节点目录, topology-summary.json}
       ↓ buildDockerImages.py
公共镜像层 + 节点配置层 → docker load
       ↓ startDocker.sh
Compose 分批启动容器，等待所有 BIRD 节点 running
       ↓ startBird.py
启动 BIRD → load1 < 50 → 固定 AS 样本协议检查
       ↓ startKernel.py
逐路由器开放 kernel export → load1 < 50 → 每个 AS 的 RIB/FIB 检查
```

容器启动和 BIRD 启动是独立阶段。初始 kernel 协议只导出直连与 OSPF
路由，确保 AS 内部可达；BGP 路由先在控制平面传播，最后由 `startKernel.py`
将导出过滤器切换为 `export all`。

## 文件职责

| 文件或目录 | 类型    | 职责 |
|---|---|---|
| `large_internrt.py` | 编译入口 | 从迁入的 `RR_214_example.py` 整理而来；按要求保留此文件名，生成通用 RR Docker 拓扑，不依赖 AnyPro 专用 AS180、PoP 或客户端。 |
| `topology/input/assignment.pkl` | 用户输入 | AS/IX 映射。只读取纯数据 pickle，不改写源文件。 |
| `topology/input/real_topology_*.txt` | 用户输入 | 不同规模的原始拓扑；输入目录只保留这两类文件。 |
| `config.yaml` | 用户配置 | 默认规模、输出路径、Compose 项目名、并发数和负载门限。 |
| `buildDockerImages.py` | 构建脚本 | 复用公共基础层，把各节点文件做成配置层，生成镜像归档并 `docker load`。 |
| `startDocker.sh` | 启动入口 | 使用当前 Python 调用 `runExperiment.py up`，分批、并行启动已构建的服务。 |
| `startBird.py` | 控制平面脚本 | 按 Compose project 隔离容器、缓存容器 ID/PID、启动 BIRD 并核验协议。 |
| `startKernel.py` | 数据平面脚本 | 验证 PID 缓存和前置协议关卡，备份 kernel 配置，逐路由器启用导出并核验 FIB。 |
| `runExperiment.py` | 总控脚本 | 按顺序执行或断点继续；同时提供阶段共用的配置读取、容器发现和运行记录函数。 |


## runExperiment.py 中合并的共用逻辑

删除独立公共模块后，以下函数并入现有总控入口：

| 逻辑 | 函数或方法 | 作用 |
|---|---|---|
| 配置与记录 | `loadSettings`、`writeJson` | 读取 YAML、解析相对路径和默认值，写入阶段 JSON 摘要。 |
| 节点发现 | `readCompose`、`ExperimentContext.discoverContainers` | 按 Compose 元数据统计实际规模，通过项目 label 限定运行容器并取得 ID/PID。 |
| 命令与负载控制 | `runCommand`、`runInNamespace`、`waitForLoad` | 统一单次命令超时、命名空间执行和本机 load 等待。 |
| 分批启动 | `startDocker` | 使用 YAML 中的批大小和并发数执行 Compose，等待全部 BIRD 节点 running；`startDocker.sh` 调用此入口。 |
| 协议/FIB 关卡 | `selectRepresentatives`、`parseNetworkCount`、`findBadProtocols`、`verifyState` | 固定 AS/路由器样本、解析路由计数、判断协议状态及 RIB/FIB 数量关系。 |

`large_internrt.py`、`startBird.py`、`startKernel.py` 直接从这里导入所需函数；
导入模块只定义函数和类，不执行阶段命令。总控自身的 `executeStage` 和
`runSelectedExperiment` 负责按 compile/build/up/bird/kernel 调用脚本、断点继续
和记录阶段结果。


#选择规模



当前提供 214、1078、1897、2599、3083、3410、3821、4192、4954、5387、5949、
6516、7001、7750、8528、9042、9245、9955 规模。

规模指 **路由器数 + IX 路由服务器数**。例如 3083 输入对应 312 个 AS、
2757 个路由器和 326 个路由服务器；4954 对应 674 个 AS、4542 个路由器和
412 个路由服务器。辅助服务不计入规模。当前 SeedEMU development 的 3083
编译结果是 3084 个服务（3083 个节点加一个基础镜像辅助服务）。

loopback 固定使用 `10.0.0.0/16`。assignment 统一由
`~/dataTopology/20260915final/build_assignment.py` 预先生成：若已有映射占用了
该段，则将它及之后的所有网段顺延一个 `/16`，保持键名、顺序和 AS/IX 编号不变。
例如原 `10.0.0.0/16` → `10.1.0.0/16`，原 `10.1.0.0/16` → `10.2.0.0/16`；
`9.255.0.0/16` 不变。编译器直接读取这个新 assignment，不临时改地址或切换 loopback。

重新生成并复制 assignment 的命令（在本目录、seedpy310 环境下执行）：

```bash
python ~/dataTopology/20260915final/build_assignment.py \
  --source-assignment topology/input/assignment.pkl \
  --output-assignment ~/dataTopology/20260915final/assignment.pkl
cp ~/dataTopology/20260915final/assignment.pkl topology/input/assignment.pkl
```

此入口保留旧 RR 文件使用的城市名/AS 键，不混用新数据库的 `pdb:*` 映射。
重复处理已经跳过 loopback 的 assignment 不会再次顺延。当前新文件共有 4695 项，
其中 2392 个网段已顺延；输入目录仍只保留 assignment 和各规模 `.txt` 文件。

本编译器接受原 RR 的五字段格式：`ixps`、`transit_asns`、`stub_asns`、
`ix_ix_transit_edges`、`as_as_ix_edges`；当前要求 `stub_asns` 为空。原目录中的
300、400 文件采用另一种 `as_ix`/`rs_groups` 格式，未混入本例输入。
添加新规模时，将同格式文件放到 `topology/input/real_topology_<规模>.txt`，
并确保 assignment 覆盖其所有 AS/IX，然后先执行输入检查：

```bash
python large_internrt.py 3083 --check-only
python large_internrt.py 4954 --check-only
```

## SeedEMU 手动准备

先用上面的 `seedemu.__file__` 输出确定 `seedpy310` 实际导入的源码目录。
以下内容由使用者手动修改 **`seedemu/layers/Routing.py`**。本例的编译脚本
不会自动应用这些修改，本轮也不执行这些框架修改的编译或运行验证。

### 1. 让容器启动时保留 BIRD 运行目录，但不启动 BIRD

在 `Routing._configure_rs()` 的 BIRD 分支中，保留创建 `/run/bird` 的命令，
注释或移除启动命令：

```python
rs_node.appendStartCommand('[ ! -d /run/bird ] && mkdir /run/bird')
# rs_node.appendStartCommand('bird -d', True)
```

在 `Routing._configure_bird_router()` 中做相同调整：

```python
rnode.appendStartCommand('[ ! -d /run/bird ] && mkdir /run/bird')
# rnode.appendStartCommand('bird -d', True)
```

Docker 编译器会根据这些节点启动命令生成 `/start.sh`，因此完成上述调整后，
`startDocker.sh` 只启动容器，BIRD 由后续的 `startBird.py` 统一启动。

### 2. 将路由器 kernel 协议拆到独立配置文件

在同一文件的 `RoutingFileTemplates["rnode_bird"]` 中，将原来内嵌的
`protocol kernel` 配置替换为一个 include。保留 router ID、`t_direct`
和 device 协议：

```python
RoutingFileTemplates["rnode_bird"] = """\
router id {routerId};
ipv4 table t_direct;
protocol device {{
}}
include "/etc/bird/conf/kernel.conf";
"""
```

这里是带 `.format()` 占位符的模板，因此 device 的大括号仍需写成 `{{`、`}}`。

### 3. 两个 kernel 模板与实验切换

`seedemu/layers/Routing.py` 现在提供两个模板：

| 模板键 | 导出规则 | 用途 |
|---|---|---|
| `rnode_kernel_conf1` | 仅接受 `RTS_DEVICE`、`RTS_OSPF`，拒绝其它来源 | 初始控制平面阶段，BGP 路由暂不写入 Linux FIB。 |
| `rnode_kernel_conf2` | `export all` | 完整数据平面阶段，向 Linux FIB 导出全部选中路由。 |

两者默认使用相同的 BIRD 协议实例名 `kernel1`，以及 `merge paths on`、
`persist`、`scan time 60000`、`import none`。这些参数可以在各自模板中修改。
模板直接传给 `setFile()`，不经过 `.format()`，所以模板内部使用普通大括号。

当前 `_configure_bird_router()` 默认选择模板 1：

```python
rnode.setFile("/etc/bird/conf/kernel.conf", RoutingFileTemplates["rnode_kernel_conf1"])
```

若要在编译时直接使用完整导出配置，将这处键改为 `rnode_kernel_conf2`。
调整选中的模板或模板选择后，需要重新编译拓扑和构建镜像；已编译目录、镜像
和运行中的容器不会自动读取源码模板的后续修改。

采用分阶段流程时，默认先编译模板 1，再由 `startBird.py` 启动 BIRD；
`startKernel.py` 将容器现有配置的导出子句改为 `export all`，保留现有协议名、
scan time、merge paths、persist、import 等参数。它切换的是已生成配置的导出
规则，不会自动读取模板 2 中另行修改的其它参数。

BIRD 自动启动的开关仍按上面第 1 步设置；这次只增加模板和默认模板引用，
没有改变路由器或 IX 路由服务器的自动启动命令。

## 逐阶段运行

默认规模在 `config.yaml` 中设置为 3083。以下两条编译命令等价：

```bash
python large_internrt.py
python large_internrt.py 3083
```

编译读取输入并生成 `output/`，其中 `topology-summary.json` 记录实际规模、
输入 SHA256、SeedEMU 包路径和平台。然后依次执行：

```bash
# 输入：output/docker-compose.yml；输出：公共基础层、节点镜像归档及本机镜像。
python buildDockerImages.py output/docker-compose.yml

# 输入：Compose + 已构建镜像；输出：运行容器与启动日志。
./startDocker.sh --experiment-dir evaluationResult/manual3083

# 输入：运行容器；输出：PID 缓存、BIRD 进程和固定协议样本结果。
sudo "$(command -v python)" startBird.py --experiment-dir evaluationResult/manual3083

# 输入：同一实验目录的成功协议检查和 PID 缓存；输出：内核导出与逐 AS FIB 结果。
sudo "$(command -v python)" startKernel.py --experiment-dir evaluationResult/manual3083
```

最后两步必须使用同一个 `--experiment-dir`。核验缓存时会同时比对 Compose
SHA256、项目名、容器 ID 和 PID；容器重建后要重新执行 `startBird.py`。

切换规模前，先结束旧规模实验，再重新编译；默认项目名相同，不能同时用它运行
两个规模。结束默认项目的命令如下，它会删除该项目的容器与网络：

```bash
docker compose -p large_scale -f output/docker-compose.yml down
python large_internrt.py 4954 --overwrite
```

`--overwrite` 只接受包含 Compose 和本编译器摘要的生成目录，并拒绝覆盖源码、
输入目录或其父目录。也可以先用 `--output output_4954` 编译到独立目录；后续脚本
通过 `--compose output_4954/docker-compose.yml` 选择该产物。

## 一次执行或断点继续

```bash
# 只检查输入和宿主依赖；不构建镜像、不创建容器。
python runExperiment.py --scale 3083 --preflight-only

# 从编译开始执行完整流程。
python runExperiment.py --scale 3083

# 已经手动编译好 output/ 时，从构建开始。
python runExperiment.py --from-stage build

# 使用原实验目录从容器启动阶段继续。
python runExperiment.py --from-stage up \
  --experiment-dir evaluationResult/3083nodes_原时间戳

# 从 kernel 阶段继续，必须复用成功执行过 BIRD 阶段的原实验目录。
python runExperiment.py --from-stage kernel \
  --experiment-dir evaluationResult/3083nodes_原时间戳
```

可选阶段是 `compile`、`build`、`up`、`bird`、`kernel`。恢复阶段会读取
`output/topology-summary.json` 获取真实规模；显式指定的 `--scale` 若不匹配会失败。
总控也支持 `--output`、`--config` 和编译时的 `--overwrite`。

## 配置字段

所有字段均可省略并采用表中的默认值。持久配置集中在 `config.yaml`，无需手动
设置环境变量。路径相对于 YAML 所在目录解析；显式 CLI 路径相对于当前目录解析。

| 字段 | 默认值 | 消费方与含义 |
|---|---|---|
| `topology.scale` | `3083` | 编译器/总控；输入文件名中的规模，CLI 规模优先。 |
| `topology.input` | `topology/input` | 编译器；assignment 与拓扑文件目录。 |
| `topology.output` | `output` | 编译器与运行脚本；默认编译产物目录。 |
| `topology.platform` | `amd` | 编译器；`amd` 或 `arm`。 |
| `docker.project` | `large_scale` | 所有阶段；Compose 项目及镜像名称，限制容器发现范围。 |
| `docker.parallel_jobs` | `8` | `startDocker.sh`；并行 Compose 进程数。 |
| `docker.batch_size` | `50` | `startDocker.sh`；每批服务数。 |
| `docker.ready_timeout` | `900` 秒 | `startDocker.sh`；等待全部 BIRD 容器 running。 |
| `bird.start_delay` | `0.02` 秒 | `startBird.py`；逐节点启动间隔。 |
| `bird.post_start_delay` | `60` 秒 | `startBird.py`；启动后的初始等待。 |
| `kernel.load_threshold` | `110` | `startKernel.py`；高于该 load1 时暂停更新。 |
| `kernel.recovery_threshold` | `50` | `startKernel.py`；暂停后等待 load1 低于该值。 |
| `kernel.update_interval` | `0.5` 秒 | `startKernel.py`；逐路由器更新间隔。 |
| `verification.load_threshold` | `50` | 两个验证关卡；验证前 load1 必须低于此值，配置不得大于 50。 |
| `verification.check_interval` | `30` 秒 | 运行脚本；负载检查间隔。 |
| `verification.command_timeout` | `60` 秒 | 运行脚本；单次发现、命名空间操作和检查命令的超时。 |

## 运行结果与验证边界

默认记录目录为 `evaluationResult/<规模>nodes_<时间戳>/`；手动阶段可通过
`--experiment-dir` 指定同一个目录。`summary.json` 保存总控结果，`compile.log`、
`build.log`、`up.log`、`bird.log`、`kernel.log` 保存阶段输出；Compose 每批日志
位于 `dockerUp/`，镜像构建细节位于 `dockerBuild/`。

- `startBird.json`：保存 `start_bird` 与独立的 `verify_after_start_bird`。
  等待本机 load1 < 50 后，按 ASN 升序取前 10 个 AS（不足则取全部），每个
  AS 取路由器编号最小的固定代表，检查 `birdc show protocols` 和路由计数；
  预期协议必须 up，BGP 必须 Established。
- `startKernel.json`：保存 `start_kernel` 与独立的 `verify_after_fib_write`。
  写入后等待 load1 < 50，再检查每个 AS 编号最小的路由器，要求 BIRD 的
  network count 严格小于 `ip route | wc -l`。失败保存计数和原始输出并停止后续阶段。
- `runtime/containerPids.json`：绑定本次 Compose/项目的容器 ID 与 PID 缓存。
- `runtime/kernelConfigs/`：开放 kernel 导出前的配置备份。配置解析或重配置
  失败时尝试恢复当前路由器配置，不把失败计作成功。

示例模板中的 kernel 协议命名为 `kernel1`，先用 `bird -p -c /etc/bird/bird.conf`
检查配置，再执行 `birdc configure` 应用变更；不再调用旧的 `birdc reload kernel`。
所有操作均在运行此脚本的机器上执行，本例不自动创建或重置虚拟机。

可选的脚本语法检查命令：

```bash
python -m py_compile large_internrt.py buildDockerImages.py \
  runExperiment.py startBird.py startKernel.py
bash -n startDocker.sh
```

这些语法检查不启动容器，也不证明真实网络已经收敛。实际运行必须以两个独立
协议/FIB 关卡的结果为准。
