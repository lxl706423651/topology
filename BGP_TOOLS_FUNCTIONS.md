# SEED BGP Tools 功能清单

> 文档状态：审阅草案  
> 更新日期：2026-08-19  
> 目标读者：SEED Emulator 开发者、BGP 实验设计者、AI Agent/MCP 工具开发者

## 1. 文档目标

本文定义一套可供 AI Agent 调用的 SEED BGP Tools。工具面向已经部署并正在运行的 SEED Emulator/BIRD 实验环境，提供运行期查询、控制、验证、故障注入、收敛测量、RIB-to-FIB 写入、MRT 分析和实验报告能力。

本文是功能审阅清单，不代表所有功能已经完成统一 API 封装。每项功能均标注建议优先级和当前代码基础，便于后续确定 MVP 范围。

## 2. 明确排除的范围

下列内容不属于 BGP Tools：

- 不生成或修改 SeedEMU 拓扑。
- 不调用 SeedEMU compiler。
- 不执行 Docker/Compose 镜像构建、Fast Build 或基础镜像构建。
- 不创建、编译或安装 BIRD 二进制。
- 不编译、安装或切换 Linux 内核。
- 不创建 VM，不负责 VM/Docker 环境初始化或清理。
- 不提供 K2C 相关工具或实验入口。
- 不提供 K3C 相关工具或实验入口。
- 不提供任何 eBPF、BPF、BCC、kprobe、TC/XDP 或 BPF Map 相关能力。
- 不封装依赖 eBPF 的 traffic-observer、`traceRtnlOps*`、`7AutoConvergenceTest.py` 或 BPF dataplane。
- 不向 AI 暴露任意 Shell 命令执行接口。

工具的运行前提是：拓扑已经编译、容器已经部署，调用方能够提供 Compose project、容器名、AS 号或其他可解析的运行期目标。

### 2.1 K1 的边界

K1 可作为可选的运行期 FIB 写入后端，但仅在目标环境已经预装并正在运行兼容 K1 的内核时使用。BGP Tools 只负责能力探测、调用和结果验证，不负责 K1 内核的编译、安装、启动项修改或重启。

## 3. 设计原则

### 3.1 结构化接口

每个工具必须使用固定参数和结构化 JSON 返回值。不得让模型拼接 `docker exec`、`nsenter`、`birdc` 或 Shell 字符串。

### 3.2 读写分离

- 查询工具默认只读，可直接执行。
- 运行控制工具必须限制目标范围并记录审计日志。
- 配置、故障、路由和 FIB 变更采用 `plan -> execute -> verify -> rollback` 流程。

### 3.3 项目隔离

所有容器发现必须优先使用 Compose project label，不能默认扫描并操作宿主机上的全部容器。

### 3.4 可重复与可审计

每次调用生成 `operation_id`，记录输入、目标、开始/结束时间、执行结果、验证证据和产物路径。变更类工具还应支持 `idempotency_key`。

### 3.5 收敛判定

路由数量稳定只能作为信号之一，不能单独作为全网收敛的充分条件。正式判定应组合：

- BGP session 状态；
- BIRD protocol 状态；
- RIB route count 或 prefix digest；
- RIB/FIB 一致性；
- 普通 ping/traceroute 等数据面探测；
- 超时和连续稳定窗口。

## 4. 公共数据模型

### 4.1 TargetSelector

所有工具复用同一目标选择器：

```json
{
  "project_name": "output_1078",
  "asns": [1277],
  "routers": ["as1277brd-r13-..."],
  "protocols": ["Ebgp_*"],
  "prefixes": ["203.0.113.0/24"],
  "sample": {
    "mode": "first-router-per-as",
    "count": 10,
    "seed": 0
  }
}
```

选择优先级建议为：精确容器名 > ASN > project 全体。任何模糊选择必须在执行前返回解析后的目标列表。

### 4.2 公共返回结构

```json
{
  "schema_version": 1,
  "operation_id": "bgp-op-...",
  "tool": "bgp.sessions.list",
  "success": true,
  "read_only": true,
  "started_at": "...",
  "finished_at": "...",
  "target": {},
  "result": {},
  "warnings": [],
  "evidence": [],
  "artifacts": [],
  "error": null
}
```

### 4.3 风险等级

| 等级 | 含义 | 执行要求 |
|---|---|---|
| R0 | 只读查询 | 可直接执行 |
| R1 | 可恢复的运行状态变更 | 精确目标、超时、后置验证 |
| R2 | 配置、路由、故障或 FIB 变更 | 必须先生成 plan，并保存前置快照 |
| R3 | 大范围停止、FIB flush 等高影响操作 | 默认禁用，需要显式审批和恢复计划 |

## 5. 功能总览

优先级含义：

- P0：首批 MVP，现有代码基础充分。
- P1：第二阶段，需要统一封装或补充验证。
- P2：AI 增强能力，需要新的分析或规划逻辑。

| 功能域 | Tool | 风险 | 优先级 | 当前基础 |
|---|---|---:|---:|---|
| 运行期发现 | `bgp.runtime.inventory` | R0 | P0 | 可复用 Docker label、inspect 和 PID 缓存逻辑 |
| 会话查询 | `bgp.sessions.list` | R0 | P0 | `bgp_health_check.py`、`birdc show protocols` |
| 会话详情 | `bgp.session.get` | R0 | P0 | BIRD protocol 明细查询 |
| 路由查询 | `bgp.route.lookup` | R0 | P0 | `network_check.py`、`birdc show route` |
| 路由统计 | `bgp.route.count` | R0 | P0 | `check_routes.py`、`birdc show route count` |
| 路由差异 | `bgp.route.diff` | R0 | P1 | BIRD primary route dump 与 delta writer 解析逻辑 |
| 选路解释 | `bgp.route.explain` | R0 | P1 | 需补充 BIRD all route/attribute 解析 |
| RIB/FIB 对比 | `bgp.rib_fib.compare` | R0 | P0 | 现有 FIB gate 和 route count 逻辑 |
| 健康检查 | `bgp.health.check` | R0 | P0 | `bgp_health_check.py`、验证脚本 |
| 收敛状态 | `bgp.convergence.status` | R0 | P0 | 稳定窗口、协议健康和负载 gate 逻辑 |
| 日志查询 | `bgp.logs.search` | R0 | P1 | BIRD 日志注入和现有日志目录 |
| BIRD 控制 | `bgp.bird.control` | R1/R2 | P0 | `start_bird*.py`、`down_bird.py`、`manage_bird.sh` |
| 协议控制 | `bgp.protocol.set_state` | R1 | P0 | `birdc enable/disable/restart` |
| 配置读取 | `bgp.config.get` | R0 | P0 | 容器内 `/etc/bird` 配置读取逻辑 |
| 配置验证 | `bgp.config.validate` | R0 | P0 | `bird -p -c /etc/bird/bird.conf` |
| 配置应用 | `bgp.config.apply` | R2 | P1 | `update_birdConf.py`、`birdc configure` |
| 配置回滚 | `bgp.config.rollback` | R2 | P1 | 需增加版本化快照 |
| Peer 变更 | `bgp.peer.change` | R2 | P1 | 需增加受管 include 文件 |
| 路由注入 | `bgp.route.inject` | R2 | P1 | 需增加受管 static protocol/include 文件 |
| 故障计划 | `bgp.fault.plan` | R0 | P0 | 目标选择和影响范围计算 |
| 故障注入 | `bgp.fault.inject` | R2/R3 | P0 | `stop_fixed_victims_verify.py`、BIRD/session 控制 |
| 故障恢复 | `bgp.fault.recover` | R2 | P0 | BIRD 启动和后置验证逻辑 |
| 收敛测量 | `bgp.convergence.measure` | R1/R2 | P0 | 非 eBPF 的轮询式收敛脚本 |
| FIB 状态 | `bgp.fib.status` | R0 | P0 | `ip route` 与 BIRD RIB 检查 |
| FIB 同步 | `bgp.fib.sync` | R2 | P1 | Stock、F、Delta-F、预置 K1 |
| FIB 清理 | `bgp.fib.flush` | R3 | P1 | 现有 `proto bird` 清理逻辑 |
| MRT 采集 | `bgp.mrt.capture` | R1 | P1 | `3InjectMrtlog.sh`、`6CollectMrt.sh` |
| MRT 分析 | `bgp.mrt.analyze` | R0 | P1 | `analyze_mrt.py`、MRT parser |
| 数据面探测 | `bgp.dataplane.probe` | R0 | P0 | 普通 ping、traceroute/tracepath、iperf3 |
| 状态快照 | `bgp.snapshot.capture` | R0 | P0 | protocols、RIB、FIB、资源数据采集 |
| 产物管理 | `bgp.artifacts.list` | R0 | P0 | `logs/`、`bird/runs/`、evaluation result |
| 报告生成 | `bgp.report.generate` | R0 | P1 | `analyze_result_logs.py` 的非 eBPF 数据源部分 |
| 根因诊断 | `bgp.diagnose` | R0 | P2 | 组合 sessions、RIB、FIB、日志和普通探测 |
| 意图规划 | `bgp.intent.plan` | R0 | P2 | 需新增意图到 BIRD 配置差异的规划逻辑 |
| What-if | `bgp.what_if` | R0 | P2 | 需新增只读影响分析，不直接操作运行环境 |
| 安全检测 | `bgp.security.analyze` | R0 | P2 | MRT、AS_PATH、前缀和 community 分析基础 |
| 修复规划 | `bgp.remediation.plan` | R0 | P2 | 基于诊断结果生成可审阅计划 |

## 6. 运行期发现与目标解析

### 6.1 `bgp.runtime.inventory`

用途：发现一个已经运行的 SEED 项目中的 BGP 相关对象。

主要输入：

- `project_name`；
- 可选 `include_stopped`；
- 可选 `router_name_pattern`。

主要返回：

- 容器总数、运行中数量；
- router、route-server、IX 和其他角色分类；
- ASN 到 router 的映射；
- container ID、PID、service name、容器名；
- BIRD 是否可执行、控制 socket 是否响应；
- BIRD 版本和配置根目录；
- 当前宿主机 load average；
- 未识别或命名异常的对象。

约束：只读取指定 Compose project，默认不读取其他项目。

### 6.2 `bgp.target.resolve`

用途：在执行其他工具前，把 ASN、router pattern 或采样规则解析为确定的容器列表。

主要返回：

- 排序后的目标列表；
- 每个目标的选择原因；
- 被过滤、缺失或重复的目标；
- 预估影响的 AS/router 数量。

该能力可以作为公共内部函数，不一定单独暴露给最终用户。

## 7. BGP 会话与协议状态

### 7.1 `bgp.sessions.list`

用途：批量查询 eBGP/iBGP 会话。

主要输入：

- `target`；
- `session_type=all|ebgp|ibgp`；
- `state_filter`；
- `include_raw=false`。

主要返回：

- router、ASN、protocol name；
- peer address、peer ASN；
- BIRD state 和 BGP state；
- Established 持续时间；
- import/export route counters（BIRD 可提供时）；
- 未建立会话的聚合统计。

### 7.2 `bgp.session.get`

用途：查询一个 BGP session 的详细状态和最近错误。

主要返回：

- local/remote ASN 和地址；
- session 状态、连接方向、上次状态变化；
- hold/keepalive timer；
- graceful restart 信息；
- route refresh、capabilities；
- 最近错误和 BIRD 原始证据。

### 7.3 `bgp.protocol.set_state`

用途：精确 enable、disable 或 restart 一个或一组 BIRD protocol。

主要输入：

- `target`；
- `protocol_names` 或受限 pattern；
- `action=enable|disable|restart`；
- `verify_timeout`。

安全要求：

- 先返回解析后的 protocol 清单；
- 禁止空 pattern 和无界 `all`，除非显式审批；
- 操作后重新查询 protocol 状态；
- 返回成功、失败和未匹配列表。

## 8. 路由查询、解释与差异

### 8.1 `bgp.route.lookup`

用途：查询指定 router 对一个或多个 prefix 的路由。

主要输入：

- `target`；
- `prefixes`；
- `view=selected|all|export|protocol`；
- 可选 `protocol_name`。

主要返回：

- prefix、是否存在；
- selected 标记；
- source protocol；
- next hop、interface；
- AS_PATH；
- local preference、MED、origin；
- standard/large community；
- BIRD preference 和时间信息。

### 8.2 `bgp.route.count`

用途：统计 router 或 AS 的网络数、路由数和 primary route 数。

主要返回：

- BIRD networks/routes/primary routes；
- Linux FIB route count；
- 按 protocol 分类的 route count；
- 多节点的 min/max/average 和异常节点。

### 8.3 `bgp.route.diff`

用途：比较两份路由视图。

支持的比较方式：

- router A 与 router B；
- 同一 router 的两个 snapshot；
- 故障前与故障后；
- BIRD RIB 与 Linux FIB。

主要返回：

- added、deleted、changed、unchanged；
- next-hop changed、AS_PATH changed、attribute-only changed；
- prefix 数量和摘要；
- 可下载的完整差异产物。

### 8.4 `bgp.route.explain`

用途：回答“为什么这个 router 选择了这条路由”。

主要返回：

- 最优路由；
- 候选路由及来源 peer；
- preference/local-pref/AS_PATH/MED 等关键属性；
- 每个候选未被选择的原因；
- 最终 next hop 是否能在 FIB 中解析；
- RIB/FIB 是否一致。

注意：BIRD 的完整决策解释可能不能从单条 CLI 输出直接获得，初期实现应明确哪些结论是原始证据，哪些是根据属性做出的推断。

## 9. 健康检查与收敛

### 9.1 `bgp.health.check`

用途：对指定范围执行综合健康检查。

检查项：

- BIRD 进程和控制 socket；
- 配置语法；
- BGP session 是否 Established；
- OSPF/device/direct/kernel 等依赖协议是否正常；
- RIB 是否为空或明显偏少；
- RIB/FIB 基本一致性；
- 宿主机 load average；
- 可选普通 ping 探测。

返回结果按 `healthy|degraded|unhealthy|unknown` 分类，并给出失败证据，避免只返回布尔值。

### 9.2 `bgp.convergence.status`

用途：只读判断当前状态是否满足指定收敛条件。

可配置条件：

- 连续稳定轮数；
- 采样间隔；
- session 必须全部 Established；
- route count 或 prefix digest 必须稳定；
- 是否要求 RIB/FIB 检查通过；
- load threshold；
- 总超时。

### 9.3 `bgp.convergence.measure`

用途：围绕一次已授权操作，测量从事件发生到控制面或数据面恢复的耗时。

支持的结束条件：

- BGP sessions 恢复；
- RIB 连续稳定；
- RIB/FIB 检查通过；
- 指定 prefix 恢复或撤销；
- 普通 ping 恢复。

该工具使用轮询和 BIRD/MRT/系统命令，不使用 eBPF 事件。

## 10. BIRD 生命周期与配置

### 10.1 `bgp.bird.control`

用途：控制指定 router 上的 BIRD。

支持动作：

- `start`；
- `graceful_down`；
- `restart`；
- `configure`；
- `status`。

安全要求：

- `start` 前清理残留 PID/socket 只能作用于目标容器；
- `restart` 前保存 protocol/RIB 摘要；
- 每个目标设置超时；
- 操作后检查 `birdc show status` 和 protocol health；
- 批量操作支持节流和最大并发数。

### 10.2 `bgp.config.get`

用途：读取 BIRD 主配置及受管 include 文件。

返回：

- 文件列表、内容摘要和 SHA-256；
- include 关系；
- protocol 列表；
- timer、filter 和关键配置摘要；
- 原始文件作为可选产物。

### 10.3 `bgp.config.validate`

用途：在不改变运行状态的情况下验证候选配置。

验证项：

- BIRD 语法检查；
- include 文件存在性；
- protocol name 冲突；
- peer 地址/ASN 基本合法性；
- filter 引用；
- 与当前配置的结构化 diff。

### 10.4 `bgp.config.apply`

用途：应用已通过验证的配置计划。

执行流程：

1. 校验 `plan_id` 和当前配置 hash；
2. 保存配置快照；
3. 原子写入受管文件；
4. 再次执行语法检查；
5. `birdc configure`；
6. 检查目标 protocol 和 session；
7. 失败时自动恢复快照。

### 10.5 `bgp.config.rollback`

用途：恢复某个历史配置快照。

必须指定 `snapshot_id`，并执行与 `config.apply` 相同的语法和后置健康检查。

## 11. Peer 和路由注入

### 11.1 `bgp.peer.change`

用途：在不重新编译拓扑的前提下，修改运行期 BIRD peer 配置。

支持动作：

- 新增 peer；
- 删除由 BGP Tools 管理的 peer；
- 修改 import/export filter；
- 修改 timer；
- enable/disable peer。

实现约束：

- 只修改 BGP Tools 专用 include 文件；
- 不直接重写 SeedEMU 生成的整个 `bird.conf`；
- 先生成配置 diff 和影响范围；
- 删除操作只能删除具有工具管理标记的 peer。

### 11.2 `bgp.route.inject`

用途：在隔离实验中宣告、修改或撤销测试前缀。

支持动作：

- `announce`；
- `withdraw`；
- `replace_attributes`。

可控属性：

- prefix；
- next hop；
- community/large community；
- MED；
- 可选测试标签。

安全要求：

- 仅允许实验地址范围或项目 allowlist；
- 默认拒绝默认路由和宿主机管理网段；
- 注入项必须带 `operation_id` 标记；
- 支持按 operation 精确撤销；
- 注入后验证目标 prefix 是否进入预期 RIB。

## 12. 故障注入与恢复

### 12.1 `bgp.fault.plan`

用途：只读生成故障计划。

支持故障类型：

- BIRD graceful down；
- BIRD process restart；
- 指定 BGP protocol disable/restart；
- 指定 prefix withdraw；
- 精确容器 stop；
- 多个不同 AS 的确定性采样故障。

主要返回：

- 精确目标；
- 预计影响的 AS/session/prefix；
- 基线采集项；
- 收敛判定条件；
- 恢复步骤；
- 风险等级。

### 12.2 `bgp.fault.inject`

用途：执行已审批的故障计划。

要求：

- 只能执行未过期的 `plan_id`；
- 故障前保存 session、RIB、FIB 和目标运行状态；
- 记录精确事件时间；
- 支持最大故障 AS/router 数限制；
- 默认禁止同时停止 route server 或所有上游；
- 返回部分成功状态，不能把部分失败报告为成功。

### 12.3 `bgp.fault.recover`

用途：按计划恢复故障目标。

恢复后依次验证：

1. 容器运行；
2. BIRD socket 可用；
3. 相关 protocol 正常；
4. BGP sessions Established；
5. RIB 稳定；
6. 可选 FIB 和普通数据面探测。

## 13. RIB-to-FIB 工具

### 13.1 `bgp.fib.status`

用途：只读检查 Linux FIB 和 BIRD RIB。

主要返回：

- BIRD network/route/primary count；
- Linux main-table route count；
- `proto bird` route count；
- 指定 prefix 的 RIB/FIB next hop；
- 缺失、stale 或 next-hop 不一致的采样结果；
- 当前可用 FIB backend 能力。

### 13.2 `bgp.rib_fib.compare`

用途：对 RIB 和 FIB 做比 route count 更严格的比较。

模式：

- `sample`：确定性抽样 prefix；
- `digest`：规范化后比较 prefix/next-hop digest；
- `full`：输出完整 added/deleted/changed 差异。

### 13.3 `bgp.fib.sync`

用途：把当前 BIRD selected routes 写入 Linux main FIB。

允许后端：

| Backend | 用途 | 边界 |
|---|---|---|
| `stock-kernel` | 使用普通 BIRD kernel protocol | 不改内核，不编译 BIRD |
| `F` | 用户态读取 BIRD primary routes，使用受控 `ip -batch` writer | 可移植的全量写入方案 |
| `delta-F` | 比较 baseline/current/FIB，只写 added/changed/deleted | 适合故障后增量刷新 |
| `bird-delta-F` | 使用已经部署的 patched BIRD delta event 和用户态 writer | 工具不负责编译或安装 patched BIRD |
| `K1` | 调用预置 K1 环境的运行期 bulk API | 仅在能力探测通过时开放；不编译/安装内核 |

明确不提供 K2C、K3C 或任何 BPF backend。

主要输入：

- `target`；
- `backend`；
- `mode=full|delta`；
- `jobs`、`batch_size`、`timeout`；
- `pre_cleanup`；
- `verify_mode=sample|digest|full`。

安全要求：

- 执行前确认 backend capability；
- 保存 RIB/FIB 摘要；
- 限制最大并发并监控 load average；
- 记录 route ops、失败数和 delete miss；
- 写入后必须执行独立 RIB/FIB 检查；
- 不因为命令 return code 为 0 就判定状态正确。

### 13.4 `bgp.fib.flush`

用途：清理由实验写入的 `proto bird` 路由。

边界：

- 默认只允许精确 project 内的 router netns；
- 默认只清理指定 protocol/metric；
- 操作前输出将删除的 route 数和样本；
- 不允许无过滤条件的主表 flush；
- 属于 R3，默认禁用。

## 14. MRT 与非 eBPF 观测

### 14.1 `bgp.mrt.capture`

用途：在指定 BIRD 节点启停 MRT 记录并回收产物。

支持动作：

- `start`：写入受管 MRT 配置并 `birdc configure`；
- `status`：查看当前文件和大小；
- `stop`：停止记录、恢复配置；
- `collect`：把 MRT 文件复制到实验目录并计算 hash。

### 14.2 `bgp.mrt.analyze`

用途：分析 MRT 中的 BGP UPDATE。

主要返回：

- UPDATE、announcement、withdrawal 数；
- 唯一 prefix 数；
- AS_PATH 长度分布；
- community/large community 统计；
- 指定 AS 邻接出现次数；
- 指定 prefix/ASN 的时间线；
- route flap 统计；
- 解析错误和不支持的消息类型。

### 14.3 `bgp.logs.search`

用途：在指定运行或时间窗口内检索 BIRD 日志。

过滤条件：

- router/ASN；
- protocol；
- state transition；
- error/warning；
- prefix、peer address；
- 时间范围。

### 14.4 `bgp.dataplane.probe`

用途：使用普通用户态工具验证数据面，不依赖 eBPF。

支持 probe：

- `ping`；
- `traceroute` 或 `tracepath`；
- `ip route get`；
- `iperf3`（目标已经提供服务时）。

主要返回：

- 命令目标和源地址；
- 成功率、RTT、hop 信息；
- `ip route get` 结果；
- 原始日志产物。

该工具不能声称获得逐容器精确转发事件；没有非 eBPF 证据时，应明确返回“仅观察到端到端结果或 traceroute 跳点”。

## 15. 快照、产物和报告

### 15.1 `bgp.snapshot.capture`

用途：为诊断、故障和实验保存一致的状态快照。

可选内容：

- runtime inventory；
- BIRD status/protocols；
- route count；
- 指定 prefix routes；
- primary RIB dump；
- Linux FIB dump；
- load、CPU、memory；
- 配置 hash；
- 普通数据面 probe。

### 15.2 `bgp.artifacts.list`

用途：列出某次 operation/experiment 的日志、JSON、MRT、CSV 和文本报告。

要求：

- 返回大小、hash、MIME/type 和生成阶段；
- 只允许访问配置的 artifact root；
- 不把任意宿主机路径暴露为下载接口。

### 15.3 `bgp.report.generate`

用途：根据结构化产物生成 Markdown 和 JSON 报告。

报告章节建议：

1. 实验目标与范围；
2. 环境和目标清单；
3. 基线健康；
4. 操作或故障时间线；
5. session/RIB/FIB 收敛；
6. MRT 更新与撤销；
7. 普通数据面验证；
8. 失败项和异常；
9. 结论、限制和产物索引。

报告生成器应忽略 K2C、K3C、eBPF/BPF 专属数据源，不在本工具集中生成相应章节。

## 16. AI 增强工具

### 16.1 `bgp.diagnose`

用途：对“为什么 session 不通”“为什么 prefix 消失”“为什么 RIB 有而 FIB 没有”等问题执行关联诊断。

诊断顺序建议：

1. 目标解析和运行状态；
2. BIRD/socket/config；
3. 依赖协议；
4. BGP session；
5. prefix RIB；
6. next-hop 可达性；
7. Linux FIB；
8. 普通数据面 probe；
9. BIRD log/MRT 时间线。

返回候选根因及置信度，并把事实、推断和建议分开。

### 16.2 `bgp.intent.plan`

用途：把自然语言策略意图转换成可审阅计划，例如：

- “AS1277 不接受经过 AS1304 的路由”；
- “只向 peer X 宣告指定测试前缀”；
- “把某个 peer 的 local preference 调低”。

只生成：

- 结构化意图；
- 候选 BIRD filter/config diff；
- 目标和影响范围；
- 验证条件；
- 回滚方案。

该工具不直接应用配置，应用必须交给 `bgp.config.apply`。

### 16.3 `bgp.what_if`

用途：只读估算 peer down、router down、prefix withdraw 或 filter 变化的影响。

返回：

- 预计受影响的 session、prefix 和 AS；
- 可能的备用路径；
- 当前证据不足的部分；
- 建议的最小化实验计划。

初期版本基于当前拓扑关系、RIB snapshot 和 AS_PATH 做静态推断，不应把推断表述为真实执行结果。

### 16.4 `bgp.security.analyze`

用途：对 RIB snapshot 或 MRT 数据执行路由安全分析。

检测项：

- 指定伪造 AS 邻接；
- AS_PATH loop；
- 私有/保留 ASN 异常；
- MOAS；
- 非预期 origin；
- 更具体前缀异常；
- route leak 候选；
- 异常长路径；
- community 变化；
- 高频 announce/withdraw flap。

输出必须注明 ground truth 来源。没有 ROA、关系数据或预设标签时，只能返回异常候选，不能直接断言劫持或泄漏。

### 16.5 `bgp.remediation.plan`

用途：根据诊断结果生成修复计划。

计划内容：

- 根因和证据；
- 最小变更目标；
- 候选配置或控制动作；
- 预期影响；
- canary 目标；
- 验证条件；
- 回滚条件和命令语义。

修复计划必须由独立的变更工具执行，诊断工具本身保持只读。

## 17. 建议的 MVP 范围

首批建议实现以下 14 个工具：

1. `bgp.runtime.inventory`
2. `bgp.sessions.list`
3. `bgp.session.get`
4. `bgp.route.lookup`
5. `bgp.route.count`
6. `bgp.rib_fib.compare`
7. `bgp.health.check`
8. `bgp.convergence.status`
9. `bgp.bird.control`
10. `bgp.protocol.set_state`
11. `bgp.config.get`
12. `bgp.config.validate`
13. `bgp.fault.plan`
14. `bgp.snapshot.capture`

MVP 首先建立统一目标模型、JSON schema、超时、并发控制和 artifact 目录，不立即开放配置应用、路由注入、FIB flush 等高风险能力。

第二阶段建议增加：

- `bgp.route.diff`；
- `bgp.route.explain`；
- `bgp.config.apply/rollback`；
- `bgp.peer.change`；
- `bgp.route.inject`；
- `bgp.fault.inject/recover`；
- `bgp.convergence.measure`；
- `bgp.fib.status/sync`；
- `bgp.mrt.capture/analyze`；
- `bgp.dataplane.probe`；
- `bgp.report.generate`。

## 18. 与当前代码的整理映射

| 现有文件组 | 建议归属 | 处理建议 |
|---|---|---|
| `bgp_health_check.py`、`check_routes.py`、`network_check.py` | health/route/rib-fib services | 提取解析器，统一 JSON 返回 |
| `start_bird.py`、`start_bird0130.py`、`start_bird0318.py`、`start_bird1.py`、`start_bird2.py`、`start_bird_ebgp.py` | bird lifecycle service | 合并为参数化实现，旧脚本暂作兼容入口 |
| `bird2_monitor.py`、`down_bird.py`、`manage_bird.sh` | bird/protocol control | 统一目标解析、超时和后置验证 |
| `start_bird0318_verify.py` | health/convergence | 抽出稳定窗口和 protocol health 逻辑，取消固定 5-AS 限制 |
| `stop_fixed_victims_verify.py` | fault/convergence | 参数化 victim 选择、基线和恢复流程 |
| `update_birdConf.py` | config service | 增加快照、原子写入、hash 检查和回滚 |
| `reconfergence_F.sh`、`reconfergence_delta_F.sh` | F/delta-F backend | 把 shell 环境变量变成结构化参数 |
| `bird/delta_fib_agent.py`、`bird/delta_fib_host_writer.py` | bird-delta-F backend | 仅封装已部署环境，不加入 BIRD 编译/安装 |
| `bird/kernel/tools/kernel_bulk_birdc_writer.py` | optional K1 backend | 仅能力探测通过时注册，不加入内核管理 |
| `3InjectMrtlog.sh`、`6CollectMrt.sh`、`analyze_mrt.py` | MRT service | 参数化目标、输出目录和分析条件 |
| `8IperfTest.py`、普通 ping/route 检查逻辑 | dataplane probe | 只保留非 eBPF probe |
| `analyze_result_logs.py` | report service | 仅复用通用和非 eBPF 解析部分 |

明确不纳入整理：

- `bird/bpf_dataplane/`；
- K2C/K3C kernel patch、writer、wrapper 和实验结果；
- `traceRtnlOps.py`、`traceRtnlOps1.py`；
- `7AutoConvergenceTest.py`；
- `4profier.py`；
- 依赖 BCC/eBPF 的 profiler、traffic observer 和路径捕获流程；
- `autocoder*.py`、`RR_1897.py`、SeedEMU compile/build/up 工具。

## 19. 审阅时需要确认的决策

1. K1 是否按本文方案保留为“预置环境可选后端”，还是从首版完全删除。
2. 是否允许 BGP Tools 管理独立的 BIRD include 文件，以支持 peer 变更和路由注入。
3. `bgp.fib.sync` 首版只支持 F/delta-F，还是同时支持 stock kernel protocol。
4. 故障注入首版是否允许 `docker stop`，还是只允许 BIRD/protocol 级故障。
5. RIB/FIB 全量比较的最大规模和超时预算。
6. AI 是否只能生成变更 plan，还是允许在显式审批后调用 execute 类工具。
7. artifact 保留时间、最大 MRT/RIB dump 大小和脱敏要求。

## 20. 验收标准

一个 BGP Tool 只有满足以下条件才视为可交付：

- 有稳定的名称、输入 schema 和输出 schema；
- 所有目标均绑定指定 project；
- 有超时、并发上限和部分失败表达；
- 只读与变更语义清晰；
- 变更类工具有 plan、前置快照、后置验证和回滚；
- 不接受任意 Shell；
- 日志和产物可由 `operation_id` 追踪；
- 有最小规模单元/集成测试；
- 有失败场景测试；
- 不包含编译、K2C、K3C 或 eBPF/BPF 能力；
- 文档说明实验性限制，不把采样验证表述为全网严格证明。
