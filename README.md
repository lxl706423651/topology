# topology

本目录是独立 Git 仓库，远程为 `git@github.com:lxl706423651/topology.git`。

当前机器的实际文件位于 `/data/lxl/seed-emulator/topology`。`~/topology` 指向这个目录，可直接使用：

```bash
cd ~/topology
git status
```

`~/seed-emulator/topology` 也指向同一目录，并通过 SeedEMU 仓库的本地 `.git/info/exclude` 忽略。实际文件与独立 Git 历史位于 SeedEMU 工作区之外，切换 SeedEMU 分支不会删除它们。

## 备份范围

- 收录顶层脚本以及 `0924`、`bird`、`fullMeshExample`、`llm`、`quant` 中的源代码、文档、构建配置、配置目录内的轻量输入文件和文档引用的本地图像。
- `quant` 和 `bird/bird-2.17.2-delta` 在本机仍有各自的 Git 元数据。本仓库保存它们的文件快照，不使用子模块；克隆本仓库即可取得这些代码和文档。
- 虚拟机磁盘、原始数据集、缓存、实验输出、生成的容器目录、构建产物和大型模型文件留在本机。已有历史中跟踪的输入文件继续保留。
- `.gitignore` 使用递归文件类型白名单，避免再次遗漏子目录代码和文档。文档引用的图像以及 BIRD 的无扩展名构建工具也纳入本次快照。

在新机器上可使用以下命令恢复代码与文档；运行实验所需的大型本地数据需要另行准备：

```bash
git clone git@github.com:lxl706423651/topology.git ~/topology
```

本次操作只保存现有文件和调整备份规则，没有运行容器、虚拟机、路由实验或量化回测。
