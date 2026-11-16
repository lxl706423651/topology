# 完整永久底仓核心队列复用接口

只新增 `quant_backtest/inventory_t_frozen_cohort.py` 和对应测试；未修改 51 份 v15 冻结运行源。模块不运行规划器、执行器或模型，也不连接标签。v15 结果和之后是否建立 close 模型均不影响本次完整机会身份。

接口：`load_frozen_inventory_core_cohort(intraday, fold, cfg, source_report, out, inputs, record_input) -> (core, metadata)`。调用方先以原缓存的完整同刻市场完成 session、provenance、timing 增强，然后调用本函数；股票过滤须发生在市场特征之后。`record_input` 接受 `(path, inputs)`。

`source_report` 必须是完整独立审计通过的 v14 永久底仓报告。`cfg` 继承该报告已解析的底仓金额、现金、费用、容量、来源隔离及当前特征参数。可选 `source_learning_audit` 指定报告内完整审计；缺省使用最新完整通过记录。建议调用方显式设置 `source_learning_audit_sha256`，并继续严格绑定原 full market 缓存及增强源码身份。新标签/模型/退出参数不是本模块参数；人工买单仅记录为训练禁用，绝不读取或改变冻结自动底仓。

模块先核完整独审绑定的配置、拆分、源码快照、两个 core 的 lineage/metadata、基准日初库存和全部旧模型矩阵文件。随后恢复全部未联标签键，精确核当前原来源及旧股股日全集，按旧 build 的列追加顺序恢复入场、全排名供体、现金竞争及股数来源。两 core 拼接后保持原稳定 `datetime/symbol` 排序。旧矩阵只是当前值和 dtype 的反证；从未把旧矩阵有效行当成机会全集。

一次真实验证只用首折的 2024/2025 两个 core：原完整截面增强包含 12,335,232 行、3,257 股，最终恢复 206,488 行、65 列。h20/h40 各 204,168 行、h60 的 178,647 行，全部 30 个原拟合当前因子、来源、股数和 dtype 精确一致；两 depth profile 的矩阵字节相同，只做一次逐值核对，仍逐文件核验各自产物 SHA。旧 h20 未覆盖的 2,320 条机会仍保留。耗时 31.67 秒，规划次数 0、拟合次数 0，没有读取外层账户或重复原 CSV 价格核验。

完整首折 core frame SHA：`3fed9679d7184f4f78f6a6f2b469d562db9a3ffc061792d66761a1b2c814c42a`。原 13 列 lineage 拼接 SHA：`f4f5597706d72a1fda73ff96fc4de326daed56326a733545f3dee3c05dbe006f`。真实完整结果另存 `complete_current_core.pkl`，来源文件、精确 dtypes/列序及每份旧矩阵核验详见 `verification.json` 和 `frozen_inventory_cohort/reuse_metadata.json`。

证据范围需要区分：原 lineage 没有保存全部 f_* 值，未进入任何旧矩阵的机会，其完整因子身份必须靠调用方的原缓存 SHA 和同一完整截面增强链保证。本次真实验证已核原缓存字节与 v14 完整独审相同，并另存新的全 current frame SHA；不把“原 lineage SHA 相等”夸大为所有因子值自动得到证明。两个相互 OOF 的 core 年仍只允许共同拟合，不能把其中一年称为独立校准集。

12 个合成反例通过，覆盖缺原键、来源改写、矩阵值/dtype 改写、原文件篡改、不完整审计、股日全集漏机会、经济配置变化、人工指令训练禁用、未来字段改写/删除不影响队列、原 NaT 来源与缺旧标签行保留，以及禁止覆盖既有输出。Quant Python 和系统 Python 编译均通过。首次合成执行发现测试夹具的标量时间为 us 而原来源合同为 ns；仅将新测试夹具改为原 ns 类型后通过，未改变生产解析或放松 dtype 核验。

`validate_real_reuse.py` 是唯一真实复用检查命令；`archive_validation.py` 只归档源码与证据，不重复测试或市场加载。`source_identity.json` 记录完整命令与冻结身份，`manifest.json` 绑定本目录文件。独立只读审查由 inspect_entry 完成，未发现实质问题。
