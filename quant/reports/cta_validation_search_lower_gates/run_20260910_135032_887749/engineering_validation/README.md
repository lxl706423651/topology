# 工程验收证据

## 架构

本目录保存两个完整阶段及中断检查点的执行证据，不能从中选另一个测试赢家。
研究输入是第七方法的已冻结过去模型、可复用724列特征和三折验证预测PKL；输出为只读复核和复盘表。
只验证2023第三季度、2024全年、2025全年内部验证，没有重新评价2026。

## 文件角色

| 文件 | 角色与含义 |
|---|---|
| README.md | 人工文档：架构、日志与调用顺序 |
| completion_checks_20260910_140250.json | 生成验收：三折均>2、174完整批次、3280套不重复规则、640旧规则/1920账户逐值一致 |
| completion_checks_20260910_140250.py | 静态快照：产生上一JSON的验收器源码，不是策略或模型 |
| closed_batch_review.csv | 生成的中文完整批次表；按年份/批次号区分独立账户，已扣双边费用 |
| cta_validation_prepare.log | 首次验证预测PKL构建，不重建原因子或读取2026预测 |
| cta_validation_reuse.log | 同一输入再次调用，确认复用原验证预测PKL |
| cta_validation_smoke.log | 初版小股票烟测，只证明流程可执行 |
| cta_validation_final_smoke.log | 性能优化之前的最终烟测，非收益验收 |
| cta_validation_smoke_audit.log | 对前述烟测重算过去模型验证预测与账本 |
| cta_validation_search_stage1.log | 首次完整搜索尝试在640规则后主动SIGINT中止；保留SystemError中断栈，不算成功完整运行 |
| cta_validation_optimized_tests.log | 性能分支加入后的44项测试通过 |
| cta_validation_optimized_smoke.log | 不可卖时跳过无用报价回溯后的烟测 |
| cta_validation_optimized_smoke_audit.log | 优化后烟测独立审计 |
| cta_validation_search_stage1_optimized.log | 相同2000套规则重新完整执行，6000账户，最低折1.4579，未达标 |
| cta_validation_stage1_audit.log | 第一阶段完整候选、24份验证预测重算、所选三账户审计通过 |
| cta_validation_search_stage2.log | 第二阶段1440套较低门槛/频率候选，4320账户，2套全部达标 |
| cta_validation_stage2_audit.log | 第二阶段独立审计，24份预测逐值一致、三账户及历史状态通过 |
| cta_validation_final_tests.log | 最终50项CTA/扩展特征/时钟相关单元测试通过 |
| cta_validation_completion.log | 最后验收及174批中文复盘表生成命令输出 |
| cta_validation_syntax.log | 系统Python及quant Python的语法检查通过记录 |
| cta_validation_diff_check.log | git diff --check的退出状态；不是Git提交或推送 |

复跑验收会增加新的时间戳JSON/源码快照，角色与已有同名前缀相同。
所有日志/JSON/CSV均为生成输出；不作为新的YAML配置，不替代原数据及原生模型。

## 顺序与命令

在quant根目录，用`/home/lxl/anaconda3/envs/quant/bin/python`执行：

1. `scripts/run_cta_validation_search.py --prepare-only`两次：第一次构建验证预测、第二次复用PKL，供后续搜索消费。
2. `-m pytest -q tests/test_cta_validation_search.py tests/test_cta_timing_rules.py tests/test_cta_expanded.py tests/test_cta_research.py`：策略、缓存接口、执行内核及时间边界检查。
3. `scripts/run_cta_validation_search.py --smoke`：小股票流程测试，报告不作为正式收益。
4. `scripts/run_cta_validation_search.py --config config/cta_validation_search.yaml`：第一阶段2000规则，每条在全部三折评价。
5. `scripts/run_cta_validation_search.py --config config/cta_validation_search_lower_gates.yaml`：第二阶段1440规则，复用同一预测PKL。
6. `scripts/audit_cta_validation_search.py --report-dir 阶段报告目录`：重算预测、排序、执行账务与历史资格，不重新调参。
7. `scripts/verify_cta_validation_completion.py --report-dir reports/cta_validation_search_lower_gates/run_20260910_135032_887749 --previous-stage reports/cta_validation_search/run_20260910_134703_432721 --interrupted-stage reports/cta_validation_search/run_20260910_133954_798660`：阶段完整性、日收益与旧检查点精确一致性、中文完整批次表。

第4步第一次尝试中断后进行性能优化并全量重跑；中断目录未删除，检查点与完整重跑逐值一致。
没有执行第三阶段时点预案。没有连接券商、修改原行情或推送Git。
