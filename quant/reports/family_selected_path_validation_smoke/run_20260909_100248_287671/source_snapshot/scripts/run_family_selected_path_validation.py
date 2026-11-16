"""在quant根目录冻结内部模型族选择，输入YAML及未启动外层的完整核心父轮。

freeze仅用父轮内部验证选择每折模型族，复制已审计模型，不重新拟合；
audit核对选择和复制指纹；evaluate审计/晋级后单次输出四年仅做多回顾。
本轮是看过父轮内部结果后增加的研究选择，不修改父轮失败结论，不连接券商。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from quant_backtest.annual_panel import write_json
from quant_backtest.annual_portfolio import run_lot_portfolio
from quant_backtest.pooled_path_models import generate_pooled_scores, METHOD_NAMES
from quant_backtest.simple_signals import create_exit_policy
from scripts.run_simple_signal_year_validation import validate_frozen, write_table
from scripts.run_pathwise_year_validation import load_frozen_data
from scripts.run_annual_method_validation import save_result, validate_result
from scripts.run_learned_exit_year_validation import calculate_sha256

TITLE = "训练内部自动选择模型族的路径净收益策略"
NOTE = "内部选模型族的留一年回顾；增加一层选参，前三折含未来年份训练，2026仅1至4月。"
NEW_SOURCES = ["scripts/run_family_selected_path_validation.py", "tests/test_family_path_selection.py"]


def select_family_rule(table: pd.DataFrame, winners: dict, cfg: dict, segment_count: int) -> tuple[str,dict,dict]:
    """仅用table内部验证选最优合格模型族；winners须为父轮各族冻结赢家，cfg给门槛，segment_count须全部有波动。

    不按年份写条件，不接受外层收益作为排序依据。无候选时保留现金，
    不能把各族测试后最好的年度拼接成策略。
    """
    if not set(table.method).issubset(METHOD_NAMES) or set(winners)!=set(METHOD_NAMES):
        raise ValueError("模型族集合不同于父轮协议")
    eligible = table.loc[np.isfinite(table.mean_sharpe)&(table.mean_sharpe>0)
        &(table.closed_lots>=cfg["min_closed_lots"])
        &(table.positive_segments>=cfg["simple_min_positive_segments"])
        &(table.active_segments==segment_count)]
    eligible = eligible.sort_values(["mean_sharpe","worst_drawdown","rule_id","method"],
                                    ascending=[False,False,True,True])
    if eligible.empty:
        method = "method1"
        rule = {**winners[method],"threshold":1e9,"rule_id":"cash_family_validation_failed",
                "selection":"两模型族均无内部合格候选，现金保护"}
        return method,rule,{"enabled":False,"mean_sharpe":None,"closed_lots":0,"positive_segments":0}
    row = eligible.iloc[0]
    parent = winners[row.method]
    if parent["rule_id"]!=row.rule_id or parent["threshold"]!=row.threshold or parent["threshold"]>=1e9:
        raise ValueError("族间最优候选不等于父轮已冻结族内赢家")
    rule = {**parent,"selection":"统一按内部有效均夏普、回撤、规则名选择模型族；外层之前冻结"}
    return row.method,rule,{"enabled":True,"mean_sharpe":float(row.mean_sharpe),
        "closed_lots":int(row.closed_lots),"positive_segments":int(row.positive_segments)}


def load_source(directory: Path) -> tuple:
    """核对directory父轮冻结和真实输入审计，拒绝已启动外层的父轮，返回配置/清单/折/审计。"""
    cfg,frozen,folds = validate_frozen(directory)
    protocol = json.loads((directory/"research_protocol.json").read_text())
    if any(importlib.metadata.version(p)!=v for p,v in protocol["packages"].items()):
        raise ValueError("父轮软件版本改变")
    if (directory/"test_started.json").exists():
        raise ValueError("父轮已经启动外层，不能事后新增模型族选择")
    audit = json.loads((directory/"training_audit.json").read_text())
    if (not audit["passed"] or audit["frozen_manifest_sha256"]!=calculate_sha256(directory/"frozen_manifest.json")
            or len(audit["models"])!=len(list(directory.glob("20*/model_*.joblib")))):
        raise ValueError("父轮缺少完整实际输入与分量重载审计")
    return cfg,frozen,folds,audit


def write_guide(directory: Path, cfg: dict, selected: list, promoted: bool) -> None:
    """在directory解释cfg交易契约与selected内部选择，promoted不是测试通过；不改父轮报告。"""
    table = pd.DataFrame(selected).rename(columns={"year":"留出年","source_method":"内部选中模型族",
        "profile":"模型配置","horizon":"退出交易日期限","threshold":"账户净收益门槛",
        "mean_sharpe":"内部均夏普","closed_lots":"完全平仓批次","positive_segments":"正收益片段",
        "enabled":"内部启用"})
    text = "# 内部模型族选择：冻结报告\n\n本轮尚无正式外层收益，不能认为平均夏普已经超过3。\n\n"
    text += table.to_markdown(index=False)+f"\n\n四折内部启用：{promoted}。\n\n"
    text += """## 与父轮的区别

父轮两类固定模型族均未四折全部通过，该结论保持不变。本轮是在看过父轮内部结果后提出的一层额外选参，
不是原协议从一开始就存在的选择，也不是新的独立数据。它只有一个统一策略：每折在非留出年份内部验证中，
按同一排序规则选择合格的直接或分解模型；不按留出年份写人工切换条件，不读取父轮外层预测或收益。
父轮没有启动外层，模型、门槛、退出、成本、仓位全部复用，只有模型族选择在新目录另行冻结。

这与按照测试年度收益事后选最好的方法不同，但新增选择仍会增加研究过拟合风险，不能当全新未见样本。
正式验收必须完整保留四年，现金或无波动年不能删除；同一统一策略平均夏普严格大于3才达标。
不能把本轮结果改贴到父轮方法一或方法二名下。

## 开平仓

每周用上一自然周高波动前300主板股。每24交易分钟，分数前5且通过冻结门槛才买，每批最多信号时总资产10%。
同股同日不重复买，隔日可新建独立批次。初始20倍/吊顶40倍简单分钟ATR或对应模型交易日同刻到期触发退出；
离池不退出，T+1之后下一分钟尝试成交，容量不足/缺价/涨跌停代理可能延期或部分成交。
万八免5、历史税费、万五滑点、100股、5%分钟共享成交额、不做空和无杠杆保持。

## 限制

"""
    for name in ["final_report.md","training_report.md"]:
        (directory/name).write_text(text+cfg["report_limitations"],encoding="utf-8")
    (directory/"输出文件说明.md").write_text("""# 文件角色

| 文件 | 用途 |
|---|---|
| final_report.md / training_report.md | 当前报告和冻结内部原文，不覆盖父轮结论 |
| resolved_config.yaml | 继承完整交易、成本、数据、标签配置及本轮说明 |
| research_protocol.json | 父轮路径/冻结/审计指纹、选择依据、新旧源码、外层前冻结时间 |
| folds.json / data_metadata.json | 继承的折边界和已有pkl身份 |
| split_audit.csv | 父轮全部候选实际训练年份/行数；只有选中配置被本轮复制，不是本轮重新拟合 |
| source_snapshot/ | 本轮及父轮实际代码快照，不是独立安装包 |
| selection_audit.csv | 每折唯一的模型族、配置、原门槛和内部选择指标 |
| 年份/frozen_rules.json | 单一family_selected策略规则，source_method指出内部选择的模型族 |
| 年份/model_*.joblib / model_*.json | 复制的已审计模型和真实输入/标签/权重指纹，没有重训 |
| 年份/booster_*.txt | 选中模型的原生分量，输出基点，分解模型相加后除10000 |
| 年份/explanation_*.csv / model_feature_manifest.csv | 选中分量重要性和中文输入定义 |
| 年份/training_rule_metrics.csv / validation_rule_metrics.csv | 父轮完整校准/验证候选，方便追溯族间选择 |
| frozen_manifest.json | 本轮全部复制文件、选择与源码的冻结指纹 |
| training_audit.json | 父轮实质审计继承、选择重算、复制字节一致，不伪称重新拟合 |
| test_started.json | 唯一一次外层执行标记 |
| test_year_metrics.csv / test_summary.csv | 外层完成后四年指标及同一策略完整均夏普 |
| 年份/family_selected/ | 外层后中文收益图、净值、成交、批次配对、未平仓、配置和逐文件说明 |

该目录只有一个模型族自动选择策略，不能把各年选择不同族说成原两固定族各自通过。
""",encoding="utf-8")


def run_freezing(config: Path, source_override: Path | None = None) -> Path:
    """读取config的source_run/output_dir，source_override仅用于明确烟测覆盖；写协议后才计算族间选择。"""
    settings = yaml.safe_load(config.read_text())
    source = (source_override or ROOT/settings["source_run"]).resolve()
    cfg,parent,folds,_ = load_source(source)
    output = str((ROOT/settings["output_dir"]).resolve())+("_smoke" if cfg["is_smoke"] else "")
    directory = Path(output)/datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True,exist_ok=False)
    cfg = {**cfg,"report_note":NOTE,"output_dir":output}
    cfg["report_limitations"] = "\n".join(line for line in cfg["report_limitations"].splitlines()
        if "至少一种方法四折内部启用" not in line)
    cfg["report_limitations"] += "\n- 本统一策略看过父轮内部结果后新增模型族选择；父轮固定族失败不改写，父轮没有启动外层，不能视为原先预注册或独立新样本。\n"
    cfg["report_limitations"] += "- 各折只依非留出年份内部验证同一排序函数选择族，不按测试年份硬编码或按外层收益切换；新增选择增加研究自由度。\n"
    cfg["report_limitations"] += "- 本轮要求统一族间选择策略四折内部全部启用再测试；不等于父轮任何固定模型族通过，其原失败结论不变。\n"
    (directory/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    protocol = json.loads((source/"research_protocol.json").read_text())
    hashes = {}
    for name in dict.fromkeys(list(protocol["source_hashes"])+NEW_SOURCES):
        target = directory/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        hashes[name] = calculate_sha256(target)
    write_json(directory/"research_protocol.json",{"created":datetime.now().isoformat(),"note":NOTE,
        "source_run":str(source),"source_frozen_sha256":calculate_sha256(source/"frozen_manifest.json"),
        "source_audit_sha256":calculate_sha256(source/"training_audit.json"),"source_hashes":hashes,
        "config_sha256":calculate_sha256(directory/"resolved_config.yaml"),"packages":protocol["packages"],
        "selection":"只在全部片段有波动、均夏普>0、足够平仓及正收益片段候选中按均夏普降序、回撤降序、规则名和方法名升序择一",
        "added_after_internal_results":True,"parent_outer_started":False,
        "criterion":"一个统一选择算法的四年有效夏普等权平均严格>3，不能删除现金或亏损年"})
    for name in ["folds.json","data_metadata.json","split_audit.csv"]:
        shutil.copyfile(source/name,directory/name)
    selected = []
    for fold in folds:
        year = fold["year"]
        original,target = source/str(year),directory/str(year)
        target.mkdir()
        table = pd.read_csv(original/"validation_rule_metrics.csv",float_precision="round_trip")
        method,rule,metrics = select_family_rule(table,json.loads((original/"frozen_rules.json").read_text()),cfg,len(fold["validation"]))
        write_json(target/"frozen_rules.json",{"family_selected":rule,"source_method":method})
        profile = rule["profile"]
        files = [original/f"model_{profile}.joblib",original/f"model_{profile}.json",
                 original/f"explanation_{profile}.csv",*original.glob(f"booster_{profile}_*.txt")]
        files += [original/n for n in ["model_feature_manifest.csv","training_rule_metrics.csv","validation_rule_metrics.csv"]]
        for file in files:
            shutil.copyfile(file,target/file.name)
        selected.append({"year":year,"source_method":method,"profile":profile,"horizon":rule["horizon"],
                         "threshold":rule["threshold"],**metrics})
    promoted = len(selected)==4 and all(r["enabled"] for r in selected)
    write_table(directory/"selection_audit.csv",selected)
    write_guide(directory,cfg,selected,promoted)
    protected = {str(p.relative_to(directory)):calculate_sha256(p) for p in directory.rglob("*") if p.is_file()}
    write_json(directory/"frozen_manifest.json",{"frozen_at":datetime.now().isoformat(),"promoted":promoted,"files":protected})
    print(f"内部模型族选择冻结：{directory}；四折启用={promoted}",flush=True)
    return directory


def run_audit(directory: Path) -> None:
    """核对directory及父轮不变、仅内部选择可重算、选中模型字节一致；明确继承父轮真实训练输入审计。"""
    cfg,frozen,folds = validate_frozen(directory)
    protocol = json.loads((directory/"research_protocol.json").read_text())
    source = Path(protocol["source_run"])
    parent_cfg,_,_,parent_audit = load_source(source)
    if any(cfg[k]!=v for k,v in parent_cfg.items() if k not in {"output_dir","report_note","report_limitations"}):
        raise ValueError("继承交易或训练配置改变，不能只称为模型族选择")
    if (calculate_sha256(source/"frozen_manifest.json")!=protocol["source_frozen_sha256"]
            or calculate_sha256(source/"training_audit.json")!=protocol["source_audit_sha256"]):
        raise ValueError("父轮冻结或实质审计改变")
    checks = []
    for fold in folds:
        year = fold["year"]
        original,target = source/str(year),directory/str(year)
        method,rule,metrics = select_family_rule(pd.read_csv(original/"validation_rule_metrics.csv",float_precision="round_trip"),
            json.loads((original/"frozen_rules.json").read_text()),cfg,len(fold["validation"]))
        if json.loads((target/"frozen_rules.json").read_text())!={"family_selected":rule,"source_method":method}:
            raise ValueError("模型族选择不能从内部表重算")
        files = [f for f in target.iterdir() if f.name!="frozen_rules.json"]
        if any(calculate_sha256(f)!=calculate_sha256(original/f.name) for f in files):
            raise ValueError("复制模型、原生分量或内部表与父轮不同")
        evidence = [r for r in parent_audit["models"] if r["year"]==year and r["profile"]==rule["profile"]]
        if len(evidence)!=1 or not evidence[0]["passed"]:
            raise ValueError("选中模型缺少父轮实际输入/分量证据")
        checks.append({"year":year,"source_method":method,"profile":rule["profile"],
                       "inherited_training_rows":evidence[0]["rows"],"components":evidence[0]["components"],"passed":True})
    write_json(directory/"training_audit.json",{"created":datetime.now().isoformat(),"passed":True,"models":checks,
        "source_audit_sha256":protocol["source_audit_sha256"],"actual_core_audit_inherited":True,
        "model_refitted":False,"outer_predictions_read":False,
        "frozen_manifest_sha256":calculate_sha256(directory/"frozen_manifest.json")})
    print(f"内部族间选择与模型字节继承审计通过：{len(checks)}模型",flush=True)


def run_evaluation(directory: Path) -> None:
    """对directory四折同一个内部选择算法单次外层，保持原仅做多引擎及成本，不按结果切换模型。"""
    cfg,frozen,folds = validate_frozen(directory)
    if (directory/"test_started.json").exists():
        raise ValueError("本轮外层已经启动，禁止重复评价")
    if not cfg["is_smoke"] and not frozen["promoted"]:
        raise ValueError("统一选择算法未四折内部启用")
    audit = json.loads((directory/"training_audit.json").read_text())
    if not audit["passed"] or audit["frozen_manifest_sha256"]!=calculate_sha256(directory/"frozen_manifest.json"):
        raise ValueError("本轮选择/继承审计缺失")
    protocol = json.loads((directory/"research_protocol.json").read_text())
    source = Path(protocol["source_run"])
    load_source(source)
    if (calculate_sha256(source/"frozen_manifest.json")!=protocol["source_frozen_sha256"]
            or calculate_sha256(source/"training_audit.json")!=protocol["source_audit_sha256"]):
        raise ValueError("父轮状态在本轮审计后改变")
    panel,market = load_frozen_data(cfg,directory)
    write_json(directory/"test_started.json",{"started":datetime.now().isoformat(),"is_smoke":cfg["is_smoke"]})
    rows = []
    for fold in folds:
        path = directory/str(fold["year"])
        selected = json.loads((path/"frozen_rules.json").read_text())
        rule = selected["family_selected"]
        frame = panel.loc[(panel.datetime>=fold["test_start"])&(panel.datetime<fold["test_end"])
                          &(panel.cross_section_count>=cfg["min_cross_section"])]
        scores = generate_pooled_scores(joblib.load(path/f"model_{rule['profile']}.joblib"),frame,cfg)
        result = run_lot_portfolio(scores,market,fold["test_start"],fold["test_end"],"prediction",
                                  rule,cfg,create_exit_policy(market,rule))
        audit_result = validate_result(result,cfg)
        result.update(report_note=NOTE,limitations=cfg["report_limitations"])
        save_result(path/"family_selected",result,f"{fold['year']}年 {'烟测：' if cfg['is_smoke'] else ''}{TITLE}",rule,cfg)
        write_json(path/"family_selected"/"execution_audit.json",audit_result)
        rows.append({"year":fold["year"],"source_method":selected["source_method"],**result["metrics"]})
    table = pd.DataFrame(rows)
    valid = len(table)==4 and (table.annual_volatility>0).all()
    mean = float(table.sharpe.mean()) if valid else None
    summary = {"策略":TITLE,"有效年数":int((table.annual_volatility>0).sum()),"完整四年平均夏普":mean,
               "超过3":bool(valid and mean>cfg["sharpe_target"] and not cfg["is_smoke"])}
    write_table(directory/"test_year_metrics.csv",rows)
    write_table(directory/"test_summary.csv",[summary])
    text = "# 内部模型族选择：年度回顾结果\n\n"+NOTE+"\n\n"
    if cfg["is_smoke"]:
        text += "**8股流程烟测，不用于盈利验收。**\n\n"
    text += pd.DataFrame([summary]).to_markdown(index=False)+"\n\n"
    text += table[["year","source_method","total_return","sharpe","max_drawdown","buy_count","closed_lot_count"]].rename(columns={
        "year":"留出年","source_method":"内部选中模型族","total_return":"收益率","sharpe":"夏普",
        "max_drawdown":"最大回撤","buy_count":"买入次数","closed_lot_count":"完全平仓批次"}).to_markdown(index=False)
    text += "\n\n现金无波动年夏普未定义，不得删掉；这是一个统一选择策略，不是父轮两个固定族各自成绩。\n\n"
    for fold in folds:
        text += f"- [{fold['year']}年收益图和逐笔记录]({fold['year']}/family_selected/final_report.md)\n"
    (directory/"final_report.md").write_text(text+"\n## 限制\n\n"+cfg["report_limitations"],encoding="utf-8")
    print(f"统一模型族选择单次外层完成：{directory}",flush=True)


def run_cli() -> None:
    """解析config/source-run/phase/run-dir；默认仅冻结族间选择，audit/evaluate须给本轮目录。"""
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--config",type=Path,default=ROOT/"config/family_selected_path_validation.yaml")
    parser.add_argument("--source-run",type=Path)
    parser.add_argument("--phase",choices=["freeze","audit","evaluate"],default="freeze")
    parser.add_argument("--run-dir",type=Path)
    args = parser.parse_args()
    if args.phase=="freeze":
        if args.run_dir:
            parser.error("freeze创建新目录，不能给run-dir")
        run_freezing(args.config,args.source_run)
    else:
        if args.run_dir is None or args.source_run:
            parser.error("audit/evaluate只接受run-dir而非source-run")
        (run_audit if args.phase=="audit" else run_evaluation)(args.run_dir.resolve())


if __name__=="__main__":
    run_cli()
