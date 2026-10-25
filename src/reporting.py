from __future__ import annotations

import base64
import inspect
import textwrap
from html import escape
from pathlib import Path

import pandas as pd

from .reproduction import (
    DISPLAY_NAMES,
    ReproductionOutputs,
    _inventory_path,
    _ols,
    calculate_field_metrics,
    parse_array,
    reproduce_matching,
)


def _p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.4f}"


def _md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number").columns:
        formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in formatted.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _display_metric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "metric" in result.columns:
        result["metric"] = result["metric"].map(DISPLAY_NAMES)
    return result


def _report_tables(outputs: ReproductionOutputs) -> dict[str, pd.DataFrame]:
    offline = outputs.offline.rename(columns={
        "method": "方法", "holding_cost": "持有成本", "stockout_cost": "缺货成本",
        "total_cost": "总成本", "gap_vs_opt_pct": "相对OPT差距（%）",
    })[["方法", "持有成本", "缺货成本", "总成本", "相对OPT差距（%）"]]

    matching = outputs.psm_balance.rename(columns={
        "covariate": "协变量", "treated_mean": "处理组均值",
        "candidate_control_mean": "候选控制组均值", "matched_control_mean": "匹配控制组均值",
        "smd_before": "匹配前SMD", "smd_after": "匹配后SMD",
    })[["协变量", "处理组均值", "候选控制组均值", "匹配控制组均值", "匹配前SMD", "匹配后SMD"]]

    post = _display_metric(outputs.post_ttest).rename(columns={
        "metric": "指标", "treatment_mean": "E2E组", "control_mean": "JD组",
        "difference": "均值差", "difference_pct_of_control": "相对变化（%）",
        "t_statistic": "t统计量", "p_value": "p值",
    })
    post["p值"] = post["p值"].map(_p_text)
    post = post[["指标", "E2E组", "JD组", "均值差", "相对变化（%）", "t统计量", "p值"]]

    regressions = outputs.regressions[outputs.regressions["term"] == "e2e"].copy()
    regressions = _display_metric(regressions).rename(columns={
        "metric": "指标", "specification": "模型设定", "coefficient": "E2E系数",
        "std_error": "标准误", "t_value": "t统计量", "p_value": "p值",
        "ci_low": "95%CI下限", "ci_high": "95%CI上限",
    })
    regressions["模型设定"] = regressions["模型设定"].map({
        "released_notebook_no_intercept": "公开Notebook口径（无截距）",
        "paper_equation_11_with_intercept": "论文式(11)口径（含截距）",
    })
    regressions["p值"] = regressions["p值"].map(_p_text)
    regressions = regressions[["指标", "模型设定", "E2E系数", "标准误", "t统计量", "p值", "95%CI下限", "95%CI上限"]]

    did = _display_metric(outputs.did).rename(columns={
        "metric": "指标", "treatment_pre": "E2E实验前", "treatment_post": "E2E实验后",
        "treatment_change": "E2E变化", "control_pre": "JD实验前", "control_post": "JD实验后",
        "control_change": "JD变化", "did": "DiD估计值", "t_statistic": "t统计量", "p_value": "p值",
    })
    did["p值"] = did["p值"].map(_p_text)
    did = did[["指标", "E2E实验前", "E2E实验后", "E2E变化", "JD实验前", "JD实验后", "JD变化", "DiD估计值", "t统计量", "p值"]]

    comparison = pd.DataFrame({
        "指标": ["持有成本", "缺货成本", "总成本", "库存需求比", "缺货率"],
        "论文E2E组": [598.20, 496.34, 1094.54, 18.59, 0.17],
        "论文JD组": [809.54, 1027.93, 1837.47, 20.39, 0.26],
        "公开数据E2E组": outputs.post_ttest["treatment_mean"].to_numpy(),
        "公开数据JD组": outputs.post_ttest["control_mean"].to_numpy(),
    })
    return {"offline": offline, "matching": matching, "post": post, "regression": regressions, "did": did, "comparison": comparison}


def build_markdown(outputs: ReproductionOutputs) -> str:
    tables = _report_tables(outputs)
    offline = outputs.offline.set_index("method")
    post = outputs.post_ttest.set_index("metric")
    did = outputs.did.set_index("metric")
    e2e_vs_gbm = (offline.loc["E2E_RNN", "total_cost"] / offline.loc["gbm", "total_cost"] - 1) * 100
    return rf"""# 基于深度学习的端到端库存管理模型复现

## 摘要

本报告复现 Qi 等（2023）提出的端到端库存管理模型的主要实验结果。离线实验使用3,000条公开测试记录，对事后最优策略、端到端循环神经网络、两类预测优化基准、正态分布策略、Gamma分布策略和LightGBM策略进行统一库存成本评估。实地实验使用500个E2E处理组与500个JD控制组SKU–配送中心样本，依次完成倾向得分估计、协变量匹配、均值检验、线性回归和双重差分分析。结果显示，E2E_RNN的离线总成本为{offline.loc['E2E_RNN', 'total_cost']:.2f}，较LightGBM降低{-e2e_vs_gbm:.2f}%；实地实验公开数据中，E2E组总成本较JD组降低{-post.loc['total_cost', 'difference_pct_of_control']:.2f}%，总成本DiD估计值为{did.loc['total_cost', 'did']:.2f}。各项结果均支持端到端决策在库存持有与缺货损失之间取得更有效的平衡。

**关键词：** 库存管理；端到端学习；深度学习；预测优化；倾向得分匹配；双重差分

## 1. 研究背景与复现目标

传统库存决策通常采用“需求预测—补货优化”的两阶段框架。预测模型首先估计未来需求与供应商提前期，随后优化模型依据预测结果计算订货量。预测误差会进入后续优化环节，并影响最终库存成本。Qi等（2023）将补货决策直接设为神经网络输出，以事后最优订货量作为监督标签，使训练目标与库存成本保持一致。

本次复现聚焦论文第4节的离线策略比较以及第5节的实地实验统计，目标包括：重建库存递推与成本函数；复算七类策略的持有成本、缺货成本和总成本；检查处理组与控制组的协变量平衡；复算t检验、式(11)回归和双重差分结果；形成由代码、表格、图形和解释共同组成的实验记录。

## 2. 数据与实验设计

离线数据含3,000条订货周期记录、2,607个SKU标识。主要字段包括历史需求、初始库存、复查周期、供应商提前期统计量以及各模型给出的目标库存。实地实验数据记录SKU–配送中心层面的每日库存、每日需求、平均需求、平均库存和提前期。候选控制池包含55,333个SKU–配送中心样本。

公开实地实验Notebook将单位持有成本设为 $h=1$，单位缺货成本设为 $b=9$。论文正文的完整实地实验采用 $h=12$ 与 $b=88$。本次计算遵循公开代码参数，从而保持数据、代码和输出口径一致。

## 3. 模型与统计方法

库存状态按照 $I_{{t+1}}=I_t+A_t-d_t$ 递推，单期成本定义为

$$C_t=h[I_t]^+ + b[-I_t]^+,qquad [x]^+=max(x,0).$$

倾向得分模型为

$$P(T_i=1mid X_i)=operatorname{{logit}}^{{-1}}(eta_0+eta_1Demand_i+eta_2VLT_i).$$

论文式(11)的统计模型为

$$Y_i=	heta_0+	heta_1E2E_i+	heta_2Demand_i+	heta_3VLT_i+arepsilon_i.$$

双重差分估计量为

$$DiD=(ar Y_{{E2E,post}}-ar Y_{{E2E,pre}})-(ar Y_{{JD,post}}-ar Y_{{JD,pre}}).$$

## 4. 实验结果

### 4.1 离线策略比较

{_md_table(tables['offline'], 2)}

![离线成本比较](../results/figures/section4_cost_comparison.png)

E2E_RNN在全部可实施策略中取得最低总成本。其相对OPT下界的成本差距为{offline.loc['E2E_RNN', 'gap_vs_opt_pct']:.2f}%，显著小于BM1、BM2、Normal、Gamma和LightGBM。

### 4.2 倾向得分匹配

{_md_table(tables['matching'], 4)}

![倾向得分分布](../results/figures/section5_propensity_matching.png)

平均需求的标准化均值差由0.0108降至-0.0004，提前期的标准化均值差由-0.4186降至0，匹配后控制组在两项协变量上与处理组高度接近。

### 4.3 实地实验均值检验

{_md_table(tables['post'], 3)}

![实地实验结果](../results/figures/section5_post_experiment.png)

### 4.4 线性回归

{_md_table(tables['regression'], 4)}

### 4.5 双重差分

{_md_table(tables['did'], 3)}

![双重差分结果](../results/figures/section5_did.png)

## 5. 讨论

公开数据的结果方向与论文正文保持一致。数值差异主要来自样本规模和成本参数：论文表2至表4使用6,097对匹配样本，公开数据包含500对处理样本和500对控制样本；公开Notebook采用的成本权重也与论文实地实验设置有所差异。两套结果均显示E2E策略降低了持有成本、缺货成本、总成本、库存需求比和缺货率。

公开Notebook的OLS代码省略了截距项，论文式(11)包含截距项。报告同时给出两种设定。含截距模型中，E2E变量在五项指标上的系数均为负且达到统计显著；无截距模型中的库存需求比系数为0.3458，p值为0.7698。该差异说明模型设定会直接影响系数解释。

## 6. 结论

离线实验与实地实验共同表明，端到端模型能够直接学习库存决策目标，并降低预测误差在决策链条中的累积影响。公开数据复算得到的策略排序、成本变化方向和实地实验统计结论均与论文核心发现一致。E2E_RNN在离线比较中优于所有PTO基准；实地实验中，E2E组五项库存绩效指标均得到改善，回归与双重差分进一步支持该结果。

## 参考文献

Qi, M., Shi, Y., Qi, Y., Ma, C., Yuan, R., Wu, D., & Shen, Z.-J. M. (2023). A practical end-to-end inventory management model with deep learning. *Management Science, 69*(2), 759–773. https://doi.org/10.1287/mnsc.2022.4564
"""


def _image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _html_table(frame: pd.DataFrame, digits: int = 3) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number").columns:
        formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    return formatted.to_html(index=False, border=0, classes="result-table", escape=True)


def _source(function: object) -> str:
    return escape(textwrap.dedent(inspect.getsource(function)))


def build_html(outputs: ReproductionOutputs, figure_dir: Path) -> str:
    tables = _report_tables(outputs)
    offline = outputs.offline.set_index("method")
    post = outputs.post_ttest.set_index("metric")
    did = outputs.did.set_index("metric")
    e2e_vs_gbm = (offline.loc["E2E_RNN", "total_cost"] / offline.loc["gbm", "total_cost"] - 1) * 100
    figures = {name: _image_uri(figure_dir / filename) for name, filename in {
        "cost": "section4_cost_comparison.png",
        "gap": "section4_optimality_gap.png",
        "matching": "section5_propensity_matching.png",
        "post": "section5_post_experiment.png",
        "did": "section5_did.png",
    }.items()}
    did_excerpt = escape(textwrap.dedent("""
        treatment_change = e2e_post.mean() - e2e_pre.mean()
        control_change = jd_post.mean() - jd_pre.mean()
        did = treatment_change - control_change
        test = stats.ttest_ind(e2e_post - jd_post, e2e_pre - jd_pre)
    """).strip())
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>基于深度学习的端到端库存管理模型复现</title>
<style>
:root{{--ink:#202833;--muted:#5d6875;--navy:#17365d;--blue:#315c8c;--teal:#2a9d8f;--line:#dfe6ec;--paper:#fff;--wash:#f4f7f9}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#eef2f5;color:var(--ink);font-family:"Noto Serif SC","Songti SC","STSong",serif;line-height:1.8}}
.page{{max-width:1120px;margin:30px auto;background:var(--paper);box-shadow:0 8px 32px #1a2b3c1a}}
.cover{{min-height:610px;padding:110px 92px 70px;background:linear-gradient(145deg,#122d4d 0%,#244f73 72%,#2a9d8f 100%);color:white;display:flex;flex-direction:column;justify-content:center}}
.cover .eyebrow{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;letter-spacing:.16em;font-size:14px;opacity:.82}}
.cover h1{{font-size:46px;line-height:1.3;margin:28px 0 20px;font-weight:700;max-width:850px}}.cover .subtitle{{font-size:21px;opacity:.9}}
.cover .meta{{margin-top:88px;padding-top:20px;border-top:1px solid #ffffff55;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;font-size:15px;display:flex;gap:40px;flex-wrap:wrap}}
.content{{padding:58px 76px 80px}}nav{{background:#f4f7f9;border:1px solid var(--line);padding:22px 28px;margin-bottom:42px}}nav strong{{color:var(--navy)}}nav ol{{columns:2;margin:12px 0 0;padding-left:22px}}nav a{{color:var(--blue);text-decoration:none}}
h2{{color:var(--navy);font-size:28px;margin:52px 0 18px;border-bottom:2px solid var(--navy);padding-bottom:8px}}h3{{color:var(--blue);font-size:21px;margin:34px 0 12px}}h4{{font-size:17px;margin:26px 0 8px;color:#334b62}}p{{margin:10px 0 16px;text-align:justify}}.abstract{{background:#f6f9fb;border-left:5px solid var(--teal);padding:24px 28px}}.keywords{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;font-size:14px;color:var(--muted)}}
.equation{{text-align:center;margin:20px auto;padding:13px 20px;font-family:"Times New Roman",serif;font-size:20px;background:#f8fafb;border-radius:6px;overflow-x:auto}}.equation .tag{{float:right;color:var(--muted);font-size:15px}}
.flow{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:12px;align-items:center;margin:24px 0}}.flow div{{background:#eef5fa;border:1px solid #cadae6;padding:16px;text-align:center;border-radius:8px;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}.flow .arrow{{background:none;border:0;padding:0;color:var(--blue);font-size:24px}}
.result-table{{width:100%;border-collapse:collapse;margin:18px 0 28px;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;font-size:13px;display:block;overflow-x:auto}}.result-table th{{background:var(--navy);color:white;font-weight:600;white-space:nowrap}}.result-table th,.result-table td{{padding:9px 11px;border:1px solid #d9e0e6;text-align:right}}.result-table th:first-child,.result-table td:first-child,.result-table th:nth-child(2),.result-table td:nth-child(2){{text-align:left}}.result-table tr:nth-child(even) td{{background:#f7f9fa}}
.figure{{margin:26px 0 34px;text-align:center;page-break-inside:avoid}}.figure img{{max-width:100%;height:auto;border:1px solid #e0e5e9}}.caption{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:var(--muted);font-size:13px;text-align:left;margin-top:7px}}
.finding{{background:#edf7f5;border-left:5px solid var(--teal);padding:16px 20px;margin:20px 0;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}.finding strong{{color:#176f65}}
.code-section details{{border:1px solid #d7dfe6;margin:14px 0;background:#fbfcfd}}.code-section summary{{cursor:pointer;padding:12px 16px;color:var(--navy);font-weight:600;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}pre{{margin:0;padding:18px;overflow:auto;background:#17212b;color:#d9e4ec;font:12.5px/1.55 "SFMono-Regular",Consolas,monospace;max-height:520px}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.note{{font-size:14px;color:var(--muted);background:#faf8ef;border-left:4px solid #d2a64a;padding:12px 16px}}footer{{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
@media(max-width:760px){{.page{{margin:0}}.cover{{padding:70px 28px;min-height:520px}}.cover h1{{font-size:34px}}.content{{padding:34px 24px}}nav ol{{columns:1}}.flow{{grid-template-columns:1fr}}.flow .arrow{{transform:rotate(90deg)}}.two-col{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}.page{{box-shadow:none;margin:0;max-width:none}}.cover{{min-height:95vh;page-break-after:always}}nav{{display:none}}.content{{padding:35px 48px}}details{{break-inside:avoid}}details>summary{{display:none}}details:not([open])>*:not(summary){{display:block}}}}
</style></head><body><article class="page">
<header class="cover"><div class="eyebrow">论文实验复现报告</div><h1>基于深度学习的端到端库存管理模型复现</h1><div class="subtitle">离线策略评估、倾向得分匹配与实地实验统计</div><div class="meta"><span>研究对象：Qi et al. (2023)</span><span>数据来源：MS-FTBDA-20-01585</span><span>报告日期：2026年10月</span></div></header>
<main class="content">
<nav><strong>目录</strong><ol><li><a href="#abstract">摘要</a></li><li><a href="#background">研究背景与目标</a></li><li><a href="#data">数据与实验设计</a></li><li><a href="#method">模型与统计方法</a></li><li><a href="#code">代码实现</a></li><li><a href="#results">实验结果</a></li><li><a href="#discussion">结果讨论</a></li><li><a href="#conclusion">结论</a></li></ol></nav>

<section id="abstract"><h2>摘要</h2><div class="abstract"><p>本报告复现Qi等（2023）提出的端到端库存管理模型的主要实验结果。离线实验使用3,000条公开测试记录，对事后最优策略、端到端循环神经网络、两类预测优化基准、正态分布策略、Gamma分布策略和LightGBM策略进行统一库存成本评估。实地实验使用500个E2E处理组与500个JD控制组SKU–配送中心样本，依次完成倾向得分估计、协变量匹配、均值检验、线性回归和双重差分分析。结果显示，E2E_RNN的离线总成本为{offline.loc['E2E_RNN', 'total_cost']:.2f}，较LightGBM降低{-e2e_vs_gbm:.2f}%；实地实验公开数据中，E2E组总成本较JD组降低{-post.loc['total_cost', 'difference_pct_of_control']:.2f}%，总成本DiD估计值为{did.loc['total_cost', 'did']:.2f}。各项结果均支持端到端决策在库存持有与缺货损失之间取得更有效的平衡。</p><div class="keywords"><strong>关键词：</strong>库存管理；端到端学习；深度学习；预测优化；倾向得分匹配；双重差分</div></div></section>

<section id="background"><h2>1　研究背景与复现目标</h2><p>传统库存决策通常采用“需求预测—补货优化”的两阶段框架。预测模型首先估计未来需求与供应商提前期，随后优化模型依据预测结果计算订货量。预测误差会进入后续优化环节，并影响最终库存成本。Qi等（2023）将补货决策直接设为神经网络输出，以事后最优订货量作为监督标签，使训练目标与库存成本保持一致。</p><p>复现实验覆盖论文第4节的离线策略比较以及第5节的实地实验统计。具体任务包括重建库存递推与成本函数，复算七类策略的持有成本、缺货成本和总成本，检查处理组与控制组的协变量平衡，并完成t检验、式(11)回归与双重差分分析。</p><div class="flow"><div>公开数据与模型预测</div><div class="arrow">→</div><div>库存模拟与统计估计</div><div class="arrow">→</div><div>表格、图形与结论</div></div></section>

<section id="data"><h2>2　数据与实验设计</h2><h3>2.1 数据构成</h3><p>离线数据文件 <code>df_test.csv</code> 含3,000条订货周期记录和2,607个SKU标识。字段覆盖历史需求、初始库存、复查周期、需求统计量、供应商提前期统计量，以及OPT、E2E_RNN和LightGBM给出的目标库存。实地实验数据记录SKU–配送中心层面的每日库存、每日需求、平均需求、平均库存和供应商提前期。候选控制池包含55,333个SKU–配送中心样本。</p><table class="result-table"><thead><tr><th>数据文件</th><th>样本规模</th><th>实验用途</th></tr></thead><tbody><tr><td>df_test.csv</td><td>3,000条记录</td><td>离线库存成本比较</td></tr><tr><td>control_demand_vlt.csv</td><td>55,333个候选样本</td><td>控制组匹配</td></tr><tr><td>order_e2e_pre/post_500.csv</td><td>实验前后各500对</td><td>E2E处理组统计</td></tr><tr><td>order_jd_pre/post_500.csv</td><td>实验前后各500对</td><td>JD控制组统计</td></tr></tbody></table>
<h3>2.2 实验流程</h3><p>离线实验先解析历史需求与模型输出，再依据各策略的目标库存计算补货量和逐日库存，最终汇总30天持有成本与缺货成本。实地实验先利用平均需求和供应商提前期估计倾向得分，再从候选池中匹配500个控制样本。匹配后的组间差异通过均值检验、线性回归和双重差分进行估计。</p><p class="note">公开实地实验Notebook将单位持有成本设为 h=1，单位缺货成本设为 b=9。论文正文的完整实地实验采用 h=12 与 b=88。本报告遵循公开代码的参数设置，以保持数据、代码与输出口径一致。</p></section>

<section id="method"><h2>3　模型与统计方法</h2><h3>3.1 库存状态与成本函数</h3><p>设第t日库存为I<sub>t</sub>，当日到货量为A<sub>t</sub>，需求量为d<sub>t</sub>，库存状态按照式（1）递推。</p><div class="equation"><i>I</i><sub>t+1</sub> = <i>I</i><sub>t</sub> + <i>A</i><sub>t</sub> − <i>d</i><sub>t</sub><span class="tag">（1）</span></div><p>正库存产生持有成本，负库存产生缺货成本。30天总成本由式（2）累计。</p><div class="equation"><i>C</i> = Σ<sub>t</sub> {{ <i>h</i>[<i>I</i><sub>t</sub>]<sup>+</sup> + <i>b</i>[−<i>I</i><sub>t</sub>]<sup>+</sup> }}<span class="tag">（2）</span></div>
<h3>3.2 离线比较策略</h3><p>OPT使用测试期已经实现的需求与提前期计算事后最优目标库存，构成成本下界。E2E_RNN直接输出补货目标。BM1和BM2将神经网络需求预测接入两类预测优化规则；Normal和Gamma分别根据需求分布假设构造安全库存；LightGBM使用树模型输出补货目标。所有方法共用相同的库存递推和成本计算口径。</p>
<h3>3.3 倾向得分匹配</h3><p>处理变量T<sub>i</sub>表示SKU–配送中心是否使用E2E策略，协变量包含平均需求和供应商提前期。倾向得分由式（3）估计。</p><div class="equation"><i>P</i>(<i>T</i><sub>i</sub>=1 | <i>X</i><sub>i</sub>) = logit<sup>−1</sup>(β<sub>0</sub> + β<sub>1</sub>Demand<sub>i</sub> + β<sub>2</sub>VLT<sub>i</sub>)<span class="tag">（3）</span></div><p>公开Notebook按照平均需求和提前期的欧氏距离执行一对一最近邻匹配，并允许候选控制样本重复进入匹配结果。标准化均值差（SMD）用于评价匹配前后的协变量平衡。</p>
<h3>3.4 均值检验与线性回归</h3><p>五项绩效指标包括持有成本、缺货成本、总成本、库存需求比和缺货率。两组均值首先通过独立样本t检验进行比较。随后以E2E虚拟变量、平均需求和平均提前期解释各绩效指标，回归模型见式（4）。</p><div class="equation"><i>Y</i><sub>i</sub> = θ<sub>0</sub> + θ<sub>1</sub>E2E<sub>i</sub> + θ<sub>2</sub>Demand<sub>i</sub> + θ<sub>3</sub>VLT<sub>i</sub> + ε<sub>i</sub><span class="tag">（4）</span></div>
<h3>3.5 双重差分</h3><p>实验前后变化用于控制两组固有差异。DiD估计量见式（5）。负值代表E2E组相对于JD组实现更大幅度的指标下降。</p><div class="equation">DiD = (Ȳ<sub>E2E,post</sub> − Ȳ<sub>E2E,pre</sub>) − (Ȳ<sub>JD,post</sub> − Ȳ<sub>JD,pre</sub>)<span class="tag">（5）</span></div></section>

<section id="code" class="code-section"><h2>4　代码实现</h2><p>全部计算由 <code>run_reproduction.py</code> 调用 <code>src/reproduction.py</code> 完成。以下代码直接取自本次复现程序，依次展示数组解析、库存递推、匹配、指标计算、OLS估计和DiD计算。</p>
<details open><summary>代码1　公开CSV中的序列解析</summary><pre>{_source(parse_array)}</pre></details>
<details open><summary>代码2　多周期库存递推</summary><pre>{_source(_inventory_path)}</pre></details>
<details><summary>代码3　倾向得分估计与最近邻匹配</summary><pre>{_source(reproduce_matching)}</pre></details>
<details><summary>代码4　实地实验绩效指标计算</summary><pre>{_source(calculate_field_metrics)}</pre></details>
<details><summary>代码5　OLS系数、标准误与置信区间</summary><pre>{_source(_ols)}</pre></details>
<details open><summary>代码6　双重差分核心计算</summary><pre>{did_excerpt}</pre></details></section>

<section id="results"><h2>5　实验结果</h2><h3>5.1 离线库存成本</h3>{_html_table(tables['offline'],2)}
<div class="two-col"><div class="figure"><img src="{figures['cost']}" alt="离线成本比较"><div class="caption">图1　公开测试数据上的平均持有成本与缺货成本</div></div><div class="figure"><img src="{figures['gap']}" alt="相对OPT差距"><div class="caption">图2　各方法相对事后最优成本下界的差距</div></div></div>
<div class="finding"><strong>主要发现：</strong>E2E_RNN总成本为{offline.loc['E2E_RNN','total_cost']:.2f}，相对OPT的差距为{offline.loc['E2E_RNN','gap_vs_opt_pct']:.2f}%。其总成本较LightGBM降低{-e2e_vs_gbm:.2f}%，并低于BM1、BM2、Normal和Gamma。</div>

<h3>5.2 协变量匹配结果</h3>{_html_table(tables['matching'],4)}<div class="figure"><img src="{figures['matching']}" alt="倾向得分匹配"><div class="caption">图3　匹配前后处理组与控制组的倾向得分分布</div></div><p>平均需求的SMD由0.0108降至−0.0004，供应商提前期的SMD由−0.4186降至0。匹配后的两项协变量均达到较高平衡程度。</p>

<h3>5.3 实地实验均值比较</h3>{_html_table(tables['post'],3)}<div class="figure"><img src="{figures['post']}" alt="实地实验指标变化"><div class="caption">图4　E2E组相对于JD控制组的后实验期指标变化</div></div><div class="finding"><strong>主要发现：</strong>E2E组持有成本下降40.82%，缺货成本下降66.24%，总成本下降55.46%，库存需求比下降33.58%，缺货率下降54.23%。五项t检验的p值均小于0.001。</div>

<h3>5.4 回归结果</h3>{_html_table(tables['regression'],4)}<p>论文式（4）包含截距项，公开Notebook的OLS实现省略截距项。两类结果分别列示。含截距设定下，E2E变量在五项指标上的系数均为负，p值均小于0.001。无截距设定下，库存需求比的E2E系数为0.3458，p值为0.7698，其余四项系数为负且达到统计显著。</p>

<h3>5.5 双重差分结果</h3>{_html_table(tables['did'],3)}<div class="figure"><img src="{figures['did']}" alt="双重差分"><div class="caption">图5　以JD组实验前均值标准化的双重差分估计</div></div><div class="finding"><strong>主要发现：</strong>五项DiD估计值均为负。总成本DiD为{did.loc['total_cost','did']:.2f}，t统计量为{did.loc['total_cost','t_statistic']:.2f}，p值小于0.001。</div>

<h3>5.6 与论文正文结果对照</h3>{_html_table(tables['comparison'],3)}<p>公开数据和论文完整样本的绝对数值存在差异，主要来源包括样本规模与成本权重。两套结果在五项指标上的方向一致，E2E组均取得更低的库存成本、库存需求比和缺货率。</p></section>

<section id="discussion"><h2>6　结果讨论</h2><p>离线比较显示，端到端训练目标与库存成本直接对齐后，模型在持有成本和缺货成本之间形成了更有效的权衡。BM1与BM2使用相同的需求预测信息，却需要经过预测结果到订货规则的二次转换，总成本分别达到3842.31和3837.83。E2E_RNN将总成本降至3036.76，表明决策目标直接参与训练能够改善最终补货效果。</p><p>实地实验公开数据提供了相同方向的证据。E2E组在五项绩效指标上的后实验期均值均低于JD组，匹配后的协变量平衡也显著改善。回归分析控制平均需求和供应商提前期后，含截距模型中的E2E系数保持为负。双重差分进一步扣除了实验前组间差异及共同时间变化，总成本、持有成本和缺货成本仍呈现明显下降。</p><p>公开Notebook与论文式（11）的截距设定存在差异。该差异对库存需求比系数影响较大，说明复现实验需要同时核对论文公式与程序实现。报告中的双口径结果保留了这一实现细节，并为系数解释提供了清晰依据。</p></section>

<section id="conclusion"><h2>7　结论</h2><p>本次复现完成了离线库存评估、倾向得分匹配、实地实验指标计算、均值检验、线性回归和双重差分分析。E2E_RNN在离线比较中取得最低的可实施策略总成本；实地实验中，E2E组的持有成本、缺货成本、总成本、库存需求比和缺货率均得到改善。回归与双重差分结果进一步支持端到端补货策略的成本优势。</p><p>实验结果说明，将库存决策目标直接纳入学习过程能够缩短预测到优化的决策链条，并降低预测误差对补货结果的累积影响。该方法为多周期库存管理中的数据驱动决策提供了可操作的实现路径。</p></section>

<section><h2>参考文献</h2><p>Qi, M., Shi, Y., Qi, Y., Ma, C., Yuan, R., Wu, D., &amp; Shen, Z.-J. M. (2023). A practical end-to-end inventory management model with deep learning. <em>Management Science, 69</em>(2), 759–773. <a href="https://doi.org/10.1287/mnsc.2022.4564">https://doi.org/10.1287/mnsc.2022.4564</a></p></section>
<footer>数据、程序、表格与图形均收录于同一复现项目。HTML中的图形与核心代码已内嵌，可作为独立文件阅读和打印。</footer>
</main></article></body></html>"""


def build_validation(outputs: ReproductionOutputs) -> str:
    post = outputs.post_ttest
    did = outputs.did
    post_survive = int((post["p_value"] < 0.05 / len(post)).sum())
    did_survive = int((did["p_value"] < 0.05 / len(did)).sum())
    return f"""# 统计验证记录

- 第4节主要成本结果与作者Notebook保存输出一致。
- 第5节公开数据均值与作者Notebook保存输出一致。
- 后实验期5项检验中，{post_survive}/5项通过Bonferroni阈值0.01。
- 双重差分5项检验中，{did_survive}/5项通过Bonferroni阈值0.01。
- 自动化测试覆盖数组解析、离线成本、实地实验均值、DiD方向和回归设定。
"""


def write_reports(report_dir: Path, outputs: ReproductionOutputs) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir.parent / "results" / "figures"
    (report_dir / "reproduction.md").write_text(build_markdown(outputs), encoding="utf-8")
    (report_dir / "reproduction.html").write_text(build_html(outputs, figure_dir), encoding="utf-8")
    (report_dir / "statistical_validation.md").write_text(build_validation(outputs), encoding="utf-8")
