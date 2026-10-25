# 端到端库存管理论文：公开结果复现（CPU）

本仓库复现 Qi et al. (2023)《A Practical End-to-End Inventory Management Model with Deep Learning》中公开数据能够支持的实验结果：

- 第4节：离线库存模拟与 OPT、E2E_RNN、BM1、BM2、Normal、Gamma、LightGBM 成本比较；
- 第5节：倾向得分匹配、现场实验指标、双样本 t 检验、线性回归和双重差分；
- 自动生成可上传的 Markdown、HTML、CSV 表格和 PNG 图片。

## 复现边界

`df_test.csv` 已经包含 `E2E_RNN_pred`、`OPT_pred` 和 `gbm_pred` 等模型输出。本仓库重新计算库存路径、成本和统计分析，但不重新训练原论文神经网络，因此不需要 GPU。

论文正文的现场实验使用6,097对处理/控制样本；作者公开的是各500对脱敏子样本。公开子样本的数值不应与论文表2–4完全相同，但计算流程和结论方向可以验证。

## 一键运行

```bash
python3 -m pip install -r requirements.txt
python3 run_reproduction.py
python3 -m unittest discover -s tests -v
```

运行时间通常为数秒到数分钟，取决于电脑性能。

## 主要输出

- `report/reproduction.md`：包含公式、步骤、表格、图片和限制的完整中文报告；
- `report/reproduction.html`：可直接用浏览器打开的结果页面；
- `report/statistical_validation.md`：重复运行结论、统计注意事项和11项谬误扫描；
- `results/tables/section4_offline_costs.csv`：第4节成本表；
- `results/tables/section5_matching_balance.csv`：匹配平衡性；
- `results/tables/section5_post_ttests.csv`：现场实验 t 检验；
- `results/tables/section5_regressions.csv`：回归结果；
- `results/tables/section5_did.csv`：双重差分；
- `results/figures/`：全部结果图片。

## 项目结构

```text
.
├── data/raw/                 # 作者公开CSV
├── notebooks/original/       # 作者公开Notebook备份
├── src/reproduction.py       # 计算逻辑
├── src/reporting.py          # 报告生成
├── tests/                    # 可复现性测试
├── results/tables/           # 机器可读结果
├── results/figures/          # 图片
├── report/                   # Markdown与HTML报告
└── run_reproduction.py       # 一键入口
```

## 重要实现说明

- 成本参数按作者公开 Notebook 设置为持有成本 `h=1`、缺货成本 `b=9`。
- 第5节中作者称为 turnover rate 的变量实际计算为 `ave_inv / ave_demand`，更接近库存覆盖天数，越低越好。
- 论文式(11)包含截距，但公开 Notebook 的 OLS 代码遗漏截距。本仓库同时输出“忠实公开代码”和“对应论文公式”两套回归口径。

## 引用

Qi, M., Shi, Y., Qi, Y., Ma, C., Yuan, R., Wu, D., & Shen, Z.-J. M. (2023). A Practical End-to-End Inventory Management Model with Deep Learning. *Management Science*, 69(2), 759–773. https://doi.org/10.1287/mnsc.2022.4564
