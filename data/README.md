# 数据说明

`raw/` 中的六个 CSV 来自论文作者公开的 `replication_package_MS-FTBDA-20-01585`：

- `df_test.csv`：3,000行离线测试记录，含作者已生成的模型预测；
- `control_demand_vlt.csv`：55,333个候选控制样本；
- `order_e2e_pre_500.csv`、`order_e2e_post_500.csv`：处理组前后期各500对；
- `order_jd_pre_500.csv`、`order_jd_post_500.csv`：控制组前后期各500对。

这些数据是脱敏公开子样本，不是京东完整业务数据。运行脚本不会修改原始 CSV。
