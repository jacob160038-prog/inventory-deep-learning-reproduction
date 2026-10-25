from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inventory-reproduction-matplotlib")

from src.reproduction import run
from src.reporting import write_reports


ROOT = Path(__file__).resolve().parent


def main() -> None:
    data_dir = ROOT / "data" / "raw"
    required = [
        "df_test.csv",
        "control_demand_vlt.csv",
        "order_e2e_pre_500.csv",
        "order_e2e_post_500.csv",
        "order_jd_pre_500.csv",
        "order_jd_post_500.csv",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing disclosed replication files: {', '.join(missing)}")

    outputs = run(data_dir, ROOT / "results")
    write_reports(ROOT / "report", outputs)
    print("CPU reproduction completed.")
    print(outputs.offline[["method", "total_cost", "gap_vs_opt_pct"]].to_string(index=False))
    print("Reports: report/reproduction.md and report/reproduction.html")


if __name__ == "__main__":
    main()
