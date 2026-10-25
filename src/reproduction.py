from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inventory-reproduction-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit
from scipy.spatial import cKDTree


HOLDING_COST = 1.0
STOCKOUT_COST = 9.0
METHODS = ["OPT", "E2E_RNN", "Bm1", "Bm2", "Normal", "Gamma", "gbm"]
METRICS = ["holding_cost", "stockout_cost", "total_cost", "turnover_days", "stockout_rate"]
DISPLAY_NAMES = {
    "holding_cost": "Holding cost",
    "stockout_cost": "Stockout cost",
    "total_cost": "Total cost",
    "turnover_days": "Inventory-to-demand ratio",
    "stockout_rate": "Stockout rate",
}


@dataclass
class ReproductionOutputs:
    offline: pd.DataFrame
    psm_balance: pd.DataFrame
    post_ttest: pd.DataFrame
    regressions: pd.DataFrame
    did: pd.DataFrame
    metadata: dict


def parse_array(value: object) -> np.ndarray:
    """Parse the bracketed numeric arrays stored in the disclosed CSV files."""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    text = str(value).strip().strip("[]").replace(",", " ")
    return np.fromstring(text, sep=" ", dtype=float)


def parse_vlt_days(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing VLT value")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(round(float(value)))
    return int(str(value).split()[0])


def _inventory_path(group: pd.Series, prediction_name: str) -> tuple[list[float], list[float]]:
    """Faithful modernization of the authors' Section 4 inventory recursion."""
    full_inventory: list[float] = [float(group["initial_stock"][0])]
    evaluation_inventory: list[float] = []
    rounds = len(group["OPT_pred"])

    for r in range(rounds):
        demand = group["demand_hist"][r]
        days = len(demand) - 1 if r < rounds - 1 else len(demand)
        lead_time = int(round(group["vlt"][r]))
        replenishment = 0
        for t in range(days):
            if t == 0:
                target = float(group[f"{prediction_name}_pred"][r])
                if r == 0:
                    reference_inventory = full_inventory[0]
                else:
                    try:
                        reference_inventory = full_inventory[-lead_time - 1]
                    except IndexError:
                        reference_inventory = full_inventory[1]
                replenishment = int(round(target - reference_inventory))

            if t < lead_time:
                if r == 0:
                    full_inventory.append(full_inventory[-1] - float(demand[t]))
            elif t == lead_time:
                if full_inventory[-1] >= 0:
                    next_inventory = full_inventory[-1] + replenishment - float(demand[t])
                else:
                    next_inventory = replenishment - float(demand[t])
                full_inventory.append(next_inventory)
                evaluation_inventory.append(next_inventory)
            else:
                next_inventory = full_inventory[-1] - float(demand[t])
                full_inventory.append(next_inventory)
                evaluation_inventory.append(next_inventory)

    return full_inventory[1:], evaluation_inventory


def reproduce_offline(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(data_dir / "df_test.csv")
    frame["sf_rnn"] = frame["sf_rnn"].map(parse_array)
    frame["demand_hist"] = frame["demand_hist"].map(parse_array)

    mean_forecast = frame["sf_rnn"].map(np.mean)
    frame["Bm1_pred"] = mean_forecast * (frame["review_period"] + frame["vlt"]).astype(int)

    def bm2(row: pd.Series) -> float:
        horizon = row["review_period"] + row["vlt"]
        days = int(horizon) if horizon <= STOCKOUT_COST else int(horizon) - horizon // (STOCKOUT_COST + HOLDING_COST)
        return row["Bm2_pred"] * days

    frame["Bm2_pred"] = mean_forecast
    frame["Bm2_pred"] = frame.apply(bm2, axis=1)

    frame["Normal_pred"] = frame.apply(
        lambda row: int(
            row["demand_mean"] * (row["review_period"] + row["vendor_vlt_mean"])
            + 1.64
            * math.sqrt(
                (row["review_period"] + row["vendor_vlt_mean"]) * row["demand_std"] ** 2
                + row["demand_std"] ** 2 * row["vendor_vlt_std"]
            )
        ),
        axis=1,
    )

    def gamma_target(row: pd.Series) -> int:
        mean = row["demand_mean"]
        variance = row["demand_std"] ** 2
        theta = variance / (mean + 1e-4)
        shape = mean / (theta + 1e-4)
        horizon_shape = int(row["review_period"] + row["vendor_vlt_mean"]) * shape
        value = stats.gamma.ppf(0.9, a=horizon_shape, scale=theta)
        return 0 if np.isnan(value) else int(value)

    frame["Gamma_pred"] = frame.apply(gamma_target, axis=1)
    grouped = frame.groupby("SKU", sort=True).agg(list).reset_index()

    result_rows: list[dict[str, float | str]] = []
    paths: list[dict[str, object]] = []
    for method in METHODS:
        holding_values: list[float] = []
        stockout_values: list[float] = []
        for _, row in grouped.iterrows():
            full_path, evaluation_path = _inventory_path(row, method)
            holding = HOLDING_COST * sum(value for value in evaluation_path if value > 0)
            stockout = STOCKOUT_COST * -sum(value for value in evaluation_path if value < 0)
            holding_values.append(holding)
            stockout_values.append(stockout)
            paths.append({"SKU": row["SKU"], "method": method, "full_path": json.dumps(full_path), "evaluation_path": json.dumps(evaluation_path)})
        result_rows.append(
            {
                "method": method,
                "total_cost": float(np.mean(np.array(holding_values) + np.array(stockout_values))),
                "holding_cost": float(np.mean(holding_values)),
                "stockout_cost": float(np.mean(stockout_values)),
                "n_skus": len(grouped),
            }
        )

    results = pd.DataFrame(result_rows)
    opt_cost = results.loc[results["method"] == "OPT", "total_cost"].iloc[0]
    results["gap_vs_opt_pct"] = (results["total_cost"] / opt_cost - 1.0) * 100.0
    return results, pd.DataFrame(paths)


def _logistic_propensity(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2-regularized logistic model compatible with the role of sklearn's default model."""
    # The original notebook does not standardize covariates before fitting.
    design = np.column_stack([np.ones(len(x)), x])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        score = design @ beta
        probability = expit(score)
        loss = np.logaddexp(0.0, score).sum() - np.dot(y, score) + 0.5 * np.dot(beta[1:], beta[1:])
        gradient = design.T @ (probability - y)
        gradient[1:] += beta[1:]
        return float(loss), gradient

    fit = optimize.minimize(
        fun=lambda beta: objective(beta)[0],
        x0=np.zeros(design.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
    )
    if not fit.success:
        raise RuntimeError(f"Propensity model failed: {fit.message}")
    return expit(design @ fit.x), fit.x


def _standardized_mean_difference(treated: np.ndarray, control: np.ndarray) -> float:
    pooled = math.sqrt((np.var(treated, ddof=1) + np.var(control, ddof=1)) / 2.0)
    return float((np.mean(treated) - np.mean(control)) / pooled) if pooled else 0.0


def reproduce_matching(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    treated = pd.read_csv(data_dir / "order_e2e_post_500.csv").copy()
    candidates = pd.read_csv(data_dir / "control_demand_vlt.csv").copy()
    treated["vlt_num"] = treated["vlt"].map(parse_vlt_days)

    feature_names = ["ave_demand", "vlt_num"]
    treated_x = treated[feature_names].to_numpy(dtype=float)
    control_x = candidates[feature_names].to_numpy(dtype=float)
    x = np.vstack([treated_x, control_x])
    y = np.r_[np.ones(len(treated_x)), np.zeros(len(control_x))]
    scores, coefficients = _logistic_propensity(x, y)
    treated_scores = scores[: len(treated_x)]
    candidate_scores = scores[len(treated_x) :]

    # The released notebook uses nearest neighbors on the two raw covariates.
    # The constant treatment indicator in its distance calculation does not change ranks.
    tree = cKDTree(control_x)
    distances, indices = tree.query(treated_x, k=1)
    matched = candidates.iloc[indices].copy().reset_index(drop=True)
    matched_scores = candidate_scores[indices]

    balance_rows = []
    for column in feature_names:
        before = _standardized_mean_difference(treated[column].to_numpy(), candidates[column].to_numpy())
        after = _standardized_mean_difference(treated[column].to_numpy(), matched[column].to_numpy())
        balance_rows.append(
            {
                "covariate": column,
                "treated_mean": treated[column].mean(),
                "candidate_control_mean": candidates[column].mean(),
                "matched_control_mean": matched[column].mean(),
                "smd_before": before,
                "smd_after": after,
                "abs_smd_reduction_pct": (1 - abs(after) / abs(before)) * 100 if before else np.nan,
            }
        )

    matched_table = pd.DataFrame(
        {
            "treated_row": np.arange(len(treated)),
            "candidate_control_row": indices,
            "distance_raw_covariates": distances,
            "treated_propensity": treated_scores,
            "matched_control_propensity": matched_scores,
        }
    )
    score_data = {
        "treated_before": treated_scores,
        "control_before": candidate_scores,
        "treated_after": treated_scores,
        "control_after": matched_scores,
        "coefficients": coefficients,
    }
    return pd.DataFrame(balance_rows), matched_table, score_data


def calculate_field_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        inventory = parse_array(row["test_inv"])
        evaluated = inventory[:-1]  # faithful to the authors' released notebook
        holding = HOLDING_COST * np.maximum(evaluated, 0).sum()
        stockout = STOCKOUT_COST * -np.minimum(evaluated, 0).sum()
        denominator = float(row["ave_demand"])
        ratio = float(row["ave_inv"] / denominator) if denominator != 0 else np.nan
        rows.append(
            {
                "holding_cost": float(holding),
                "stockout_cost": float(stockout),
                "total_cost": float(holding + stockout),
                "turnover_days": max(0.0, ratio) if np.isfinite(ratio) else np.nan,
                "stockout_rate": float(np.sum(evaluated < 0) / len(inventory)),
                "ave_demand": float(row["ave_demand"]),
                "vlt_num": float(row["vlt_num"] if "vlt_num" in frame.columns else parse_vlt_days(row["vlt"])),
            }
        )
    return pd.DataFrame(rows)


def _ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> pd.DataFrame:
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    standard_error = np.sqrt(np.diag(covariance))
    t_value = beta / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_value), dof)
    return pd.DataFrame(
        {
            "term": names,
            "coefficient": beta,
            "std_error": standard_error,
            "t_value": t_value,
            "p_value": p_value,
            "ci_low": beta - stats.t.ppf(0.975, dof) * standard_error,
            "ci_high": beta + stats.t.ppf(0.975, dof) * standard_error,
        }
    )


def reproduce_field_experiment(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    raw = {
        "e2e_pre": pd.read_csv(data_dir / "order_e2e_pre_500.csv"),
        "e2e_post": pd.read_csv(data_dir / "order_e2e_post_500.csv"),
        "jd_pre": pd.read_csv(data_dir / "order_jd_pre_500.csv"),
        "jd_post": pd.read_csv(data_dir / "order_jd_post_500.csv"),
    }
    metric_frames = {name: calculate_field_metrics(frame) for name, frame in raw.items()}

    post_rows = []
    for metric in METRICS:
        treated = metric_frames["e2e_post"][metric].dropna().to_numpy()
        control = metric_frames["jd_post"][metric].dropna().to_numpy()
        test = stats.ttest_ind(treated, control, equal_var=True)
        treated_mean = treated.mean()
        control_mean = control.mean()
        post_rows.append(
            {
                "metric": metric,
                "treatment_mean": treated_mean,
                "control_mean": control_mean,
                "difference": treated_mean - control_mean,
                "difference_pct_of_control": (treated_mean / control_mean - 1) * 100,
                "t_statistic": test.statistic,
                "p_value": test.pvalue,
            }
        )

    regressions = []
    combined = pd.concat([metric_frames["e2e_post"], metric_frames["jd_post"]], ignore_index=True)
    treatment_indicator = np.r_[np.ones(len(metric_frames["e2e_post"])), np.zeros(len(metric_frames["jd_post"]))]
    for metric in METRICS:
        y = combined[metric].to_numpy()
        finite = np.isfinite(y)
        # Faithful specification in the released notebook: no intercept.
        x_notebook = np.column_stack(
            [treatment_indicator, combined["vlt_num"].to_numpy(), combined["ave_demand"].to_numpy()]
        )
        fit_notebook = _ols(y[finite], x_notebook[finite], ["e2e", "vlt", "ave_demand"])
        fit_notebook.insert(0, "specification", "released_notebook_no_intercept")
        fit_notebook.insert(0, "metric", metric)
        regressions.append(fit_notebook)

        # Equation (11) in the paper includes theta_0, so we also report the intercept-corrected audit.
        x_paper = np.column_stack([np.ones(len(combined)), x_notebook])
        fit_paper = _ols(y[finite], x_paper[finite], ["intercept", "e2e", "vlt", "ave_demand"])
        fit_paper.insert(0, "specification", "paper_equation_11_with_intercept")
        fit_paper.insert(0, "metric", metric)
        regressions.append(fit_paper)

    did_rows = []
    for metric in METRICS:
        e2e_pre = metric_frames["e2e_pre"][metric].to_numpy()
        e2e_post = metric_frames["e2e_post"][metric].to_numpy()
        jd_pre = metric_frames["jd_pre"][metric].to_numpy()
        jd_post = metric_frames["jd_post"][metric].to_numpy()
        valid = np.isfinite(e2e_pre) & np.isfinite(e2e_post) & np.isfinite(jd_pre) & np.isfinite(jd_post)
        post_difference = e2e_post[valid] - jd_post[valid]
        pre_difference = e2e_pre[valid] - jd_pre[valid]
        test = stats.ttest_ind(post_difference, pre_difference, equal_var=True)
        treatment_change = e2e_post[valid].mean() - e2e_pre[valid].mean()
        control_change = jd_post[valid].mean() - jd_pre[valid].mean()
        did_rows.append(
            {
                "metric": metric,
                "treatment_pre": e2e_pre[valid].mean(),
                "treatment_post": e2e_post[valid].mean(),
                "treatment_change": treatment_change,
                "control_pre": jd_pre[valid].mean(),
                "control_post": jd_post[valid].mean(),
                "control_change": control_change,
                "did": treatment_change - control_change,
                "t_statistic": test.statistic,
                "p_value": test.pvalue,
            }
        )

    return pd.DataFrame(post_rows), pd.concat(regressions, ignore_index=True), pd.DataFrame(did_rows), metric_frames


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def make_figures(
    figure_dir: Path,
    offline: pd.DataFrame,
    score_data: dict[str, np.ndarray],
    post_ttest: pd.DataFrame,
    did: pd.DataFrame,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 130})

    x = np.arange(len(offline))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(x, offline["holding_cost"], label="Holding cost", color="#315C8C")
    ax.bar(x, offline["stockout_cost"], bottom=offline["holding_cost"], label="Stockout cost", color="#D97757")
    ax.set_xticks(x, offline["method"])
    ax.set_ylabel("Average 30-day cost per SKU")
    ax.set_title("Section 4 offline evaluation on the disclosed 3,000-row sample")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "section4_cost_comparison.png", dpi=220)
    plt.close(fig)

    non_opt = offline[offline["method"] != "OPT"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = ["#2A9D8F" if name == "E2E_RNN" else "#8FA6B8" for name in non_opt["method"]]
    ax.bar(non_opt["method"], non_opt["gap_vs_opt_pct"], color=colors)
    ax.set_ylabel("Total-cost gap vs. ex-post OPT (%)")
    ax.set_title("Optimality gap: lower is better")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "section4_optimality_gap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
    for ax, treated, control, title in [
        (axes[0], score_data["treated_before"], score_data["control_before"], "Before matching"),
        (axes[1], score_data["treated_after"], score_data["control_after"], "After matching"),
    ]:
        bins = np.linspace(min(treated.min(), control.min()), max(treated.max(), control.max()), 18)
        ax.hist(treated, bins=bins, density=True, alpha=0.55, label="Treatment", color="#D97757")
        ax.hist(control, bins=bins, density=True, alpha=0.55, label="Control", color="#315C8C")
        ax.set_xlabel("Estimated propensity score")
        ax.set_title(title)
        ax.legend(frameon=False)
        _style_axes(ax)
    axes[0].set_ylabel("Density")
    fig.suptitle("Section 5 propensity-score overlap")
    fig.tight_layout()
    fig.savefig(figure_dir / "section5_propensity_matching.png", dpi=220)
    plt.close(fig)

    labels = [DISPLAY_NAMES[m] for m in post_ttest["metric"]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(labels, post_ttest["difference_pct_of_control"], color="#2A9D8F")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("E2E difference relative to JD control (%)")
    ax.tick_params(axis="x", rotation=15)
    ax.set_title("Section 5 post-experiment improvement (public 500-pair subsets)")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "section5_post_experiment.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    did_relative = did["did"] / did["control_pre"].abs() * 100.0
    colors = ["#2A9D8F" if value < 0 else "#D97757" for value in did_relative]
    ax.bar([DISPLAY_NAMES[m] for m in did["metric"]], did_relative, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("DiD relative to control pre-period mean (%)")
    ax.set_title("Normalized difference-in-differences estimates")
    ax.tick_params(axis="x", rotation=15)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(figure_dir / "section5_did.png", dpi=220)
    plt.close(fig)


def run(data_dir: Path, results_dir: Path) -> ReproductionOutputs:
    tables = results_dir / "tables"
    figures = results_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    offline, paths = reproduce_offline(data_dir)
    balance, matched, scores = reproduce_matching(data_dir)
    post_ttest, regressions, did, metric_frames = reproduce_field_experiment(data_dir)

    offline.to_csv(tables / "section4_offline_costs.csv", index=False)
    paths.to_csv(tables / "section4_inventory_paths.csv", index=False)
    balance.to_csv(tables / "section5_matching_balance.csv", index=False)
    matched.to_csv(tables / "section5_matched_indices.csv", index=False)
    post_ttest.to_csv(tables / "section5_post_ttests.csv", index=False)
    regressions.to_csv(tables / "section5_regressions.csv", index=False)
    did.to_csv(tables / "section5_did.csv", index=False)
    for name, frame in metric_frames.items():
        frame.to_csv(tables / f"section5_metrics_{name}.csv", index=False)

    make_figures(figures, offline, scores, post_ttest, did)
    metadata = {
        "data_source": "MS-FTBDA-20-01585 disclosed replication package",
        "offline_rows": int(pd.read_csv(data_dir / "df_test.csv", usecols=["SKU"]).shape[0]),
        "offline_unique_skus": int(pd.read_csv(data_dir / "df_test.csv", usecols=["SKU"])["SKU"].nunique()),
        "treatment_pairs": 500,
        "control_pairs": 500,
        "candidate_controls": int(pd.read_csv(data_dir / "control_demand_vlt.csv", usecols=["ave_demand"]).shape[0]),
        "holding_cost_parameter": HOLDING_COST,
        "stockout_cost_parameter": STOCKOUT_COST,
        "gpu_used": False,
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return ReproductionOutputs(offline, balance, post_ttest, regressions, did, metadata)
