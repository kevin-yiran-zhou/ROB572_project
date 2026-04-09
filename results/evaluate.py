from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


def compute_metrics(gt: np.ndarray, est: np.ndarray) -> dict[str, float]:
    if gt.size == 0 or est.size == 0:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "ratio": float("nan"),
        }

    err = est - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    gt_pos = gt > 0
    ratio = float(np.mean(est[gt_pos] / gt[gt_pos])) if np.any(gt_pos) else float("nan")

    return {
        "count": float(gt.size),
        "mae": mae,
        "rmse": rmse,
        "ratio": ratio,
    }


def format_metrics_row(label: str, m: dict[str, float]) -> str:
    return (
        f"{label:<10} "
        f"{int(m['count']):>8} "
        f"{m['mae']:>10.3f} "
        f"{m['rmse']:>10.3f} "
        f"{m['ratio']:>10.3f}"
    )


def build_report_text(model_data: dict[str, pd.DataFrame]) -> str:
    # Cumulative GT ranges in meters: [0,10), [0,20), [0,50)
    ranges = [(0.0, 10.0), (0.0, 20.0), (0.0, 50.0)]
    labels = ["0-10", "0-20", "0-50"]

    lines: list[str] = []
    lines.append("=== Evaluation Metrics (MAE, RMSE, Est/GT) ===")
    lines.append("All metrics are grouped by GT distance bins (including ALL).")
    lines.append("")
    model_order = ["small", "base", "large"]
    for model_name in model_order:
        if model_name not in model_data:
            continue
        df = model_data[model_name]
        lines.append(f"[{model_name}]")
        gt_all = df["GT"].to_numpy(dtype=float)
        est_all = df["est"].to_numpy(dtype=float)
        lines.append(
            f"{'bin':<10} {'count':>8} {'mae(m)':>10} {'rmse(m)':>10} {'est/gt':>10}"
        )

        m_all = compute_metrics(gt_all, est_all)
        lines.append(format_metrics_row("ALL", m_all))

        for i in range(len(labels)):
            lo, hi = ranges[i]
            mask = (gt_all >= lo) & (gt_all < hi)

            m = compute_metrics(gt_all[mask], est_all[mask])
            lines.append(format_metrics_row(labels[i], m))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    results_dir = Path(__file__).resolve().parent
    csv_paths = sorted(
        p for p in results_dir.glob("*.csv") if p.stem in {"small", "base", "large"}
    )
    if not csv_paths:
        raise FileNotFoundError(
            f"No model CSV files found in {results_dir}. "
            "Expected one or more of: small.csv, base.csv, large.csv"
        )

    model_data: dict[str, pd.DataFrame] = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "GT" not in df.columns or "est" not in df.columns:
            raise ValueError(
                f"{csv_path.name} missing required columns. "
                f"Found columns: {list(df.columns)}; expected ['GT', 'est']"
            )
        model_data[csv_path.stem] = df[["GT", "est"]].dropna()

    report_text = build_report_text(model_data)
    out_path = results_dir / "evaluate.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote evaluation report: {out_path}")


if __name__ == "__main__":
    main()
