#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def safe_ratio(a, b):
    return float(a / b) if b and np.isfinite(b) and b != 0 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-out", default="network_out_core")
    ap.add_argument("--ba-out", default="ba_out_core")
    ap.add_argument("--out", default="ba_out_core/compare_real_vs_ba.csv")
    args = ap.parse_args()

    real_out = Path(args.real_out)
    ba_out = Path(args.ba_out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # REAL
    real_summary = pd.read_csv(real_out / "metrics_summary_by_language.csv")
    real_per_book = pd.read_csv(real_out / "metrics_per_book.csv")

    # mean shortest path per language from per-book
    spl = (
        real_per_book
        .groupby("language", as_index=False)["avg_shortest_path_lcc"]
        .mean()
        .rename(columns={"avg_shortest_path_lcc": "real_avg_shortest_path_lcc_mean"})
    )
    real = real_summary.merge(spl, on="language", how="left")

    # BA
    ba = pd.read_csv(ba_out / "metrics_ba_summary.csv")
    ba = ba[[
        "language", "ba_m",
        "avg_degree_mean", "avg_clustering_mean", "avg_shortest_path_lcc_mean",
        "avg_degree_ci95_lo", "avg_degree_ci95_hi",
        "avg_clustering_ci95_lo", "avg_clustering_ci95_hi",
        "avg_shortest_path_lcc_ci95_lo", "avg_shortest_path_lcc_ci95_hi",
    ]].rename(columns={
        "avg_degree_mean": "ba_avg_degree_mean",
        "avg_clustering_mean": "ba_avg_clustering_mean",
        "avg_shortest_path_lcc_mean": "ba_avg_shortest_path_lcc_mean",
    })

    df = real.merge(ba, on="language", how="inner")

    rows = []
    for r in df.itertuples(index=False):
        real_avg_degree = r.avg_degree
        real_avg_clustering = r.avg_clustering
        real_spl = r.real_avg_shortest_path_lcc_mean

        ba_avg_degree = r.ba_avg_degree_mean
        ba_clust = r.ba_avg_clustering_mean
        ba_spl = r.ba_avg_shortest_path_lcc_mean

        rows.append({
            "language": r.language,

            "real_n_nodes": r.n_nodes,
            "real_n_edges": r.n_edges,
            "real_avg_degree": real_avg_degree,

            "ba_m": r.ba_m,
            "ba_avg_degree_mean": ba_avg_degree,
            "delta_avg_degree": real_avg_degree - ba_avg_degree,
            "ratio_avg_degree": safe_ratio(real_avg_degree, ba_avg_degree),

            "real_avg_clustering": real_avg_clustering,
            "ba_avg_clustering_mean": ba_clust,
            "delta_avg_clustering": real_avg_clustering - ba_clust,
            "ratio_avg_clustering": safe_ratio(real_avg_clustering, ba_clust),

            "real_avg_shortest_path_lcc_mean": real_spl,
            "ba_avg_shortest_path_lcc_mean": ba_spl,
            "delta_avg_shortest_path_lcc": real_spl - ba_spl,
            "ratio_avg_shortest_path_lcc": safe_ratio(real_spl, ba_spl),

            "ba_avg_degree_ci95_lo": r.avg_degree_ci95_lo,
            "ba_avg_degree_ci95_hi": r.avg_degree_ci95_hi,
            "ba_avg_clustering_ci95_lo": r.avg_clustering_ci95_lo,
            "ba_avg_clustering_ci95_hi": r.avg_clustering_ci95_hi,
            "ba_avg_shortest_path_ci95_lo": r.avg_shortest_path_lcc_ci95_lo,
            "ba_avg_shortest_path_ci95_hi": r.avg_shortest_path_lcc_ci95_hi,
        })

    out_df = pd.DataFrame(rows).sort_values("language")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved comparison: {out_path}")

if __name__ == "__main__":
    main()