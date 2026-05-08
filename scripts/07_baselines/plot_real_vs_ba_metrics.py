#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def plot_metric(
    merged: pd.DataFrame,
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
):
    """
    merged columns expected:
      language,
      real_<metric_key>,
      ba_<metric_key>_mean,
      ba_<metric_key>_ci95_lo,
      ba_<metric_key>_ci95_hi
    """
    langs = list(merged["language"])

    real = merged[f"real_{metric_key}"].tolist()
    ba_mean = merged[f"ba_{metric_key}_mean"].tolist()

    ba_lo_raw = merged[f"ba_{metric_key}_ci95_lo"].tolist()
    ba_hi_raw = merged[f"ba_{metric_key}_ci95_hi"].tolist()

    # Normalize (lo <= hi) and build non-negative yerr
    ba_lo = []
    ba_hi = []
    for lo, hi in zip(ba_lo_raw, ba_hi_raw):
        if lo is None or hi is None:
            ba_lo.append(None)
            ba_hi.append(None)
            continue
        lo2 = min(lo, hi)
        hi2 = max(lo, hi)
        ba_lo.append(lo2)
        ba_hi.append(hi2)

    # errorbars = distance from mean; must be >= 0
    yerr_lo = []
    yerr_hi = []
    for m, lo, hi in zip(ba_mean, ba_lo, ba_hi):
        if m is None or lo is None or hi is None:
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
            continue
        yerr_lo.append(max(0.0, m - lo))
        yerr_hi.append(max(0.0, hi - m))

    yerr = [yerr_lo, yerr_hi]

    x = list(range(len(langs)))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Real as points
    ax.plot(x, real, marker="o", linestyle="none", label="Real")

    # BA as points + CI
    ax.errorbar(
        x,
        ba_mean,
        yerr=yerr,
        fmt="o",
        capsize=4,
        label="BA mean ± CI95",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="network_out_core/metrics_summary_by_language.csv")
    ap.add_argument("--ba", default="ba_out_core/metrics_ba_summary.csv")
    ap.add_argument("--outdir", default="ba_out_core/plots")
    ap.add_argument("--langs", default="en,de,nl")
    args = ap.parse_args()

    real_path = Path(args.real)
    ba_path = Path(args.ba)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    real = pd.read_csv(real_path)
    ba = pd.read_csv(ba_path)

    # Keep only selected languages
    real = real[real["language"].isin(langs)].copy()
    ba = ba[ba["language"].isin(langs)].copy()

    # Normalize numeric types (some CSVs can have strings)
    for col in real.columns:
        if col != "language":
            real[col] = real[col].map(_safe_float)

    for col in ba.columns:
        if col != "language":
            ba[col] = ba[col].map(_safe_float)

    # Build merged frame with the columns we need
    # REAL columns in your file:
    # language,n_nodes,n_edges,avg_degree,density,n_components,lcc_nodes,lcc_fraction,avg_clustering
    # BA columns:
    # language,... avg_degree_mean, avg_clustering_mean, avg_shortest_path_lcc_mean, ... CI columns

    merged = pd.DataFrame({"language": langs}).merge(real, on="language", how="left").merge(ba, on="language", how="left")

    # Rename to consistent keys for plotting
    merged = merged.rename(
        columns={
            "avg_degree": "real_avg_degree",
            "avg_clustering": "real_avg_clustering",
            # real summary does NOT have shortest path (only per book has it)
            # we'll take real shortest path from per_book means if you want; for now we detect missing.
        }
    )

    # BA columns already have *_mean and CI columns; map them
    merged = merged.rename(
        columns={
            "avg_degree_mean": "ba_avg_degree_mean",
            "avg_degree_ci95_lo": "ba_avg_degree_ci95_lo",
            "avg_degree_ci95_hi": "ba_avg_degree_ci95_hi",
            "avg_clustering_mean": "ba_avg_clustering_mean",
            "avg_clustering_ci95_lo": "ba_avg_clustering_ci95_lo",
            "avg_clustering_ci95_hi": "ba_avg_clustering_ci95_hi",
            "avg_shortest_path_lcc_mean": "ba_avg_shortest_path_mean",
            "avg_shortest_path_lcc_ci95_lo": "ba_avg_shortest_path_ci95_lo",
            "avg_shortest_path_lcc_ci95_hi": "ba_avg_shortest_path_ci95_hi",
        }
    )

    # 1) avg_degree plot
    plot_metric(
        merged,
        metric_key="avg_degree",
        title="Real vs BA: average degree",
        ylabel="avg_degree",
        out_path=outdir / "metrics_real_vs_ba_avg_degree.png",
    )

    # 2) avg_clustering plot
    plot_metric(
        merged,
        metric_key="avg_clustering",
        title="Real vs BA: average clustering coefficient",
        ylabel="avg_clustering",
        out_path=outdir / "metrics_real_vs_ba_avg_clustering.png",
    )

    # 3) shortest path needs REAL value – your summary file doesn't include it.
    # We'll compute REAL shortest path mean from metrics_per_book.csv and re-merge.
    per_book_path = Path("network_out_core/metrics_per_book.csv")
    if per_book_path.exists():
        per_book = pd.read_csv(per_book_path)
        per_book = per_book[per_book["language"].isin(langs)].copy()
        per_book["avg_shortest_path_lcc"] = per_book["avg_shortest_path_lcc"].map(_safe_float)

        real_sp = (
            per_book.groupby("language", as_index=False)["avg_shortest_path_lcc"]
            .mean()
            .rename(columns={"avg_shortest_path_lcc": "real_avg_shortest_path"})
        )

        merged2 = merged.merge(real_sp, on="language", how="left")

        # rename BA to expected key
        merged2 = merged2.rename(columns={"ba_avg_shortest_path_mean": "ba_avg_shortest_path_mean"})
        merged2 = merged2.rename(
            columns={
                "ba_avg_shortest_path_mean": "ba_avg_shortest_path_mean",
                "ba_avg_shortest_path_ci95_lo": "ba_avg_shortest_path_ci95_lo",
                "ba_avg_shortest_path_ci95_hi": "ba_avg_shortest_path_ci95_hi",
            }
        )

        # Align names for plot_metric (expects real_<key> and ba_<key>_mean/ci)
        merged2 = merged2.rename(
            columns={
                "real_avg_shortest_path": "real_avg_shortest_path",
                "ba_avg_shortest_path_mean": "ba_avg_shortest_path_mean",
                "ba_avg_shortest_path_ci95_lo": "ba_avg_shortest_path_ci95_lo",
                "ba_avg_shortest_path_ci95_hi": "ba_avg_shortest_path_ci95_hi",
            }
        )

        plot_metric(
            merged2.rename(
                columns={
                    "real_avg_shortest_path": "real_avg_shortest_path",
                    "ba_avg_shortest_path_mean": "ba_avg_shortest_path_mean",
                    "ba_avg_shortest_path_ci95_lo": "ba_avg_shortest_path_ci95_lo",
                    "ba_avg_shortest_path_ci95_hi": "ba_avg_shortest_path_ci95_hi",
                }
            ),
            metric_key="avg_shortest_path",
            title="Real vs BA: average shortest path (LCC)",
            ylabel="avg_shortest_path_lcc",
            out_path=outdir / "metrics_real_vs_ba_avg_shortest_path.png",
        )
    else:
        print("[WARN] network_out_core/metrics_per_book.csv not found -> skipping shortest path plot.")


if __name__ == "__main__":
    main()