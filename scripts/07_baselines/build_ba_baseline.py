#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt

# networks
import networkx as nx


def safe_m_from_avg_degree(avg_degree: float) -> int:
    """BA parameter m ~ avg_degree/2, must be >=1."""
    m = int(round(float(avg_degree) / 2.0))
    return max(1, m)


def approx_avg_shortest_path_length(G: nx.Graph, samples: int = 200, seed: int = 42) -> float:
    """
    Fast approximation of average shortest path length:
    sample 'samples' source nodes and average distances from them.
    """
    if G.number_of_nodes() == 0:
        return float("nan")
    if G.number_of_nodes() == 1:
        return 0.0

    rng = random.Random(seed)
    nodes = list(G.nodes())
    s = min(samples, len(nodes))
    sources = rng.sample(nodes, s)

    total = 0
    count = 0
    for src in sources:
        lengths = nx.single_source_shortest_path_length(G, src)
        total += sum(lengths.values())
        count += len(lengths) - 1  # exclude distance to itself

    return (total / count) if count > 0 else float("nan")


def graph_metrics_like_yours(G: nx.Graph, sp_samples: int = 200, seed: int = 42) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0.0
    density = nx.density(G) if n_nodes > 1 else 0.0

    # components / LCC
    if n_nodes == 0:
        n_components = 0
        lcc_nodes = 0
        lcc_fraction = 0.0
        avg_clustering = float("nan")
        avg_spl_lcc = float("nan")
    else:
        comps = list(nx.connected_components(G))
        n_components = len(comps)
        lcc = max(comps, key=len) if comps else set()
        lcc_nodes = len(lcc)
        lcc_fraction = lcc_nodes / n_nodes if n_nodes > 0 else 0.0

        # clustering on whole graph (as in your summary)
        avg_clustering = nx.average_clustering(G) if n_nodes > 1 else 0.0

        # avg shortest path on LCC only (approx so it doesn't take ages)
        if lcc_nodes <= 1:
            avg_spl_lcc = 0.0
        else:
            Glcc = G.subgraph(lcc).copy()
            avg_spl_lcc = approx_avg_shortest_path_length(Glcc, samples=sp_samples, seed=seed)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": float(avg_degree),
        "density": float(density),
        "n_components": int(n_components),
        "lcc_nodes": int(lcc_nodes),
        "lcc_fraction": float(lcc_fraction),
        "avg_clustering": float(avg_clustering),
        "avg_shortest_path_lcc": float(avg_spl_lcc),
    }


def degree_histogram_pk(G: nx.Graph) -> pd.DataFrame:
    degs = [d for _, d in G.degree()]
    if not degs:
        return pd.DataFrame({"k": [], "count": [], "pk": []})

    vals, counts = np.unique(np.array(degs, dtype=int), return_counts=True)
    pk = counts / counts.sum()
    return pd.DataFrame({"k": vals, "count": counts, "pk": pk})


def load_real_degree_hist(real_dir: Path, lang: str) -> pd.DataFrame:
    """
    Tries to load real degree histogram from:
      network_out_core/degree_histograms/degree_hist_<lang>.csv
    Accepts flexible column names.
    """
    p = real_dir / "degree_histograms" / f"degree_hist_{lang}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing real degree histogram: {p}")

    df = pd.read_csv(p)

    # normalize columns
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # common possibilities: k/count/pk OR degree/freq/prob
    if "k" not in df.columns:
        if "degree" in df.columns:
            df = df.rename(columns={"degree": "k"})
        elif cols:
            df = df.rename(columns={cols[0]: "k"})

    if "count" not in df.columns:
        if "freq" in df.columns:
            df = df.rename(columns={"freq": "count"})
        elif "counts" in df.columns:
            df = df.rename(columns={"counts": "count"})

    if "pk" not in df.columns:
        if "p" in df.columns:
            df = df.rename(columns={"p": "pk"})
        elif "prob" in df.columns:
            df = df.rename(columns={"prob": "pk"})
        elif "probability" in df.columns:
            df = df.rename(columns={"probability": "pk"})

    # if pk missing but count exists, compute
    if "pk" not in df.columns and "count" in df.columns:
        s = df["count"].sum()
        df["pk"] = df["count"] / s if s > 0 else 0.0

    df = df[["k", "count", "pk"]].copy()
    return df.sort_values("k")


def aggregate_pk_over_runs(pk_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Align degrees k across runs and average pk.
    """
    if not pk_dfs:
        return pd.DataFrame({"k": [], "pk_mean": []})

    all_k = sorted(set(int(k) for df in pk_dfs for k in df["k"].tolist()))
    rows = []
    for k in all_k:
        vals = []
        for df in pk_dfs:
            m = df[df["k"] == k]
            if not m.empty:
                vals.append(float(m["pk"].iloc[0]))
            else:
                vals.append(0.0)
        rows.append((k, float(np.mean(vals))))
    return pd.DataFrame(rows, columns=["k", "pk_mean"])


def plot_real_vs_ba_pk(real_df: pd.DataFrame, ba_pk_mean: pd.DataFrame, out_png: Path, title: str):
    plt.figure(figsize=(6.5, 4.5))

    # avoid zeros in log-scale by filtering k>=1 and pk>0
    r = real_df[(real_df["k"] >= 1) & (real_df["pk"] > 0)]
    b = ba_pk_mean[(ba_pk_mean["k"] >= 1) & (ba_pk_mean["pk_mean"] > 0)]

    plt.plot(r["k"], r["pk"], marker="o", linestyle="none", markersize=3, label="Real")
    plt.plot(b["k"], b["pk_mean"], marker="o", linestyle="none", markersize=3, label="BA (mean over runs)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("k (degree)")
    plt.ylabel("P(k)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-out", default="network_out_core", help="Folder with real network outputs")
    ap.add_argument("--out", default="ba_out_core", help="Output folder for BA baseline")
    ap.add_argument("--langs", default="en,de,nl", help="Languages")
    ap.add_argument("--n-runs", type=int, default=20, help="BA runs per language")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    ap.add_argument("--sp-samples", type=int, default=200, help="Samples for approx avg shortest path on LCC")
    args = ap.parse_args()

    real_dir = Path(args.real_out)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_plots = out_dir / "plots"
    out_hists = out_dir / "degree_histograms"
    out_plots.mkdir(parents=True, exist_ok=True)
    out_hists.mkdir(parents=True, exist_ok=True)

    summary_path = real_dir / "metrics_summary_by_language.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")

    real_summary = pd.read_csv(summary_path)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_runs_rows = []
    all_ba_summary_rows = []

    for lang in langs:
        row = real_summary[real_summary["language"] == lang]
        if row.empty:
            print(f"[WARN] language not found in summary: {lang}")
            continue

        n_nodes_real = int(round(float(row["n_nodes"].iloc[0])))
        avg_degree_real = float(row["avg_degree"].iloc[0])
        m = safe_m_from_avg_degree(avg_degree_real)

        # BA constraints
        if m >= n_nodes_real:
            m = max(1, n_nodes_real - 1)

        print(f"[INFO] {lang}: n={n_nodes_real}, avg_degree={avg_degree_real:.3f} => BA m={m}")

        # generate BA runs
        pk_runs = []
        for i in range(args.n_runs):
            run_seed = args.seed + 1000 * (hash(lang) % 1000) + i
            G = nx.barabasi_albert_graph(n=n_nodes_real, m=m, seed=run_seed)

            met = graph_metrics_like_yours(G, sp_samples=args.sp_samples, seed=run_seed)
            met.update({"language": lang, "run": i, "ba_m": m, "ba_seed": run_seed})
            all_runs_rows.append(met)

            pk_df = degree_histogram_pk(G)
            pk_runs.append(pk_df)

        # aggregate pk over runs
        pk_mean = aggregate_pk_over_runs(pk_runs)
        pk_mean_out = out_hists / f"degree_hist_ba_{lang}.csv"
        pk_mean.to_csv(pk_mean_out, index=False)

        # plot real vs BA pk
        real_deg = load_real_degree_hist(real_dir, lang)
        out_png = out_plots / f"degree_pk_real_vs_ba_{lang}.png"
        plot_real_vs_ba_pk(
            real_df=real_deg,
            ba_pk_mean=pk_mean,
            out_png=out_png,
            title=f"Degree distribution P(k): Real vs BA ({lang})",
        )

        # summary of BA metrics for this language
        runs_lang = [r for r in all_runs_rows if r["language"] == lang]
        df_runs_lang = pd.DataFrame(runs_lang)

        def q(x, p):  # quantile helper
            return float(np.quantile(x, p))

        summ = {
            "language": lang,
            "n_runs": int(args.n_runs),
            "ba_m": int(m),
            "n_nodes_mean": float(df_runs_lang["n_nodes"].mean()),
            "n_edges_mean": float(df_runs_lang["n_edges"].mean()),
            "avg_degree_mean": float(df_runs_lang["avg_degree"].mean()),
            "avg_clustering_mean": float(df_runs_lang["avg_clustering"].mean()),
            "avg_shortest_path_lcc_mean": float(df_runs_lang["avg_shortest_path_lcc"].mean()),

            "avg_degree_ci95_lo": q(df_runs_lang["avg_degree"], 0.025),
            "avg_degree_ci95_hi": q(df_runs_lang["avg_degree"], 0.975),

            "avg_clustering_ci95_lo": q(df_runs_lang["avg_clustering"], 0.025),
            "avg_clustering_ci95_hi": q(df_runs_lang["avg_clustering"], 0.975),

            "avg_shortest_path_lcc_ci95_lo": q(df_runs_lang["avg_shortest_path_lcc"], 0.025),
            "avg_shortest_path_lcc_ci95_hi": q(df_runs_lang["avg_shortest_path_lcc"], 0.975),
        }
        all_ba_summary_rows.append(summ)

        print(f"[OK] {lang}: saved {pk_mean_out} and {out_png}")

    # save run metrics and summary
    runs_out = out_dir / "metrics_ba_runs.csv"
    summary_out = out_dir / "metrics_ba_summary.csv"
    pd.DataFrame(all_runs_rows).to_csv(runs_out, index=False)
    pd.DataFrame(all_ba_summary_rows).to_csv(summary_out, index=False)

    print(f"[OK] Saved BA runs: {runs_out}")
    print(f"[OK] Saved BA summary: {summary_out}")


if __name__ == "__main__":
    main()
