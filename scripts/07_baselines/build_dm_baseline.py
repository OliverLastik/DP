#!/usr/bin/env python3
"""
Dorogovtsev-Mendes (DM) pseudofractal scale-free network baseline.

DM model (Dorogovtsev, Goltsev, Mendes, Phys Rev E 65, 2002):
  - Zacina trojuholnikom (3 uzly, 3 hrany)
  - V kazdom kroku: pre kazdu EXISTUJUCU hranu pridaj novy uzol
    pripojeny k obom koncom tej hrany
  - Vysledok: scale-free (gamma=ln3/ln2 ~ 1.585 pre kumul.),
    vysoky clustering, small-world

Variant pouzity tu: "simplified DM" -- v kazdom kroku nahodne vyber
podmnozinu hran a pripoj novy uzol. Toto umoznuje kontrolovat velkost
siete (target n_nodes).

Porovnava s realnymi sietami aj BA baselinami.
"""
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def build_dm_network(n_target: int, seed: int = 42) -> nx.Graph:
    """
    Simplified Dorogovtsev-Mendes growing network.

    Zacina trojuholnikom. V kazdom kroku:
    - Nahodne vyber hranu (u, v)
    - Pridaj novy uzol w s hranami (w, u) a (w, v)
    Opakuj az kym n_nodes >= n_target.
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    while G.number_of_nodes() < n_target:
        edges = list(G.edges())
        u, v = rng.choice(edges)
        w = G.number_of_nodes()
        G.add_edges_from([(w, u), (w, v)])

    return G


def approx_avg_shortest_path_length(G: nx.Graph, samples: int = 200, seed: int = 42) -> float:
    """Aproximacia avg shortest path na LCC."""
    if G.number_of_nodes() <= 1:
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
        count += len(lengths) - 1

    return (total / count) if count > 0 else float("nan")


def graph_metrics(G: nx.Graph, sp_samples: int = 200, seed: int = 42) -> dict:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    avg_deg = (2 * e / n) if n > 0 else 0.0

    if n == 0:
        return {"n_nodes": 0, "n_edges": 0, "avg_degree": 0,
                "avg_clustering": 0, "avg_shortest_path_lcc": 0, "density": 0}

    comps = list(nx.connected_components(G))
    lcc = max(comps, key=len) if comps else set()
    lcc_frac = len(lcc) / n

    avg_cl = nx.average_clustering(G) if n > 1 else 0.0

    if len(lcc) <= 1:
        avg_spl = 0.0
    else:
        Glcc = G.subgraph(lcc).copy()
        avg_spl = approx_avg_shortest_path_length(Glcc, samples=sp_samples, seed=seed)

    return {
        "n_nodes": n, "n_edges": e,
        "avg_degree": float(avg_deg),
        "density": nx.density(G),
        "avg_clustering": float(avg_cl),
        "avg_shortest_path_lcc": float(avg_spl),
        "lcc_fraction": float(lcc_frac),
    }


def degree_histogram_pk(G: nx.Graph):
    degs = [d for _, d in G.degree()]
    if not degs:
        return np.array([]), np.array([])
    vals, counts = np.unique(np.array(degs, dtype=int), return_counts=True)
    pk = counts / counts.sum()
    return vals.astype(float), pk


def logbin_pk(ks, pk, n_bins=40):
    mask = (ks > 0) & (pk > 0)
    k, p = ks[mask], pk[mask]
    if len(k) == 0:
        return np.array([]), np.array([])
    log_k = np.log10(k)
    edges = np.linspace(log_k.min(), log_k.max(), n_bins + 1)
    bin_k, bin_p = [], []
    for i in range(len(edges) - 1):
        m = (log_k >= edges[i]) & (log_k < edges[i + 1])
        if m.sum() > 0:
            bin_k.append(10 ** np.mean(log_k[m]))
            bin_p.append(np.mean(p[m]))
    return np.array(bin_k), np.array(bin_p)


def load_real_degree_hist(path: Path):
    ks, counts = [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = int(float(row["k"]))
                c = int(float(row["count"]))
            except Exception:
                continue
            if k > 0 and c > 0:
                ks.append(k)
                counts.append(c)
    ks = np.array(ks, dtype=float)
    counts = np.array(counts, dtype=float)
    return ks, counts / counts.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-out", default="network_out_core")
    ap.add_argument("--ba-out", default="ba_out_core")
    ap.add_argument("--out", default="dm_out_core")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--n-runs", type=int, default=10, help="DM runs per language")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sp-samples", type=int, default=200)
    args = ap.parse_args()

    real_dir = Path(args.real_out)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "degree_histograms").mkdir(parents=True, exist_ok=True)

    summary_path = real_dir / "metrics_summary_by_language.csv"
    real_summary = {}
    with summary_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            real_summary[row["language"]] = row

    # Load BA summary
    ba_summary = {}
    ba_path = Path(args.ba_out) / "metrics_ba_summary.csv"
    if ba_path.exists():
        with ba_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ba_summary[row["language"]] = row

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_runs = []
    all_summary = []

    for lang in langs:
        if lang not in real_summary:
            print(f"[WARN] {lang} not in real summary")
            continue

        n_target = int(round(float(real_summary[lang]["n_nodes"])))
        print(f"\n[INFO] {lang}: target n={n_target}, generating {args.n_runs} DM networks...")

        pk_runs = []
        lang_metrics = []

        for i in range(args.n_runs):
            run_seed = args.seed + 1000 * (hash(lang) % 1000) + i
            G = build_dm_network(n_target, seed=run_seed)
            met = graph_metrics(G, sp_samples=args.sp_samples, seed=run_seed)
            met.update({"language": lang, "run": i, "seed": run_seed})
            all_runs.append(met)
            lang_metrics.append(met)

            ks, pk = degree_histogram_pk(G)
            pk_runs.append((ks, pk))

            if (i + 1) % 5 == 0:
                print(f"  run {i+1}/{args.n_runs} done")

        # aggregate metrics
        df_m = [{k: v for k, v in m.items()} for m in lang_metrics]
        avg_cl = np.mean([m["avg_clustering"] for m in lang_metrics])
        avg_deg = np.mean([m["avg_degree"] for m in lang_metrics])
        avg_spl = np.mean([m["avg_shortest_path_lcc"] for m in lang_metrics])

        summ = {
            "language": lang,
            "n_runs": args.n_runs,
            "n_nodes_mean": np.mean([m["n_nodes"] for m in lang_metrics]),
            "avg_degree_mean": avg_deg,
            "avg_clustering_mean": avg_cl,
            "avg_shortest_path_lcc_mean": avg_spl,
            "avg_clustering_ci95_lo": float(np.quantile([m["avg_clustering"] for m in lang_metrics], 0.025)),
            "avg_clustering_ci95_hi": float(np.quantile([m["avg_clustering"] for m in lang_metrics], 0.975)),
            "avg_spl_ci95_lo": float(np.quantile([m["avg_shortest_path_lcc"] for m in lang_metrics], 0.025)),
            "avg_spl_ci95_hi": float(np.quantile([m["avg_shortest_path_lcc"] for m in lang_metrics], 0.975)),
        }
        all_summary.append(summ)

        print(f"  [OK] {lang}: avg_clustering={avg_cl:.4f}, avg_degree={avg_deg:.2f}, avg_spl={avg_spl:.3f}")

        # aggregate degree distribution
        all_k = sorted(set(int(k) for ks, _ in pk_runs for k in ks))
        pk_agg = []
        for k in all_k:
            vals = []
            for ks_r, pk_r in pk_runs:
                idx = np.where(ks_r == k)[0]
                vals.append(float(pk_r[idx[0]]) if len(idx) > 0 else 0.0)
            pk_agg.append((k, np.mean(vals)))

        # save degree hist
        hist_csv = out_dir / "degree_histograms" / f"degree_hist_dm_{lang}.csv"
        with hist_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "pk_mean"])
            for k, p in pk_agg:
                w.writerow([k, p])

        # ── Plot: Real vs BA vs DM ──
        # load real
        real_hist = real_dir / "degree_histograms" / f"degree_hist_{lang}.csv"
        if real_hist.exists():
            r_ks, r_pk = load_real_degree_hist(real_hist)
            r_ks_lb, r_pk_lb = logbin_pk(r_ks, r_pk)

            dm_ks = np.array([k for k, _ in pk_agg], dtype=float)
            dm_pk = np.array([p for _, p in pk_agg])
            dm_ks_lb, dm_pk_lb = logbin_pk(dm_ks, dm_pk)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(r_ks_lb, r_pk_lb, "o-", color="C0", markersize=5, linewidth=1.5, label="Real")
            ax.plot(dm_ks_lb, dm_pk_lb, "s-", color="C2", markersize=4, linewidth=1.2, label="DM")

            # load BA
            ba_hist = Path(args.ba_out) / "degree_histograms" / f"degree_hist_ba_{lang}.csv"
            if ba_hist.exists():
                ba_data = []
                with ba_hist.open("r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        try:
                            ba_data.append((float(row["k"]), float(row["pk_mean"])))
                        except Exception:
                            continue
                if ba_data:
                    ba_ks = np.array([k for k, _ in ba_data])
                    ba_pk = np.array([p for _, p in ba_data])
                    ba_ks_lb, ba_pk_lb = logbin_pk(ba_ks, ba_pk)
                    ax.plot(ba_ks_lb, ba_pk_lb, "^--", color="C3", markersize=4, linewidth=1.0, label="BA")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("k (degree)", fontsize=11)
            ax.set_ylabel("P(k)", fontsize=11)
            ax.set_title(f"Degree distribution: Real vs BA vs DM -- {lang.upper()}", fontsize=12)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "plots" / f"degree_real_ba_dm_{lang}.png", dpi=300)
            plt.close(fig)

    # ── Save CSVs ──
    runs_csv = out_dir / "metrics_dm_runs.csv"
    fields = ["language", "run", "seed", "n_nodes", "n_edges", "avg_degree",
              "density", "avg_clustering", "avg_shortest_path_lcc", "lcc_fraction"]
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_runs:
            w.writerow({k: r.get(k, "") for k in fields})

    summ_csv = out_dir / "metrics_dm_summary.csv"
    with summ_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
        w.writeheader()
        w.writerows(all_summary)

    # ── Comparison bar chart: Real vs BA vs DM ──
    # Zisti ktore metriky su dostupne v real summary
    available_real_metrics = set(real_summary[langs[0]].keys()) if langs else set()
    metrics_to_compare = []
    if "avg_clustering" in available_real_metrics:
        metrics_to_compare.append(("avg_clustering", "Avg clustering"))
    if "avg_shortest_path_lcc" in available_real_metrics:
        metrics_to_compare.append(("avg_shortest_path_lcc", "Avg shortest path (LCC)"))

    # DM ma avg_shortest_path_lcc, tak ho porovname s BA aj ked real nema
    # Pridame clustering porovnanie vzdy
    for metric, label in metrics_to_compare:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(langs))
        width = 0.25

        real_vals = [float(real_summary[l].get(metric, 0)) for l in langs]
        dm_vals = [s[f"{metric}_mean"] for s in all_summary]
        ba_vals = [float(ba_summary[l].get(f"{metric}_mean", 0)) if l in ba_summary else 0 for l in langs]

        ax.bar(x - width, real_vals, width, label="Real", color="C0", alpha=0.85)
        ax.bar(x, dm_vals, width, label="DM", color="C2", alpha=0.85)
        ax.bar(x + width, ba_vals, width, label="BA", color="C3", alpha=0.85)

        for i, v in enumerate(real_vals):
            ax.text(i - width, v + max(real_vals) * 0.02, f"{v:.3f}", ha="center", fontsize=7)
        for i, v in enumerate(dm_vals):
            ax.text(i, v + max(dm_vals) * 0.02, f"{v:.3f}", ha="center", fontsize=7)
        for i, v in enumerate(ba_vals):
            ax.text(i + width, v + max(max(ba_vals, default=0.01), 0.001) * 0.02, f"{v:.3f}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in langs])
        ax.set_ylabel(label)
        ax.set_title(f"{label}: Real vs DM vs BA")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"compare_{metric}.png", dpi=300)
        plt.close(fig)

    # Aj DM vs BA shortest path (bez real)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(langs))
    width = 0.3
    dm_spl = [s["avg_shortest_path_lcc_mean"] for s in all_summary]
    ba_spl = [float(ba_summary[l].get("avg_shortest_path_lcc_mean", 0)) if l in ba_summary else 0 for l in langs]
    ax.bar(x - width/2, dm_spl, width, label="DM", color="C2", alpha=0.85)
    ax.bar(x + width/2, ba_spl, width, label="BA", color="C3", alpha=0.85)
    for i, v in enumerate(dm_spl):
        ax.text(i - width/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(ba_spl):
        ax.text(i + width/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs])
    ax.set_ylabel("Avg shortest path (LCC)")
    ax.set_title("Avg shortest path: DM vs BA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "compare_avg_spl_dm_ba.png", dpi=300)
    plt.close(fig)

    print(f"\n[OK] All saved to {out_dir}")
    print("Hotovo!")


if __name__ == "__main__":
    main()
