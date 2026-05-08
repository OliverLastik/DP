#!/usr/bin/env python3
"""
BA baseline scale-up: generuje BA siete s vacsim N (default 50 000)
nez realna siet (N ~ 4500), aby sa overilo, ze namerany gamma exponent
sa priblizi k teoretickej asymptotickej hodnote 3.

Motivacia: existujuca BA evaluacia (build_ba_baseline.py) generuje
siete s rovnakym N ako realna siet (~4500 uzlov). Pri tejto velkosti
finite-size efekty znizuju empiricky gamma na ~2.7. Vacsie N posunie
gamma blizsie k 3.

Vystup:
  ba_out_core/scaleup/metrics_runs.csv     - metriky kazdeho runu
  ba_out_core/scaleup/gamma_fit_summary.csv - aggregated gamma per lang
  ba_out_core/scaleup/degree_hist_<lang>.csv - mean P(k) per lang
  ba_out_core/scaleup/plots/ba_scaleup_pk_<lang>.png
  ba_out_core/scaleup/plots/gamma_vs_N.png  - ak je viac N hodnot
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def degree_pk(G):
    degs = np.array([d for _, d in G.degree()], dtype=int)
    if degs.size == 0:
        return np.array([]), np.array([])
    vals, counts = np.unique(degs, return_counts=True)
    pk = counts / counts.sum()
    return vals, pk


def logbin(ks, pk, n_bins=50):
    mask = (ks > 0) & (pk > 0)
    k = ks[mask]; p = pk[mask]
    if len(k) == 0:
        return np.array([]), np.array([])
    log_k = np.log10(k)
    edges = np.linspace(log_k.min(), log_k.max(), n_bins + 1)
    bk, bp = [], []
    for i in range(len(edges) - 1):
        in_bin = (log_k >= edges[i]) & (log_k < edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bk.append(10 ** np.mean(log_k[in_bin]))
        bp.append(np.mean(p[in_bin]))
    return np.array(bk), np.array(bp)


def fit_powerlaw(bk, bp, lo, hi):
    """Linear fit on log-log within [lo, hi]. Returns gamma, C, R^2."""
    mask = (bk >= lo) & (bk <= hi) & (bp > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.log10(bk[mask])
    y = np.log10(bp[mask])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -slope, 10 ** intercept, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50000, help="BA network size")
    ap.add_argument("--m", type=int, default=4, help="BA edges per new node")
    ap.add_argument("--n-runs", type=int, default=10, help="Runs per language")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="ba_out_core/scaleup")
    ap.add_argument("--fit-lo-frac", type=float, default=0.001,
                    help="Lower fit bound as fraction of k_max (default 0.001)")
    ap.add_argument("--fit-hi-frac", type=float, default=0.3,
                    help="Upper fit bound as fraction of k_max (default 0.3)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    print(f"[INFO] BA scale-up: N={args.N}, m={args.m}, n_runs={args.n_runs}, "
          f"langs={langs}")

    runs_rows = []
    summary_rows = []

    for lang in langs:
        print(f"\n[{lang}] generating {args.n_runs} BA networks...")
        per_lang_pk_aligned = {}  # k -> list of pk values across runs

        gammas, clusterings, k_maxes = [], [], []
        t_lang = time.time()
        for i in range(args.n_runs):
            seed = args.seed + 1000 * (hash(lang) % 1000) + i
            t0 = time.time()
            G = nx.barabasi_albert_graph(n=args.N, m=args.m, seed=seed)
            t_build = time.time() - t0

            ks, pk = degree_pk(G)
            k_max = int(ks.max()) if ks.size else 0
            k_maxes.append(k_max)

            # gamma fit on log-binned data
            bk, bp = logbin(ks, pk, n_bins=60)
            fit_lo = max(args.m + 1, int(k_max * args.fit_lo_frac))
            fit_hi = max(fit_lo + 5, int(k_max * args.fit_hi_frac))
            gamma, C, r2 = fit_powerlaw(bk, bp, fit_lo, fit_hi)
            gammas.append(gamma)

            # clustering: only on first run (expensive at large N)
            if i == 0:
                t_cl = time.time()
                avg_cl = nx.average_clustering(G)
                t_cl = time.time() - t_cl
                clusterings.append(avg_cl)
                print(f"  run {i}: build {t_build:.1f}s, k_max={k_max}, "
                      f"gamma={gamma:.3f} (fit [{fit_lo}, {fit_hi}], R^2={r2:.4f}), "
                      f"avg_cl={avg_cl:.5f} ({t_cl:.1f}s)")
            else:
                avg_cl = float("nan")
                print(f"  run {i}: build {t_build:.1f}s, k_max={k_max}, "
                      f"gamma={gamma:.3f}, R^2={r2:.4f}")

            for k, p in zip(ks.tolist(), pk.tolist()):
                per_lang_pk_aligned.setdefault(int(k), []).append(float(p))

            runs_rows.append({
                "lang": lang, "run": i, "N": args.N, "m": args.m, "seed": seed,
                "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
                "k_max": k_max, "gamma": gamma, "C": C, "r2": r2,
                "fit_lo": fit_lo, "fit_hi": fit_hi,
                "avg_clustering": avg_cl,
            })

        print(f"  ({time.time() - t_lang:.1f}s total for lang)")

        # Aggregate P(k) over runs
        all_k = sorted(per_lang_pk_aligned.keys())
        pk_mean = []
        pk_std = []
        for k in all_k:
            vals = per_lang_pk_aligned[k]
            # pad with zeros for runs that didn't have this k
            padded = vals + [0.0] * (args.n_runs - len(vals))
            pk_mean.append(np.mean(padded))
            pk_std.append(np.std(padded))

        hist_path = out_dir / f"degree_hist_{lang}.csv"
        with hist_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "pk_mean", "pk_std"])
            for k, m, s in zip(all_k, pk_mean, pk_std):
                w.writerow([k, m, s])

        # Aggregated fit
        ks_arr = np.array(all_k)
        pk_arr = np.array(pk_mean)
        bk, bp = logbin(ks_arr, pk_arr, n_bins=60)
        fit_lo = max(args.m + 1, int(np.mean(k_maxes) * args.fit_lo_frac))
        fit_hi = max(fit_lo + 5, int(np.mean(k_maxes) * args.fit_hi_frac))
        gamma_agg, C_agg, r2_agg = fit_powerlaw(bk, bp, fit_lo, fit_hi)

        summary_rows.append({
            "lang": lang, "N": args.N, "m": args.m, "n_runs": args.n_runs,
            "gamma_mean": float(np.mean(gammas)),
            "gamma_std": float(np.std(gammas)),
            "gamma_lo": float(np.percentile(gammas, 2.5)),
            "gamma_hi": float(np.percentile(gammas, 97.5)),
            "gamma_aggregated": gamma_agg,
            "r2_aggregated": r2_agg,
            "fit_lo": fit_lo, "fit_hi": fit_hi,
            "k_max_mean": float(np.mean(k_maxes)),
            "avg_clustering": clusterings[0] if clusterings else float("nan"),
        })
        print(f"  -> gamma_mean = {np.mean(gammas):.3f} +- {np.std(gammas):.3f}, "
              f"gamma_aggregated = {gamma_agg:.3f}")

        # Plot per-lang P(k)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ks_arr, pk_arr, "o", color="#1f77b4", markersize=3, alpha=0.6,
                label=f"BA N={args.N} (mean over {args.n_runs} runs)")
        ax.plot(bk, bp, "s-", color="#d62728", markersize=5, linewidth=1.5,
                label="log-binned")
        # fit line
        kl = np.logspace(np.log10(fit_lo), np.log10(fit_hi), 80)
        ax.plot(kl, C_agg * kl ** (-gamma_agg), "--", color="black", linewidth=2,
                label=f"power-law fit: gamma={gamma_agg:.2f}, R²={r2_agg:.4f}")
        # reference: gamma = 3
        kl3 = np.logspace(np.log10(args.m + 1), np.log10(int(np.mean(k_maxes))), 80)
        # normalize at fit_lo
        if fit_lo > 0:
            C3 = C_agg * (fit_lo ** (-gamma_agg)) / (fit_lo ** (-3))
            ax.plot(kl3, C3 * kl3 ** (-3), ":", color="gray", linewidth=1.5,
                    label="reference: gamma=3 (asymptotic)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("k (degree)", fontsize=11)
        ax.set_ylabel("P(k)", fontsize=11)
        ax.set_title(f"BA scale-up P(k) — {lang.upper()} (N={args.N}, m={args.m})",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(plot_dir / f"ba_scaleup_pk_{lang}.png", dpi=300)
        plt.close(fig)
        print(f"  [OK] plot: {plot_dir / f'ba_scaleup_pk_{lang}.png'}")

    # Save run + summary CSVs
    runs_csv = out_dir / "metrics_runs.csv"
    pd.DataFrame(runs_rows).to_csv(runs_csv, index=False)
    print(f"\n[OK] runs: {runs_csv}")

    summary_csv = out_dir / "gamma_fit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[OK] summary: {summary_csv}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in summary_rows:
        print(f"  {r['lang']:3s}  N={r['N']:>6d}  gamma = {r['gamma_mean']:.3f} "
              f"+- {r['gamma_std']:.3f}  (95% CI [{r['gamma_lo']:.3f}, {r['gamma_hi']:.3f}])  "
              f"clustering={r['avg_clustering']:.5f}")

    print("\n[OK] BA scale-up hotovy.")


if __name__ == "__main__":
    main()
