#!/usr/bin/env python3
"""
Temporalna analyza: rovnaky autor, rozne casove obdobia.

Porovna interpunkcnu distribuciu napriec knihami jedneho autora
z roznych obdobi jeho kariery.

Analyzy:
  1. Rank-frequency distribúcia interpunkcie (+ ZM fit)
  2. Markov transition matica P(to|from)
  3. Jensen-Shannon divergencia medzi obdobiami
  4. Sietove metriky (clustering, degree, density)
  5. Recurrence time / burstiness (CV)
  6. Inter-punctuation intervals
"""
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit, minimize
from scipy.spatial.distance import jensenshannon
from scipy.stats import weibull_min, kstest


PUNCT_SET = {".", ",", "!", "?", ";", ":"}
PUNCT_LIST = [".", ",", ";", ":", "?", "!"]

PERIOD_COLORS = {"early": "#1f77b4", "middle": "#ff7f0e", "late": "#2ca02c"}
PERIOD_LABELS = {"early": "Early", "middle": "Middle", "late": "Late"}


# ── Helpers ──

def load_tokens(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="replace").split()


def zipf_mandelbrot(R, C, alpha, c):
    return C * (R + c) ** (-alpha)


def build_graph(tokens):
    G = nx.Graph()
    for i in range(len(tokens) - 1):
        u, v = tokens[i], tokens[i + 1]
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def compute_markov_matrix(tokens, symbols=PUNCT_LIST):
    """Compute P(to|from) for punctuation transitions."""
    # Extract punct-only sequence
    punct_seq = [t for t in tokens if t in PUNCT_SET]

    trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(punct_seq) - 1):
        trans[punct_seq[i]][punct_seq[i + 1]] += 1

    # Normalize to probabilities
    mat = np.zeros((len(symbols), len(symbols)))
    for i, fr in enumerate(symbols):
        total = sum(trans[fr].values())
        if total > 0:
            for j, to in enumerate(symbols):
                mat[i, j] = trans[fr][to] / total

    return mat


def compute_js_divergence(mat1, mat2):
    """Row-wise JS divergence between two Markov matrices."""
    divs = []
    for i in range(mat1.shape[0]):
        p = mat1[i]
        q = mat2[i]
        if p.sum() > 0 and q.sum() > 0:
            divs.append(jensenshannon(p, q) ** 2)  # JSD squared
    return np.mean(divs) if divs else float("nan")


def compute_recurrence_cv(tokens):
    """CV of recurrence times for each punct token."""
    positions = defaultdict(list)
    for i, t in enumerate(tokens):
        if t in PUNCT_SET:
            positions[t].append(i)

    results = {}
    for tok, pos in positions.items():
        if len(pos) < 10:
            continue
        intervals = np.diff(pos)
        mu = intervals.mean()
        if mu > 0:
            results[tok] = {
                "count": len(pos),
                "mean_interval": float(mu),
                "cv": float(intervals.std() / mu),
            }
    return results


def compute_inter_punct_intervals(tokens):
    """Pocet slov medzi po sebe iducimi interpunkcnymi znackami."""
    intervals = []
    word_count = 0
    seen_first_punct = False

    for t in tokens:
        if t in PUNCT_SET:
            if seen_first_punct:
                intervals.append(word_count)
            seen_first_punct = True
            word_count = 0
        else:
            word_count += 1

    return np.array(intervals)


def fit_weibull_discrete(data):
    """
    Diskretny Weibull fit cez MLE.
    P(X=k) = exp(-(k/beta)^alpha) - exp(-((k+1)/beta)^alpha),  k = 0, 1, 2, ...
    (Stanisz et al. 2014)
    """
    data = np.asarray(data, dtype=float)
    data = data[data >= 0]
    if len(data) < 20:
        return {"alpha": float("nan"), "beta": float("nan"), "loglik": float("nan"), "n": len(data)}

    def neg_log_lik(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return 1e18
        k = data
        try:
            p = np.exp(-(k / beta) ** alpha) - np.exp(-((k + 1) / beta) ** alpha)
        except (OverflowError, FloatingPointError):
            return 1e18
        p = np.clip(p, 1e-300, None)
        return -np.sum(np.log(p))

    mean = data.mean()
    x0 = [1.0, max(mean, 1.0)]
    try:
        res = minimize(neg_log_lik, x0, method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
        alpha, beta = res.x
        return {"alpha": float(alpha), "beta": float(beta),
                "loglik": float(-res.fun), "n": len(data)}
    except Exception:
        return {"alpha": float("nan"), "beta": float("nan"), "loglik": float("nan"), "n": len(data)}


def weibull_discrete_pmf(k, alpha, beta):
    k = np.asarray(k, dtype=float)
    return np.exp(-(k / beta) ** alpha) - np.exp(-((k + 1) / beta) ** alpha)


def fit_weibull_continuous(data):
    """Spojity Weibull fit cez scipy (weibull_min, loc=0)."""
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    if len(data) < 20:
        return {"c": float("nan"), "scale": float("nan"), "ks_stat": float("nan"),
                "ks_p": float("nan"), "n": len(data)}
    try:
        c, loc, scale = weibull_min.fit(data, floc=0)
        ks_stat, ks_p = kstest(data, "weibull_min", args=(c, loc, scale))
        return {"c": float(c), "scale": float(scale),
                "ks_stat": float(ks_stat), "ks_p": float(ks_p), "n": len(data)}
    except Exception:
        return {"c": float("nan"), "scale": float("nan"),
                "ks_stat": float("nan"), "ks_p": float("nan"), "n": len(data)}


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out")
    args = ap.parse_args()

    token_dir = Path(args.token_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find token files
    files = sorted(token_dir.glob("*.txt"))
    if not files:
        print(f"[ERR] No token files in {token_dir}")
        return

    # Parse metadata from filenames. Two supported schemes:
    #   lang_author_period_year_title  (new, 5 parts)
    #   author_period_year_title       (legacy, 4 parts)
    books = []
    for fp in files:
        parts = fp.stem.split("_")
        if len(parts) >= 5 and parts[0] in {"en", "de", "nl", "da", "sv"}:
            lang = parts[0]
            author = parts[1]
            period = parts[2]
            year = parts[3]
            title = "_".join(parts[4:])
        elif len(parts) >= 4:
            lang = "??"
            author = parts[0]
            period = parts[1]
            year = parts[2]
            title = "_".join(parts[3:])
        else:
            continue
        try:
            year_int = int(year)
        except ValueError:
            continue
        books.append({
            "path": fp,
            "lang": lang,
            "author": author,
            "period": period,
            "year": year_int,
            "title": title,
            "label": f"{period.capitalize()} ({year})",
        })

    if not books:
        print("[ERR] Cannot parse book metadata from filenames")
        return

    books.sort(key=lambda b: b["year"])
    print(f"[INFO] Found {len(books)} books:")
    for b in books:
        print(f"  {b['label']}: {b['title']}")

    # ── Analyze each book ──
    all_data = []

    for book in books:
        tokens = load_tokens(book["path"])
        print(f"\n{'='*50}")
        print(f"  {book['label']}: {book['title']} ({len(tokens)} tokens)")
        print(f"{'='*50}")

        # Frequency counts
        freq = Counter(tokens)
        total = len(tokens)

        # Punct frequencies
        punct_freq = {p: freq.get(p, 0) for p in PUNCT_LIST}
        punct_total = sum(punct_freq.values())
        print(f"  Punct: {punct_total} ({punct_total/total*100:.1f}%)")
        for p in PUNCT_LIST:
            if punct_freq[p] > 0:
                print(f"    {p}: {punct_freq[p]} ({punct_freq[p]/total*100:.2f}%)")

        # Rank-frequency
        sorted_freq = sorted(freq.values(), reverse=True)
        ranks = np.arange(1, len(sorted_freq) + 1, dtype=float)
        freqs = np.array(sorted_freq, dtype=float)
        probs = freqs / freqs.sum()

        # ZM fit on rank 20-5000
        mask = (ranks >= 20) & (ranks <= min(5000, len(ranks)))
        if mask.sum() > 10:
            try:
                popt, _ = curve_fit(zipf_mandelbrot, ranks[mask], probs[mask],
                                    p0=[0.1, 1.0, 5.0], maxfev=10000)
                zm_C, zm_alpha, zm_c = popt
            except Exception:
                zm_C, zm_alpha, zm_c = float("nan"), float("nan"), float("nan")
        else:
            zm_C, zm_alpha, zm_c = float("nan"), float("nan"), float("nan")

        print(f"  ZM fit: alpha={zm_alpha:.3f}, c={zm_c:.2f}")

        # Network metrics
        G = build_graph(tokens)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        avg_deg = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        avg_cl = nx.average_clustering(G) if n_nodes > 1 else 0
        density = nx.density(G)

        print(f"  Network: {n_nodes} nodes, {n_edges} edges, <k>={avg_deg:.2f}, C={avg_cl:.4f}")

        # Markov matrix
        markov_mat = compute_markov_matrix(tokens)

        # Recurrence CV
        cv_data = compute_recurrence_cv(tokens)

        # Inter-punctuation intervals + Weibull (Stanisz et al.)
        ipi = compute_inter_punct_intervals(tokens)
        weib_disc = fit_weibull_discrete(ipi)
        weib_cont = fit_weibull_continuous(ipi)
        print(f"  IPI: n={len(ipi)}, mean={ipi.mean():.2f}, median={np.median(ipi):.1f}")
        print(f"  Weibull (discrete MLE): alpha={weib_disc['alpha']:.3f}, beta={weib_disc['beta']:.3f}")
        print(f"  Weibull (continuous):  c={weib_cont['c']:.3f}, scale={weib_cont['scale']:.3f}, "
              f"KS={weib_cont['ks_stat']:.3f}")

        all_data.append({
            "book": book,
            "tokens": tokens,
            "freq": freq,
            "ranks": ranks,
            "probs": probs,
            "zm_alpha": zm_alpha,
            "zm_c": zm_c,
            "zm_C": zm_C,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "avg_degree": avg_deg,
            "avg_clustering": avg_cl,
            "density": density,
            "markov_mat": markov_mat,
            "cv_data": cv_data,
            "ipi": ipi,
            "weib_disc": weib_disc,
            "weib_cont": weib_cont,
            "punct_freq": punct_freq,
            "total_tokens": total,
        })

    # ══════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════

    author_name = all_data[0]["book"]["author"].capitalize()
    lang_code = all_data[0]["book"]["lang"]
    author_label = f"{author_name} ({lang_code})"

    # 1. Rank-frequency overlay
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for d in all_data:
        b = d["book"]
        color = PERIOD_COLORS[b["period"]]
        ax.plot(d["ranks"], d["probs"], "-", color=color, linewidth=1.5, alpha=0.8,
                label=f"{b['label']}")

        # annotate punct positions
        for p in PUNCT_LIST:
            if p in d["freq"]:
                rank = sorted(d["freq"].values(), reverse=True).index(d["freq"][p]) + 1
                prob = d["freq"][p] / d["total_tokens"]
                ax.plot(rank, prob, "D", color=color, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rank $R$", fontsize=12)
    ax.set_ylabel("$P(R)$", fontsize=12)
    ax.set_title(f"Rank-frequency: {author_label} across career", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_rankfreq.png", dpi=300)
    plt.close(fig)
    print("\n[OK] temporal_rankfreq.png")

    # 2. Punct proportion bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PUNCT_LIST))
    width = 0.25
    for i, d in enumerate(all_data):
        b = d["book"]
        vals = [d["punct_freq"][p] / d["total_tokens"] * 100 for p in PUNCT_LIST]
        ax.bar(x + (i - 1) * width, vals, width,
               color=PERIOD_COLORS[b["period"]], alpha=0.8,
               label=b["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(PUNCT_LIST, fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency (%)", fontsize=12)
    ax.set_title(f"Punctuation usage: {author_label} across career", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_punct_freq.png", dpi=300)
    plt.close(fig)
    print("[OK] temporal_punct_freq.png")

    # 3. Markov heatmaps side-by-side
    fig, axes = plt.subplots(1, len(all_data), figsize=(5.5 * len(all_data), 5))
    if len(all_data) == 1:
        axes = [axes]

    for ax, d in zip(axes, all_data):
        b = d["book"]
        mat = d["markov_mat"]
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=0.6, cmap="YlOrRd")
        ax.set_xticks(range(len(PUNCT_LIST)))
        ax.set_xticklabels(PUNCT_LIST, fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(PUNCT_LIST)))
        ax.set_yticklabels(PUNCT_LIST, fontsize=12, fontweight="bold")
        ax.set_title(f"{b['label']}", fontsize=12, fontweight="bold")

        for i in range(len(PUNCT_LIST)):
            for j in range(len(PUNCT_LIST)):
                val = mat[i, j]
                if val >= 0.02:
                    color = "white" if val > 0.35 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

    fig.colorbar(im, ax=axes, label="P(to|from)", shrink=0.7, pad=0.02)
    fig.suptitle(f"Markov transitions: {author_label} across career", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_markov_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] temporal_markov_heatmaps.png")

    # 4. JS divergence between periods
    n = len(all_data)
    js_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                js_matrix[i, j] = compute_js_divergence(
                    all_data[i]["markov_mat"], all_data[j]["markov_mat"]
                )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(js_matrix, cmap="Blues", vmin=0)
    labels = [d["book"]["label"] for d in all_data]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{js_matrix[i,j]:.4f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, label="JS divergence", shrink=0.8)
    ax.set_title("JS divergence of Markov matrices", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_js_divergence.png", dpi=300)
    plt.close(fig)
    print("[OK] temporal_js_divergence.png")

    # 5. Network metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    periods = [d["book"]["label"] for d in all_data]
    colors = [PERIOD_COLORS[d["book"]["period"]] for d in all_data]

    for ax, (metric, ylabel, title) in zip(axes, [
        ("avg_clustering", "$\\langle C \\rangle$", "Clustering"),
        ("avg_degree", "$\\langle k \\rangle$", "Avg degree"),
        ("density", "density", "Density"),
    ]):
        vals = [d[metric] for d in all_data]
        bars = ax.bar(range(len(all_data)), vals, color=colors, alpha=0.8)
        ax.set_xticks(range(len(all_data)))
        ax.set_xticklabels(periods, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Network metrics: {author_label} across career", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_network_metrics.png", dpi=300)
    plt.close(fig)
    print("[OK] temporal_network_metrics.png")

    # 6. Recurrence CV comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    for d in all_data:
        b = d["book"]
        color = PERIOD_COLORS[b["period"]]
        for tok, info in d["cv_data"].items():
            ax.scatter(info["count"], info["cv"],
                       c=color, s=60 if tok in PUNCT_SET else 15,
                       marker="*" if tok in PUNCT_SET else "o",
                       alpha=0.8, zorder=5 if tok in PUNCT_SET else 3)
            if tok in PUNCT_SET:
                ax.annotate(tok, (info["count"], info["cv"]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=8, color=color, fontweight="bold")

        # dummy for legend
        ax.scatter([], [], c=color, s=60, marker="*", label=b["label"])

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Token frequency", fontsize=12)
    ax.set_ylabel("CV of recurrence time", fontsize=12)
    ax.set_title(f"Burstiness: {author_label} across career", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_recurrence_cv.png", dpi=300)
    plt.close(fig)
    print("[OK] temporal_recurrence_cv.png")

    # 7. Inter-punctuation interval histograms + Weibull fits
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in all_data:
        b = d["book"]
        color = PERIOD_COLORS[b["period"]]
        ipi = d["ipi"]
        if len(ipi) == 0:
            continue
        bins = np.arange(0, min(50, int(ipi.max())) + 1) - 0.5
        counts, edges = np.histogram(ipi, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, "o", color=color, markersize=4, alpha=0.7,
                label=f"{b['label']} (mean={ipi.mean():.1f})")

        wd = d["weib_disc"]
        if not np.isnan(wd["alpha"]):
            k_grid = np.arange(0, int(centers.max()) + 1)
            pmf = weibull_discrete_pmf(k_grid, wd["alpha"], wd["beta"])
            ax.plot(k_grid, pmf, "-", color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Words between punctuation marks $k$", fontsize=12)
    ax.set_ylabel("$P(k)$", fontsize=12)
    ax.set_title("Inter-punctuation intervals + discrete Weibull (Stanisz et al.)",
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_ipi_histogram.png", dpi=300)
    plt.close(fig)
    print("[OK] temporal_ipi_histogram.png")

    # 7b. Weibull on log-y scale (highlights tail)
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in all_data:
        b = d["book"]
        color = PERIOD_COLORS[b["period"]]
        ipi = d["ipi"]
        if len(ipi) == 0:
            continue
        bins = np.arange(0, min(80, int(ipi.max())) + 1) - 0.5
        counts, edges = np.histogram(ipi, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        ax.plot(centers[mask], counts[mask], "o", color=color,
                markersize=4, alpha=0.7,
                label=f"{b['label']}")

        wd = d["weib_disc"]
        if not np.isnan(wd["alpha"]):
            k_grid = np.arange(0, int(centers.max()) + 1)
            pmf = weibull_discrete_pmf(k_grid, wd["alpha"], wd["beta"])
            ax.plot(k_grid, pmf, "-", color=color, linewidth=1.5, alpha=0.9,
                    label=f"  fit: $\\alpha$={wd['alpha']:.2f}, $\\beta$={wd['beta']:.2f}")

    ax.set_yscale("log")
    ax.set_xlabel("Words between punctuation marks $k$", fontsize=12)
    ax.set_ylabel("$P(k)$ (log)", fontsize=12)
    ax.set_title("IPI tail + discrete Weibull fit", fontsize=12)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_ipi_weibull_logy.png", dpi=300)
    plt.close(fig)
    print("[OK] temporal_ipi_weibull_logy.png")

    # ── Summary CSV ──
    summary_rows = []
    for d in all_data:
        b = d["book"]
        row = {
            "lang": b["lang"],
            "author": b["author"],
            "period": b["period"],
            "year": b["year"],
            "title": b["title"],
            "n_tokens": d["total_tokens"],
            "n_types": len(d["freq"]),
            "zm_alpha": d["zm_alpha"],
            "zm_c": d["zm_c"],
            "n_nodes": d["n_nodes"],
            "n_edges": d["n_edges"],
            "avg_degree": d["avg_degree"],
            "avg_clustering": d["avg_clustering"],
            "density": d["density"],
            "ipi_mean": float(d["ipi"].mean()) if len(d["ipi"]) > 0 else float("nan"),
            "ipi_median": float(np.median(d["ipi"])) if len(d["ipi"]) > 0 else float("nan"),
            "weibull_disc_alpha": d["weib_disc"]["alpha"],
            "weibull_disc_beta": d["weib_disc"]["beta"],
            "weibull_disc_loglik": d["weib_disc"]["loglik"],
            "weibull_cont_c": d["weib_cont"]["c"],
            "weibull_cont_scale": d["weib_cont"]["scale"],
            "weibull_cont_ks_stat": d["weib_cont"]["ks_stat"],
            "weibull_cont_ks_p": d["weib_cont"]["ks_p"],
        }
        # punct percentages
        for p in PUNCT_LIST:
            pname = {".": "dot", ",": "comma", ";": "semicol", ":": "colon",
                     "?": "question", "!": "excl"}[p]
            row[f"pct_{pname}"] = d["punct_freq"][p] / d["total_tokens"] * 100

        summary_rows.append(row)

    csv_path = out_dir / "temporal_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[OK] CSV: {csv_path}")

    # JS divergence summary
    print("\nJS divergence between periods:")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"  {labels[i]} vs {labels[j]}: {js_matrix[i, j]:.4f}")

    print("\nWeibull parameters (discrete, Stanisz et al.):")
    for d in all_data:
        b = d["book"]
        w = d["weib_disc"]
        print(f"  {b['label']:20s}  alpha={w['alpha']:.3f}  beta={w['beta']:.3f}  "
              f"loglik={w['loglik']:.1f}")

    print("\n[OK] Temporalna analyza hotova.")


if __name__ == "__main__":
    main()
