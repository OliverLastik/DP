#!/usr/bin/env python3
"""
A) Empiricka hazard funkcia h(k) per kniha + theoreticka Weibull hazard (alpha/beta)(k/beta)^(alpha-1).
   Vizualna diagnostika modelu - ukazuje ci IPI realne ma "fatigue dynamics".

E) Goh-Barabasi burstiness B = (sigma - mu)/(sigma + mu)
   + memory coefficient M = autokorelacia 1. radu na IPI sekvencii.
   Plot scatter (B, M) per book - kazda kniha ma jeden bod v 2D priestore.

Vystupy:
  temporal_out/_hazard_burst/
    hazard_per_book.csv
    burstiness_memory.csv
    plots/
      hazard_<book_key>.png  (9x)
      hazard_overlay_all.png  (vsetkych 9 v 3x3 grid)
      burstiness_memory_scatter.png
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model_comparison import fit_weibull, PUNCT_SET  # type: ignore


PERIOD_COLORS = {"early": "#1f77b4", "middle": "#ff7f0e", "late": "#2ca02c"}
AUTHOR_MARKERS = {"dickens": "o", "fontane": "s", "couperus": "^"}


def load_tokens(p: Path):
    return p.read_text(encoding="utf-8", errors="replace").split()


def compute_ipi(tokens):
    out, w, seen = [], 0, False
    for t in tokens:
        if t in PUNCT_SET:
            if seen:
                out.append(w)
            seen = True
            w = 0
        else:
            w += 1
    return np.array(out, dtype=np.int64)


def empirical_hazard(K, k_max=None):
    """h(k) = P(K=k) / P(K >= k) = count(K=k) / count(K >= k)."""
    if k_max is None:
        k_max = int(np.percentile(K, 99))
    ks = np.arange(0, k_max + 1)
    hs = []
    n = len(K)
    for k in ks:
        n_ge = int(np.sum(K >= k))
        n_eq = int(np.sum(K == k))
        if n_ge == 0:
            hs.append(np.nan)
        else:
            hs.append(n_eq / n_ge)
    return ks, np.array(hs)


def theoretical_weibull_hazard(k, alpha, beta):
    """h_W(k) = (alpha/beta) * (k/beta)^(alpha-1) — continuous-limit hazard.

    Diskretna verzia presne: h_d(k) = 1 - exp(-((k+1)^alpha - k^alpha)/beta^alpha).
    Pouzivam diskretnu verziu pre presnost.
    """
    k = np.asarray(k, dtype=float)
    # P(K >= k) = exp(-(k/beta)^alpha)
    # P(K = k) = exp(-(k/beta)^alpha) - exp(-((k+1)/beta)^alpha)
    # h(k) = P(K=k)/P(K>=k) = 1 - exp(-((k+1)/beta)^alpha + (k/beta)^alpha)
    delta = ((k + 1) / beta) ** alpha - (k / beta) ** alpha
    return 1 - np.exp(-delta)


def burstiness(K):
    """Goh-Barabasi B = (sigma - mu) / (sigma + mu). B in [-1, 1]."""
    mu = float(K.mean())
    sigma = float(K.std())
    if mu + sigma == 0:
        return float("nan")
    return (sigma - mu) / (sigma + mu)


def memory_coefficient(K):
    """
    M = (1/(N-1)) sum_i (K_i - mu_1)(K_{i+1} - mu_2) / (sigma_1 * sigma_2)
    where mu_1, sigma_1 are mean/std of K[0:N-1] and mu_2, sigma_2 of K[1:N].
    M in [-1, 1]: 0 = no autocorrelation, >0 = positive memory.
    """
    if len(K) < 3:
        return float("nan")
    a = K[:-1].astype(float)
    b = K[1:].astype(float)
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.std(), b.std()
    if s1 == 0 or s2 == 0:
        return float("nan")
    return float(((a - mu1) * (b - mu2)).mean() / (s1 * s2))


def parse_meta(stem):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", "--tok-root", dest="token_dir",
                    default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_hazard_burst")
    args = ap.parse_args()

    tok_root = Path(args.token_dir)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    books = []
    for sub in sorted(tok_root.iterdir()):
        if not sub.is_dir() or "validation" in sub.name:
            continue
        for fp in sorted(sub.glob("*.txt")):
            meta = parse_meta(fp.stem)
            if not meta:
                continue
            lang, author, period, year, title = meta
            books.append({
                "path": fp, "lang": lang, "author": author,
                "period": period, "year": year, "title": title,
                "key": f"{author[:4]}_{year}",
                "label": f"{author.capitalize()} {year}",
            })

    print(f"[INFO] {len(books)} books")

    hazard_rows = []
    burst_rows = []
    book_data = []  # cache for per-book plots

    for b in books:
        tokens = load_tokens(b["path"])
        K = compute_ipi(tokens)
        if len(K) < 50:
            continue

        # ── A: hazard ──
        # fit Weibull
        wf = fit_weibull(K)
        alpha = wf["params"]["alpha"]
        beta = wf["params"]["beta"]

        ks, h_emp = empirical_hazard(K, k_max=int(np.percentile(K, 99)))
        h_theo = theoretical_weibull_hazard(ks, alpha, beta)

        # ── E: burstiness + memory ──
        B = burstiness(K)
        M = memory_coefficient(K)

        print(f"[{b['key']}] alpha={alpha:.3f} beta={beta:.3f}  "
              f"B={B:.3f}  M={M:.3f}  n_K={len(K)}")

        for k, he, ht in zip(ks, h_emp, h_theo):
            hazard_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "k": int(k), "hazard_empirical": float(he),
                "hazard_weibull": float(ht),
            })

        burst_rows.append({
            "lang": b["lang"], "author": b["author"], "period": b["period"],
            "year": b["year"], "title": b["title"],
            "n_ipi": len(K), "mean_K": float(K.mean()), "std_K": float(K.std()),
            "burstiness_B": B, "memory_M": M,
            "weibull_alpha": alpha, "weibull_beta": beta,
        })

        book_data.append({
            "b": b, "ks": ks, "h_emp": h_emp, "h_theo": h_theo,
            "alpha": alpha, "beta": beta, "B": B, "M": M, "K": K,
        })

    # Save CSVs
    with (out_dir / "hazard_per_book.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hazard_rows[0].keys()))
        w.writeheader(); w.writerows(hazard_rows)
    with (out_dir / "burstiness_memory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(burst_rows[0].keys()))
        w.writeheader(); w.writerows(burst_rows)
    print(f"\n[OK] CSVs saved to {out_dir}")

    # ════════════════════════════════════════
    # PLOT A.1 — per book hazard
    # ════════════════════════════════════════
    for d in book_data:
        b = d["b"]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        mask = ~np.isnan(d["h_emp"]) & (d["h_emp"] > 0)
        ax.plot(d["ks"][mask], d["h_emp"][mask], "o", color="#1f77b4",
                markersize=5, alpha=0.7, label="empirical h(k)")
        ax.plot(d["ks"], d["h_theo"], "-", color="#d62728", linewidth=1.8,
                label=f"Weibull h(k) (α={d['alpha']:.2f}, β={d['beta']:.2f})")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("k (words since last punctuation)", fontsize=11)
        ax.set_ylabel("hazard h(k) = P(K=k | K ≥ k)", fontsize=11)
        ax.set_title(f"Hazard function — {b['label']} ({b['lang']})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"hazard_{b['key']}.png", dpi=300)
        plt.close(fig)

    # ════════════════════════════════════════
    # PLOT A.2 — 3x3 grid overlay (all books)
    # ════════════════════════════════════════
    n = len(book_data)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.3 * rows),
                              sharex=False, sharey=False)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    # sort: dickens, fontane, couperus, then by year
    order = ["dickens", "fontane", "couperus"]
    book_data_sorted = sorted(book_data,
                              key=lambda d: (order.index(d["b"]["author"]),
                                             d["b"]["year"]))

    for ax, d in zip(axes, book_data_sorted):
        b = d["b"]
        mask = ~np.isnan(d["h_emp"]) & (d["h_emp"] > 0)
        color = PERIOD_COLORS[b["period"]]
        ax.plot(d["ks"][mask], d["h_emp"][mask], "o", color=color,
                markersize=4, alpha=0.7, label="empirical")
        ax.plot(d["ks"], d["h_theo"], "-", color="black", linewidth=1.5,
                label=f"Weibull α={d['alpha']:.2f}")
        ax.set_title(f"{b['label']}", fontsize=10)
        ax.set_xlabel("k", fontsize=9)
        ax.set_ylabel("h(k)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    for ax in axes[len(book_data_sorted):]:
        ax.axis("off")
    fig.suptitle("Empirical vs Weibull hazard h(k) — všetkých 9 kníh",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(plot_dir / "hazard_overlay_all.png", dpi=300)
    plt.close(fig)
    print(f"[OK] hazard plots saved")

    # ════════════════════════════════════════
    # PLOT E — Goh-Barabasi (B, M) scatter
    # ════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for d in book_data:
        b = d["b"]
        marker = AUTHOR_MARKERS[b["author"]]
        color = PERIOD_COLORS[b["period"]]
        ax.scatter(d["B"], d["M"], c=color, marker=marker, s=140,
                   alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.annotate(f"{b['key']}", (d["B"], d["M"]),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=9)

    # quadrant lines
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    # reference: Poisson process at origin (B=0, M=0)
    ax.scatter([0], [0], marker="*", c="red", s=200, zorder=5,
               label="Poisson reference (B=0, M=0)")

    # legend for authors
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=11, label="Dickens"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=11, label="Fontane"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=11, label="Couperus"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markersize=14, label="Poisson"),
    ]
    period_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PERIOD_COLORS["early"],
               markersize=11, label="Early"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PERIOD_COLORS["middle"],
               markersize=11, label="Middle"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PERIOD_COLORS["late"],
               markersize=11, label="Late"),
    ]
    leg1 = ax.legend(handles=handles, fontsize=9, loc="upper left", title="Autor")
    ax.add_artist(leg1)
    ax.legend(handles=period_handles, fontsize=9, loc="lower right", title="Obdobie")

    ax.set_xlabel("Burstiness B = (σ−μ)/(σ+μ)", fontsize=11)
    ax.set_ylabel("Memory M = autokorelácia 1. rádu IPI", fontsize=11)
    ax.set_title("Goh–Barabási (B, M) priestor pre 9 temporálnych kníh",
                 fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "burstiness_memory_scatter.png", dpi=300)
    plt.close(fig)
    print(f"[OK] burstiness scatter saved")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("BURSTINESS / MEMORY SUMMARY")
    print("=" * 70)
    print(f"  {'book':16s}  {'B':>7s}  {'M':>7s}  {'alpha':>7s}  {'beta':>7s}")
    for r in burst_rows:
        print(f"  {r['author'][:4]}_{r['year']:>11d}  "
              f"{r['burstiness_B']:7.3f}  {r['memory_M']:7.3f}  "
              f"{r['weibull_alpha']:7.3f}  {r['weibull_beta']:7.3f}")
    print("\n  Note: B<0 = sub-Poisson (regular), B≈0 = Poisson, B>0 = bursty")
    print("        M>0 = positive autocorrelation, M<0 = anti-correlation")


if __name__ == "__main__":
    main()
