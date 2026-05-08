#!/usr/bin/env python3
"""
Bootstrap CI na diskretne Weibull parametre (alpha, beta) pre kazdu knihu.

Pre kazdu knihu:
  1. spocitaj IPI
  2. N-krat resample IPI s navratom
  3. fituj diskretny Weibull (MLE, Nelder-Mead) na kazdy bootstrap sample
  4. uloz 2.5% / 50% / 97.5% percentily ako CI

Vystupom je CSV s CI a prekreslene drift grafy s error bars.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PUNCT_SET = {".", ",", "!", "?", ";", ":"}

AUTHOR_COLORS = {
    "dickens":  "#1f77b4",
    "fontane":  "#d62728",
    "couperus": "#2ca02c",
}
AUTHOR_MARKERS = {
    "dickens":  "o",
    "fontane":  "s",
    "couperus": "D",
}


def load_tokens(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="replace").split()


def compute_ipi(tokens):
    intervals = []
    w = 0
    seen = False
    for t in tokens:
        if t in PUNCT_SET:
            if seen:
                intervals.append(w)
            seen = True
            w = 0
        else:
            w += 1
    return np.array(intervals, dtype=np.int64)


def fit_weibull_discrete(data, x0=None):
    data = np.asarray(data, dtype=float)
    data = data[data >= 0]
    if len(data) < 20:
        return float("nan"), float("nan")

    def neg_log_lik(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return 1e18
        try:
            p = np.exp(-(data / beta) ** alpha) - np.exp(-((data + 1) / beta) ** alpha)
        except (OverflowError, FloatingPointError):
            return 1e18
        p = np.clip(p, 1e-300, None)
        return -np.sum(np.log(p))

    if x0 is None:
        x0 = [1.5, max(data.mean(), 1.0)]
    try:
        res = minimize(neg_log_lik, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})
        return float(res.x[0]), float(res.x[1])
    except Exception:
        return float("nan"), float("nan")


def bootstrap_weibull(ipi, n_boot, rng):
    n = len(ipi)
    # warm-start from point estimate
    alpha0, beta0 = fit_weibull_discrete(ipi)
    x0 = [alpha0, beta0] if not np.isnan(alpha0) else None

    alphas = np.empty(n_boot)
    betas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = ipi[idx]
        a, b = fit_weibull_discrete(sample, x0=x0)
        alphas[i] = a
        betas[i] = b
    return alpha0, beta0, alphas, betas


def parse_meta(stem: str):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl", "da", "sv"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_bootstrap")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok_root = Path(args.token_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    books = []
    for sub in sorted(tok_root.iterdir()):
        if not sub.is_dir():
            continue
        for fp in sorted(sub.glob("*.txt")):
            meta = parse_meta(fp.stem)
            if not meta:
                continue
            lang, author, period, year, title = meta
            books.append({
                "path": fp, "lang": lang, "author": author,
                "period": period, "year": year, "title": title,
            })

    print(f"[INFO] {len(books)} books, n_boot={args.n_boot}")

    rows = []
    for b in books:
        tokens = load_tokens(b["path"])
        ipi = compute_ipi(tokens)
        print(f"  {b['lang']}/{b['author']}/{b['period']} {b['year']}  "
              f"(n_ipi={len(ipi)}) ... ", end="", flush=True)

        alpha0, beta0, alphas, betas = bootstrap_weibull(ipi, args.n_boot, rng)
        alphas = alphas[np.isfinite(alphas)]
        betas = betas[np.isfinite(betas)]

        a_lo, a_med, a_hi = np.percentile(alphas, [2.5, 50, 97.5])
        b_lo, b_med, b_hi = np.percentile(betas, [2.5, 50, 97.5])

        print(f"alpha={alpha0:.3f} [{a_lo:.3f}, {a_hi:.3f}]  "
              f"beta={beta0:.3f} [{b_lo:.3f}, {b_hi:.3f}]")

        rows.append({
            "lang": b["lang"], "author": b["author"], "period": b["period"],
            "year": b["year"], "title": b["title"],
            "n_ipi": len(ipi),
            "alpha_point": alpha0, "alpha_lo": a_lo, "alpha_med": a_med, "alpha_hi": a_hi,
            "beta_point": beta0, "beta_lo": b_lo, "beta_med": b_med, "beta_hi": b_hi,
            "alpha_std": float(np.std(alphas)),
            "beta_std": float(np.std(betas)),
            "n_valid_boot": len(alphas),
        })

    # CSV
    csv_path = out / "weibull_bootstrap.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {csv_path}")

    # Group by author for plotting
    by_author = defaultdict(list)
    for r in rows:
        by_author[(r["lang"], r["author"])].append(r)
    for k in by_author:
        by_author[k].sort(key=lambda r: r["year"])

    # ── Plot: drift with CI ──
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for (lang, author), arr in sorted(by_author.items()):
        color = AUTHOR_COLORS.get(author, "gray")
        marker = AUTHOR_MARKERS.get(author, "o")
        label = f"{author.capitalize()} ({lang})"

        years = np.array([r["year"] for r in arr])
        alpha_p = np.array([r["alpha_point"] for r in arr])
        alpha_lo = np.array([r["alpha_lo"] for r in arr])
        alpha_hi = np.array([r["alpha_hi"] for r in arr])
        beta_p = np.array([r["beta_point"] for r in arr])
        beta_lo = np.array([r["beta_lo"] for r in arr])
        beta_hi = np.array([r["beta_hi"] for r in arr])

        axes[0].errorbar(years, alpha_p,
                         yerr=[alpha_p - alpha_lo, alpha_hi - alpha_p],
                         fmt=f"-{marker}", color=color, markersize=9,
                         linewidth=2, capsize=4, label=label)
        axes[1].errorbar(years, beta_p,
                         yerr=[beta_p - beta_lo, beta_hi - beta_p],
                         fmt=f"-{marker}", color=color, markersize=9,
                         linewidth=2, capsize=4, label=label)

    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Year", fontsize=11)
    axes[0].set_ylabel(r"Weibull shape $\alpha$", fontsize=11)
    axes[0].set_title(r"$\alpha$ drift with 95% CI", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Year", fontsize=11)
    axes[1].set_ylabel(r"Weibull scale $\beta$", fontsize=11)
    axes[1].set_title(r"$\beta$ drift with 95% CI", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Bootstrap 95% CI on discrete Weibull parameters "
                 f"(n_boot={args.n_boot})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "weibull_drift_with_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] weibull_drift_with_ci.png")

    # ── Check which trends survive CI overlap ──
    print("\n== CI overlap check (does early↔late CI overlap?) ==")
    for (lang, author), arr in sorted(by_author.items()):
        if len(arr) < 2:
            continue
        early = arr[0]
        late = arr[-1]
        a_overlap = not (early["alpha_hi"] < late["alpha_lo"]
                         or late["alpha_hi"] < early["alpha_lo"])
        b_overlap = not (early["beta_hi"] < late["beta_lo"]
                         or late["beta_hi"] < early["beta_lo"])
        print(f"  {author} ({lang}): "
              f"alpha_early=[{early['alpha_lo']:.3f},{early['alpha_hi']:.3f}] "
              f"alpha_late=[{late['alpha_lo']:.3f},{late['alpha_hi']:.3f}] "
              f"{'OVERLAP' if a_overlap else 'DISJOINT'}  |  "
              f"beta_early=[{early['beta_lo']:.3f},{early['beta_hi']:.3f}] "
              f"beta_late=[{late['beta_lo']:.3f},{late['beta_hi']:.3f}] "
              f"{'OVERLAP' if b_overlap else 'DISJOINT'}")

    print("\n[OK] Bootstrap hotovo.")


if __name__ == "__main__":
    main()
