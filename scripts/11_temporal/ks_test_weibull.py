#!/usr/bin/env python3
"""
C) Kolmogorov-Smirnov goodness-of-fit pre diskretny Weibull
   s bootstrap p-value (parametricky bootstrap).

Otazka: Je samotne pozorovane data (per kniha) konzistentne s diskretnym
Weibull? AIC vs alternativy je relativne kriterium, KS je absolutne.

Procedura (parametricky bootstrap, Stute et al. 1993):
  1. Fit Weibull na data X -> theta_hat = (alpha_hat, beta_hat)
  2. Spocitaj KS statistiku D_obs = max_k |F_emp(k) - F_W(k; theta_hat)|
  3. Bootstrap loop (B = 200 replicates):
     a. Generuj X_b ~ Weibull(theta_hat) rovnakej velkosti ako X
     b. Fit X_b -> theta_b
     c. Spocitaj D_b = max_k |F_emp_b(k) - F_W(k; theta_b)|
  4. p-value = (# {D_b >= D_obs}) / B

p < 0.05 -> data NIE konzistentne s Weibull
p >= 0.05 -> data je konzistentne s Weibull (Weibull nie je odmietnuty)

Vystupy:
  temporal_out/_ks_test/
    ks_results.csv      - per kniha: D_obs, p_value, alpha, beta, n
    plots/
      ks_pvalue_bars.png   - barplot p-value per kniha
      ks_diagnostic_<key>.png  - empirical CDF vs theoretical CDF + D_obs
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model_comparison import fit_weibull, PUNCT_SET  # type: ignore


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


def weibull_cdf(k, alpha, beta):
    """F(k) = 1 - exp(-((k+1)/beta)^alpha) for k = 0, 1, 2, ..."""
    k = np.asarray(k, dtype=float)
    return 1 - np.exp(-((k + 1) / beta) ** alpha)


def empirical_cdf(K, k_grid):
    """F_emp(k) = mean(K <= k) for k in k_grid."""
    K_sorted = np.sort(K)
    return np.searchsorted(K_sorted, k_grid, side="right") / len(K)


def ks_statistic(K, alpha, beta):
    """D = max_k |F_emp(k) - F_W(k; alpha, beta)|."""
    k_max = int(K.max())
    k_grid = np.arange(0, k_max + 1)
    F_emp = empirical_cdf(K, k_grid)
    F_th = weibull_cdf(k_grid, alpha, beta)
    return float(np.max(np.abs(F_emp - F_th)))


def sample_discrete_weibull(n, alpha, beta, rng):
    """Sample n IPI values from DiscreteWeibull(alpha, beta) via inverse CDF."""
    u = rng.random(n)
    # F(k) = 1 - exp(-((k+1)/beta)^alpha) = u
    # ((k+1)/beta)^alpha = -ln(1-u)
    # k+1 = beta * (-ln(1-u))^(1/alpha)
    k = beta * (-np.log(1 - u)) ** (1 / alpha) - 1
    return np.clip(np.floor(k + 1).astype(int), 0, None)


def parse_meta(stem):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", "--tok-root", dest="token_dir",
                    default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_ks_test")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
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

    print(f"[INFO] {len(books)} books, B={args.n_bootstrap} bootstrap reps")

    results = []
    for b in books:
        tokens = load_tokens(b["path"])
        K = compute_ipi(tokens)
        if len(K) < 100:
            continue

        # 1. Fit
        f = fit_weibull(K)
        alpha = f["params"]["alpha"]
        beta = f["params"]["beta"]
        D_obs = ks_statistic(K, alpha, beta)

        # 2. Bootstrap
        rng = np.random.default_rng(args.seed + hash(b["key"]) % 1000)
        D_boot = []
        n = len(K)
        for boot_i in range(args.n_bootstrap):
            X_b = sample_discrete_weibull(n, alpha, beta, rng)
            f_b = fit_weibull(X_b)
            a_b = f_b["params"]["alpha"]
            be_b = f_b["params"]["beta"]
            if not (np.isnan(a_b) or np.isnan(be_b)):
                D_b = ks_statistic(X_b, a_b, be_b)
                D_boot.append(D_b)

        D_boot = np.array(D_boot)
        p_value = float((D_boot >= D_obs).sum() / len(D_boot)) if len(D_boot) > 0 else float("nan")

        # also: 95th percentile of D_boot (rejection threshold under null)
        D_crit_95 = float(np.percentile(D_boot, 95)) if len(D_boot) > 0 else float("nan")

        print(f"  [{b['key']}]  n={n:>6d}  alpha={alpha:.3f} beta={beta:.3f}  "
              f"D_obs={D_obs:.4f}  D_crit_95={D_crit_95:.4f}  p={p_value:.3f}")

        results.append({
            "lang": b["lang"], "author": b["author"], "period": b["period"],
            "year": b["year"], "title": b["title"],
            "n_ipi": n, "alpha": alpha, "beta": beta,
            "D_obs": D_obs, "D_crit_95": D_crit_95,
            "p_value": p_value, "n_bootstrap": len(D_boot),
            "weibull_acceptable_at_5pct": bool(p_value >= 0.05),
        })

        # Diagnostic plot
        k_max = int(K.max())
        k_grid = np.arange(0, k_max + 1)
        F_emp = empirical_cdf(K, k_grid)
        F_th = weibull_cdf(k_grid, alpha, beta)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(k_grid, F_emp, "-", color="#1f77b4", linewidth=1.8,
                label="empirical CDF")
        ax.plot(k_grid, F_th, "--", color="#d62728", linewidth=1.8,
                label=f"Weibull CDF (α={alpha:.2f}, β={beta:.2f})")
        # mark D_obs
        diff = np.abs(F_emp - F_th)
        k_max_diff = int(np.argmax(diff))
        ax.plot([k_max_diff, k_max_diff], [F_emp[k_max_diff], F_th[k_max_diff]],
                "k-", linewidth=2, label=f"D_obs={D_obs:.4f}")
        ax.scatter([k_max_diff], [F_emp[k_max_diff]], color="black", zorder=5)
        ax.scatter([k_max_diff], [F_th[k_max_diff]], color="black", zorder=5)

        ax.set_xlabel("k (words between punctuation)", fontsize=11)
        ax.set_ylabel("CDF F(k)", fontsize=11)
        ax.set_title(f"KS diagnostika — {b['label']} (p-value={p_value:.3f}, "
                     f"{'OK' if p_value >= 0.05 else 'REJECTED'})",
                     fontsize=11)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3)
        # zoom on the area where D_obs is
        ax.set_xlim(0, min(k_max, 4 * beta))
        fig.tight_layout()
        fig.savefig(plot_dir / f"ks_diagnostic_{b['key']}.png", dpi=300)
        plt.close(fig)

    # Save CSV
    with (out_dir / "ks_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\n[OK] ks_results.csv")

    # P-value barplot
    book_keys = [r["author"][:4] + "_" + str(r["year"]) for r in results]
    p_vals = [r["p_value"] for r in results]
    colors = ["#2ca02c" if p >= 0.05 else "#d62728" for p in p_vals]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(np.arange(len(book_keys)), p_vals, color=colors, alpha=0.85)
    ax.axhline(0.05, color="black", linewidth=1.2, linestyle="--",
               label="α = 0.05 hranica")
    ax.set_xticks(np.arange(len(book_keys)))
    ax.set_xticklabels(book_keys, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("KS bootstrap p-value", fontsize=11)
    ax.set_title("Goodness-of-fit Weibull pre IPI distribuciu (B=200 bootstrap)",
                 fontsize=12)
    for bar, p in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{p:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "ks_pvalue_bars.png", dpi=300)
    plt.close(fig)
    print(f"[OK] ks_pvalue_bars.png")

    # Summary
    print("\n" + "=" * 70)
    print("KS GoF SUMMARY")
    print("=" * 70)
    n_acc = sum(1 for r in results if r["weibull_acceptable_at_5pct"])
    print(f"  Weibull acceptable (p >= 0.05): {n_acc}/{len(results)}")
    print(f"\n  {'book':16s}  {'D_obs':>7s}  {'D_crit95':>9s}  {'p':>7s}  {'verdict':>10s}")
    for r in results:
        verdict = "ACCEPT" if r["p_value"] >= 0.05 else "REJECT"
        print(f"  {r['author'][:4]}_{r['year']:>11d}  {r['D_obs']:7.4f}  "
              f"{r['D_crit_95']:9.4f}  {r['p_value']:7.3f}  {verdict:>10s}")


if __name__ == "__main__":
    main()
