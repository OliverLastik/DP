#!/usr/bin/env python3
"""
D) Conditional Weibull: K_i ~ DiscreteWeibull(alpha(tau_prev), beta(tau_prev))

Motivacia: independence test (sekcia 14.4.1) ukazal Cramer V = 0.06-0.26 medzi
K_i a tau_{i-1}. T.j. dat distribucia K zavisi od typu predoslej interpunkcie.
Tu fitujem Weibull osobitne pre kazdy tau_prev v {., ,, !, ?, ;, :} per kniha.

Vystupy:
  temporal_out/_cond_weibull/
    conditional_alpha_beta.csv   - per book x prev_punct: alpha, beta, n
    conditional_summary.csv      - per book: range_alpha, range_beta
    plots/
      conditional_alpha_per_book.png   - bar chart alpha per tau_prev x book
      conditional_beta_per_book.png    - bar chart beta per tau_prev x book
      pmf_overlay_<book_key>.png       - PMF for each tau_prev overlaid
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model_comparison import fit_weibull, PUNCT_SET, loglik_weibull  # type: ignore

PUNCT_SYMBOLS = [".", ",", "!", "?", ";", ":"]


def load_tokens(p: Path):
    return p.read_text(encoding="utf-8", errors="replace").split()


def compute_ipi_with_prev(tokens):
    """Vracia (K_i, tau_{i-1}) pary."""
    out = []
    word_count = 0
    last_punct = None
    for t in tokens:
        if t in PUNCT_SET:
            if last_punct is not None:
                out.append((word_count, last_punct))
            last_punct = t
            word_count = 0
        else:
            word_count += 1
    return out


def parse_meta(stem):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", "--tok-root", dest="token_dir",
                    default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_cond_weibull")
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

    cond_rows = []
    summary_rows = []
    book_data = []

    for b in books:
        tokens = load_tokens(b["path"])
        pairs = compute_ipi_with_prev(tokens)
        if len(pairs) < 100:
            continue

        # marginal fit (for reference)
        K_all = np.array([p[0] for p in pairs])
        marg_fit = fit_weibull(K_all)
        a_marg = marg_fit["params"]["alpha"]
        b_marg = marg_fit["params"]["beta"]

        # per tau_prev fit
        per_prev = defaultdict(list)
        for k, t in pairs:
            per_prev[t].append(k)

        per_prev_results = {}
        alphas = []
        betas = []
        for sym in PUNCT_SYMBOLS:
            arr = np.array(per_prev.get(sym, []))
            if len(arr) < 50:
                per_prev_results[sym] = None
                continue
            f = fit_weibull(arr)
            a = f["params"]["alpha"]
            be = f["params"]["beta"]
            ll = f["loglik"]
            per_prev_results[sym] = {
                "alpha": a, "beta": be, "n": len(arr), "loglik": ll,
                "mean": float(arr.mean()),
            }
            alphas.append(a)
            betas.append(be)

            cond_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "tau_prev": sym, "n": len(arr),
                "mean_K": float(arr.mean()),
                "alpha": a, "beta": be, "loglik": ll,
                "alpha_marginal": a_marg, "beta_marginal": b_marg,
                "alpha_ratio": a / a_marg if a_marg > 0 else float("nan"),
                "beta_ratio": be / b_marg if b_marg > 0 else float("nan"),
            })

        # joint conditional likelihood vs marginal likelihood (LR-test-like)
        # joint_ll = sum over symbols of per-symbol loglik
        # marg_ll = loglik using marginal (a_marg, b_marg) on full data
        joint_ll = sum(r["loglik"] for r in per_prev_results.values()
                       if r is not None)
        marg_ll = -float(loglik_weibull([a_marg, b_marg], K_all))
        # AIC: marginal has 2 params; conditional has 2 * n_symbols_used params
        n_sym_used = sum(1 for r in per_prev_results.values() if r is not None)
        aic_marg = 2 * 2 - 2 * marg_ll
        aic_cond = 2 * (2 * n_sym_used) - 2 * joint_ll
        delta_aic = aic_cond - aic_marg  # negative = conditional is better

        summary_rows.append({
            "lang": b["lang"], "author": b["author"], "period": b["period"],
            "year": b["year"], "title": b["title"],
            "alpha_marginal": a_marg, "beta_marginal": b_marg,
            "alpha_min": float(min(alphas)) if alphas else float("nan"),
            "alpha_max": float(max(alphas)) if alphas else float("nan"),
            "alpha_range": float(max(alphas) - min(alphas)) if alphas else float("nan"),
            "beta_min": float(min(betas)) if betas else float("nan"),
            "beta_max": float(max(betas)) if betas else float("nan"),
            "beta_range": float(max(betas) - min(betas)) if betas else float("nan"),
            "marg_loglik": marg_ll, "cond_loglik": joint_ll,
            "marg_aic": aic_marg, "cond_aic": aic_cond,
            "delta_aic": delta_aic,
            "n_sym_used": n_sym_used,
        })

        print(f"\n[{b['key']}] marg alpha={a_marg:.3f} beta={b_marg:.3f}")
        for sym in PUNCT_SYMBOLS:
            r = per_prev_results[sym]
            if r is None:
                print(f"  prev={sym}  n<50, skip")
            else:
                print(f"  prev={sym}  n={r['n']:>5d}  mean={r['mean']:5.2f}  "
                      f"alpha={r['alpha']:.3f}  beta={r['beta']:.3f}")
        print(f"  delta_AIC (cond - marg) = {delta_aic:+.1f}  "
              f"({'cond better' if delta_aic < 0 else 'marg better'})")

        book_data.append({
            "b": b, "per_prev": per_prev_results,
            "marg": {"alpha": a_marg, "beta": b_marg},
            "delta_aic": delta_aic,
        })

    # Save CSVs
    with (out_dir / "conditional_alpha_beta.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cond_rows[0].keys()))
        w.writeheader(); w.writerows(cond_rows)
    with (out_dir / "conditional_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\n[OK] CSVs saved")

    # ════════════════════════════════════════
    # PLOT: alpha per tau_prev x book
    # ════════════════════════════════════════
    book_keys = [d["b"]["key"] for d in book_data]
    n_books = len(book_data)
    x = np.arange(n_books)
    width = 0.13
    colors = {".": "#1f77b4", ",": "#ff7f0e", "!": "#2ca02c",
              "?": "#d62728", ";": "#9467bd", ":": "#8c564b"}

    for metric, ylabel, fname in [
        ("alpha", "Weibull shape α", "conditional_alpha_per_book.png"),
        ("beta",  "Weibull scale β", "conditional_beta_per_book.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for i, sym in enumerate(PUNCT_SYMBOLS):
            vals = []
            for d in book_data:
                r = d["per_prev"].get(sym)
                vals.append(r[metric] if r is not None else np.nan)
            ax.bar(x + (i - 2.5) * width, vals, width,
                   color=colors[sym], alpha=0.85, label=f"prev = '{sym}'")

        # marginal as black line
        marg_vals = [d["marg"][metric] for d in book_data]
        ax.plot(x, marg_vals, "k--", linewidth=1.5, marker="D",
                markersize=8, label="marginal (sekcia 10)")

        ax.set_xticks(x)
        ax.set_xticklabels(book_keys, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Conditional Weibull {metric} per τ_prev (vs marginal)",
                     fontsize=12)
        ax.legend(fontsize=9, ncol=4, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        if metric == "alpha":
            ax.axhline(1.0, color="red", linewidth=0.8, linestyle=":",
                       label="α=1 (memoryless)")
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=300)
        plt.close(fig)
    print(f"[OK] alpha/beta plots saved")

    # ── Summary printing ──
    print("\n" + "=" * 70)
    print("CONDITIONAL WEIBULL SUMMARY")
    print("=" * 70)
    print(f"  {'book':16s}  {'a_min':>7s}  {'a_max':>7s}  {'a_range':>8s}  "
          f"{'dAIC(cond-marg)':>16s}")
    for r in summary_rows:
        print(f"  {r['author'][:4]}_{r['year']:>11d}  "
              f"{r['alpha_min']:7.3f}  {r['alpha_max']:7.3f}  "
              f"{r['alpha_range']:8.3f}  {r['delta_aic']:>+16.1f}")
    print("\nInterpretacia delta_AIC:")
    print("  dAIC < -10  ->  conditional model JEDNOZNACNE lepsi")
    print("  -10 < dAIC < 0  ->  conditional mierne lepsi")
    print("  0 < dAIC < 10  ->  marginal model dostacuje (Occam)")
    print("  dAIC > 10  ->  marginal jednoznacne lepsi (cond overfit)")


if __name__ == "__main__":
    main()
