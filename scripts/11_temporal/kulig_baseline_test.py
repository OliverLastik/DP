#!/usr/bin/env python3
"""
Kulig-style IPI test pre nulove modely (Shuffled + Bernoulli-punct).

Otazka ktoru zodpoveda:
  "Ci Kulig/Stanisz Weibull shape (alpha ~ 1.5) moze vyprodukovat
   aj proces bez pamate. Ak ano, nasledok je ze Weibull alpha > 1
   je artefakt. Ak nie, je to skutocna signatura jazyka."

Postup pre kazdu z 9 temporalnych knih:
  1. Real IPI -> fit 4 distribucii -> AIC winner
  2. Shuffled null: 100x random permutacia tokenov -> IPI -> fit
     -> mean/std Weibull alpha, beta, AIC winner distribution
  3. Bernoulli null: 100x sample punct pozicie uniformne s pravdepodobnostou
     p = n_punct / n_tokens -> IPI -> fit
  4. Vystup CSV + porovnavaci plot

Pouzije sa priamo fit_weibull / fit_nbinom / fit_lognorm / fit_geometric
z model_comparison.py (re-import).
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse fit functions
sys.path.insert(0, str(Path(__file__).parent))
from model_comparison import (  # type: ignore
    fit_weibull, fit_nbinom, fit_lognorm, fit_geometric,
    aic, PUNCT_SET,
)


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


def shuffled_tokens(tokens, rng):
    """Random permutacia vsetkych tokenov (nici poradie aj pozicie punct)."""
    shuf = tokens[:]
    rng.shuffle(shuf)
    return shuf


def bernoulli_punct_tokens(tokens, rng):
    """
    Zachova slova v povodnom poradi, nahodne umiestni punct s pravdepodobnostou
    p = n_punct / n_tokens. Distribucia typov punct sa zachova proporcionalne
    aby bola porovnatelna s real (inak by to bol dalsi nulovy model navyse).

    Vysledkom je rovnaky pocet slov + rovnaky pocet punct, ale pozicie punct
    su jednotne nahodne -> IPI by malo byt geometricke (memoryless).
    """
    words = [t for t in tokens if t not in PUNCT_SET]
    puncts = [t for t in tokens if t in PUNCT_SET]
    if not words or not puncts:
        return tokens[:]

    n_words = len(words)
    n_punct = len(puncts)
    total = n_words + n_punct

    # choose n_punct positions uniformly from total slots
    positions = sorted(rng.sample(range(total), n_punct))
    rng.shuffle(puncts)  # rozhadze typy punct

    result = [None] * total
    pi = 0
    wi = 0
    punct_set_pos = set(positions)
    for i in range(total):
        if i in punct_set_pos:
            result[i] = puncts[pi]
            pi += 1
        else:
            result[i] = words[wi]
            wi += 1
    return result


def fit_all_models(ipi):
    """Fitne vsetky 4 distribucie, vrati dict s params, aic, delta_aic, winner."""
    if len(ipi) < 20:
        return None

    fits = {
        "weibull":   fit_weibull(ipi),
        "nbinom":    fit_nbinom(ipi),
        "lognormal": fit_lognorm(ipi),
        "geometric": fit_geometric(ipi),
    }

    result = {}
    for name, f in fits.items():
        result[name] = {
            "loglik": f["loglik"],
            "aic": aic(f["loglik"], f["k"]),
            "params": f["params"],
        }
    best_aic = min(r["aic"] for r in result.values())
    for name in result:
        result[name]["delta_aic"] = result[name]["aic"] - best_aic
    result["_winner"] = min(
        (name for name in ["weibull", "nbinom", "lognormal", "geometric"]),
        key=lambda n: result[n]["aic"],
    )
    return result


def parse_meta(stem):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-root", default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_baseline")
    ap.add_argument("--n-reps", type=int, default=100,
                    help="Pocet null replikatov na knihu a model")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok_root = Path(args.tok_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            })

    print(f"[INFO] {len(books)} books, n_reps={args.n_reps}")

    per_rep_rows = []     # raw: one row per book x variant x rep
    summary_rows = []     # aggregated: one row per book x variant

    for b in books:
        tokens = load_tokens(b["path"])
        n_tokens = len(tokens)
        n_punct = sum(1 for t in tokens if t in PUNCT_SET)
        print(f"\n[{b['lang']}/{b['author']}/{b['period']}] {b['title'][:30]}"
              f"  n_tokens={n_tokens:,}  n_punct={n_punct:,}")

        # 1. Real
        ipi_real = compute_ipi(tokens)
        real_fit = fit_all_models(ipi_real)
        if real_fit is not None:
            r = {
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "variant": "real", "rep": 0, "n_ipi": len(ipi_real),
                "alpha": real_fit["weibull"]["params"]["alpha"],
                "beta": real_fit["weibull"]["params"]["beta"],
                "winner": real_fit["_winner"],
                "weibull_delta_aic": real_fit["weibull"]["delta_aic"],
                "geometric_delta_aic": real_fit["geometric"]["delta_aic"],
                "nbinom_delta_aic": real_fit["nbinom"]["delta_aic"],
                "lognormal_delta_aic": real_fit["lognormal"]["delta_aic"],
            }
            per_rep_rows.append(r)
            print(f"  real:      alpha={r['alpha']:.3f}  beta={r['beta']:.3f}"
                  f"  winner={r['winner']}  dAIC_weibull={r['weibull_delta_aic']:.1f}"
                  f"  dAIC_geom={r['geometric_delta_aic']:.1f}")

        # 2. Shuffled null
        rng = random.Random(args.seed + b["year"])
        shuf_alphas, shuf_betas, shuf_winners = [], [], []
        shuf_d_weib, shuf_d_geom = [], []
        t0 = time.time()
        for rep in range(args.n_reps):
            toks_s = shuffled_tokens(tokens, rng)
            ipi_s = compute_ipi(toks_s)
            f = fit_all_models(ipi_s)
            if f is None:
                continue
            shuf_alphas.append(f["weibull"]["params"]["alpha"])
            shuf_betas.append(f["weibull"]["params"]["beta"])
            shuf_winners.append(f["_winner"])
            shuf_d_weib.append(f["weibull"]["delta_aic"])
            shuf_d_geom.append(f["geometric"]["delta_aic"])
            per_rep_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "variant": "shuffled", "rep": rep, "n_ipi": len(ipi_s),
                "alpha": f["weibull"]["params"]["alpha"],
                "beta": f["weibull"]["params"]["beta"],
                "winner": f["_winner"],
                "weibull_delta_aic": f["weibull"]["delta_aic"],
                "geometric_delta_aic": f["geometric"]["delta_aic"],
                "nbinom_delta_aic": f["nbinom"]["delta_aic"],
                "lognormal_delta_aic": f["lognormal"]["delta_aic"],
            })
        print(f"  shuffled:  alpha={np.mean(shuf_alphas):.3f} +- {np.std(shuf_alphas):.3f}"
              f"  beta={np.mean(shuf_betas):.3f}"
              f"  winner_mode={max(set(shuf_winners), key=shuf_winners.count)}"
              f"  ({time.time() - t0:.1f}s)")

        # 3. Bernoulli-punct null
        rng = random.Random(args.seed + b["year"] + 10000)
        bern_alphas, bern_betas, bern_winners = [], [], []
        bern_d_weib, bern_d_geom = [], []
        t0 = time.time()
        for rep in range(args.n_reps):
            toks_b = bernoulli_punct_tokens(tokens, rng)
            ipi_b = compute_ipi(toks_b)
            f = fit_all_models(ipi_b)
            if f is None:
                continue
            bern_alphas.append(f["weibull"]["params"]["alpha"])
            bern_betas.append(f["weibull"]["params"]["beta"])
            bern_winners.append(f["_winner"])
            bern_d_weib.append(f["weibull"]["delta_aic"])
            bern_d_geom.append(f["geometric"]["delta_aic"])
            per_rep_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "variant": "bernoulli", "rep": rep, "n_ipi": len(ipi_b),
                "alpha": f["weibull"]["params"]["alpha"],
                "beta": f["weibull"]["params"]["beta"],
                "winner": f["_winner"],
                "weibull_delta_aic": f["weibull"]["delta_aic"],
                "geometric_delta_aic": f["geometric"]["delta_aic"],
                "nbinom_delta_aic": f["nbinom"]["delta_aic"],
                "lognormal_delta_aic": f["lognormal"]["delta_aic"],
            })
        print(f"  bernoulli: alpha={np.mean(bern_alphas):.3f} +- {np.std(bern_alphas):.3f}"
              f"  beta={np.mean(bern_betas):.3f}"
              f"  winner_mode={max(set(bern_winners), key=bern_winners.count)}"
              f"  ({time.time() - t0:.1f}s)")

        # Summary row per variant
        def _summ(variant, alphas, betas, winners, d_weib, d_geom):
            if not alphas:
                return None
            winner_counts = {w: winners.count(w) for w in set(winners)}
            dominant = max(winner_counts, key=winner_counts.get)
            return {
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "variant": variant,
                "n_reps": len(alphas),
                "alpha_mean": float(np.mean(alphas)),
                "alpha_std": float(np.std(alphas)),
                "alpha_lo": float(np.percentile(alphas, 2.5)),
                "alpha_hi": float(np.percentile(alphas, 97.5)),
                "beta_mean": float(np.mean(betas)),
                "beta_std": float(np.std(betas)),
                "dominant_winner": dominant,
                "geometric_win_pct": 100 * winner_counts.get("geometric", 0) / len(winners),
                "weibull_win_pct": 100 * winner_counts.get("weibull", 0) / len(winners),
                "nbinom_win_pct": 100 * winner_counts.get("nbinom", 0) / len(winners),
                "lognormal_win_pct": 100 * winner_counts.get("lognormal", 0) / len(winners),
                "mean_delta_aic_weibull": float(np.mean(d_weib)),
                "mean_delta_aic_geometric": float(np.mean(d_geom)),
            }

        # real as a summary row with n_reps=1
        if real_fit is not None:
            summary_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "variant": "real",
                "n_reps": 1,
                "alpha_mean": real_fit["weibull"]["params"]["alpha"],
                "alpha_std": 0.0,
                "alpha_lo": real_fit["weibull"]["params"]["alpha"],
                "alpha_hi": real_fit["weibull"]["params"]["alpha"],
                "beta_mean": real_fit["weibull"]["params"]["beta"],
                "beta_std": 0.0,
                "dominant_winner": real_fit["_winner"],
                "geometric_win_pct": 100.0 if real_fit["_winner"] == "geometric" else 0.0,
                "weibull_win_pct": 100.0 if real_fit["_winner"] == "weibull" else 0.0,
                "nbinom_win_pct": 100.0 if real_fit["_winner"] == "nbinom" else 0.0,
                "lognormal_win_pct": 100.0 if real_fit["_winner"] == "lognormal" else 0.0,
                "mean_delta_aic_weibull": real_fit["weibull"]["delta_aic"],
                "mean_delta_aic_geometric": real_fit["geometric"]["delta_aic"],
            })

        s = _summ("shuffled", shuf_alphas, shuf_betas, shuf_winners, shuf_d_weib, shuf_d_geom)
        if s:
            summary_rows.append(s)
        s = _summ("bernoulli", bern_alphas, bern_betas, bern_winners, bern_d_weib, bern_d_geom)
        if s:
            summary_rows.append(s)

    # ── Save CSVs ──
    per_rep_csv = out_dir / "kulig_ipi_baseline_perrep.csv"
    if per_rep_rows:
        with per_rep_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_rep_rows[0].keys()))
            w.writeheader()
            w.writerows(per_rep_rows)
    print(f"\n[OK] per-rep CSV: {per_rep_csv}")

    summ_csv = out_dir / "kulig_ipi_baseline_summary.csv"
    if summary_rows:
        with summ_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"[OK] summary CSV: {summ_csv}")

    # ── Plot: alpha real vs shuffled vs bernoulli per book ──
    books_sorted = sorted({(r["author"], r["year"]) for r in summary_rows})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(books_sorted))
    width = 0.27

    variants_plot = [
        ("real", "#1f77b4", "Real"),
        ("shuffled", "#d62728", "Shuffled null"),
        ("bernoulli", "#2ca02c", "Bernoulli-punct null"),
    ]
    for i, (var, color, label) in enumerate(variants_plot):
        means, errs = [], []
        for author, year in books_sorted:
            row = next((r for r in summary_rows
                        if r["author"] == author and r["year"] == year
                        and r["variant"] == var), None)
            if row is None:
                means.append(np.nan); errs.append(0)
                continue
            means.append(row["alpha_mean"])
            errs.append(max(row["alpha_hi"] - row["alpha_mean"],
                            row["alpha_mean"] - row["alpha_lo"]))
        ax.bar(x + (i - 1) * width, means, width, yerr=errs,
               color=color, alpha=0.85, label=label, capsize=3)

    labels = [f"{a[:4]}\n{y}" for a, y in books_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="α=1 (memoryless)")
    ax.set_ylabel(r"Weibull shape parameter $\alpha$", fontsize=11)
    ax.set_title("Kulig-style test on IPI: real vs memoryless null models",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "kulig_ipi_baseline_alpha.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[OK] plot: {plot_path}")

    # ── Plot 2: IPI histogram pre 1 vybranu knihu (Pickwick) ──
    pickwick_tokens = None
    for b in books:
        if "Pickwick" in b["title"]:
            pickwick_tokens = load_tokens(b["path"])
            pickwick_book = b
            break
    if pickwick_tokens:
        ipi_real = compute_ipi(pickwick_tokens)
        rng = random.Random(args.seed)
        ipi_shuf = compute_ipi(shuffled_tokens(pickwick_tokens, rng))
        ipi_bern = compute_ipi(bernoulli_punct_tokens(pickwick_tokens, rng))

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.arange(0, 60) - 0.5
        for data, color, label in [
            (ipi_real, "#1f77b4", f"Real (n={len(ipi_real):,})"),
            (ipi_shuf, "#d62728", f"Shuffled (n={len(ipi_shuf):,})"),
            (ipi_bern, "#2ca02c", f"Bernoulli (n={len(ipi_bern):,})"),
        ]:
            counts, edges = np.histogram(data, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            mask = counts > 0
            ax.plot(centers[mask], counts[mask], "o-", color=color,
                    markersize=4, alpha=0.85, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Words between punctuation marks k", fontsize=11)
        ax.set_ylabel("P(k)", fontsize=11)
        ax.set_title(f"IPI distribution: {pickwick_book['title']} "
                     "(real vs null models)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path2 = out_dir / "kulig_ipi_baseline_hist_pickwick.png"
        fig.savefig(plot_path2, dpi=300)
        plt.close(fig)
        print(f"[OK] plot: {plot_path2}")

    # ── Summary print ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for row in summary_rows:
        print(f"  {row['lang']}/{row['author'][:8]:8s}/{row['period']:6s} "
              f"{row['year']} {row['variant']:10s}  "
              f"alpha={row['alpha_mean']:.3f}+-{row['alpha_std']:.3f}  "
              f"winner={row['dominant_winner']:10s}  "
              f"geom_win%={row['geometric_win_pct']:5.1f}")

    print("\n[OK] Kulig baseline test hotovy.")


if __name__ == "__main__":
    main()
