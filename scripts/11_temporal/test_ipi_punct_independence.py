#!/usr/bin/env python3
"""
Empiricky test nezavislosti K_i (IPI) a tau_{i-1} (typ predoslej interpunkcie).

Otazka: Plati P(K | tau_prev = ',') == P(K | tau_prev = '.') == ... ?
Ak ano, faktorizacia v Sekcii 14.4 (timing nezavisle od typu) je opravnena.
Ak nie, treba model rozsirit na P(K | tau_prev).

Test:
  1. Pre kazdu knihu, group K_i podla tau_{i-1}.
  2. Spocitaj per-group mean, median, std.
  3. KL divergencia kazdej sub-distribucie voci marginalu P(K).
  4. Globalny chi-square test nezavislosti (po binovani K).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUNCT_SYMBOLS = [".", ",", "!", "?", ";", ":"]
PUNCT_SET = set(PUNCT_SYMBOLS)


def load_tokens(p: Path):
    return p.read_text(encoding="utf-8", errors="replace").split()


def parse_meta(stem):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def compute_ipi_with_prev_punct(tokens):
    """Vracia zoznam (K_i, tau_{i-1}) tuplov."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", "--tok-root", dest="token_dir",
                    default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_independence_test")
    args = ap.parse_args()

    tok_root = Path(args.token_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    chi2_rows = []

    books = []
    for sub in sorted(tok_root.iterdir()):
        if not sub.is_dir() or "validation" in sub.name:
            continue
        for fp in sorted(sub.glob("*.txt")):
            meta = parse_meta(fp.stem)
            if not meta:
                continue
            books.append((meta, fp))

    print(f"[INFO] {len(books)} books")

    for (lang, author, period, year, title), fp in books:
        tokens = load_tokens(fp)
        pairs = compute_ipi_with_prev_punct(tokens)
        if not pairs:
            continue
        Ks = np.array([p[0] for p in pairs])
        prevs = [p[1] for p in pairs]

        # marginal
        marg_mean = float(Ks.mean())
        marg_med = float(np.median(Ks))

        # group by tau_prev
        groups = defaultdict(list)
        for k, t in pairs:
            groups[t].append(k)

        per_group_stats = {}
        for sym in PUNCT_SYMBOLS:
            arr = np.array(groups.get(sym, []))
            if len(arr) < 30:
                per_group_stats[sym] = None
                continue
            per_group_stats[sym] = {
                "n": len(arr),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
            }

        # KL divergence per group vs marginal (after binning)
        K_max = int(np.percentile(Ks, 99))
        bins = np.arange(0, K_max + 2)
        marg_hist, _ = np.histogram(Ks, bins=bins)
        marg_p = marg_hist / marg_hist.sum() + 1e-12
        kl_divs = {}
        for sym in PUNCT_SYMBOLS:
            arr = np.array(groups.get(sym, []))
            if len(arr) < 30:
                kl_divs[sym] = float("nan")
                continue
            sub_hist, _ = np.histogram(arr, bins=bins)
            sub_p = sub_hist / sub_hist.sum() + 1e-12
            kl_divs[sym] = float(entropy(sub_p, marg_p))

        # Chi-square test of independence
        # Build contingency table: rows = K bins (coarsened), cols = tau_prev
        # use coarse binning (5 bins on quantiles) to avoid sparse cells
        qs = np.percentile(Ks, [20, 40, 60, 80])
        K_bins = np.digitize(Ks, qs)  # 0..4
        prev_idx = [PUNCT_SYMBOLS.index(p) if p in PUNCT_SYMBOLS else -1 for p in prevs]

        # Build contingency
        table = np.zeros((5, 6), dtype=int)
        for kb, pi in zip(K_bins, prev_idx):
            if pi >= 0:
                table[kb, pi] += 1

        # Drop columns with zero counts (otherwise chi2 fails)
        col_sum = table.sum(axis=0)
        keep_cols = col_sum >= 30
        table_red = table[:, keep_cols]

        try:
            chi2, p_val, dof, _ = chi2_contingency(table_red)
            cramers_v = np.sqrt(chi2 / (table_red.sum() * (min(table_red.shape) - 1)))
        except Exception:
            chi2, p_val, dof, cramers_v = float("nan"), float("nan"), 0, float("nan")

        key = f"{author[:4]}_{year}"
        print(f"\n[{key}] n_pairs={len(pairs)}, marg_mean={marg_mean:.2f}")
        for sym in PUNCT_SYMBOLS:
            s = per_group_stats[sym]
            kl = kl_divs[sym]
            if s is None:
                print(f"  P(K | {sym})  n<30, skip")
                continue
            print(f"  P(K | {sym})  n={s['n']:5d}  mean={s['mean']:5.2f}  "
                  f"median={s['median']:4.1f}  KL_to_marginal={kl:.4f}")
        print(f"  Chi^2 ind. test: chi2={chi2:.1f} dof={dof} p={p_val:.2e} "
              f"Cramer V={cramers_v:.3f}")

        chi2_rows.append({
            "lang": lang, "author": author, "period": period, "year": year,
            "title": title, "n_pairs": len(pairs),
            "chi2": chi2, "dof": dof, "p_value": p_val,
            "cramers_v": cramers_v,
            "marg_mean": marg_mean, "marg_median": marg_med,
        })

        for sym in PUNCT_SYMBOLS:
            s = per_group_stats[sym]
            if s is None:
                continue
            row = {
                "lang": lang, "author": author, "period": period, "year": year,
                "title": title, "tau_prev": sym, **s,
                "kl_to_marginal": kl_divs[sym],
                "ratio_mean_to_marginal": s["mean"] / marg_mean,
            }
            summary_rows.append(row)

    # Save CSVs
    with (out_dir / "per_group_stats.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\n[OK] per_group_stats.csv")

    with (out_dir / "chi2_independence.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chi2_rows[0].keys()))
        w.writeheader(); w.writerows(chi2_rows)
    print(f"[OK] chi2_independence.csv")

    # Plot: ratio of mean(K|tau) / mean(K) per book per symbol
    fig, ax = plt.subplots(figsize=(10, 5.5))
    book_keys = sorted(set(f"{r['author'][:4]}_{r['year']}" for r in summary_rows))
    x = np.arange(len(book_keys))
    width = 0.13
    colors = {".": "#1f77b4", ",": "#ff7f0e", "!": "#2ca02c",
              "?": "#d62728", ";": "#9467bd", ":": "#8c564b"}

    for i, sym in enumerate(PUNCT_SYMBOLS):
        ratios = []
        for k in book_keys:
            row = next((r for r in summary_rows
                        if r["tau_prev"] == sym and f"{r['author'][:4]}_{r['year']}" == k), None)
            ratios.append(row["ratio_mean_to_marginal"] if row else np.nan)
        ax.bar(x + (i - 2.5) * width, ratios, width, color=colors[sym], alpha=0.85,
               label=f"prev = '{sym}'")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="independence (= 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(book_keys, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("mean(K | prev_punct) / mean(K)", fontsize=11)
    ax.set_title("Test nezavislosti K (IPI) ⊥ τ_prev (predosly punct typ)",
                 fontsize=12)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "ipi_punct_dependence.png", dpi=300)
    plt.close(fig)
    print(f"[OK] plot: ipi_punct_dependence.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — INDEPENDENCE TEST")
    print("=" * 70)
    print(f"  {'book':16s}  {'n_pairs':>8s}  {'chi2':>8s}  {'p':>10s}  {'Cramer V':>10s}")
    for r in chi2_rows:
        print(f"  {r['author'][:4]}_{r['year']:>11d}  {r['n_pairs']:>8d}  "
              f"{r['chi2']:>8.0f}  {r['p_value']:>10.2e}  {r['cramers_v']:>10.3f}")

    print("\nInterpretacia:")
    print("  - p < 0.001 -> nezavislost zamietnuta")
    print("  - Cramer V < 0.10 -> dependence prakticky zanedbatelna")
    print("  - Cramer V 0.10-0.30 -> mierna dependence")
    print("  - Cramer V > 0.30 -> silna dependence (model treba opravit)")


if __name__ == "__main__":
    main()
