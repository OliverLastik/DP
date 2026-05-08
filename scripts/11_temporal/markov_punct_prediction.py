#!/usr/bin/env python3
"""
Markov punct-prediction + Bayesovske vyhladenie (2A + 2B z feedbacku).

Co robi:
  1. Pre kazdu z 9 temporalnych knih nacita punct-only sekvenciu
     (extrahuje len {., ,, !, ?, ;, :}).
  2. Postavi Markov bigram model P(to|from) v dvoch variantoch:
     - MLE: raw frekvencie (sucasna metodika)
     - Bayesian: Dirichletovsky prior (Laplace alpha=1, alebo Jeffreys alpha=0.5)
  3. Pre kazdu knihu spocita:
     - Self-perplexity (in-sample) per variant
     - Top-1 accuracy na hold-out (80/20 split)
     - Cross-entropy
  4. Cross-book matica: trenuje na knihe A, testuje perplexitu na knihe B.
     Vystupom je 9x9 matica per variant.
  5. Per-symbol entropy: H(P(.|from)) per zdrojovy symbol per kniha
     - moze ukazat autorske rozdiely v pouzivani konkretnej interpunkcie.

Vystup:
  temporal_out/_markov_pred/
    self_perplexity.csv         - per kniha x variant (mle, laplace, jeffreys)
    cv_accuracy.csv             - 80/20 CV per kniha x variant
    cross_perplexity_<variant>.csv - 9x9 cross-perplexity matica
    per_symbol_entropy.csv      - H(P(.|from)) per kniha x symbol
    plots/
      cross_perplexity_heatmap_<variant>.png
      smoothing_comparison.png
      per_symbol_entropy.png
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


PUNCT_SYMBOLS = [".", ",", "!", "?", ";", ":"]
PUNCT_SET = set(PUNCT_SYMBOLS)
V = len(PUNCT_SYMBOLS)
SYM2IDX = {s: i for i, s in enumerate(PUNCT_SYMBOLS)}


def load_punct_seq(path: Path):
    toks = path.read_text(encoding="utf-8", errors="replace").split()
    return [t for t in toks if t in PUNCT_SET]


def parse_meta(stem):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def build_transition_counts(seq):
    """Counts matrix N_ij = count(i -> j) in punct sequence."""
    N = np.zeros((V, V), dtype=np.float64)
    for a, b in zip(seq[:-1], seq[1:]):
        i = SYM2IDX[a]; j = SYM2IDX[b]
        N[i, j] += 1
    return N


def smooth(N, alpha):
    """
    Bayesian Dirichlet smoothing. Returns row-normalized P(to|from).
    P(j|i) = (N_ij + alpha) / (sum_j N_ij + alpha * V)

    alpha = 0     -> MLE (no smoothing)
    alpha = 0.5   -> Jeffreys prior
    alpha = 1     -> Laplace (add-one) smoothing
    """
    M = N + alpha
    row_sums = M.sum(axis=1, keepdims=True)
    # avoid /0 for never-seen "from" symbols
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return M / row_sums


def perplexity(P, seq):
    """
    Per-token perplexity of seq under transition matrix P.
      ppl = exp( - (1/n) * sum_t log P(seq[t+1] | seq[t]) )
    Returns +inf if any zero-prob transition (only with MLE on unseen pairs).
    """
    if len(seq) < 2:
        return float("nan"), 0
    log_sum = 0.0
    n = 0
    for a, b in zip(seq[:-1], seq[1:]):
        p = P[SYM2IDX[a], SYM2IDX[b]]
        if p <= 0:
            return float("inf"), n
        log_sum += math.log(p)
        n += 1
    return math.exp(-log_sum / n), n


def top1_accuracy(P, seq):
    """Fraction of next-symbol predictions where argmax matches truth."""
    if len(seq) < 2:
        return float("nan"), 0
    correct = 0
    n = 0
    for a, b in zip(seq[:-1], seq[1:]):
        i = SYM2IDX[a]; j = SYM2IDX[b]
        pred = int(np.argmax(P[i]))
        if pred == j:
            correct += 1
        n += 1
    return correct / n, n


def row_entropy(P):
    """Shannon entropy of each row (per-source symbol entropy)."""
    H = []
    for i in range(V):
        p = P[i]
        p = p[p > 0]
        H.append(-float(np.sum(p * np.log2(p))))
    return H  # length V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-root", default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_markov_pred")
    ap.add_argument("--cv-frac", type=float, default=0.2,
                    help="Hold-out fraction for CV accuracy/perplexity (default 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok_root = Path(args.tok_root)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Smoothing variants
    SMOOTHINGS = [("mle", 0.0), ("jeffreys", 0.5), ("laplace", 1.0)]

    # Load all books
    books = []
    for sub in sorted(tok_root.iterdir()):
        if not sub.is_dir() or "validation" in sub.name:
            continue
        for fp in sorted(sub.glob("*.txt")):
            meta = parse_meta(fp.stem)
            if not meta:
                continue
            lang, author, period, year, title = meta
            seq = load_punct_seq(fp)
            books.append({
                "path": fp, "lang": lang, "author": author,
                "period": period, "year": year, "title": title,
                "key": f"{author[:4]}_{year}",
                "label": f"{author[:4]} {year}",
                "seq": seq, "n_punct": len(seq),
            })

    print(f"[INFO] {len(books)} books loaded")
    for b in books:
        print(f"  {b['lang']}/{b['key']}: n_punct={b['n_punct']:,}")

    # ── Self-perplexity + train-test split CV ──
    self_rows = []
    cv_rows = []
    rng = random.Random(args.seed)

    for b in books:
        seq = b["seq"]
        # 80/20 split of bigram pairs (random — no order, since we treat
        # punct sequence as i.i.d. realization of Markov bigrams)
        # Actually keep order: train on first 80% of sequence, test on last 20%
        n = len(seq)
        split = int(n * (1 - args.cv_frac))
        train_seq = seq[:split]
        test_seq = seq[split:]

        train_counts = build_transition_counts(train_seq)
        full_counts = build_transition_counts(seq)

        for name, alpha in SMOOTHINGS:
            # self perplexity (full book, in-sample)
            P_full = smooth(full_counts, alpha)
            ppl_self, n_self = perplexity(P_full, seq)

            # CV
            P_train = smooth(train_counts, alpha)
            ppl_test, n_test = perplexity(P_train, test_seq)
            acc_test, _ = top1_accuracy(P_train, test_seq)

            self_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "smoothing": name, "alpha": alpha,
                "n_punct": b["n_punct"],
                "perplexity_self": ppl_self,
            })
            cv_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "smoothing": name, "alpha": alpha,
                "n_train": len(train_seq), "n_test": len(test_seq),
                "perplexity_test": ppl_test,
                "top1_accuracy": acc_test,
            })

        print(f"  [OK] {b['key']}  n={n:,}  split {len(train_seq)}/{len(test_seq)}")

    # Save self + CV csvs
    with (out_dir / "self_perplexity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(self_rows[0].keys()))
        w.writeheader(); w.writerows(self_rows)

    with (out_dir / "cv_accuracy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cv_rows[0].keys()))
        w.writeheader(); w.writerows(cv_rows)
    print(f"\n[OK] self_perplexity.csv, cv_accuracy.csv -> {out_dir}")

    # ── Cross-book perplexity matrix per variant ──
    book_keys = [b["key"] for b in books]
    n_books = len(books)
    cross_matrices = {}  # name -> matrix
    for name, alpha in SMOOTHINGS:
        # train P_train on each book (full book)
        P_models = []
        for b in books:
            counts = build_transition_counts(b["seq"])
            P_models.append(smooth(counts, alpha))

        M = np.full((n_books, n_books), np.nan)
        for i in range(n_books):
            for j in range(n_books):
                ppl, _ = perplexity(P_models[i], books[j]["seq"])
                M[i, j] = ppl
        cross_matrices[name] = M

        # save csv
        with (out_dir / f"cross_perplexity_{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train_on"] + book_keys)
            for i, k in enumerate(book_keys):
                w.writerow([k] + [f"{M[i, j]:.4f}" for j in range(n_books)])

        print(f"[OK] cross_perplexity_{name}.csv")

    # ── Per-symbol entropy per book (only with Laplace, robust) ──
    entropy_rows = []
    for b in books:
        counts = build_transition_counts(b["seq"])
        P = smooth(counts, 1.0)
        H = row_entropy(P)
        for sym, h in zip(PUNCT_SYMBOLS, H):
            entropy_rows.append({
                "lang": b["lang"], "author": b["author"], "period": b["period"],
                "year": b["year"], "title": b["title"],
                "from_symbol": sym, "entropy_bits": h,
            })
    with (out_dir / "per_symbol_entropy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(entropy_rows[0].keys()))
        w.writeheader(); w.writerows(entropy_rows)
    print(f"[OK] per_symbol_entropy.csv")

    # ══════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════

    # 1. Smoothing comparison: perplexity_test for mle vs laplace vs jeffreys
    #    per book, side-by-side bars
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_books)
    width = 0.27
    colors = {"mle": "#d62728", "jeffreys": "#ff7f0e", "laplace": "#2ca02c"}
    for i, (name, alpha) in enumerate(SMOOTHINGS):
        ppls = [r["perplexity_test"] for r in cv_rows if r["smoothing"] == name]
        # replace inf with very large
        ppls_plot = [min(p, 1e6) if not math.isinf(p) else 1e6 for p in ppls]
        ax.bar(x + (i - 1) * width, ppls_plot, width,
               color=colors[name], alpha=0.85, label=f"{name} (alpha={alpha})")
    ax.set_xticks(x)
    ax.set_xticklabels([b["label"] for b in books], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Test perplexity (held-out 20%)", fontsize=11)
    ax.set_title("Markov punct-prediction: smoothing comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "smoothing_comparison.png", dpi=300)
    plt.close(fig)
    print(f"[OK] plot: smoothing_comparison.png")

    # 2. Cross-perplexity heatmaps (one per smoothing)
    for name, alpha in SMOOTHINGS:
        M = cross_matrices[name]
        fig, ax = plt.subplots(figsize=(9, 7.5))
        # cap at 99-percentile to avoid one outlier squashing the colormap
        vmax = np.nanpercentile(M, 99)
        vmin = np.nanmin(M)
        im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_books))
        ax.set_xticklabels(book_keys, fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(n_books))
        ax.set_yticklabels(book_keys, fontsize=9)
        ax.set_xlabel("Test on (book)", fontsize=11)
        ax.set_ylabel("Train on (book)", fontsize=11)
        ax.set_title(f"Cross-book perplexity ({name}, alpha={alpha}) — "
                     "diagonal = self, off-diagonal = generalization",
                     fontsize=11)
        for i in range(n_books):
            for j in range(n_books):
                if not np.isnan(M[i, j]) and not np.isinf(M[i, j]):
                    color = "white" if M[i, j] > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)
        fig.colorbar(im, ax=ax, label="perplexity")
        fig.tight_layout()
        fig.savefig(plot_dir / f"cross_perplexity_heatmap_{name}.png", dpi=300)
        plt.close(fig)
        print(f"[OK] plot: cross_perplexity_heatmap_{name}.png")

    # 3. Per-symbol entropy heatmap (book x symbol)
    H_matrix = np.zeros((n_books, V))
    for i, b in enumerate(books):
        rows = [r for r in entropy_rows if r["title"] == b["title"]]
        for r in rows:
            j = PUNCT_SYMBOLS.index(r["from_symbol"])
            H_matrix[i, j] = r["entropy_bits"]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(H_matrix, cmap="viridis", aspect="auto",
                   vmin=0, vmax=np.log2(V))
    ax.set_xticks(range(V))
    ax.set_xticklabels(PUNCT_SYMBOLS, fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_books))
    ax.set_yticklabels([b["label"] for b in books], fontsize=9)
    ax.set_xlabel("Source symbol (P(. | from))", fontsize=11)
    ax.set_ylabel("Book", fontsize=11)
    ax.set_title("Per-symbol next-punct entropy (Laplace) — "
                 "higher = less predictable next punct",
                 fontsize=11)
    for i in range(n_books):
        for j in range(V):
            ax.text(j, i, f"{H_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if H_matrix[i, j] < 1.5 else "black")
    fig.colorbar(im, ax=ax, label="entropy (bits)")
    fig.tight_layout()
    fig.savefig(plot_dir / "per_symbol_entropy.png", dpi=300)
    plt.close(fig)
    print(f"[OK] plot: per_symbol_entropy.png")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Smoothing comparison (test perplexity, lower=better) ---")
    print(f"  {'book':16s}  {'mle':>10s}  {'jeffreys':>10s}  {'laplace':>10s}")
    for b in books:
        ppls = {}
        for r in cv_rows:
            if r["title"] == b["title"]:
                ppls[r["smoothing"]] = r["perplexity_test"]
        mle_str = "INF" if math.isinf(ppls['mle']) else f"{ppls['mle']:.3f}"
        print(f"  {b['key']:16s}  {mle_str:>10s}  "
              f"{ppls['jeffreys']:10.3f}  {ppls['laplace']:10.3f}")

    print("\n--- Cross-perplexity (laplace) — train on row, test on col ---")
    M = cross_matrices["laplace"]
    print("  " + "  ".join([f"{k:>10s}" for k in [""] + book_keys]))
    for i, k in enumerate(book_keys):
        row = "  ".join([f"{M[i, j]:>10.3f}" for j in range(n_books)])
        print(f"  {k:>10s}  {row}")

    print("\n--- Self vs cross asymmetry (laplace) ---")
    diag = [M[i, i] for i in range(n_books)]
    offdiag_means = [np.nanmean([M[i, j] for j in range(n_books) if j != i])
                     for i in range(n_books)]
    print(f"  {'book':16s}  {'self':>10s}  {'cross_mean':>12s}  {'gap':>10s}")
    for i, b in enumerate(books):
        gap = offdiag_means[i] - diag[i]
        print(f"  {b['key']:16s}  {diag[i]:10.3f}  {offdiag_means[i]:12.3f}  "
              f"{gap:+10.3f}")

    print("\n[OK] Markov punct-prediction hotovy.")


if __name__ == "__main__":
    main()
