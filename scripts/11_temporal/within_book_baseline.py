#!/usr/bin/env python3
"""
Within-book JSD baseline.

Kazdu knihu rozdeli na 3 rovnake okna a spocita JSD IPI distribucii
medzi oknami. Potom porovna tento intra-book sum s inter-period JSD
z temporalnej analyzy. Ak intra-book JSD >= inter-period JSD, temporalny
drift je v sume; ak intra-book JSD << inter-period JSD, drift je skutocny.
"""
import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

PUNCT_SET = {".", ",", "!", "?", ";", ":"}

AUTHOR_COLORS = {
    "dickens":  "#1f77b4",
    "fontane":  "#d62728",
    "couperus": "#2ca02c",
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
    return np.array(intervals)


def ipi_pmf(ipi, max_k=50):
    bins = np.arange(0, max_k + 1)
    h, _ = np.histogram(ipi, bins=bins, density=False)
    s = h.sum()
    return h.astype(float) / s if s > 0 else h.astype(float)


def jsd(p, q):
    return float(jensenshannon(p, q) ** 2)


def parse_meta(stem: str):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl", "da", "sv"}:
        return parts[0], parts[1], parts[2], int(parts[3])
    if len(parts) >= 4:
        return "??", parts[0], parts[1], int(parts[2])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_baseline")
    ap.add_argument("--n-windows", type=int, default=3)
    args = ap.parse_args()

    tok_root = Path(args.token_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Gather all books grouped by author
    by_author = defaultdict(list)
    for sub in sorted(tok_root.iterdir()):
        if not sub.is_dir():
            continue
        for fp in sorted(sub.glob("*.txt")):
            meta = parse_meta(fp.stem)
            if not meta:
                continue
            lang, author, period, year = meta
            by_author[(lang, author)].append({
                "path": fp,
                "period": period,
                "year": year,
                "title": "_".join(fp.stem.split("_")[4:]),
            })

    if not by_author:
        print(f"[ERR] No books under {tok_root}")
        return

    rows = []  # per-window records (for CSV)
    intra_rows = []  # intra-book JSD
    inter_rows = []  # inter-period JSD (full books)

    # ── Per-author: split each book into N windows, compute intra-book JSD ──
    per_author_books = {}  # (lang,author) -> list of {period,year,title,full_pmf,window_pmfs}

    for (lang, author), books in by_author.items():
        books.sort(key=lambda b: b["year"])
        book_records = []
        for b in books:
            tokens = load_tokens(b["path"])
            n = len(tokens)
            w = args.n_windows
            chunk_size = n // w
            window_pmfs = []
            for i in range(w):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < w - 1 else n
                ipi = compute_ipi(tokens[start:end])
                window_pmfs.append(ipi_pmf(ipi))
                rows.append({
                    "lang": lang, "author": author, "period": b["period"],
                    "year": b["year"], "title": b["title"],
                    "window": i, "n_tokens": end - start,
                    "ipi_mean": float(ipi.mean()) if len(ipi) else float("nan"),
                })

            # intra-book pairwise JSD
            intra_jsds = [jsd(window_pmfs[i], window_pmfs[j])
                          for i, j in combinations(range(w), 2)]
            for (i, j), v in zip(combinations(range(w), 2), intra_jsds):
                intra_rows.append({
                    "lang": lang, "author": author,
                    "period": b["period"], "year": b["year"], "title": b["title"],
                    "win_i": i, "win_j": j, "jsd": v,
                })

            full_ipi = compute_ipi(tokens)
            book_records.append({
                "period": b["period"], "year": b["year"], "title": b["title"],
                "full_pmf": ipi_pmf(full_ipi),
                "intra_jsds": intra_jsds,
                "intra_mean": float(np.mean(intra_jsds)),
                "intra_max": float(np.max(intra_jsds)),
            })

        per_author_books[(lang, author)] = book_records

        # inter-period JSD (full-book pmfs)
        for i in range(len(book_records)):
            for j in range(i + 1, len(book_records)):
                bi = book_records[i]
                bj = book_records[j]
                v = jsd(bi["full_pmf"], bj["full_pmf"])
                inter_rows.append({
                    "lang": lang, "author": author,
                    "period_i": bi["period"], "year_i": bi["year"],
                    "period_j": bj["period"], "year_j": bj["year"],
                    "jsd": v,
                })

    # Write CSVs
    if rows:
        with (out / "window_records.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] window_records.csv ({len(rows)} rows)")

    if intra_rows:
        with (out / "intra_book_jsd.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(intra_rows[0].keys()))
            w.writeheader()
            w.writerows(intra_rows)
        print(f"[OK] intra_book_jsd.csv ({len(intra_rows)} rows)")

    if inter_rows:
        with (out / "inter_period_jsd.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inter_rows[0].keys()))
            w.writeheader()
            w.writerows(inter_rows)
        print(f"[OK] inter_period_jsd.csv ({len(inter_rows)} rows)")

    # ── Signal vs noise plot ──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    author_keys = sorted(per_author_books.keys(), key=lambda k: (k[1],))
    x_base = np.arange(len(author_keys))
    width = 0.35

    intra_means = []
    intra_maxes = []
    inter_means = []
    inter_maxes = []
    labels = []
    for (lang, author) in author_keys:
        labels.append(f"{author.capitalize()}\n({lang})")
        brs = per_author_books[(lang, author)]
        all_intra = [v for b in brs for v in b["intra_jsds"]]
        all_inter = [r["jsd"] for r in inter_rows
                     if r["author"] == author and r["lang"] == lang]
        intra_means.append(np.mean(all_intra) if all_intra else 0)
        intra_maxes.append(np.max(all_intra) if all_intra else 0)
        inter_means.append(np.mean(all_inter) if all_inter else 0)
        inter_maxes.append(np.max(all_inter) if all_inter else 0)

    ax.bar(x_base - width / 2, intra_means, width,
           color="#cccccc", edgecolor="black",
           label="Intra-book JSD (mean)", zorder=2)
    ax.bar(x_base - width / 2, np.array(intra_maxes) - np.array(intra_means),
           width, bottom=intra_means,
           color="#eeeeee", edgecolor="black", hatch="//",
           label="Intra-book JSD (max - mean)", zorder=2)
    ax.bar(x_base + width / 2, inter_means, width,
           color=[AUTHOR_COLORS.get(a, "gray") for _, a in author_keys],
           alpha=0.9, edgecolor="black",
           label="Inter-period JSD (mean)", zorder=2)
    ax.bar(x_base + width / 2, np.array(inter_maxes) - np.array(inter_means),
           width, bottom=inter_means,
           color=[AUTHOR_COLORS.get(a, "gray") for _, a in author_keys],
           alpha=0.4, edgecolor="black", hatch="\\\\",
           label="Inter-period JSD (max - mean)", zorder=2)

    ax.set_xticks(x_base)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Jensen-Shannon divergence (IPI distribution)", fontsize=11)
    ax.set_title("Temporal drift vs within-book noise floor", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out / "signal_vs_noise.png", dpi=300)
    plt.close(fig)
    print("[OK] signal_vs_noise.png")

    # ── Per-book intra JSD scatter (zero-in on Couperus late outlier) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    author_keys_for_scatter = sorted(per_author_books.keys())
    x_off = {k: i for i, k in enumerate(author_keys_for_scatter)}
    for (lang, author), brs in per_author_books.items():
        color = AUTHOR_COLORS.get(author, "gray")
        xs = []
        ys_mean = []
        ys_max = []
        for b in brs:
            xs.append(b["year"])
            ys_mean.append(b["intra_mean"])
            ys_max.append(b["intra_max"])
        ax.plot(xs, ys_mean, "-o", color=color, markersize=8, linewidth=2,
                label=f"{author.capitalize()} ({lang}) intra-mean")
        ax.plot(xs, ys_max, ":^", color=color, markersize=6, linewidth=1, alpha=0.6,
                label=f"{author.capitalize()} ({lang}) intra-max")
    ax.set_xlabel("Publication year", fontsize=11)
    ax.set_ylabel("Intra-book JSD (IPI, 3 windows)", fontsize=11)
    ax.set_title("Within-book variability per book", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "intra_book_scatter.png", dpi=300)
    plt.close(fig)
    print("[OK] intra_book_scatter.png")

    # ── Text summary ──
    print("\n== Signal-vs-noise summary ==")
    print(f"{'author':20s} {'intra_mean':>12s} {'intra_max':>12s} "
          f"{'inter_mean':>12s} {'inter_max':>12s} {'ratio':>10s}")
    ratio_rows = []
    for (lang, author), im, ix, em, ex in zip(
            author_keys, intra_means, intra_maxes, inter_means, inter_maxes):
        ratio = em / im if im > 0 else float("nan")
        print(f"  {author} ({lang}) {'':6s} {im:12.5f} {ix:12.5f} "
              f"{em:12.5f} {ex:12.5f} {ratio:10.2f}")
        ratio_rows.append({
            "lang": lang, "author": author,
            "intra_jsd_mean": im, "intra_jsd_max": ix,
            "inter_jsd_mean": em, "inter_jsd_max": ex,
            "ratio_inter_over_intra": ratio,
        })

    with (out / "signal_noise_ratio.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
        w.writeheader()
        w.writerows(ratio_rows)
    print(f"[OK] signal_noise_ratio.csv")

    print("\n[OK] Within-book baseline hotovy.")


if __name__ == "__main__":
    main()
