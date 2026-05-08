#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PUNCT_KEYS = [".", ",", ";", ":", "?", "!", "…", "—", "-", '"', "'", "(", ")", "[", "]"]

def bootstrap_ci(x: np.ndarray, stat="median", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    f = np.median if stat == "median" else np.mean
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        boots.append(f(samp))
    boots = np.array(boots)
    return (f(x), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="punct_stats_core.csv")
    ap.add_argument("--outdir", default="punct_out_core")
    ap.add_argument("--agg", choices=["median", "mean"], default="median")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inp, dtype={"language": str, "gutenberg_id": str})
    df = df[df["language"].isin(["en", "de", "nl"])].copy()

    # 1) summary per language
    rows = []
    for lang, g in df.groupby("language"):
        row = {
            "language": lang,
            "n_books": len(g),
            "words_median": float(np.median(g["words"]))
        }
        for k in PUNCT_KEYS:
            col = f"per1k_{k}"
            if col not in g.columns:
                continue
            val, lo, hi = bootstrap_ci(
                g[col].to_numpy(dtype=float),
                stat=args.agg,
                n_boot=args.boot,
                seed=args.seed
            )
            row[f"{args.agg}_per1k_{k}"] = val
            row[f"ci95_lo_per1k_{k}"] = lo
            row[f"ci95_hi_per1k_{k}"] = hi
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("language")
    summary.to_csv(outdir / "punct_summary_by_language.csv", index=False, encoding="utf-8")

    # 2) tidy per-book
    tidy_rows = []
    for _, r in df.iterrows():
        for k in PUNCT_KEYS:
            col = f"per1k_{k}"
            if col in df.columns:
                tidy_rows.append({
                    "language": r["language"],
                    "gutenberg_id": r["gutenberg_id"],
                    "punct": k,
                    "per1k": float(r[col]),
                    "words": int(r["words"]),
                })
    tidy = pd.DataFrame(tidy_rows)
    tidy.to_csv(outdir / "punct_per_book_tidy.csv", index=False, encoding="utf-8")

    # 3) ktoré znaky sa medzi jazykmi líšia najviac
    cols = [c for c in summary.columns if c.startswith(f"{args.agg}_per1k_")]
    dist_rows = []
    for c in cols:
        vals = summary[c].to_numpy(dtype=float)
        dist_rows.append({
            "punct": c.replace(f"{args.agg}_per1k_", ""),
            "range_across_langs": float(np.nanmax(vals) - np.nanmin(vals)),
        })
    dist = pd.DataFrame(dist_rows).sort_values("range_across_langs", ascending=False)
    dist.to_csv(outdir / "punct_most_different_across_langs.csv", index=False, encoding="utf-8")

    # 4) krátka LaTeX tabuľka pre . , ; : ? !
    keep = [".", ",", ";", ":", "?", "!"]
    latex = summary[["language", "n_books"] + [f"{args.agg}_per1k_{k}" for k in keep]].copy()
    latex.to_latex(outdir / "punct_summary_table.tex", index=False, float_format="%.3f")

    print("Done.")
    print(f"Output dir: {outdir.resolve()}")

if __name__ == "__main__":
    main()
