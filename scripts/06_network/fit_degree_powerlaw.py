#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_degree_samples_from_hist(hist_csv: Path):
    """
    Ocakava histogram s columns: k, count  (alebo degree,count alebo k,pk_mean)
    Vrati pole stupnov (samples) replikovane podla count.
    Pre pk_mean (BA output) konvertuje na pseudo-counts.
    """
    with hist_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = {c.lower(): c for c in (r.fieldnames or [])}
        k_col = cols.get("k") or cols.get("degree")
        # support pk_mean (BA histograms) as well as count
        c_col = cols.get("count") or cols.get("n") or cols.get("freq")
        pk_col = cols.get("pk_mean") or cols.get("pk")
        is_pk = False
        if not c_col and pk_col:
            c_col = pk_col
            is_pk = True
        if not (k_col and c_col):
            raise ValueError(f"Neviem najst stlpce k/count v {hist_csv} (mam: {r.fieldnames})")

        raw = []
        for row in r:
            try:
                k = int(float(row[k_col]))
                val = float(row[c_col])
            except Exception:
                continue
            if k <= 0 or val <= 0:
                continue
            raw.append((k, val))

    if is_pk:
        # convert pk to pseudo-counts (scale to ~100k samples)
        total_target = 100000
        samples = []
        for k, pk in raw:
            cnt = max(1, int(round(pk * total_target)))
            samples.extend([k] * cnt)
    else:
        samples = []
        for k, cnt in raw:
            samples.extend([k] * int(cnt))

    return np.array(samples, dtype=np.int32)

def plot_ccdf(samples, title, out_png: Path):
    # CCDF: P(K >= k)
    vals = np.sort(samples)
    unique = np.unique(vals)
    ccdf = np.array([(vals.size - np.searchsorted(vals, k, side="left")) / vals.size for k in unique])

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(unique, ccdf)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k (degree)")
    ax.set_ylabel("CCDF P(K ≥ k)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-dir", default="network_out_core/degree_histograms", help="kde sú degree_hist_*.csv")
    ap.add_argument("--outdir", default="network_out_core/powerlaw", help="kam ukladať výsledky")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--use-ba", action="store_true", help="fitovať aj BA histogramy (ak sú v ba_out_core/degree_histograms)")
    ap.add_argument("--ba-hist-dir", default="ba_out_core/degree_histograms")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    # powerlaw je extra balík
    import powerlaw

    rows = []
    for lang in langs:
        hist_path = Path(args.hist_dir) / f"degree_hist_{lang}.csv"
        if not hist_path.exists():
            print(f"[WARN] missing {hist_path}")
            continue

        samples = load_degree_samples_from_hist(hist_path)
        if samples.size < 1000:
            print(f"[WARN] {lang}: málo samples ({samples.size})")
            continue

        # CCDF plot (real)
        plot_ccdf(samples, f"CCDF degree (real) - {lang}", outdir / f"ccdf_real_{lang}.png")

        fit = powerlaw.Fit(samples, discrete=True, verbose=False)
        alpha = float(fit.power_law.alpha)
        xmin = int(fit.power_law.xmin)
        try:
            ks = float(fit.power_law.KS())
        except Exception:
            ks = float("nan")

        # compare to lognormal
        try:
            R, p = fit.distribution_compare("power_law", "lognormal")
            R = float(R); p = float(p)
        except Exception:
            R, p = float("nan"), float("nan")

        rows.append({
            "lang": lang,
            "kind": "real",
            "n_samples": int(samples.size),
            "alpha": alpha,
            "xmin": xmin,
            "ks": ks,
            "R_pl_vs_lognormal": R,
            "p_pl_vs_lognormal": p,
        })

        # powerlaw package plot (optional nicer overlay)
        fig = fit.plot_ccdf(label="empirical")
        fit.power_law.plot_ccdf(ax=fig, linestyle="--", label="power-law fit")
        plt.legend()
        plt.title(f"CCDF + power-law fit (real) - {lang}")
        plt.tight_layout()
        plt.savefig(outdir / f"ccdf_fit_real_{lang}.png", dpi=300)
        plt.close()

        print(f"[OK] {lang} REAL: alpha={alpha:.3f}, xmin={xmin}, KS={ks:.4f}, R={R:.3f}, p={p:.4g}")

        # Optional BA
        if args.use_ba:
            ba_hist = Path(args.ba_hist_dir) / f"degree_hist_ba_{lang}.csv"
            if ba_hist.exists():
                ba_samples = load_degree_samples_from_hist(ba_hist)
                plot_ccdf(ba_samples, f"CCDF degree (BA) - {lang}", outdir / f"ccdf_ba_{lang}.png")

                ba_fit = powerlaw.Fit(ba_samples, discrete=True, verbose=False)
                ba_alpha = float(ba_fit.power_law.alpha)
                ba_xmin = int(ba_fit.power_law.xmin)
                try:
                    ba_ks = float(ba_fit.power_law.KS())
                except Exception:
                    ba_ks = float("nan")
                try:
                    R2, p2 = ba_fit.distribution_compare("power_law", "lognormal")
                except Exception:
                    R2, p2 = float("nan"), float("nan")
                rows.append({
                    "lang": lang,
                    "kind": "ba",
                    "n_samples": int(ba_samples.size),
                    "alpha": ba_alpha,
                    "xmin": ba_xmin,
                    "ks": ba_ks,
                    "R_pl_vs_lognormal": float(R2),
                    "p_pl_vs_lognormal": float(p2),
                })

                fig = ba_fit.plot_ccdf(label="empirical")
                ba_fit.power_law.plot_ccdf(ax=fig, linestyle="--", label="power-law fit")
                plt.legend()
                plt.title(f"CCDF + power-law fit (BA) - {lang}")
                plt.tight_layout()
                plt.savefig(outdir / f"ccdf_fit_ba_{lang}.png", dpi=300)
                plt.close()

                print(f"[OK] {lang} BA: alpha={ba_alpha:.3f}, xmin={ba_xmin}, KS={ba_ks:.4f}, R={float(R2):.3f}, p={float(p2):.4g}")
            else:
                print(f"[WARN] missing BA hist {ba_hist}")

    # Save summary CSV
    out_csv = outdir / "degree_powerlaw_fit_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang","kind","n_samples","alpha","xmin","ks","R_pl_vs_lognormal","p_pl_vs_lognormal"
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] saved {out_csv}")

if __name__ == "__main__":
    main()