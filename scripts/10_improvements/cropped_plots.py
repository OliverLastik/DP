#!/usr/bin/env python3
"""
Orezane (cropped) verzie degree a rank-frequency plotov.

Motivacia: raw a log-binned ploty zobrazuju cely rozsah vratane
"ohnutej" hlavy (k < 4, rank < 10) a hlucneho chvosta (single-count
bins). Lineárna (power-law) oblast je stred. Tento skript:

  1. Nacita histogramy (degree) alebo CSV (rankfreq) pre en/de/nl
  2. Log-binne ich
  3. Nacita existujuce fit ranges z improved_plots (fit_lo, fit_hi)
  4. Crop xlim na [fit_lo, fit_hi] + male marginaly
  5. Zobrazi iba stredovu linearnu cast + fit line + R^2
  6. Ulozi do improved_plots/{degree,rankfreq}_cropped/

Vystupy:
  improved_plots/degree_cropped/degree_cropped_{lang}.png
  improved_plots/rankfreq_cropped/rankfreq_cropped_{lang}_{mode}.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG_COLORS = {"en": "#1f77b4", "de": "#d62728", "nl": "#2ca02c"}


def load_hist(csv_path: Path):
    ks, vals = [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        k_col = cols.get("k") or cols.get("degree") or cols.get("rank")
        if "pk_mean" in cols:
            v_col = cols["pk_mean"]
        elif "pk" in cols:
            v_col = cols["pk"]
        elif "frequency" in cols:
            v_col = cols["frequency"]
        else:
            v_col = cols.get("count") or cols.get("n") or cols.get("freq")
        for row in reader:
            try:
                k = float(row[k_col])
                v = float(row[v_col])
            except Exception:
                continue
            if k > 0 and v > 0:
                ks.append(k)
                vals.append(v)
    ks = np.array(ks, dtype=float)
    vals = np.array(vals, dtype=float)
    total = vals.sum()
    pk = vals / total if total > 0 else vals
    return ks, pk


def logbin(ks, pk, n_bins=50):
    mask = (ks > 0) & (pk > 0)
    k = ks[mask]
    p = pk[mask]
    if len(k) == 0:
        return np.array([]), np.array([])
    log_k = np.log10(k)
    edges = np.linspace(log_k.min(), log_k.max(), n_bins + 1)
    bin_k, bin_p = [], []
    for i in range(len(edges) - 1):
        in_bin = (log_k >= edges[i]) & (log_k < edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_k.append(10 ** np.mean(log_k[in_bin]))
        bin_p.append(np.mean(p[in_bin]))
    return np.array(bin_k), np.array(bin_p)


def load_fit_summary(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def crop_degree_plots(args):
    in_dir = Path(args.deg_hist_dir)
    ba_dir = Path(args.ba_hist_dir)
    dm_dir = Path(args.dm_hist_dir)
    out_dir = Path(args.out_dir) / "degree_cropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_path = Path(args.out_dir) / "degree_logbin_fit_improved.csv"
    fit_rows = load_fit_summary(fit_path)
    fit_map = {(r["lang"], r["kind"]): r for r in fit_rows}

    for lang in args.langs.split(","):
        lang = lang.strip()
        real_fit = fit_map.get((lang, "real"))
        if real_fit is None:
            print(f"[WARN] no fit for {lang}")
            continue

        fit_lo = float(real_fit["fit_lo"])
        fit_hi = float(real_fit["fit_hi"])
        gamma = float(real_fit["gamma"])
        C = float(real_fit["C"])
        r2 = float(real_fit["r2"])

        # margins in log space
        lx_lo = np.log10(fit_lo) - 0.1
        lx_hi = np.log10(fit_hi) + 0.1
        x_lo = 10 ** lx_lo
        x_hi = 10 ** lx_hi

        fig, ax = plt.subplots(figsize=(7.5, 5.2))

        # Real
        real_path = in_dir / f"degree_hist_{lang}.csv"
        if not real_path.exists():
            print(f"[WARN] missing {real_path}")
            continue
        ks_r, pk_r = load_hist(real_path)
        ks_rlb, pk_rlb = logbin(ks_r, pk_r, n_bins=args.n_bins)
        ax.plot(ks_rlb, pk_rlb, "o-", color="#1f77b4", markersize=7,
                linewidth=1.8, label="Real (log-bin)", zorder=4)

        # BA
        ba_path = ba_dir / f"degree_hist_ba_{lang}.csv"
        if ba_path.exists():
            ks_ba, pk_ba = load_hist(ba_path)
            ks_ba_lb, pk_ba_lb = logbin(ks_ba, pk_ba, n_bins=args.n_bins)
            ax.plot(ks_ba_lb, pk_ba_lb, "^--", color="#d62728", markersize=5,
                    linewidth=1.2, alpha=0.75, label="BA", zorder=3)
            ba_fit = fit_map.get((lang, "ba"))
            if ba_fit:
                kl = np.logspace(np.log10(float(ba_fit["fit_lo"])),
                                 np.log10(float(ba_fit["fit_hi"])), 80)
                ax.plot(kl, float(ba_fit["C"]) * kl ** (-float(ba_fit["gamma"])),
                        ":", color="#d62728", linewidth=1.4,
                        label=f"BA fit: γ={float(ba_fit['gamma']):.2f}", zorder=2)

        # DM
        dm_path = dm_dir / f"degree_hist_dm_{lang}.csv"
        if dm_path.exists():
            ks_dm, pk_dm = load_hist(dm_path)
            ks_dm_lb, pk_dm_lb = logbin(ks_dm, pk_dm, n_bins=args.n_bins)
            ax.plot(ks_dm_lb, pk_dm_lb, "s--", color="#2ca02c", markersize=5,
                    linewidth=1.2, alpha=0.75, label="DM", zorder=3)
            dm_fit = fit_map.get((lang, "dm"))
            if dm_fit:
                kl = np.logspace(np.log10(float(dm_fit["fit_lo"])),
                                 np.log10(float(dm_fit["fit_hi"])), 80)
                ax.plot(kl, float(dm_fit["C"]) * kl ** (-float(dm_fit["gamma"])),
                        ":", color="#2ca02c", linewidth=1.4,
                        label=f"DM fit: γ={float(dm_fit['gamma']):.2f}", zorder=2)

        # Real fit line
        kl = np.logspace(np.log10(fit_lo), np.log10(fit_hi), 100)
        ax.plot(kl, C * kl ** (-gamma), "--", color="#1f77b4", linewidth=2.2,
                label=f"Real fit: γ={gamma:.2f}, R²={r2:.4f}", zorder=5)

        # Crop axes to linear region
        ax.set_xlim(x_lo, x_hi)

        # y limits from real data in fit range
        in_range = (ks_rlb >= x_lo) & (ks_rlb <= x_hi)
        if in_range.sum() > 0:
            y_vals = pk_rlb[in_range]
            y_lo = y_vals.min() / 5
            y_hi = y_vals.max() * 5
            ax.set_ylim(y_lo, y_hi)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k$ (degree)", fontsize=12)
        ax.set_ylabel(r"$P(k)$", fontsize=12)
        ax.set_title(f"Degree distribution — linear region only — {lang.upper()}\n"
                     f"fit range $k \\in [{fit_lo:.0f}, {fit_hi:.0f}]$",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        out_path = out_dir / f"degree_cropped_{lang}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[OK] {out_path}")


def crop_rankfreq_plots(args):
    in_dir = Path(args.rf_csv_dir)
    out_dir = Path(args.out_dir) / "rankfreq_cropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_path = Path(args.out_dir) / "rankfreq_logbin_fit_improved.csv"
    fit_rows = load_fit_summary(fit_path)
    fit_map = {(r["lang"], r["mode"]): r for r in fit_rows}

    for lang in args.langs.split(","):
        lang = lang.strip()
        for mode in ["words", "words_plus_punct"]:
            csv_path = in_dir / f"rankfreq_{mode}_{lang}.csv"
            if not csv_path.exists():
                print(f"[WARN] missing {csv_path}")
                continue
            fit = fit_map.get((lang, mode))
            if fit is None:
                print(f"[WARN] no fit for {lang} {mode}")
                continue

            ks, pk = load_hist(csv_path)
            ks_lb, pk_lb = logbin(ks, pk, n_bins=args.n_bins)

            fit_lo = float(fit["fit_lo"])
            fit_hi = float(fit["fit_hi"])
            alpha = float(fit["alpha"])
            C = float(fit["C"])
            r2 = float(fit["r2"])

            lx_lo = np.log10(fit_lo) - 0.1
            lx_hi = np.log10(fit_hi) + 0.1
            x_lo = 10 ** lx_lo
            x_hi = 10 ** lx_hi

            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            color = LANG_COLORS.get(lang, "#1f77b4")
            ax.plot(ks_lb, pk_lb, "o-", color=color, markersize=6,
                    linewidth=1.6, label="log-bin", zorder=3)

            kl = np.logspace(np.log10(fit_lo), np.log10(fit_hi), 100)
            ax.plot(kl, C * kl ** (-alpha), "--", color="black", linewidth=2.0,
                    label=f"fit: α={alpha:.2f}, R²={r2:.4f}", zorder=4)

            ax.set_xlim(x_lo, x_hi)
            in_range = (ks_lb >= x_lo) & (ks_lb <= x_hi)
            if in_range.sum() > 0:
                y_vals = pk_lb[in_range]
                ax.set_ylim(y_vals.min() / 5, y_vals.max() * 5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("rank $r$", fontsize=12)
            ax.set_ylabel(r"$P(r)$", fontsize=12)
            pretty_mode = "words" if mode == "words" else "words + punct"
            ax.set_title(f"Rank-frequency — linear region only — {lang.upper()} ({pretty_mode})\n"
                         f"fit range $r \\in [{fit_lo:.0f}, {fit_hi:.0f}]$",
                         fontsize=12)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
            ax.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            out_path = out_dir / f"rankfreq_cropped_{lang}_{mode}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"[OK] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg-hist-dir", default="network_out_core/degree_histograms")
    ap.add_argument("--ba-hist-dir", default="ba_out_core/degree_histograms")
    ap.add_argument("--dm-hist-dir", default="dm_out_core/degree_histograms")
    ap.add_argument("--rf-csv-dir", default="rankfreq_out_core/csv")
    ap.add_argument("--out-dir", default="improved_plots")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--n-bins", type=int, default=50)
    ap.add_argument("--skip-rankfreq", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("DEGREE cropped plots")
    print("=" * 60)
    crop_degree_plots(args)

    if not args.skip_rankfreq:
        print("\n" + "=" * 60)
        print("RANK-FREQUENCY cropped plots")
        print("=" * 60)
        crop_rankfreq_plots(args)


if __name__ == "__main__":
    main()
