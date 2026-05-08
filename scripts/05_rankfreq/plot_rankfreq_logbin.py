#!/usr/bin/env python3
"""
Logaritmicky binovanane rank-frequency ploty.

Log-binning rozdeluje x-os (rank) do binov s exponencialne rastucou sirkou,
co vyhladzuje sumovy chvost a umoznuje cistejsi power-law fit.

Vystupy:
  - Log-bin CSV pre kazdy jazyk a rezim
  - Log-bin ploty (words only vs words+punct overlay)
  - Power-law fit len na linearnu cast (orezany zaciatok a chvost)
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}
PUNCT_LABELS = {".": "#dot", ",": "#com", "!": "#ex", "?": "#qu", ";": "#scol", ":": "#col"}


def load_rankfreq(csv_path: Path):
    """Nacita rank-frequency CSV."""
    ranks, freqs, tokens = [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["rank"]))
            freqs.append(float(row["freq"]))
            tokens.append(row["token"])
    return np.array(ranks, dtype=float), np.array(freqs), tokens


def logbin(ranks, freqs, n_bins=50, base=None):
    """
    Logaritmicky binuje rank-frequency data.

    Rozdeluje log(rank) na rovnomerne intervaly, v kazdom bine
    priemerne rank a priemerne freq.
    """
    mask = (ranks > 0) & (freqs > 0)
    r = ranks[mask]
    f = freqs[mask]

    log_r = np.log10(r)
    r_min, r_max = log_r.min(), log_r.max()

    if base is None:
        edges = np.linspace(r_min, r_max, n_bins + 1)
    else:
        edges = np.arange(r_min, r_max + base, base)

    bin_ranks = []
    bin_freqs = []

    for i in range(len(edges) - 1):
        in_bin = (log_r >= edges[i]) & (log_r < edges[i + 1])
        if in_bin.sum() == 0:
            continue
        # geometricky priemer ranku, aritmeticky priemer frekvencie
        bin_ranks.append(10 ** np.mean(log_r[in_bin]))
        bin_freqs.append(np.mean(f[in_bin]))

    return np.array(bin_ranks), np.array(bin_freqs)


def pure_powerlaw(R, C, alpha):
    """P(R) = C * R^(-alpha)"""
    return C * R ** (-alpha)


def fit_powerlaw_on_range(ranks, freqs, lo, hi):
    """Fituje cisty power law na [lo, hi] range."""
    mask = (ranks >= lo) & (ranks <= hi) & (freqs > 0)
    r_fit = np.log10(ranks[mask])
    f_fit = np.log10(freqs[mask])

    if len(r_fit) < 5:
        return None, None

    # linearny fit v log-log priestore: log(f) = log(C) - alpha * log(r)
    coeffs = np.polyfit(r_fit, f_fit, 1)
    alpha = -coeffs[0]
    C = 10 ** coeffs[1]
    return C, alpha


def plot_logbin_overlay(langs_data, title, out_png, fit_lo=10, fit_hi=5000):
    """
    Overlay log-bin kriviek pre viacero jazykov s power-law fitom.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"en": "C0", "de": "C1", "nl": "C2"}

    for lang, (ranks_raw, freqs_raw, ranks_lb, freqs_lb, tokens) in langs_data.items():
        color = colors.get(lang, "C3")

        # raw data (priehladne)
        ax.plot(ranks_raw, freqs_raw, ".", color=color, markersize=1.5,
                alpha=0.15, zorder=1)

        # log-binned data
        ax.plot(ranks_lb, freqs_lb, "o-", color=color, markersize=4,
                linewidth=1.2, label=f"{lang.upper()} (log-bin)", zorder=3)

        # power-law fit na linearnej casti
        C, alpha = fit_powerlaw_on_range(ranks_lb, freqs_lb, fit_lo, fit_hi)
        if C is not None:
            r_line = np.logspace(0, np.log10(max(ranks_raw)), 200)
            ax.plot(r_line, pure_powerlaw(r_line, C, alpha), "--", color=color,
                    linewidth=1.0, alpha=0.7,
                    label=f"{lang.upper()} fit: alpha={alpha:.2f}", zorder=2)

    ax.axvline(fit_lo, color="0.7", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axvline(fit_hi, color="0.7", linestyle=":", linewidth=0.7, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank R", fontsize=11)
    ax.set_ylabel("P(R)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_single_lang_logbin(ranks_raw, freqs_raw, ranks_lb, freqs_lb, tokens,
                            lang, mode, out_png, fit_lo=10, fit_hi=5000):
    """
    Log-bin plot pre jeden jazyk s anotovanou interpunkciou.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # raw data
    ax.plot(ranks_raw, freqs_raw, ".", color="0.6", markersize=1.5, alpha=0.3, zorder=1)

    # log-binned
    ax.plot(ranks_lb, freqs_lb, "o-", color="C0", markersize=5,
            linewidth=1.5, label="log-binned", zorder=3)

    # power-law fit
    C, alpha = fit_powerlaw_on_range(ranks_lb, freqs_lb, fit_lo, fit_hi)
    if C is not None:
        r_line = np.logspace(0, np.log10(max(ranks_raw)), 200)
        ax.plot(r_line, pure_powerlaw(r_line, C, alpha), "--", color="C3",
                linewidth=1.5, label=f"power-law fit: alpha={alpha:.2f}", zorder=2)

    # anotacia interpunkcie (ak mode = words_plus_punct)
    if "punct" in mode:
        for i, tok in enumerate(tokens):
            if tok in PUNCT_TOKENS:
                label = PUNCT_LABELS.get(tok, tok)
                ax.plot(ranks_raw[i], freqs_raw[i], "D", color="C1", markersize=8,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)
                ax.annotate(label, (ranks_raw[i], freqs_raw[i]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, fontweight="bold", color="C1")

    # top slova
    if "punct" in mode:
        word_count = 0
        for i in range(min(30, len(tokens))):
            if tokens[i] not in PUNCT_TOKENS and word_count < 6:
                ax.annotate(tokens[i], (ranks_raw[i], freqs_raw[i]),
                            textcoords="offset points", xytext=(6, -6),
                            fontsize=7, color="0.3", fontstyle="italic")
                word_count += 1

    ax.axvline(fit_lo, color="0.7", linestyle=":", linewidth=0.7)
    ax.axvline(fit_hi, color="0.7", linestyle=":", linewidth=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank R", fontsize=11)
    ax.set_ylabel("P(R)", fontsize=11)
    mode_label = "words + punct" if "punct" in mode else "words only"
    ax.set_title(f"Rank-frequency (log-binned) -- {lang.upper()} ({mode_label})", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="rankfreq_out_core/csv")
    ap.add_argument("--out-dir", default="rankfreq_out_core/logbin")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--n-bins", type=int, default=50, help="Pocet log-binov")
    ap.add_argument("--fit-lo", type=int, default=10, help="Spodna hranica pre power-law fit")
    ap.add_argument("--fit-hi", type=int, default=5000, help="Horna hranica pre power-law fit")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    modes = [
        ("rankfreq_words", "words"),
        ("rankfreq_words_plus_punct", "words_plus_punct"),
    ]

    fit_results = []

    for mode_prefix, mode_name in modes:
        overlay_data = {}

        for lang in langs:
            csv_path = csv_dir / f"{mode_prefix}_{lang}.csv"
            if not csv_path.exists():
                print(f"[WARN] missing {csv_path}")
                continue

            ranks_raw, freqs_raw, tokens = load_rankfreq(csv_path)

            # log-binning
            ranks_lb, freqs_lb = logbin(ranks_raw, freqs_raw, n_bins=args.n_bins)

            # save log-bin CSV
            lb_csv = out_dir / "csv" / f"logbin_{mode_name}_{lang}.csv"
            with lb_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bin_rank", "bin_freq", "log_rank", "log_freq"])
                for r, freq in zip(ranks_lb, freqs_lb):
                    w.writerow([r, freq, math.log10(r), math.log10(freq) if freq > 0 else ""])

            # power-law fit
            C, alpha = fit_powerlaw_on_range(ranks_lb, freqs_lb, args.fit_lo, args.fit_hi)
            if C is not None:
                fit_results.append({
                    "language": lang, "mode": mode_name,
                    "alpha": alpha, "C": C,
                    "fit_lo": args.fit_lo, "fit_hi": args.fit_hi,
                })
                print(f"[OK] {lang}/{mode_name}: alpha={alpha:.3f}")

            # single lang plot
            plot_single_lang_logbin(
                ranks_raw, freqs_raw, ranks_lb, freqs_lb, tokens,
                lang, mode_name,
                out_dir / "plots" / f"logbin_{mode_name}_{lang}.png",
                fit_lo=args.fit_lo, fit_hi=args.fit_hi,
            )

            overlay_data[lang] = (ranks_raw, freqs_raw, ranks_lb, freqs_lb, tokens)

        # overlay plot pre vsetky jazyky
        if overlay_data:
            mode_label = "words + punct" if "punct" in mode_name else "words only"
            plot_logbin_overlay(
                overlay_data,
                f"Rank-frequency log-binned ({mode_label})",
                out_dir / "plots" / f"logbin_overlay_{mode_name}.png",
                fit_lo=args.fit_lo, fit_hi=args.fit_hi,
            )

    # summary CSV
    if fit_results:
        summary_csv = out_dir / "logbin_powerlaw_fit_summary.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["language", "mode", "alpha", "C", "fit_lo", "fit_hi"])
            w.writeheader()
            w.writerows(fit_results)
        print(f"\n[OK] Fit summary: {summary_csv}")

    print("Hotovo!")


if __name__ == "__main__":
    main()
