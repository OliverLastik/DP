#!/usr/bin/env python3
"""
Vylepseny logbin fit pre rank-frequency a degree distribucii.

Klucove vylepsenia oproti povodnym skriptom:
  - Auto-detekcia linearnej oblasti v log-log priestore
  - R^2 goodness-of-fit metrika
  - Orezanie zaiatku (low-rank bump) a chvosta (finite-size cutoff)
  - Vizualizacia fit regionu (shadovana oblast)
  - Overlay words vs words+punct s fitom
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Rank-frequency loading ──

def load_rankfreq(csv_path: Path):
    ranks, freqs, tokens = [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["rank"]))
            freqs.append(float(row["freq"]))
            tokens.append(row["token"])
    return np.array(ranks, dtype=float), np.array(freqs), tokens


def load_degree_hist(csv_path: Path):
    ks, vals = [], []
    is_pk = False
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        k_col = cols.get("k") or cols.get("degree")
        if "pk_mean" in cols:
            v_col = cols["pk_mean"]; is_pk = True
        elif "pk" in cols:
            v_col = cols["pk"]; is_pk = True
        else:
            v_col = cols.get("count") or cols.get("n") or cols.get("freq")
        for row in reader:
            try:
                k = int(float(row[k_col]))
                v = float(row[v_col])
            except Exception:
                continue
            if k > 0 and v > 0:
                ks.append(k)
                vals.append(v)
    ks = np.array(ks, dtype=float)
    vals = np.array(vals, dtype=float)
    if not is_pk:
        vals = vals / vals.sum()
    else:
        vals = vals / vals.sum()
    return ks, vals


# ── Log-binning ──

def logbin(x, y, n_bins=50):
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    log_x = np.log10(x)
    edges = np.linspace(log_x.min(), log_x.max(), n_bins + 1)
    bx, by = [], []
    for i in range(len(edges) - 1):
        m = (log_x >= edges[i]) & (log_x < edges[i + 1])
        if m.sum() > 0:
            bx.append(10 ** np.mean(log_x[m]))
            by.append(np.mean(y[m]))
    return np.array(bx), np.array(by)


# ── Auto linear region detection ──

def find_linear_region(log_x, log_y, min_points=5, max_residual=0.05):
    """
    Najde najdlhsi usek kde linear fit ma male rezidua.
    Pouziva sliding window s rastucou dlzkou.
    Vracia (best_lo_idx, best_hi_idx, best_r2).
    """
    n = len(log_x)
    if n < min_points:
        return 0, n - 1, 0.0

    best_lo, best_hi, best_r2, best_len = 0, n - 1, 0.0, 0

    for lo in range(n - min_points + 1):
        for hi in range(lo + min_points - 1, n):
            seg_x = log_x[lo:hi + 1]
            seg_y = log_y[lo:hi + 1]

            coeffs = np.polyfit(seg_x, seg_y, 1)
            predicted = np.polyval(coeffs, seg_x)
            ss_res = np.sum((seg_y - predicted) ** 2)
            ss_tot = np.sum((seg_y - seg_y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            seg_len = hi - lo + 1
            # prioritizuj dlhsie segmenty s dobrym fitom
            if r2 >= 0.995 and seg_len > best_len:
                best_lo, best_hi, best_r2, best_len = lo, hi, r2, seg_len
            elif r2 > best_r2 and seg_len >= min_points and best_len < min_points:
                best_lo, best_hi, best_r2, best_len = lo, hi, r2, seg_len

    return best_lo, best_hi, best_r2


def fit_powerlaw_auto(x_lb, y_lb):
    """
    Auto-fit power law na logbinned data.
    Vracia (alpha, C, r2, fit_lo, fit_hi) alebo None.
    """
    mask = (x_lb > 0) & (y_lb > 0)
    x, y = x_lb[mask], y_lb[mask]
    if len(x) < 5:
        return None

    log_x = np.log10(x)
    log_y = np.log10(y)

    lo_idx, hi_idx, r2 = find_linear_region(log_x, log_y)

    seg_x = log_x[lo_idx:hi_idx + 1]
    seg_y = log_y[lo_idx:hi_idx + 1]
    coeffs = np.polyfit(seg_x, seg_y, 1)
    alpha = -coeffs[0]
    C = 10 ** coeffs[1]

    return {
        "alpha": alpha, "C": C, "r2": r2,
        "fit_lo": float(x[lo_idx]), "fit_hi": float(x[hi_idx]),
        "lo_idx": lo_idx, "hi_idx": hi_idx,
    }


def fit_powerlaw_range(x_lb, y_lb, lo, hi):
    """Fit na fixnom rozsahu."""
    mask = (x_lb >= lo) & (x_lb <= hi) & (y_lb > 0)
    if mask.sum() < 3:
        return None
    lx = np.log10(x_lb[mask])
    ly = np.log10(y_lb[mask])
    coeffs = np.polyfit(lx, ly, 1)
    alpha = -coeffs[0]
    C = 10 ** coeffs[1]

    predicted = np.polyval(coeffs, lx)
    ss_res = np.sum((ly - predicted) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {"alpha": alpha, "C": C, "r2": r2, "fit_lo": lo, "fit_hi": hi}


# ── Plotting ──

PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}
PUNCT_LABELS = {".": ".", ",": ",", "!": "!", "?": "?", ";": ";", ":": ":"}
LANG_COLORS = {"en": "#1f77b4", "de": "#ff7f0e", "nl": "#2ca02c"}


def plot_rankfreq_improved(lang, mode, ranks_raw, freqs_raw, tokens,
                           ranks_lb, freqs_lb, fit_info, out_dir):
    """Vylepseny rank-freq plot s auto-fit oblastou."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # raw
    ax.plot(ranks_raw, freqs_raw, ".", color="0.7", markersize=1.5, alpha=0.2, zorder=1)
    # logbin
    ax.plot(ranks_lb, freqs_lb, "o-", color="#1f77b4", markersize=5, linewidth=1.5,
            label="log-binned", zorder=3)

    if fit_info:
        # shade fit region
        ax.axvspan(fit_info["fit_lo"], fit_info["fit_hi"],
                   color="#1f77b4", alpha=0.08, zorder=0)
        ax.axvline(fit_info["fit_lo"], color="#1f77b4", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(fit_info["fit_hi"], color="#1f77b4", linestyle=":", linewidth=0.8, alpha=0.5)

        r_line = np.logspace(np.log10(fit_info["fit_lo"]),
                             np.log10(fit_info["fit_hi"]), 100)
        ax.plot(r_line, fit_info["C"] * r_line ** (-fit_info["alpha"]),
                "--", color="red", linewidth=2.0,
                label=f"fit: $\\alpha$={fit_info['alpha']:.3f}, $R^2$={fit_info['r2']:.4f}",
                zorder=4)

    # annotate punctuation
    if "punct" in mode:
        for i, tok in enumerate(tokens):
            if tok in PUNCT_TOKENS:
                ax.plot(ranks_raw[i], freqs_raw[i], "D", color="#ff7f0e", markersize=8,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)
                ax.annotate(PUNCT_LABELS.get(tok, tok),
                            (ranks_raw[i], freqs_raw[i]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=9, fontweight="bold", color="#ff7f0e")

    mode_label = "words + punct" if "punct" in mode else "words only"
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rank $R$", fontsize=12)
    ax.set_ylabel("$P(R)$", fontsize=12)
    ax.set_title(f"Rank-frequency (log-binned) — {lang.upper()} ({mode_label})", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"logbin_improved_{mode}_{lang}.png", dpi=300)
    plt.close(fig)


def plot_degree_improved(lang, ks_lb, pk_lb, fit_info, out_dir,
                         ba_ks_lb=None, ba_pk_lb=None, dm_ks_lb=None, dm_pk_lb=None,
                         ba_fit=None, dm_fit=None):
    """Vylepseny degree distribution plot."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # real
    ax.plot(ks_lb, pk_lb, "o-", color="#1f77b4", markersize=5, linewidth=1.5,
            label="Real", zorder=3)

    if fit_info:
        ax.axvspan(fit_info["fit_lo"], fit_info["fit_hi"],
                   color="#1f77b4", alpha=0.08, zorder=0)
        k_line = np.logspace(np.log10(fit_info["fit_lo"]),
                             np.log10(fit_info["fit_hi"]), 100)
        ax.plot(k_line, fit_info["C"] * k_line ** (-fit_info["alpha"]),
                "--", color="#1f77b4", linewidth=2.0,
                label=f"Real fit: $\\gamma$={fit_info['alpha']:.2f}, $R^2$={fit_info['r2']:.4f}",
                zorder=4)

    # BA
    if ba_ks_lb is not None:
        ax.plot(ba_ks_lb, ba_pk_lb, "^--", color="#d62728", markersize=4, linewidth=1.0,
                alpha=0.7, label="BA", zorder=2)
        if ba_fit:
            k_line = np.logspace(np.log10(ba_fit["fit_lo"]),
                                 np.log10(ba_fit["fit_hi"]), 100)
            ax.plot(k_line, ba_fit["C"] * k_line ** (-ba_fit["alpha"]),
                    ":", color="#d62728", linewidth=1.5,
                    label=f"BA fit: $\\gamma$={ba_fit['alpha']:.2f}")

    # DM
    if dm_ks_lb is not None:
        ax.plot(dm_ks_lb, dm_pk_lb, "s--", color="#2ca02c", markersize=4, linewidth=1.0,
                alpha=0.7, label="DM", zorder=2)
        if dm_fit:
            k_line = np.logspace(np.log10(dm_fit["fit_lo"]),
                                 np.log10(dm_fit["fit_hi"]), 100)
            ax.plot(k_line, dm_fit["C"] * k_line ** (-dm_fit["alpha"]),
                    ":", color="#2ca02c", linewidth=1.5,
                    label=f"DM fit: $\\gamma$={dm_fit['alpha']:.2f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$k$ (degree)", fontsize=12)
    ax.set_ylabel("$P(k)$", fontsize=12)
    ax.set_title(f"Degree distribution — {lang.upper()}", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"degree_improved_{lang}.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-csv-dir", default="rankfreq_out_core/csv")
    ap.add_argument("--deg-hist-dir", default="network_out_core/degree_histograms")
    ap.add_argument("--ba-hist-dir", default="ba_out_core/degree_histograms")
    ap.add_argument("--dm-hist-dir", default="dm_out_core/degree_histograms")
    ap.add_argument("--out-dir", default="improved_plots")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--n-bins", type=int, default=50)
    # Manual override for fit range (0 = auto)
    ap.add_argument("--rf-fit-lo", type=int, default=0)
    ap.add_argument("--rf-fit-hi", type=int, default=0)
    ap.add_argument("--deg-fit-lo", type=int, default=0)
    ap.add_argument("--deg-fit-hi", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rankfreq").mkdir(parents=True, exist_ok=True)
    (out_dir / "degree").mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    rf_fit_rows = []
    deg_fit_rows = []

    # ── RANK-FREQUENCY ──
    print("=" * 60)
    print("  RANK-FREQUENCY (log-bin + auto-fit)")
    print("=" * 60)

    modes = [
        ("rankfreq_words", "words"),
        ("rankfreq_words_plus_punct", "words_plus_punct"),
    ]

    # Data for overlay
    overlay_data = {mode: {} for _, mode in modes}

    for mode_prefix, mode_name in modes:
        for lang in langs:
            csv_path = Path(args.rf_csv_dir) / f"{mode_prefix}_{lang}.csv"
            if not csv_path.exists():
                print(f"[WARN] missing {csv_path}")
                continue

            ranks_raw, freqs_raw, tokens = load_rankfreq(csv_path)
            ranks_lb, freqs_lb = logbin(ranks_raw, freqs_raw, n_bins=args.n_bins)

            if args.rf_fit_lo > 0 and args.rf_fit_hi > 0:
                fit_info = fit_powerlaw_range(ranks_lb, freqs_lb,
                                             args.rf_fit_lo, args.rf_fit_hi)
            else:
                fit_info = fit_powerlaw_auto(ranks_lb, freqs_lb)

            if fit_info:
                rf_fit_rows.append({
                    "lang": lang, "mode": mode_name,
                    "alpha": fit_info["alpha"], "C": fit_info["C"],
                    "r2": fit_info["r2"],
                    "fit_lo": fit_info["fit_lo"], "fit_hi": fit_info["fit_hi"],
                })
                print(f"  {lang}/{mode_name}: alpha={fit_info['alpha']:.3f}, "
                      f"R2={fit_info['r2']:.4f}, range=[{fit_info['fit_lo']:.0f}, {fit_info['fit_hi']:.0f}]")

            plot_rankfreq_improved(lang, mode_name, ranks_raw, freqs_raw, tokens,
                                   ranks_lb, freqs_lb, fit_info,
                                   out_dir / "rankfreq")

            overlay_data[mode_name][lang] = (ranks_lb, freqs_lb, fit_info)

    # Overlay: words vs words+punct per language
    for lang in langs:
        if lang in overlay_data["words"] and lang in overlay_data["words_plus_punct"]:
            fig, ax = plt.subplots(figsize=(8, 5.5))

            for mode, color, marker in [("words", "#1f77b4", "o"),
                                         ("words_plus_punct", "#ff7f0e", "s")]:
                r_lb, f_lb, fit = overlay_data[mode][lang]
                label = "words" if mode == "words" else "words+punct"
                ax.plot(r_lb, f_lb, f"{marker}-", color=color, markersize=4,
                        linewidth=1.2, label=label, zorder=3)

                if fit:
                    r_line = np.logspace(np.log10(fit["fit_lo"]),
                                         np.log10(fit["fit_hi"]), 100)
                    ax.plot(r_line, fit["C"] * r_line ** (-fit["alpha"]),
                            "--", color=color, linewidth=1.8,
                            label=f"{label} fit: $\\alpha$={fit['alpha']:.3f}",
                            zorder=4)

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("Rank $R$", fontsize=12)
            ax.set_ylabel("$P(R)$", fontsize=12)
            ax.set_title(f"Rank-frequency: words vs words+punct — {lang.upper()}", fontsize=13)
            ax.legend(fontsize=9, loc="lower left")
            fig.tight_layout()
            fig.savefig(out_dir / "rankfreq" / f"overlay_w_vs_wp_{lang}.png", dpi=300)
            plt.close(fig)

    # ── DEGREE DISTRIBUTION ──
    print("\n" + "=" * 60)
    print("  DEGREE DISTRIBUTION (log-bin + auto-fit)")
    print("=" * 60)

    for lang in langs:
        # Real
        real_path = Path(args.deg_hist_dir) / f"degree_hist_{lang}.csv"
        if not real_path.exists():
            print(f"[WARN] missing {real_path}")
            continue
        ks_raw, pk_raw = load_degree_hist(real_path)
        ks_lb, pk_lb = logbin(ks_raw, pk_raw, n_bins=args.n_bins)

        if args.deg_fit_lo > 0 and args.deg_fit_hi > 0:
            fit_info = fit_powerlaw_range(ks_lb, pk_lb,
                                          args.deg_fit_lo, args.deg_fit_hi)
        else:
            fit_info = fit_powerlaw_auto(ks_lb, pk_lb)

        if fit_info:
            deg_fit_rows.append({
                "lang": lang, "kind": "real",
                "gamma": fit_info["alpha"], "C": fit_info["C"],
                "r2": fit_info["r2"],
                "fit_lo": fit_info["fit_lo"], "fit_hi": fit_info["fit_hi"],
            })
            print(f"  {lang} real: gamma={fit_info['alpha']:.3f}, "
                  f"R2={fit_info['r2']:.4f}, range=[{fit_info['fit_lo']:.0f}, {fit_info['fit_hi']:.0f}]")

        # BA
        ba_path = Path(args.ba_hist_dir) / f"degree_hist_ba_{lang}.csv"
        ba_ks_lb, ba_pk_lb, ba_fit = None, None, None
        if ba_path.exists():
            ba_ks, ba_pk = load_degree_hist(ba_path)
            ba_ks_lb, ba_pk_lb = logbin(ba_ks, ba_pk, n_bins=args.n_bins)
            ba_fit = fit_powerlaw_auto(ba_ks_lb, ba_pk_lb)
            if ba_fit:
                deg_fit_rows.append({
                    "lang": lang, "kind": "ba",
                    "gamma": ba_fit["alpha"], "C": ba_fit["C"],
                    "r2": ba_fit["r2"],
                    "fit_lo": ba_fit["fit_lo"], "fit_hi": ba_fit["fit_hi"],
                })
                print(f"  {lang} BA:   gamma={ba_fit['alpha']:.3f}, R2={ba_fit['r2']:.4f}")

        # DM
        dm_path = Path(args.dm_hist_dir) / f"degree_hist_dm_{lang}.csv"
        dm_ks_lb, dm_pk_lb, dm_fit = None, None, None
        if dm_path.exists():
            dm_ks, dm_pk = load_degree_hist(dm_path)
            dm_ks_lb, dm_pk_lb = logbin(dm_ks, dm_pk, n_bins=args.n_bins)
            dm_fit = fit_powerlaw_auto(dm_ks_lb, dm_pk_lb)
            if dm_fit:
                deg_fit_rows.append({
                    "lang": lang, "kind": "dm",
                    "gamma": dm_fit["alpha"], "C": dm_fit["C"],
                    "r2": dm_fit["r2"],
                    "fit_lo": dm_fit["fit_lo"], "fit_hi": dm_fit["fit_hi"],
                })
                print(f"  {lang} DM:   gamma={dm_fit['alpha']:.3f}, R2={dm_fit['r2']:.4f}")

        plot_degree_improved(lang, ks_lb, pk_lb, fit_info, out_dir / "degree",
                             ba_ks_lb, ba_pk_lb, dm_ks_lb, dm_pk_lb,
                             ba_fit, dm_fit)

    # ── Save CSVs ──
    if rf_fit_rows:
        csv_path = out_dir / "rankfreq_logbin_fit_improved.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["lang", "mode", "alpha", "C", "r2",
                                              "fit_lo", "fit_hi"])
            w.writeheader()
            w.writerows(rf_fit_rows)
        print(f"\n[OK] Rankfreq fit CSV: {csv_path}")

    if deg_fit_rows:
        csv_path = out_dir / "degree_logbin_fit_improved.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["lang", "kind", "gamma", "C", "r2",
                                              "fit_lo", "fit_hi"])
            w.writeheader()
            w.writerows(deg_fit_rows)
        print(f"[OK] Degree fit CSV: {csv_path}")

    print("\n[OK] Vsetky vylepsene logbin ploty hotove.")


if __name__ == "__main__":
    main()
