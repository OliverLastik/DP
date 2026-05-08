#!/usr/bin/env python3
"""
Log-binned degree distribution P(k) pre word-adjacency siete.

Nacita existujuce degree histogramy a vytvori log-binned verzie
s power-law fitom na linearnej casti.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_degree_hist(csv_path: Path):
    """Nacita degree histogram (k, count alebo k, pk_mean)."""
    ks, vals = [], []
    is_pk = False
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        k_col = cols.get("k") or cols.get("degree")
        # skus najprv pk_mean (BA output), potom count
        if "pk_mean" in cols:
            v_col = cols["pk_mean"]
            is_pk = True
        elif "pk" in cols:
            v_col = cols["pk"]
            is_pk = True
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
    if is_pk:
        pk = vals / vals.sum()  # re-normalize
        counts = vals
    else:
        counts = vals
        pk = counts / counts.sum()
    return ks, pk, counts


def logbin_pk(ks, pk, n_bins=40):
    """Log-bin degree distribution."""
    mask = (ks > 0) & (pk > 0)
    k = ks[mask]
    p = pk[mask]

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


def fit_powerlaw_loglog(x, y, lo, hi):
    """Linearny fit v log-log priestore na [lo, hi]."""
    mask = (x >= lo) & (x <= hi) & (y > 0)
    if mask.sum() < 3:
        return None, None
    lx = np.log10(x[mask])
    ly = np.log10(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    gamma = -coeffs[0]
    C = 10 ** coeffs[1]
    return C, gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="network_out_core/degree_histograms")
    ap.add_argument("--ba-dir", default="ba_out_core/degree_histograms")
    ap.add_argument("--out-dir", default="network_out_core/logbin_degree")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--fit-lo", type=int, default=5, help="Min k pre fit")
    ap.add_argument("--fit-hi", type=int, default=500, help="Max k pre fit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    colors = {"en": "C0", "de": "C1", "nl": "C2"}

    fit_rows = []

    # ── overlay plot vsetkych jazykov ──
    fig_overlay, ax_overlay = plt.subplots(figsize=(8, 5.5))

    for lang in langs:
        color = colors.get(lang, "C3")

        # Real network
        real_path = Path(args.real_dir) / f"degree_hist_{lang}.csv"
        if not real_path.exists():
            print(f"[WARN] missing {real_path}")
            continue

        ks_raw, pk_raw, counts_raw = load_degree_hist(real_path)
        ks_lb, pk_lb = logbin_pk(ks_raw, pk_raw, n_bins=args.n_bins)

        # BA baseline
        ba_path = Path(args.ba_dir) / f"degree_hist_ba_{lang}.csv"
        ba_ks_lb, ba_pk_lb = None, None
        if ba_path.exists():
            ba_ks, ba_pk, _ = load_degree_hist(ba_path)
            if len(ba_ks) > 0:
                ba_ks_lb, ba_pk_lb = logbin_pk(ba_ks, ba_pk, n_bins=args.n_bins)

        # ── Single language plot ──
        fig, ax = plt.subplots(figsize=(7, 5))

        # raw
        ax.plot(ks_raw, pk_raw, ".", color="0.6", markersize=2, alpha=0.3, label="raw P(k)")
        # log-bin real
        ax.plot(ks_lb, pk_lb, "o-", color="C0", markersize=5, linewidth=1.5, label="real (log-bin)")

        # log-bin BA
        if ba_ks_lb is not None:
            ax.plot(ba_ks_lb, ba_pk_lb, "s--", color="C3", markersize=4, linewidth=1.2,
                    alpha=0.7, label="BA (log-bin)")

        # power-law fit (real)
        C, gamma = fit_powerlaw_loglog(ks_lb, pk_lb, args.fit_lo, args.fit_hi)
        if C is not None:
            k_line = np.logspace(0, np.log10(max(ks_raw)), 200)
            ax.plot(k_line, C * k_line ** (-gamma), ":", color="C0", linewidth=1.2,
                    label=f"fit real: gamma={gamma:.2f}")
            fit_rows.append({"language": lang, "kind": "real", "gamma": gamma, "C": C})

        # power-law fit (BA)
        if ba_ks_lb is not None:
            C_ba, gamma_ba = fit_powerlaw_loglog(ba_ks_lb, ba_pk_lb, args.fit_lo, args.fit_hi)
            if C_ba is not None:
                ax.plot(k_line, C_ba * k_line ** (-gamma_ba), ":", color="C3", linewidth=1.0,
                        label=f"fit BA: gamma={gamma_ba:.2f}")
                fit_rows.append({"language": lang, "kind": "ba", "gamma": gamma_ba, "C": C_ba})

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k (degree)", fontsize=11)
        ax.set_ylabel("P(k)", fontsize=11)
        ax.set_title(f"Degree distribution (log-binned) -- {lang.upper()}", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"degree_logbin_{lang}.png", dpi=300)
        plt.close(fig)

        # ── overlay ──
        ax_overlay.plot(ks_lb, pk_lb, "o-", color=color, markersize=4, linewidth=1.2,
                        label=f"{lang.upper()} (real)")
        if ba_ks_lb is not None:
            ax_overlay.plot(ba_ks_lb, ba_pk_lb, "x--", color=color, markersize=4,
                            linewidth=0.8, alpha=0.5, label=f"{lang.upper()} (BA)")

        print(f"[OK] {lang}: saved degree_logbin_{lang}.png")

    ax_overlay.set_xscale("log")
    ax_overlay.set_yscale("log")
    ax_overlay.set_xlabel("k (degree)", fontsize=11)
    ax_overlay.set_ylabel("P(k)", fontsize=11)
    ax_overlay.set_title("Degree distributions (log-binned): Real vs BA", fontsize=12)
    ax_overlay.legend(fontsize=7, loc="upper right")
    fig_overlay.tight_layout()
    fig_overlay.savefig(out_dir / "plots" / "degree_logbin_overlay.png", dpi=300)
    plt.close(fig_overlay)

    # summary CSV
    if fit_rows:
        summary = out_dir / "degree_logbin_fit_summary.csv"
        with summary.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["language", "kind", "gamma", "C"])
            w.writeheader()
            w.writerows(fit_rows)
        print(f"\n[OK] Fit summary: {summary}")

    print("Hotovo!")


if __name__ == "__main__":
    main()
