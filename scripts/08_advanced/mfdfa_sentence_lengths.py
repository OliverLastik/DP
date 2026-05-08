#!/usr/bin/env python3
"""
Multifractal Detrended Fluctuation Analysis (MF-DFA) dlzok viet.
Vety su definovane ako sekvencie tokenov medzi bodkami (. ! ?).
MF-DFA skuma long-range korelacie a multifraktalitu.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SENTENCE_ENDS = {".", "!", "?"}


def extract_sentence_lengths(tokens: list[str]) -> np.ndarray:
    """Extrahuje dlzky viet (pocet tokenov medzi interpunkcnymi ukonceniami)."""
    lengths = []
    current = 0
    for tok in tokens:
        if tok in SENTENCE_ENDS:
            if current > 0:
                lengths.append(current)
            current = 0
        else:
            current += 1
    if current > 0:
        lengths.append(current)
    return np.array(lengths, dtype=float)


def dfa_fluctuation(profile: np.ndarray, scale: int, q: float = 2.0) -> float:
    """
    Spocita DFA fluktuaciu pre danu skalu.
    profile: kumulativny profil (integrovaný signál)
    scale: velkost segmentu
    q: moment (q=2 je standard DFA)
    """
    N = len(profile)
    n_seg = N // scale
    if n_seg < 1:
        return float("nan")

    F2_list = []
    for i in range(n_seg):
        seg = profile[i * scale : (i + 1) * scale]
        x = np.arange(scale)
        # linear detrend
        coeffs = np.polyfit(x, seg, 1)
        trend = np.polyval(coeffs, x)
        var = np.mean((seg - trend) ** 2)
        if var > 0:
            F2_list.append(var)

    # backwards segments
    for i in range(n_seg):
        seg = profile[N - (i + 1) * scale : N - i * scale]
        x = np.arange(scale)
        coeffs = np.polyfit(x, seg, 1)
        trend = np.polyval(coeffs, x)
        var = np.mean((seg - trend) ** 2)
        if var > 0:
            F2_list.append(var)

    if not F2_list:
        return float("nan")

    F2 = np.array(F2_list)

    if q == 0:
        return np.exp(0.5 * np.mean(np.log(F2)))
    else:
        return (np.mean(F2 ** (q / 2))) ** (1.0 / q)


def mfdfa(signal: np.ndarray, q_list: list[float],
          scales: np.ndarray = None) -> dict:
    """
    Full MF-DFA.
    Vrati dict s klucom q -> (scales, Fq) a fitovane h(q).
    """
    N = len(signal)
    if N < 50:
        return {}

    # Integrujeme signal (kumulativny profil)
    profile = np.cumsum(signal - np.mean(signal))

    if scales is None:
        s_min = 10
        s_max = N // 4
        if s_max <= s_min:
            s_max = s_min + 10
        scales = np.unique(np.geomspace(s_min, s_max, num=30).astype(int))

    result = {}
    for q in q_list:
        Fq = []
        valid_scales = []
        for s in scales:
            f = dfa_fluctuation(profile, s, q)
            if not np.isnan(f) and f > 0:
                Fq.append(f)
                valid_scales.append(s)

        if len(valid_scales) < 5:
            continue

        valid_scales = np.array(valid_scales)
        Fq = np.array(Fq)

        # Fit h(q) from log-log regression
        log_s = np.log(valid_scales)
        log_f = np.log(Fq)
        coeffs = np.polyfit(log_s, log_f, 1)
        h = coeffs[0]

        result[q] = {
            "scales": valid_scales,
            "Fq": Fq,
            "h": h,
        }

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--outdir", default="rankfreq_out_core/mfdfa")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--lowercase", action="store_true", default=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    q_list = [-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]

    summary_rows = []
    hq_rows = []

    for lang in langs:
        lang_dir = Path(args.token_dir) / lang
        if not lang_dir.exists():
            print(f"[WARN] {lang}: missing {lang_dir}")
            continue

        # Concatenate all books
        all_tokens = []
        for f in sorted(lang_dir.glob("*.txt")):
            text = f.read_text(encoding="utf-8", errors="replace")
            toks = text.split()
            if args.lowercase:
                toks = [t.lower() for t in toks]
            all_tokens.extend(toks)

        sent_lens = extract_sentence_lengths(all_tokens)
        print(f"[INFO] {lang}: {len(all_tokens)} tokens, {len(sent_lens)} sentences, "
              f"mean len={sent_lens.mean():.1f}, median={np.median(sent_lens):.0f}")

        if len(sent_lens) < 100:
            print(f"[WARN] {lang}: too few sentences")
            continue

        # Run MF-DFA
        result = mfdfa(sent_lens, q_list)
        if not result:
            print(f"[WARN] {lang}: MF-DFA failed")
            continue

        # --- Plot 1: log F(q,s) vs log s for different q ---
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.cm.coolwarm
        q_vals = sorted(result.keys())
        for i, q in enumerate(q_vals):
            r = result[q]
            color = cmap(i / max(1, len(q_vals) - 1))
            ax.plot(r["scales"], r["Fq"], "o-", color=color, markersize=3,
                    label=f"q={q:.1f}, h={r['h']:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Scale s")
        ax.set_ylabel("F(q,s)")
        ax.set_title(f"MF-DFA fluctuation function - {lang.upper()}")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / f"mfdfa_fluctuation_{lang}.png", dpi=300)
        plt.close(fig)

        # --- Plot 2: h(q) spectrum ---
        h_vals = [result[q]["h"] for q in q_vals]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(q_vals, h_vals, "o-", color="C0", markersize=6)
        ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="h=0.5 (uncorrelated)")
        ax2.set_xlabel("q")
        ax2.set_ylabel("h(q)")
        ax2.set_title(f"Generalized Hurst exponent h(q) - {lang.upper()}")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(outdir / f"mfdfa_hq_{lang}.png", dpi=300)
        plt.close(fig2)

        # --- Plot 3: Multifractal spectrum f(alpha) via Legendre transform ---
        q_arr = np.array(q_vals)
        h_arr = np.array(h_vals)
        tau = q_arr * h_arr - 1  # tau(q) = q*h(q) - 1

        # f(alpha) via numerical derivative
        # alpha = d(tau)/d(q), f = q*alpha - tau
        if len(q_arr) >= 3:
            alpha_spec = np.gradient(tau, q_arr)
            f_spec = q_arr * alpha_spec - tau

            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.plot(alpha_spec, f_spec, "o-", color="C1", markersize=5)
            ax3.set_xlabel("$\\alpha$ (singularity strength)")
            ax3.set_ylabel("$f(\\alpha)$ (singularity spectrum)")
            ax3.set_title(f"Multifractal spectrum - {lang.upper()}")
            fig3.tight_layout()
            fig3.savefig(outdir / f"mfdfa_spectrum_{lang}.png", dpi=300)
            plt.close(fig3)

            # Width of spectrum = measure of multifractality
            delta_alpha = alpha_spec.max() - alpha_spec.min()
        else:
            delta_alpha = float("nan")

        # h(2) is the standard Hurst exponent
        h2 = result.get(2, {}).get("h", float("nan"))

        summary_rows.append({
            "lang": lang,
            "n_sentences": len(sent_lens),
            "mean_sent_len": float(sent_lens.mean()),
            "h2": h2,
            "delta_alpha": delta_alpha,
        })

        # Collect per-q values for h(q) CSV
        for q in q_vals:
            hq_rows.append({"lang": lang, "q": q, "h_q": result[q]["h"]})

        print(f"  {lang}: h(2)={h2:.4f}, delta_alpha={delta_alpha:.4f}")
        print(f"[OK] {lang}: MF-DFA plots saved")

    # Save summary
    out_csv = outdir / "mfdfa_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "n_sentences", "mean_sent_len", "h2", "delta_alpha"])
        w.writeheader()
        w.writerows(summary_rows)

    # Save h(q) per lang
    hq_csv = outdir / "mfdfa_hq.csv"
    with hq_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "q", "h_q"])
        w.writeheader()
        w.writerows(hq_rows)

    print(f"\n[OK] CSV: {out_csv}")
    print(f"[OK] h(q) CSV: {hq_csv}")


if __name__ == "__main__":
    main()
