#!/usr/bin/env python3
"""
Zipf-Mandelbrot fit: P(R) = C · (R + c)^(-α)

Fituje na rank-frequency dáta pre oba režimy (words only / words+punct)
a porovnáva parameter c. Kľúčový výsledok: pridanie interpunkcie znižuje c
(bližšie k čistému Zipfovmu zákonu).

Výstupy:
  - CSV so sumárnymi parametrami (α, c) pre každý jazyk a režim
  - Zipf-Mandelbrot ploty s fitom a anotovanou interpunkciou
  - Porovnávací plot c (words vs words+punct) pre všetky jazyky
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


# ── interpunkčné tokeny pre anotáciu ──────────────────────────────
PUNCT_TOKENS = {".", ",", "!", "?", ";", ":"}
PUNCT_LABELS = {
    ".": "#dot",
    ",": "#com",
    "!": "#ex",
    "?": "#qu",
    ";": "#scol",
    ":": "#col",
}


# ── Zipf-Mandelbrot model ────────────────────────────────────────
def zipf_mandelbrot(R, C, alpha, c):
    """P(R) = C · (R + c)^(-alpha)"""
    return C * (R + c) ** (-alpha)


def fit_zm(ranks, freqs, fit_lo=10, fit_hi=10000):
    """
    Fituje Zipf-Mandelbrot na rank-frequency dáta v rozsahu [fit_lo, fit_hi].
    Vracia (C, alpha, c) a ich kovariančnú maticu.
    """
    mask = (ranks >= fit_lo) & (ranks <= fit_hi) & (freqs > 0)
    r_fit = ranks[mask]
    f_fit = freqs[mask]

    if len(r_fit) < 50:
        raise ValueError(f"Málo bodov na fit ({len(r_fit)}) v rozsahu [{fit_lo}, {fit_hi}]")

    # počiatočný odhad
    p0 = [f_fit[0] * r_fit[0], 1.0, 1.0]
    bounds = ([0, 0.5, 0], [np.inf, 3.0, 50.0])

    popt, pcov = curve_fit(
        zipf_mandelbrot, r_fit, f_fit,
        p0=p0, bounds=bounds,
        maxfev=20000,
    )
    return popt, pcov, r_fit, f_fit


def load_rankfreq(csv_path: Path):
    """Načítanie rank-frequency CSV."""
    ranks, freqs, tokens = [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["rank"]))
            freqs.append(float(row["freq"]))
            tokens.append(row["token"])
    return np.array(ranks), np.array(freqs), tokens


def find_punct_positions(ranks, tokens):
    """Nájde ranky interpunkčných značiek."""
    punct_info = []
    for i, tok in enumerate(tokens):
        if tok in PUNCT_TOKENS:
            label = PUNCT_LABELS.get(tok, tok)
            punct_info.append((ranks[i], freqs[i], label) if False else None)
    # robím to poriadne
    punct_info = []
    for i, tok in enumerate(tokens):
        if tok in PUNCT_TOKENS:
            label = PUNCT_LABELS.get(tok, tok)
            punct_info.append({"rank": ranks[i], "token": tok, "label": label, "idx": i})
    return punct_info


# ── Ploty ─────────────────────────────────────────────────────────
def plot_zipf_with_fit(ranks, freqs, tokens, popt_words, popt_wpp,
                       lang, out_png, fit_lo=10, fit_hi=10000):
    """
    Hlavný Zipf plot: words+punct dáta, Zipf-Mandelbrot fit,
    a anotácia interpunkčných značiek.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # dátové body
    ax.plot(ranks, freqs, ".", color="0.4", markersize=2, alpha=0.5, label="data (words + punct)", zorder=1)

    # Zipf-Mandelbrot fit (words+punct)
    C_wpp, alpha_wpp, c_wpp = popt_wpp
    r_line = np.logspace(0, np.log10(max(ranks)), 300)
    f_line_wpp = zipf_mandelbrot(r_line, C_wpp, alpha_wpp, c_wpp)
    ax.plot(r_line, f_line_wpp, "-", color="C0", linewidth=1.8,
            label=f"ZM fit w+p: α={alpha_wpp:.2f}, c={c_wpp:.1f}", zorder=3)

    # Zipf-Mandelbrot fit (words only) -- čiarkovaná čiara
    if popt_words is not None:
        C_w, alpha_w, c_w = popt_words
        f_line_w = zipf_mandelbrot(r_line, C_w, alpha_w, c_w)
        ax.plot(r_line, f_line_w, "--", color="C3", linewidth=1.5,
                label=f"ZM fit words: α={alpha_w:.2f}, c={c_w:.1f}", zorder=2)

    # anotácia interpunkcie
    punct_info = find_punct_positions(ranks, tokens)
    for pi in punct_info:
        r_p = pi["rank"]
        f_p = freqs[pi["idx"]]
        ax.plot(r_p, f_p, "D", color="C1", markersize=7, zorder=5,
                markeredgecolor="black", markeredgewidth=0.5)
        # label offset -- vyhneme sa prekrývaniu
        ax.annotate(pi["label"], (r_p, f_p),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, fontweight="bold", color="C1")

    # anotácia top slov (nie interpunkcia, top 10)
    word_count = 0
    for i in range(min(30, len(tokens))):
        if tokens[i] not in PUNCT_TOKENS and word_count < 8:
            ax.annotate(tokens[i], (ranks[i], freqs[i]),
                        textcoords="offset points", xytext=(6, -6),
                        fontsize=7, color="0.3", fontstyle="italic")
            word_count += 1

    # fit range indikátor
    ax.axvline(fit_lo, color="0.7", linestyle=":", linewidth=0.8)
    ax.axvline(fit_hi, color="0.7", linestyle=":", linewidth=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank R", fontsize=11)
    ax.set_ylabel("P(R)", fontsize=11)
    ax.set_title(f"Zipf-Mandelbrot fit — {lang.upper()}", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(0.8, ranks[-1] * 1.2)
    ax.set_ylim(freqs[freqs > 0].min() * 0.5, freqs.max() * 2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_c_comparison(results, out_png):
    """
    Bar plot porovnávajúci parameter c: words vs words+punct pre každý jazyk.
    """
    langs = sorted(set(r["language"] for r in results))

    words_c = []
    wpp_c = []
    for lang in langs:
        for r in results:
            if r["language"] == lang and r["mode"] == "words":
                words_c.append(r["c"])
            if r["language"] == lang and r["mode"] == "words_plus_punct":
                wpp_c.append(r["c"])

    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, words_c, width, label="words only", color="C3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, wpp_c, width, label="words + punct", color="C0", alpha=0.8)

    # hodnotové labely
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Parameter c", fontsize=11)
    ax.set_title("Zipf-Mandelbrot: parameter c\n(nižšie c = bližšie k čistému Zipf zákonu)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(words_c + wpp_c) * 1.4)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_alpha_comparison(results, out_png):
    """Bar plot porovnávajúci exponent α."""
    langs = sorted(set(r["language"] for r in results))
    words_a, wpp_a = [], []
    for lang in langs:
        for r in results:
            if r["language"] == lang and r["mode"] == "words":
                words_a.append(r["alpha"])
            if r["language"] == lang and r["mode"] == "words_plus_punct":
                wpp_a.append(r["alpha"])

    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, words_a, width, label="words only", color="C3", alpha=0.8)
    ax.bar(x + width / 2, wpp_a, width, label="words + punct", color="C0", alpha=0.8)

    for i, v in enumerate(words_a):
        ax.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(wpp_a):
        ax.text(i + width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_ylabel("Exponent α", fontsize=11)
    ax.set_title("Zipf-Mandelbrot: exponent α", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs], fontsize=11)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Zipf-Mandelbrot fit: words vs words+punct")
    ap.add_argument("--csv-dir", default="rankfreq_out_core/csv")
    ap.add_argument("--out-dir", default="rankfreq_out_core/zipf_mandelbrot")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--fit-lo", type=int, default=20, help="Spodna hranica rank pre fit (default 20 kvoli rank-shift pri words+punct)")
    ap.add_argument("--fit-hi", type=int, default=10000, help="Horna hranica rank pre fit")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_results = []

    for lang in langs:
        print(f"\n{'='*60}")
        print(f"  {lang.upper()}")
        print(f"{'='*60}")

        # ── Načítanie dát ──
        path_words = csv_dir / f"rankfreq_words_{lang}.csv"
        path_wpp = csv_dir / f"rankfreq_words_plus_punct_{lang}.csv"

        if not path_words.exists() or not path_wpp.exists():
            print(f"[WARN] Chýba CSV pre {lang}, preskakujem")
            continue

        ranks_w, freqs_w, tokens_w = load_rankfreq(path_words)
        ranks_wpp, freqs_wpp, tokens_wpp = load_rankfreq(path_wpp)

        # ── Fit: words only ──
        print(f"\n  Fit (words only) na [{args.fit_lo}, {args.fit_hi}]:")
        try:
            popt_w, pcov_w, _, _ = fit_zm(ranks_w, freqs_w, args.fit_lo, args.fit_hi)
            C_w, alpha_w, c_w = popt_w
            print(f"    C = {C_w:.6f}")
            print(f"    alpha = {alpha_w:.4f}")
            print(f"    c = {c_w:.2f}")
            all_results.append({
                "language": lang, "mode": "words",
                "C": C_w, "alpha": alpha_w, "c": c_w,
            })
        except Exception as e:
            print(f"    [FAIL] {e}")
            popt_w = None

        # ── Fit: words + punct ──
        print(f"\n  Fit (words + punct) na [{args.fit_lo}, {args.fit_hi}]:")
        try:
            popt_wpp, pcov_wpp, _, _ = fit_zm(ranks_wpp, freqs_wpp, args.fit_lo, args.fit_hi)
            C_wpp, alpha_wpp, c_wpp = popt_wpp
            print(f"    C = {C_wpp:.6f}")
            print(f"    alpha = {alpha_wpp:.4f}")
            print(f"    c = {c_wpp:.2f}")
            all_results.append({
                "language": lang, "mode": "words_plus_punct",
                "C": C_wpp, "alpha": alpha_wpp, "c": c_wpp,
            })
        except Exception as e:
            print(f"    [FAIL] {e}")
            popt_wpp = None

        # ── Výpis rozdielu ──
        if popt_w is not None and popt_wpp is not None:
            print(f"\n  delta_c = {c_w:.2f} -> {c_wpp:.2f}  (zmena: {c_wpp - c_w:+.2f})")
            print(f"  delta_alpha = {alpha_w:.3f} -> {alpha_wpp:.3f}  (zmena: {alpha_wpp - alpha_w:+.3f})")

        # ── Zipf plot s fitom a anotáciou (words+punct dáta) ──
        if popt_wpp is not None:
            plot_zipf_with_fit(
                ranks_wpp, freqs_wpp, tokens_wpp,
                popt_w, popt_wpp,
                lang,
                out_dir / "plots" / f"zipf_mandelbrot_{lang}.png",
                fit_lo=args.fit_lo, fit_hi=args.fit_hi,
            )
            print(f"  [OK] Plot uložený: zipf_mandelbrot_{lang}.png")

    # ── Sumárne CSV ──
    if all_results:
        out_csv = out_dir / "zipf_mandelbrot_fit_summary.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["language", "mode", "C", "alpha", "c"])
            w.writeheader()
            w.writerows(all_results)
        print(f"\n[OK] Sumárne CSV: {out_csv}")

        # ── Porovnávacie bar ploty ──
        plot_c_comparison(all_results, out_dir / "plots" / "comparison_c.png")
        print(f"[OK] Comparison c plot uložený")

        plot_alpha_comparison(all_results, out_dir / "plots" / "comparison_alpha.png")
        print(f"[OK] Comparison alpha plot ulozeny")

    # ── LaTeX tabuľka ──
    if all_results:
        tex_path = out_dir / "zipf_mandelbrot_table.tex"
        with tex_path.open("w", encoding="utf-8") as f:
            f.write("\\begin{table}[ht]\n\\centering\n")
            f.write("\\caption{Zipf-Mandelbrot fit parameters}\n")
            f.write("\\begin{tabular}{llcc}\n\\hline\n")
            f.write("Language & Mode & $\\alpha$ & $c$ \\\\\n\\hline\n")
            for r in sorted(all_results, key=lambda x: (x["language"], x["mode"])):
                mode_label = "words" if r["mode"] == "words" else "words + punct"
                f.write(f"{r['language'].upper()} & {mode_label} & "
                        f"{r['alpha']:.3f} & {r['c']:.2f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        print(f"[OK] LaTeX tabuľka: {tex_path}")

    print("\nHotovo!")


if __name__ == "__main__":
    # fix pre find_punct_positions -- freqs je globálna v plot funcke,
    # tu len zavoláme main
    main()
