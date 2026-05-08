#!/usr/bin/env python3
"""
Heaps' law analysis: V(n) ~ n^beta
  V(n) = vocabulary size after reading n tokens
Porovnanie words-only vs words+punct pre kazdy jazyk.
Vystup: CSV s fitovanymi beta, ploty V(n) pre kazdy jazyk + overlay.
"""
import argparse
import csv
import math
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def heaps_curve(tokens: list[str], sample_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Spocita V(n) pre dany token stream.
    Vrati (n_array, V_array) s logaritmicky rozlozenymi bodmi.
    """
    N = len(tokens)
    if N < 10:
        return np.array([]), np.array([])

    # logarithmicky rozlozene body + posledny bod
    indices = np.unique(np.geomspace(1, N, num=sample_points).astype(int))
    if indices[-1] != N:
        indices = np.append(indices, N)

    vocab = set()
    n_vals = []
    v_vals = []
    idx_ptr = 0  # pointer do indices

    for i, tok in enumerate(tokens, 1):
        vocab.add(tok)
        if idx_ptr < len(indices) and i == indices[idx_ptr]:
            n_vals.append(i)
            v_vals.append(len(vocab))
            idx_ptr += 1

    return np.array(n_vals), np.array(v_vals)


def heaps_fit(n_arr: np.ndarray, v_arr: np.ndarray):
    """
    Fituje V = K * n^beta na log-log.
    Vrati (K, beta, r_squared).
    """
    mask = (n_arr > 0) & (v_arr > 0)
    ln = np.log(n_arr[mask])
    lv = np.log(v_arr[mask])

    # linear regression on log-log
    coeffs = np.polyfit(ln, lv, 1)
    beta = coeffs[0]
    K = math.exp(coeffs[1])

    # R^2
    lv_pred = np.polyval(coeffs, ln)
    ss_res = np.sum((lv - lv_pred) ** 2)
    ss_tot = np.sum((lv - np.mean(lv)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return K, beta, r2


def load_tokens(path: Path) -> list[str]:
    """Nacita tokeny zo suboru (space-separated, jeden riadok alebo viac)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.split()


def main():
    ap = argparse.ArgumentParser(description="Heaps' law: V(n) ~ K * n^beta")
    ap.add_argument("--words-dir", default="token_out_core/tokens_words",
                     help="Priecinok s words-only tokenmi (per lang subdir)")
    ap.add_argument("--punct-dir", default="token_out_core/tokens_words_plus_punct",
                     help="Priecinok s words+punct tokenmi")
    ap.add_argument("--outdir", default="rankfreq_out_core/heaps_law")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--lowercase", action="store_true", default=True,
                     help="Lowercase normalizacia tokenov")
    ap.add_argument("--sample-points", type=int, default=500)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    summary_rows = []

    # Global overlay plot
    fig_all, ax_all = plt.subplots(figsize=(7, 5))
    colors = {"en": "C0", "de": "C1", "nl": "C2"}
    styles = {"words": "-", "words+punct": "--"}

    for lang in langs:
        words_dir = Path(args.words_dir) / lang
        punct_dir = Path(args.punct_dir) / lang

        if not words_dir.exists():
            print(f"[WARN] {lang}: missing {words_dir}")
            continue

        # Nacitaj vsetky knihy a spoj
        words_all = []
        punct_all = []

        book_files = sorted(words_dir.glob("*.txt"))
        for bf in book_files:
            toks = load_tokens(bf)
            if args.lowercase:
                toks = [t.lower() for t in toks]
            words_all.extend(toks)

        punct_files = sorted(punct_dir.glob("*.txt")) if punct_dir.exists() else []
        for pf in punct_files:
            toks = load_tokens(pf)
            if args.lowercase:
                toks = [t.lower() for t in toks]
            punct_all.extend(toks)

        print(f"[INFO] {lang}: words={len(words_all)}, words+punct={len(punct_all)}")

        # Per-book Heaps curves + fit
        book_betas_w = []
        book_betas_p = []

        for bf in book_files:
            toks_w = load_tokens(bf)
            if args.lowercase:
                toks_w = [t.lower() for t in toks_w]
            n_w, v_w = heaps_curve(toks_w, args.sample_points)
            if len(n_w) > 10:
                _, beta_w, _ = heaps_fit(n_w, v_w)
                book_betas_w.append(beta_w)

            # matching punct file
            pf = punct_dir / bf.name
            if pf.exists():
                toks_p = load_tokens(pf)
                if args.lowercase:
                    toks_p = [t.lower() for t in toks_p]
                n_p, v_p = heaps_curve(toks_p, args.sample_points)
                if len(n_p) > 10:
                    _, beta_p, _ = heaps_fit(n_p, v_p)
                    book_betas_p.append(beta_p)

        # Aggregate Heaps curve (concatenated corpus)
        n_w, v_w = heaps_curve(words_all, args.sample_points)
        n_p, v_p = heaps_curve(punct_all, args.sample_points)

        K_w, beta_w, r2_w = heaps_fit(n_w, v_w)
        K_p, beta_p, r2_p = heaps_fit(n_p, v_p) if len(n_p) > 10 else (0, 0, 0)

        print(f"  words:       beta={beta_w:.4f}, K={K_w:.2f}, R2={r2_w:.4f}")
        print(f"  words+punct: beta={beta_p:.4f}, K={K_p:.2f}, R2={r2_p:.4f}")

        summary_rows.append({
            "lang": lang,
            "mode": "words",
            "n_tokens": len(words_all),
            "final_vocab": int(v_w[-1]) if len(v_w) > 0 else 0,
            "K": K_w, "beta": beta_w, "R2": r2_w,
            "beta_median_perbook": float(np.median(book_betas_w)) if book_betas_w else float("nan"),
            "beta_std_perbook": float(np.std(book_betas_w)) if book_betas_w else float("nan"),
        })
        summary_rows.append({
            "lang": lang,
            "mode": "words+punct",
            "n_tokens": len(punct_all),
            "final_vocab": int(v_p[-1]) if len(v_p) > 0 else 0,
            "K": K_p, "beta": beta_p, "R2": r2_p,
            "beta_median_perbook": float(np.median(book_betas_p)) if book_betas_p else float("nan"),
            "beta_std_perbook": float(np.std(book_betas_p)) if book_betas_p else float("nan"),
        })

        # --- Per-language plot ---
        fig, ax = plt.subplots(figsize=(6, 4.5))

        ax.plot(n_w, v_w, "-", color="C0", label=f"words ($\\beta$={beta_w:.3f})")
        # fit line
        ax.plot(n_w, K_w * n_w**beta_w, "--", color="C0", alpha=0.5, label="_fit_w")

        if len(n_p) > 10:
            ax.plot(n_p, v_p, "-", color="C1", label=f"words+punct ($\\beta$={beta_p:.3f})")
            ax.plot(n_p, K_p * n_p**beta_p, "--", color="C1", alpha=0.5, label="_fit_p")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (tokens read)")
        ax.set_ylabel("V(n) (vocabulary size)")
        ax.set_title(f"Heaps' law - {lang.upper()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"heaps_{lang}.png", dpi=300)
        plt.close(fig)

        # --- Per-book beta boxplot ---
        if book_betas_w or book_betas_p:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            data_box = []
            labels_box = []
            if book_betas_w:
                data_box.append(book_betas_w)
                labels_box.append("words")
            if book_betas_p:
                data_box.append(book_betas_p)
                labels_box.append("words+punct")
            bp = ax2.boxplot(data_box, tick_labels=labels_box, patch_artist=True)
            for patch, c in zip(bp["boxes"], ["C0", "C1"]):
                patch.set_facecolor(c)
                patch.set_alpha(0.4)
            ax2.set_ylabel("$\\beta$ (Heaps exponent)")
            ax2.set_title(f"Per-book Heaps $\\beta$ - {lang.upper()}")
            fig2.tight_layout()
            fig2.savefig(outdir / f"heaps_beta_boxplot_{lang}.png", dpi=300)
            plt.close(fig2)

        # --- overlay plot ---
        c = colors.get(lang, "C3")
        ax_all.plot(n_w, v_w, styles["words"], color=c,
                    label=f"{lang} words ($\\beta$={beta_w:.3f})")
        if len(n_p) > 10:
            ax_all.plot(n_p, v_p, styles["words+punct"], color=c,
                        label=f"{lang} w+p ($\\beta$={beta_p:.3f})")

    # Save overlay
    ax_all.set_xscale("log")
    ax_all.set_yscale("log")
    ax_all.set_xlabel("n (tokens read)")
    ax_all.set_ylabel("V(n) (vocabulary size)")
    ax_all.set_title("Heaps' law - all languages")
    ax_all.legend(fontsize=8)
    fig_all.tight_layout()
    fig_all.savefig(outdir / "heaps_overlay.png", dpi=300)
    plt.close(fig_all)

    # Save CSV
    out_csv = outdir / "heaps_law_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang", "mode", "n_tokens", "final_vocab", "K", "beta", "R2",
            "beta_median_perbook", "beta_std_perbook"
        ])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[OK] CSV: {out_csv}")
    print(f"[OK] Plots in {outdir}")


if __name__ == "__main__":
    main()
