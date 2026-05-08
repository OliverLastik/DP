#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def infer_cols(df: pd.DataFrame):
    """
    Očakáva long CSV s aspoň:
      language, from, to, prob
    Názvy sa môžu mierne líšiť, skúsime ich nájsť robustne.
    """
    lower = {c.lower(): c for c in df.columns}

    lang = lower.get("language") or lower.get("lang")
    frm = lower.get("from") or lower.get("from_token") or lower.get("from_symbol")
    to = lower.get("to") or lower.get("to_token") or lower.get("to_symbol")
    prob = lower.get("prob") or lower.get("p") or lower.get("probability")

    if not all([lang, frm, to, prob]):
        raise ValueError(f"Neviem nájsť stĺpce. Našiel som: {list(df.columns)}")

    return lang, frm, to, prob


def build_matrix(df: pd.DataFrame, lang_value: str, lang_col: str, from_col: str, to_col: str, prob_col: str,
                 order: list[str] | None = None):
    sub = df[df[lang_col] == lang_value].copy()

    tokens = sorted(set(sub[from_col].astype(str)).union(set(sub[to_col].astype(str))))
    if order is not None:
        # zachovaj zadané poradie, plus doplň čo chýba
        base = [t for t in order if t in tokens]
        rest = [t for t in tokens if t not in base]
        tokens = base + rest

    mat = (
        sub.pivot_table(index=from_col, columns=to_col, values=prob_col, aggfunc="mean")
        .reindex(index=tokens, columns=tokens)
        .fillna(0.0)
    )
    return tokens, mat.to_numpy(dtype=float)


def plot_delta(delta: np.ndarray, labels: list[str], title: str, out_path: Path):
    plt.figure(figsize=(8.5, 6.8))
    vmax = float(np.max(np.abs(delta))) if delta.size else 1.0
    if vmax == 0:
        vmax = 1e-9

    plt.imshow(delta, aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="ΔP (A - B)")
    plt.title(title)

    # popisky osí
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markov-long", default="markov_out_core/markov_conditional_probs_long.csv",
                    help="Long CSV s P(to|from) pre každý jazyk")
    ap.add_argument("--out-dir", default="markov_out_core/plots", help="Kam uložiť PNG")
    ap.add_argument("--langs", default="en,de,nl", help="Jazyky, napr. en,de,nl")
    args = ap.parse_args()

    markov_path = Path(args.markov_long)
    out_dir = Path(args.out_dir)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    df = pd.read_csv(markov_path)
    lang_col, from_col, to_col, prob_col = infer_cols(df)

    # vybuduj matice pre každý jazyk v jednom konzistentnom poradí tokenov:
    # vezmeme poradie tokenov z prvého jazyka a použijeme ho pre všetky
    base_labels, base_mat = build_matrix(df, langs[0], lang_col, from_col, to_col, prob_col, order=None)

    mats = {langs[0]: base_mat}
    labels = base_labels

    for lang in langs[1:]:
        _, m = build_matrix(df, lang, lang_col, from_col, to_col, prob_col, order=labels)
        mats[lang] = m

    pairs = [(langs[i], langs[j]) for i in range(len(langs)) for j in range(i + 1, len(langs))]

    for a, b in pairs:
        delta = mats[a] - mats[b]
        out_path = out_dir / f"markov_delta_{a}_minus_{b}.png"
        plot_delta(
            delta=delta,
            labels=labels,
            title=f"Markov delta heatmap: {a.upper()} - {b.upper()} (ΔP(to|from))",
            out_path=out_path,
        )
        print("Wrote:", out_path.resolve())

    print("Done. Pairwise delta heatmaps generated.")


if __name__ == "__main__":
    main()