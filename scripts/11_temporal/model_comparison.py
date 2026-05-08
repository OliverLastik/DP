#!/usr/bin/env python3
"""
AIC/BIC model comparison na IPI distribuciach.

Porovna 4 kandidatske modely per kniha:
  1. Discrete Weibull     (2 parametre)  - Stanisz et al., nas navrh
  2. Negative binomial    (2 parametre)  - standardny model pre count data
  3. Discrete log-normal  (2 parametre)  - heavy-tailed alternativa
  4. Geometric            (1 parameter)  - memoryless null

Pre kazdu knihu:
  - MLE fit kazdeho modelu
  - loglik, AIC, BIC
  - rank podla AIC
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import nbinom, norm

PUNCT_SET = {".", ",", "!", "?", ";", ":"}

AUTHOR_COLORS = {
    "dickens":  "#1f77b4",
    "fontane":  "#d62728",
    "couperus": "#2ca02c",
}

MODEL_COLORS = {
    "weibull":   "#2ca02c",
    "nbinom":    "#1f77b4",
    "lognormal": "#ff7f0e",
    "geometric": "#d62728",
}


def load_tokens(p: Path):
    return p.read_text(encoding="utf-8", errors="replace").split()


def compute_ipi(tokens):
    out, w, seen = [], 0, False
    for t in tokens:
        if t in PUNCT_SET:
            if seen:
                out.append(w)
            seen = True
            w = 0
        else:
            w += 1
    return np.array(out, dtype=np.int64)


# ── Discrete Weibull (Stanisz) ──

def loglik_weibull(params, data):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return 1e18
    try:
        p = np.exp(-(data / beta) ** alpha) - np.exp(-((data + 1) / beta) ** alpha)
    except (OverflowError, FloatingPointError):
        return 1e18
    p = np.clip(p, 1e-300, None)
    return -np.sum(np.log(p))


def fit_weibull(data):
    x0 = [1.3, max(float(data.mean()), 1.0)]
    r = minimize(loglik_weibull, x0, args=(data,), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return {"params": {"alpha": r.x[0], "beta": r.x[1]},
            "loglik": -r.fun, "k": 2}


# ── Negative binomial ──

def loglik_nbinom(params, data):
    r, p = params
    if r <= 0 or p <= 0 or p >= 1:
        return 1e18
    return -np.sum(nbinom.logpmf(data, r, p))


def fit_nbinom(data):
    mu = float(data.mean())
    var = float(data.var())
    if var > mu:
        r0 = mu * mu / (var - mu)
        p0 = mu / var
    else:
        r0, p0 = 5.0, 0.5
    r0 = max(r0, 0.5)
    p0 = min(max(p0, 0.05), 0.95)
    res = minimize(loglik_nbinom, [r0, p0], args=(data,),
                   method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return {"params": {"r": res.x[0], "p": res.x[1]},
            "loglik": -res.fun, "k": 2}


# ── Discrete log-normal (shift by 0.5 to handle k=0) ──

def loglik_lognorm(params, data):
    mu, sigma = params
    if sigma <= 0:
        return 1e18
    lo = data - 0.5
    hi = data + 0.5
    lo = np.clip(lo, 1e-9, None)
    z_hi = (np.log(hi) - mu) / sigma
    z_lo = (np.log(lo) - mu) / sigma
    cdf_hi = norm.cdf(z_hi)
    cdf_lo = norm.cdf(z_lo)
    p = cdf_hi - cdf_lo
    p = np.clip(p, 1e-300, None)
    return -np.sum(np.log(p))


def fit_lognorm(data):
    pos = data[data > 0]
    if len(pos) < 10:
        return {"params": {}, "loglik": -1e18, "k": 2}
    log_pos = np.log(pos.astype(float))
    mu0 = float(log_pos.mean())
    sigma0 = float(log_pos.std() + 1e-3)
    res = minimize(loglik_lognorm, [mu0, sigma0], args=(data,),
                   method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return {"params": {"mu": res.x[0], "sigma": res.x[1]},
            "loglik": -res.fun, "k": 2}


# ── Geometric (k = 0,1,2,...) ──

def fit_geometric(data):
    # MLE: p = 1 / (1 + mean)
    mean = float(data.mean())
    p = 1.0 / (1.0 + mean)
    # P(X=k) = (1-p)^k * p
    ll = float(np.sum(data * np.log(1 - p) + np.log(p)))
    return {"params": {"p": p}, "loglik": ll, "k": 1}


def aic(loglik, k):
    return 2 * k - 2 * loglik


def bic(loglik, k, n):
    return k * np.log(n) - 2 * loglik


MODELS = {
    "weibull":   fit_weibull,
    "nbinom":    fit_nbinom,
    "lognormal": fit_lognorm,
    "geometric": fit_geometric,
}


def parse_meta(stem: str):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] in {"en", "de", "nl"}:
        return parts[0], parts[1], parts[2], int(parts[3]), "_".join(parts[4:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", "--tok-root", dest="token_dir",
                    default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_aic")
    args = ap.parse_args()

    tok_root = Path(args.token_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    books = []
    for sub in sorted(tok_root.iterdir()):
        if not sub.is_dir() or "validation" in sub.name:
            continue
        for fp in sorted(sub.glob("*.txt")):
            meta = parse_meta(fp.stem)
            if not meta:
                continue
            lang, author, period, year, title = meta
            books.append({
                "path": fp, "lang": lang, "author": author,
                "period": period, "year": year, "title": title,
            })

    print(f"[INFO] {len(books)} books to compare")
    if not books:
        print(f"[WARN] No temporal books found in {tok_root}")
        return

    rows = []
    for b in books:
        tokens = load_tokens(b["path"])
        ipi = compute_ipi(tokens)
        n = len(ipi)

        fits = {}
        for name, fn in MODELS.items():
            fit = fn(ipi)
            fits[name] = {
                "loglik": fit["loglik"],
                "k": fit["k"],
                "aic": aic(fit["loglik"], fit["k"]),
                "bic": bic(fit["loglik"], fit["k"], n),
                "params": fit["params"],
            }

        aics = {name: f["aic"] for name, f in fits.items()}
        best_aic = min(aics.values())
        best_model = min(aics, key=aics.get)
        deltas = {name: a - best_aic for name, a in aics.items()}

        print(f"\n  {b['lang']}/{b['author']}/{b['period']} {b['year']} "
              f"{b['title'][:22]:24s}  (n_ipi={n})")
        for name in MODELS:
            f = fits[name]
            mark = " *" if name == best_model else "  "
            print(f"    {mark}{name:10s}  ll={f['loglik']:>12.1f}  "
                  f"AIC={f['aic']:>12.1f}  dAIC={deltas[name]:>9.1f}")

        row = {
            "lang": b["lang"], "author": b["author"], "period": b["period"],
            "year": b["year"], "title": b["title"], "n_ipi": n,
            "best_model": best_model,
        }
        for name in MODELS:
            row[f"{name}_loglik"] = fits[name]["loglik"]
            row[f"{name}_aic"] = fits[name]["aic"]
            row[f"{name}_delta_aic"] = deltas[name]
        rows.append(row)

    rows.sort(key=lambda r: (r["author"], r["year"]))

    csv_path = out / "model_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {csv_path}")

    # ── Win count ──
    win_counts = {name: 0 for name in MODELS}
    for r in rows:
        win_counts[r["best_model"]] += 1
    print(f"\nWin count across {len(rows)} books:")
    for name, c in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}  {c}")

    # ── Plot: delta AIC per book, grouped by author ──
    fig, ax = plt.subplots(figsize=(11, 5.5))
    model_order = ["weibull", "nbinom", "lognormal", "geometric"]
    n_books = len(rows)
    n_models = len(model_order)
    x = np.arange(n_books)
    width = 0.20

    for i, m in enumerate(model_order):
        deltas = [r[f"{m}_delta_aic"] for r in rows]
        ax.bar(x + (i - 1.5) * width, deltas, width,
               color=MODEL_COLORS[m], label=m, alpha=0.85)

    labels = [f"{r['author'][:4]}\n{r['year']}" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\Delta$AIC (vs best model per book)", fontsize=11)
    ax.set_title("Model comparison: inter-punctuation interval distribution",
                 fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(10, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("symlog", linthresh=10)
    fig.tight_layout()
    fig.savefig(out / "delta_aic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] delta_aic.png")

    # ── Plot: win pie chart ──
    fig, ax = plt.subplots(figsize=(5, 5))
    wins = [(name, win_counts[name]) for name in model_order if win_counts[name] > 0]
    ax.pie([w for _, w in wins], labels=[n for n, _ in wins],
           colors=[MODEL_COLORS[n] for n, _ in wins],
           autopct=lambda p: f"{int(round(p * len(rows) / 100))}",
           startangle=90, textprops={"fontsize": 11})
    ax.set_title(f"Best model by AIC (n={len(rows)} books)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "aic_winners.png", dpi=300)
    plt.close(fig)
    print("[OK] aic_winners.png")

    print("\n[OK] Model comparison hotovo.")


if __name__ == "__main__":
    main()
