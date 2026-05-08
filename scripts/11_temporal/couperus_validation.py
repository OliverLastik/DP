#!/usr/bin/env python3
"""
Couperus validacia: Weibull fit + bootstrap CI pre 5 kniha (3 original + 2 val).

Otazka: je alpha ~ 1 kolaps neskorej fazy temporalny, alebo len specifikum
knihy De komedianten (antický Rím, dialogovo husty)?

Kontrolne knihy:
  - De stille kracht (1900)  -- realisticky kolonialny, rovnaky rok ako Langs lijnen
  - De verliefde ezel (1918) -- klasicko-satiricka, rok po De komedianten
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PUNCT_SET = {".", ",", "!", "?", ";", ":"}


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


def fit_weibull(data, x0=None):
    data = np.asarray(data, dtype=float)
    data = data[data >= 0]
    if len(data) < 20:
        return float("nan"), float("nan")
    def nll(p):
        a, b = p
        if a <= 0 or b <= 0:
            return 1e18
        try:
            pr = np.exp(-(data / b) ** a) - np.exp(-((data + 1) / b) ** a)
        except (OverflowError, FloatingPointError):
            return 1e18
        pr = np.clip(pr, 1e-300, None)
        return -np.sum(np.log(pr))
    if x0 is None:
        x0 = [1.3, max(data.mean(), 1.0)]
    try:
        r = minimize(nll, x0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})
        return float(r.x[0]), float(r.x[1])
    except Exception:
        return float("nan"), float("nan")


def bootstrap(ipi, n_boot, rng):
    a0, b0 = fit_weibull(ipi)
    x0 = [a0, b0] if not np.isnan(a0) else None
    n = len(ipi)
    A = np.empty(n_boot)
    B = np.empty(n_boot)
    for i in range(n_boot):
        s = ipi[rng.integers(0, n, n)]
        A[i], B[i] = fit_weibull(s, x0=x0)
    return a0, b0, A, B


def main():
    tok_root = Path("temporal_data/tokens")
    out = Path("temporal_out/_couperus_val")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    N_BOOT = 300

    books = []
    # original 3-period set
    for fp in sorted((tok_root / "nl_couperus").glob("*.txt")):
        parts = fp.stem.split("_")
        period = parts[2]
        year = int(parts[3])
        title = "_".join(parts[4:])
        books.append({"path": fp, "period": period, "year": year,
                      "title": title, "source": "original"})
    # validation set
    for fp in sorted((tok_root / "nl_couperus_validation").glob("*.txt")):
        parts = fp.stem.split("_")
        period = parts[2]
        year = int(parts[3])
        title = "_".join(parts[4:])
        books.append({"path": fp, "period": period, "year": year,
                      "title": title, "source": "validation"})

    books.sort(key=lambda b: b["year"])
    print(f"[INFO] {len(books)} Couperus books")

    rows = []
    for b in books:
        tokens = load_tokens(b["path"])
        ipi = compute_ipi(tokens)
        n_punct = sum(1 for t in tokens if t in PUNCT_SET)
        pct = n_punct / len(tokens) * 100
        print(f"  {b['year']} {b['period']:8s} {b['title'][:28]:30s} "
              f"(n_tok={len(tokens)}, punct={pct:.1f}%)  ", end="", flush=True)

        a0, bet0, A, Bv = bootstrap(ipi, N_BOOT, rng)
        A = A[np.isfinite(A)]
        Bv = Bv[np.isfinite(Bv)]
        a_lo, a_hi = np.percentile(A, [2.5, 97.5])
        b_lo, b_hi = np.percentile(Bv, [2.5, 97.5])

        print(f"alpha={a0:.3f}[{a_lo:.3f},{a_hi:.3f}]  "
              f"beta={bet0:.3f}[{b_lo:.3f},{b_hi:.3f}]")

        rows.append({
            "year": b["year"], "period": b["period"], "title": b["title"],
            "source": b["source"], "n_tokens": len(tokens), "punct_pct": pct,
            "ipi_mean": float(ipi.mean()), "n_ipi": len(ipi),
            "alpha": a0, "alpha_lo": a_lo, "alpha_hi": a_hi,
            "beta": bet0, "beta_lo": b_lo, "beta_hi": b_hi,
        })

    csv_path = out / "couperus_validation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {csv_path}")

    # ── Plot: alpha + beta across 5 books ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for r in rows:
        years = np.array([r["year"]])
        if r["source"] == "original":
            color = "#2ca02c"
            marker = "D"
            ms = 12
            label = None
        else:
            color = "#9467bd"
            marker = "*"
            ms = 16
            label = None

        axes[0].errorbar(years, [r["alpha"]],
                         yerr=[[r["alpha"] - r["alpha_lo"]],
                               [r["alpha_hi"] - r["alpha"]]],
                         fmt=marker, color=color, markersize=ms,
                         markeredgecolor="black", capsize=5, zorder=5)
        axes[1].errorbar(years, [r["beta"]],
                         yerr=[[r["beta"] - r["beta_lo"]],
                               [r["beta_hi"] - r["beta"]]],
                         fmt=marker, color=color, markersize=ms,
                         markeredgecolor="black", capsize=5, zorder=5)

        axes[0].annotate(f"{r['title'][:16]}",
                         (r["year"], r["alpha"]),
                         textcoords="offset points", xytext=(7, 7),
                         fontsize=7)
        axes[1].annotate(f"{r['title'][:16]}",
                         (r["year"], r["beta"]),
                         textcoords="offset points", xytext=(7, 7),
                         fontsize=7)

    orig_years = [r["year"] for r in rows if r["source"] == "original"]
    orig_alpha = [r["alpha"] for r in rows if r["source"] == "original"]
    orig_beta = [r["beta"] for r in rows if r["source"] == "original"]
    order = np.argsort(orig_years)
    axes[0].plot(np.array(orig_years)[order], np.array(orig_alpha)[order],
                 "-", color="#2ca02c", linewidth=1.5, alpha=0.5, zorder=2)
    axes[1].plot(np.array(orig_years)[order], np.array(orig_beta)[order],
                 "-", color="#2ca02c", linewidth=1.5, alpha=0.5, zorder=2)

    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Publication year", fontsize=11)
    axes[0].set_ylabel(r"Weibull shape $\alpha$", fontsize=11)
    axes[0].set_title(r"$\alpha$ — original timeline + validation", fontsize=12)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Publication year", fontsize=11)
    axes[1].set_ylabel(r"Weibull scale $\beta$", fontsize=11)
    axes[1].set_title(r"$\beta$ — original timeline + validation", fontsize=12)
    axes[1].grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=11, label="Original (early/middle/late)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#9467bd",
               markeredgecolor="black", markersize=15, label="Validation (midreal/latefan)"),
    ]
    axes[0].legend(handles=legend_items, fontsize=9, loc="best")

    fig.suptitle("Couperus — 5 books (timeline + genre validation) with 95% CI",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "couperus_validation_alpha_beta.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] couperus_validation_alpha_beta.png")

    # ── Interpretation ──
    print("\n== Interpretation ==")
    langs_row = next(r for r in rows if r["title"].startswith("Langs"))
    stille_row = next(r for r in rows if r["title"].startswith("De_stille"))
    kom_row = next(r for r in rows if r["title"].startswith("De_komedianten"))
    ezel_row = next(r for r in rows if r["title"].startswith("De_verliefde_ezel"))

    print(f"Langs lijnen (1900 middle):     alpha={langs_row['alpha']:.3f}")
    print(f"De stille kracht (1900 mid-real ctrl):  alpha={stille_row['alpha']:.3f}")
    print(f"  -> same-year realist control: {'MATCH' if abs(langs_row['alpha']-stille_row['alpha']) < 0.05 else 'DIFFER'}")
    print()
    print(f"De komedianten (1917 late):     alpha={kom_row['alpha']:.3f}")
    print(f"De verliefde ezel (1918 late-fan ctrl): alpha={ezel_row['alpha']:.3f}")
    print(f"  -> late-period control:       {'MATCH' if abs(kom_row['alpha']-ezel_row['alpha']) < 0.05 else 'DIFFER'}")

    print("\n[OK] Couperus validacia hotova.")


if __name__ == "__main__":
    main()
