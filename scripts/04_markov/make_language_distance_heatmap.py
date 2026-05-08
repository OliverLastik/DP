import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Cesty k súborom ---
csv_path = Path("markov_out_core2/rowwise_language_distances.csv")
output_dir = Path("markov_out_core/plots")
output_dir.mkdir(parents=True, exist_ok=True)
output_png = output_dir / "language_distance_heatmap.png"

# --- Načítanie dát ---
df = pd.read_csv(csv_path)

# Vyberieme jazyky (všetky, ktoré sa objavia v lang_a/lang_b)
langs = sorted(set(df["lang_a"]).union(df["lang_b"]))

# Inicializujeme maticu vzdialeností (diagonála = 0)
dist_matrix = pd.DataFrame(
    np.zeros((len(langs), len(langs))),
    index=langs,
    columns=langs,
)

# Naplníme symetrickú maticu z CSV (použijeme rowwise_jsd_weighted)
for row in df.itertuples(index=False):
    a = row.lang_a
    b = row.lang_b
    d = row.rowwise_jsd_weighted
    dist_matrix.loc[a, b] = d
    dist_matrix.loc[b, a] = d

print("Distance matrix:")
print(dist_matrix)

# --- Vykreslenie heatmapy ---
fig, ax = plt.subplots(figsize=(4, 3.5))

im = ax.imshow(dist_matrix.values)

# Osi
ax.set_xticks(range(len(langs)))
ax.set_yticks(range(len(langs)))
ax.set_xticklabels(langs)
ax.set_yticklabels(langs)

# Otočenie popisov X osi (nech sa to nezliepa)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Číselné hodnoty do políčok
for i in range(len(langs)):
    for j in range(len(langs)):
        value = dist_matrix.iloc[i, j]
        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=8,
        )

# Farebná legenda
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("JSD vzdialenosť")

ax.set_title("Párové vzdialenosti jazykov (Markov, JSD)")

fig.tight_layout()
fig.savefig(output_png, dpi=300)
print(f"Saved to {output_png}")
