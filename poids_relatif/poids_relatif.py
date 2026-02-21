#!/usr/bin/env python3
"""
Graphique : poids relatif du score scolaire vs bonus IPS 400
=============================================================
Nouvelles stats brutes (sans tranchage)

Score scolaire = K × Σ [ H_champ × poids_champ ]
avec H = 10 × (10 + (note − µ) / σ)

On compare l'amplitude scolaire (écart entre un élève à 19/20 et un élève
à 10/20) au bonus IPS de 400 points, pour K allant de 2.0 à 2.5.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════════════
# NOUVELLES STATS BRUTES (SANS TRANCHAGE) — ACADÉMIE PARIS 2025
# ═══════════════════════════════════════════════════════════════════════════
STATS = {
    "MATHEMATIQUES":      {"mu": 12.152, "sigma": 4.989},
    "FRANCAIS":           {"mu": 12.631, "sigma": 3.945},
    "HISTOIRE-GEO":       {"mu": 13.119, "sigma": 4.076},
    "LANGUES VIVANTES":   {"mu": 13.822, "sigma": 4.211},
    "SCIENCES-TECHNO-DP": {"mu": 13.458, "sigma": 3.660},
    "ARTS":               {"mu": 15.930, "sigma": 2.960},
    "EPS":                {"mu": 15.883, "sigma": 2.889},
}

POIDS = {
    "MATHEMATIQUES": 5, "FRANCAIS": 5,
    "HISTOIRE-GEO": 4, "LANGUES VIVANTES": 4,
    "SCIENCES-TECHNO-DP": 4, "ARTS": 4, "EPS": 4,
}

BONUS_IPS = 400

# ═══════════════════════════════════════════════════════════════════════════
# CALCUL DU SCORE SCOLAIRE HARMONISÉ
# ═══════════════════════════════════════════════════════════════════════════
def score_scolaire(note_partout):
    """Score harmonisé (avant coef K) pour un élève ayant `note_partout` partout."""
    total = 0.0
    for champ, s in STATS.items():
        H = 10 * (10 + (note_partout - s["mu"]) / s["sigma"])
        total += H * POIDS[champ]
    return total

# Profils repères
score_19 = score_scolaire(19)   # excellent
score_15 = score_scolaire(15)   # bon
score_12 = score_scolaire(12)   # correct
score_10 = score_scolaire(10)   # moyen

# Coefficients étudiés
coefs = np.arange(2.0, 2.55, 0.1)

print(f"Score harmonisé (avant K) :")
print(f"  Élève 19/20 : {score_19:.0f}")
print(f"  Élève 15/20 : {score_15:.0f}")
print(f"  Élève 12/20 : {score_12:.0f}")
print(f"  Élève 10/20 : {score_10:.0f}")
print(f"  Amplitude 19−10 : {score_19 - score_10:.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# GRAPHIQUE
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 9))
plt.style.use("seaborn-v0_8-whitegrid")

bar_width = 0.06

for i, K in enumerate(coefs):
    x = K
    s19 = K * score_19
    s15 = K * score_15
    s12 = K * score_12
    s10 = K * score_10

    # --- Barre : score élève 10/20 (base) ---
    ax.bar(x, s10, width=bar_width, color="#95a5a6", alpha=0.20, zorder=2)

    # --- Barre : amplitude scolaire 10 → 19 (gradient central = distribution) ---
    amplitude = s19 - s10
    n_slices = 200
    slice_h = amplitude / n_slices
    for s_idx in range(n_slices):
        # position normalisée 0..1 du bas vers le haut de la barre
        t = (s_idx + 0.5) / n_slices
        # profil gaussien centré à 0.5 → plus dense au centre
        alpha_val = 0.35 + 0.60 * np.exp(-((t - 0.5) ** 2) / (2 * 0.18 ** 2))
        ax.bar(x, slice_h, bottom=s10 + s_idx * slice_h, width=bar_width,
               color="#2c3e50", alpha=alpha_val, zorder=2, linewidth=0)

    # --- Barre bonus IPS 400 accolée ---
    x_bonus = x + bar_width * 0.8
    ax.bar(x_bonus, BONUS_IPS, bottom=s10, width=bar_width * 0.5,
           color="#e67e22", alpha=0.9, zorder=3)

    # --- Repères intermédiaires (traits horizontaux dans la barre) ---
    for note, score_val, color, label_txt in [
        (15, s15, "#27ae60", "15/20"),
        (12, s12, "#2980b9", "12/20"),
    ]:
        ax.plot([x - bar_width/2, x + bar_width/2], [score_val, score_val],
                color=color, linewidth=2, zorder=4)
        if i == 0:  # légende seulement sur le premier
            ax.text(x - bar_width/2 - 0.01, score_val, label_txt,
                    ha="right", va="center", fontsize=8, color=color,
                    fontweight="bold")

    # --- Annotations ---
    # Amplitude scolaire
    y_center = (s19 + s10) / 2
    ax.text(x, y_center, f"{amplitude:.0f}",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="white", zorder=5)

    # Score top
    ax.text(x, s19 + 80, f"{s19:.0f}",
            ha="center", va="bottom", fontsize=9, color="#2c3e50",
            fontweight="bold")

    # Score bas
    ax.text(x, s10 - 80, f"{s10:.0f}",
            ha="center", va="top", fontsize=9, color="#7f8c8d")

    # Bonus / amplitude en %
    pct = 100 * BONUS_IPS / amplitude
    ax.text(x_bonus, s10 + BONUS_IPS + 60, f"400\n({pct:.0f}%)",
            ha="center", va="bottom", fontsize=9, color="#e67e22",
            fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# DROITES HORIZONTALES — AMPLITUDE NOTES 2025
# ═══════════════════════════════════════════════════════════════════════════
ax.axhspan(6800, 8100, color="#e74c3c", alpha=0.07, zorder=0)
ax.axhline(y=6800, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
ax.axhline(y=8100, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
ax.text(coefs[0] - 0.18, 6800 + 40, "6 800 (~score min 2025)", va="bottom", fontsize=9,
        color="#e74c3c", fontweight="bold")
ax.text(coefs[0] - 0.18, 8100 + 40, "8 100 (score max 2025)", va="bottom", fontsize=9,
        color="#e74c3c", fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# ANNOTATIONS GLOBALES
# ═══════════════════════════════════════════════════════════════════════════

# Flèche explicative sur le premier K
K0 = coefs[0]
s10_0 = K0 * score_10
s19_0 = K0 * score_19
ax.annotate("Amplitude\nscolaire\n(19−10/20)",
            xy=(K0, (s19_0 + s10_0)/2),
            xytext=(K0 - 0.15, (s19_0 + s10_0)/2 + 400),
            fontsize=10, color="#2c3e50", fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

# Légende
legend_elements = [
    mpatches.Patch(facecolor="#2c3e50", alpha=0.85,
                   label="Amplitude scolaire estimée (19/20 − 10/20)"),
    mpatches.Patch(facecolor="#95a5a6", alpha=0.5,
                   label="Score min estimé (élève 10/20)"),
    mpatches.Patch(facecolor="#e67e22", alpha=0.9,
                   label=f"Bonus IPS = {BONUS_IPS} pts"),
    plt.Line2D([0], [0], color="#27ae60", linewidth=2, label="Repère 15/20"),
    plt.Line2D([0], [0], color="#2980b9", linewidth=2, label="Repère 12/20"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=True,
          fontsize=10, framealpha=0.95)

# Axes
ax.set_xlabel("Coefficient multiplicateur K", fontsize=13, labelpad=10)
ax.set_ylabel("Score scolaire", fontsize=13, labelpad=10)
ax.set_xticks(coefs)
ax.set_xticklabels([f"K = {k:.1f}" for k in coefs], fontsize=11)

# Marge
x_min = coefs[0] - 0.2
x_max = coefs[-1] + 0.2
ax.set_xlim(x_min, x_max)

y_min = min(K * score_10 for K in coefs) - 600
y_max = max(K * score_19 for K in coefs) + 600
ax.set_ylim(y_min, y_max)

# Titre
ax.set_title(
    "ESTIMATION 2026 DU POIDS RELATIF DU SCORE SCOLAIRE vs BONUS IPS (400 pts)",
    fontsize=15, fontweight="bold", pad=20
)

# Sous-titre
fig.text(0.5, 0.01,
         "Le % indique la part du bonus IPS dans l'amplitude scolaire. "
         "Plus K augmente, plus le scolaire domine et plus le bonus perd "
         "son pouvoir de rééquilibrage.",
         ha="center", fontsize=10, color="#7f8c8d", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_path = "poids_scolaire_vs_bonus_ips.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n→ Graphique sauvegardé : {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# TABLEAU RÉCAPITULATIF
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'K':>5s} {'Score 19':>10s} {'Score 10':>10s} {'Amplitude':>10s} "
      f"{'Bonus':>7s} {'Bonus/Amp':>10s}")
print("─" * 58)
for K in coefs:
    s19 = K * score_19
    s10 = K * score_10
    amp = s19 - s10
    pct = 100 * BONUS_IPS / amp
    print(f"{K:>5.1f} {s19:>10.0f} {s10:>10.0f} {amp:>10.0f} "
          f"{BONUS_IPS:>7d} {pct:>9.1f}%")
