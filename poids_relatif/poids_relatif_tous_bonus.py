#!/usr/bin/env python3
"""
Graphique : poids relatif du score scolaire vs TOUS les bonus IPS
==================================================================
Bonus IPS : 0, 400, 800, 1200
Coefficient K : 2.0 à 2.5
Stats brutes (sans tranchage) — Académie de Paris 2025.
Un encadré d'équivalence bonus → points de moyenne pour chaque K.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════════════
# STATS BRUTES — ACADÉMIE PARIS 2025
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

BONUS_LIST = [400, 800, 1200]
BONUS_COLORS = {400: "#e67e22", 800: "#2980b9", 1200: "#27ae60"}

def score_scolaire(note_partout):
    total = 0.0
    for champ, s in STATS.items():
        H = 10 * (10 + (note_partout - s["mu"]) / s["sigma"])
        total += H * POIDS[champ]
    return total

score_19 = score_scolaire(19)
score_15 = score_scolaire(15)
score_12 = score_scolaire(12)
score_10 = score_scolaire(10)

coefs = np.arange(2.0, 2.55, 0.1)

# ═══════════════════════════════════════════════════════════════════════════
# POSITIONNEMENT
# ═══════════════════════════════════════════════════════════════════════════
n_bars = 4
bar_w = 0.12
gap = 0.03
step = bar_w + gap

x_ticks = np.arange(len(coefs), dtype=float)

fig, ax = plt.subplots(figsize=(17, 11))
plt.style.use("seaborn-v0_8-whitegrid")

for i, K in enumerate(coefs):
    xc = x_ticks[i]
    s19 = K * score_19
    s15 = K * score_15
    s12 = K * score_12
    s10 = K * score_10
    amplitude = s19 - s10

    offsets = [(j - (n_bars - 1) / 2) * step for j in range(n_bars)]

    # --- Barre scolaire ---
    x0 = xc + offsets[0]
    ax.bar(x0, s10, width=bar_w, color="#95a5a6", alpha=0.20, zorder=2)
    # Gradient gaussien sur l'amplitude scolaire
    n_slices = 200
    slice_h = amplitude / n_slices
    for s_idx in range(n_slices):
        t = (s_idx + 0.5) / n_slices
        alpha_val = 0.35 + 0.60 * np.exp(-((t - 0.5) ** 2) / (2 * 0.18 ** 2))
        ax.bar(x0, slice_h, bottom=s10 + s_idx * slice_h, width=bar_w,
               color="#2c3e50", alpha=alpha_val, zorder=2, linewidth=0)

    hw = bar_w / 2
    for score_val, color, lbl in [(s15, "#27ae60", "15"), (s12, "#2980b9", "12")]:
        ax.plot([x0 - hw, x0 + hw], [score_val, score_val],
                color=color, linewidth=2, zorder=4)
        if i == 0:
            ax.text(x0 - hw - 0.02, score_val, f"{lbl}/20",
                    ha="right", va="center", fontsize=8, color=color,
                    fontweight="bold")

    ax.text(x0, (s19 + s10) / 2, f"{amplitude:.0f}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=5)
    ax.text(x0, s19 + 60, f"{s19:.0f}", ha="center", va="bottom",
            fontsize=8, color="#2c3e50", fontweight="bold")

    # --- Barres bonus IPS ---
    for j, bonus in enumerate(BONUS_LIST):
        xb = xc + offsets[j + 1]
        ax.bar(xb, bonus, bottom=s10, width=bar_w,
               color=BONUS_COLORS[bonus], alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.5)

        pct = 100 * bonus / amplitude
        if bonus >= 600:
            ax.text(xb, s10 + bonus / 2, f"{bonus}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=5)
        else:
            ax.text(xb, s10 + bonus + 50, f"{bonus}\n({pct:.0f}%)",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=BONUS_COLORS[bonus], zorder=5)

    # --- Encadré équivalence pour ce K, sous les barres ---
    pts_per_note = amplitude / 9  # amplitude couvre 9 pts de moyenne (10→19)
    equiv_lines = f"K = {K:.1f}\n"
    for bonus in BONUS_LIST:
        equiv = bonus / pts_per_note
        equiv_lines += f" {bonus:>4d} ≈ {equiv:.1f} pts\n"
    equiv_lines = equiv_lines.rstrip("\n")

    ax.text(xc, s10 - 150, equiv_lines,
            ha="center", va="top", fontsize=7.5, fontfamily="monospace",
            color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#bdc3c7",
                      alpha=0.92),
            zorder=6)

# ═══════════════════════════════════════════════════════════════════════════
# DROITES HORIZONTALES — AMPLITUDE NOTES 2025
# ═══════════════════════════════════════════════════════════════════════════
ax.axhspan(6800, 8100, color="#e74c3c", alpha=0.07, zorder=0)
ax.axhline(y=6800, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
ax.axhline(y=8100, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
ax.text(x_ticks[0] - 0.48, 6800 + 40, "6 800 (~score min 2025)", va="bottom", fontsize=9,
        color="#e74c3c", fontweight="bold")
ax.text(x_ticks[0] - 0.48, 8100 + 40, "8 100 (score max 2025)", va="bottom", fontsize=9,
        color="#e74c3c", fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# LÉGENDE, AXES, TITRE
# ═══════════════════════════════════════════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor="#2c3e50", alpha=0.85,
                   label="Amplitude scolaire (19/20 − 10/20)"),
    mpatches.Patch(facecolor="#95a5a6", alpha=0.20,
                   label="Score socle (élève 10/20)"),
    mpatches.Patch(facecolor=BONUS_COLORS[400], alpha=0.85, label="Bonus IPS 400"),
    mpatches.Patch(facecolor=BONUS_COLORS[800], alpha=0.85, label="Bonus IPS 800"),
    mpatches.Patch(facecolor=BONUS_COLORS[1200], alpha=0.85, label="Bonus IPS 1200"),
    plt.Line2D([0], [0], color="#27ae60", linewidth=2, label="Repère 15/20"),
    plt.Line2D([0], [0], color="#2980b9", linewidth=2, label="Repère 12/20"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=True,
          fontsize=10, framealpha=0.95, ncol=2)

ax.set_xticks(x_ticks)
ax.set_xticklabels([f"K = {k:.1f}" for k in coefs], fontsize=12)
ax.set_xlabel("Coefficient multiplicateur K", fontsize=13, labelpad=10)
ax.set_ylabel("Score scolaire", fontsize=13, labelpad=10)
ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 0.5)

# Marge basse élargie pour les encadrés
y_min = min(K * score_10 for K in coefs) - 900
y_max = max(K * score_19 for K in coefs) + 500
ax.set_ylim(y_min, y_max)

ax.set_title(
    "ESTMATION 2026 DU POIDS RELATIF DU SCORE SCOLAIRE vs BONUS IPS (0 / 400 / 800 / 1200)",
    fontsize=15, fontweight="bold", pad=20
)

fig.text(0.5, 0.005,
         "Le % indique la part du bonus IPS dans l'amplitude scolaire (19−10/20). "
         "Encadrés : équivalence bonus → points de moyenne générale.",
         ha="center", fontsize=10, color="#7f8c8d", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])

out_path = "poids_scolaire_vs_tous_bonus_ips.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"→ Graphique sauvegardé : {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# TABLEAU RÉCAPITULATIF
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'K':>5s} {'Amp.':>7s} {'B400':>6s} {'%':>6s} "
      f"{'B800':>6s} {'%':>6s} {'B1200':>6s} {'%':>6s}")
print("─" * 58)
for K in coefs:
    amp = K * (score_19 - score_10)
    line = f"{K:>5.1f} {amp:>7.0f}"
    for b in BONUS_LIST:
        pct = 100 * b / amp
        line += f" {b:>6d} {pct:>5.1f}%"
    print(line)