#!/usr/bin/env python3
"""
Gain d'un point de moyenne générale exprimé en % d'une tranche IPS (400 pts)
=============================================================================
K de référence : 2.0 à 2.5
Stats brutes (sans tranchage)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

BONUS_IPS = 400

def score_scolaire(note_partout):
    total = 0.0
    for champ, s in STATS.items():
        H = 10 * (10 + (note_partout - s["mu"]) / s["sigma"])
        total += H * POIDS[champ]
    return total

score_19 = score_scolaire(19)
score_10 = score_scolaire(10)
amplitude_base = score_19 - score_10  # pour K=1

# Valeur d'1 point de moyenne = K × amplitude_base / 9
# En % de bonus 400 = (K × amplitude_base / 9) / 400 × 100

def gain_1pt_pct(K):
    return K * amplitude_base / 9 / BONUS_IPS * 100

# ═══════════════════════════════════════════════════════════════════════════
# TABLEAU
# ═══════════════════════════════════════════════════════════════════════════
coefs_discrets = np.arange(2.0, 2.55, 0.1)

print("=" * 60)
print("  ESTIMATION DU GAIN D'1 POINT DE MOYENNE EN % D'UNE TRANCHE IPS (400)")
print("=" * 60)
print(f"\n  Amplitude de base (K=1) : {amplitude_base:.1f} pts pour 9 pts de moyenne")
print(f"  → 1 pt de moyenne (K=1) = {amplitude_base/9:.1f} pts Affelnet\n")

print(f"  {'K':>5s} │ {'1 pt (score)':>12s} │ {'% de 400':>10s} │ {'Interprétation'}")
print("  " + "─" * 65)
for K in coefs_discrets:
    val_1pt = K * amplitude_base / 9
    pct = gain_1pt_pct(K)
    # Combien de pts de moyenne pour "valoir" un bonus 400 ?
    pts_pour_400 = BONUS_IPS / val_1pt
    print(f"  {K:>5.1f} │ {val_1pt:>12.1f} │ {pct:>9.1f}% │ "
          f"1 bonus 400 = {pts_pour_400:.1f} pts de moy.")

# ═══════════════════════════════════════════════════════════════════════════
# COURBE CONTINUE
# ═══════════════════════════════════════════════════════════════════════════
K_cont = np.linspace(2.0, 2.5, 200)
pct_cont = gain_1pt_pct(K_cont)

fig, ax1 = plt.subplots(figsize=(12, 7))
plt.style.use("seaborn-v0_8-whitegrid")

# Courbe principale
ax1.plot(K_cont, pct_cont, color="#2c3e50", linewidth=3, zorder=3)
ax1.fill_between(K_cont, pct_cont, alpha=0.08, color="#2c3e50")

# Points discrets
pct_discrets = gain_1pt_pct(coefs_discrets)
ax1.scatter(coefs_discrets, pct_discrets, s=120, color="#e67e22",
            edgecolors="white", linewidth=2, zorder=4)

# Annotations sur chaque point
for K, pct in zip(coefs_discrets, pct_discrets):
    val_1pt = K * amplitude_base / 9
    pts_pour_400 = BONUS_IPS / val_1pt
    ax1.annotate(f"{pct:.1f}%\n(400 = {pts_pour_400:.1f} pts moy.)",
                 xy=(K, pct), xytext=(0, 22), textcoords="offset points",
                 ha="center", fontsize=9, fontweight="bold", color="#2c3e50",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa",
                           ec="#bdc3c7", alpha=0.9))

# Axes
ax1.set_xlabel("Coefficient multiplicateur K", fontsize=13, labelpad=10)
ax1.set_ylabel("Valeur d'1 point de moyenne\n(en % d'une tranche IPS = 400 pts)",
               fontsize=12, labelpad=10)
ax1.set_xlim(1.95, 2.55)
ax1.set_ylim(pct_cont.min() - 3, pct_cont.max() + 8)
ax1.set_xticks(coefs_discrets)
ax1.set_xticklabels([f"{k:.1f}" for k in coefs_discrets], fontsize=11)

# Lignes de référence
for ref_pct, ref_label, ref_color in [
    (50, "1 pt = ½ tranche IPS", "#e74c3c"),
    (25, "1 pt = ¼ tranche IPS", "#2980b9"),
]:
    if ax1.get_ylim()[0] < ref_pct < ax1.get_ylim()[1]:
        ax1.axhline(y=ref_pct, color=ref_color, linestyle="--",
                    linewidth=1.5, alpha=0.6)
        ax1.text(2.52, ref_pct + 0.5, ref_label, color=ref_color,
                 fontsize=9, ha="right", fontweight="bold", alpha=0.8)

# Titre
ax1.set_title(
    "ESTIMATION 2026 DE LA VALEUR D'UN POINT DE MOYENNE GÉNÉRALE\n"
    "exprimée en % d'une tranche de bonus IPS (400 pts)\n",
    fontsize=14, fontweight="bold", pad=20
)

# Note de lecture
fig.text(0.5, 0.01,
         "Lecture : à K=2.3, gagner 1 point de moyenne vaut 46% d'une tranche IPS. "
         "Il faut donc ~2.2 pts de moyenne pour compenser un bonus de 400.",
         ha="center", fontsize=10, color="#7f8c8d", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_path = "gain_1pt_moyenne_vs_bonus_ips.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n→ Graphique sauvegardé : {out_path}")
