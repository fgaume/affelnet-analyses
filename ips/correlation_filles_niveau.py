"""
Corrélation entre le niveau scolaire d'un lycée (% mentions TB au bac)
et la proportion de filles en terminale générale.

Deux contextes : France entière puis Paris uniquement.
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ==============================================================================
# 1. DONNÉES BAC — Taux de mentions TB (année 2023, toute la France)
# ==============================================================================
print("▶ Récupération des résultats du bac 2023 (France entière)…")

BAC_YEAR = 2023
bac_url = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-indicateurs-de-resultat-des-lycees-gt_v2/exports/json"
)
bac_params = {
    "select": "uai,libelle_uai,libelle_academie,nb_mentions_tb_avecf_g,"
              "nb_mentions_tb_sansf_g,presents_gnle",
    "where": (
        f"(`annee` >= date'{BAC_YEAR}-01-01') "
        f"AND (`annee` < date'{BAC_YEAR + 1}-01-01') "
        f"AND (`presents_gnle` > 0)"
    ),
    "timezone": "Europe/Berlin",
    "limit": "-1",
}

r_bac = requests.get(bac_url, params=bac_params)
r_bac.raise_for_status()
df_bac = pd.DataFrame(r_bac.json())

for col in ["nb_mentions_tb_avecf_g", "nb_mentions_tb_sansf_g", "presents_gnle"]:
    df_bac[col] = pd.to_numeric(df_bac[col], errors="coerce")

df_bac["nb_mentions_tb_avecf_g"] = df_bac["nb_mentions_tb_avecf_g"].fillna(0)
df_bac["nb_mentions_tb_sansf_g"] = df_bac["nb_mentions_tb_sansf_g"].fillna(0)
df_bac = df_bac.dropna(subset=["presents_gnle"])
df_bac = df_bac[df_bac["presents_gnle"] > 0]

df_bac["taux_tb"] = (
    (df_bac["nb_mentions_tb_avecf_g"] + df_bac["nb_mentions_tb_sansf_g"])
    / df_bac["presents_gnle"]
)

print(f"  {len(df_bac)} lycées avec résultats bac {BAC_YEAR}")

# ==============================================================================
# 2. DONNÉES EFFECTIFS — Proportion de filles en terminale G (rentrée 2022)
# ==============================================================================
# Les élèves en terminale à la rentrée N passent le bac en année N+1
RENTREE = BAC_YEAR - 1
print(f"\n▶ Récupération des effectifs terminale G, rentrée {RENTREE}…")

eff_url = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-lycee_gt-effectifs-niveau-sexe-lv/exports/json"
)
eff_params = {
    "select": "numero_lycee,terminales_g,terminales_g_filles",
    "where": f"`rentree_scolaire` = '{RENTREE}' AND `terminales_g` > 0",
    "timezone": "Europe/Berlin",
    "limit": "-1",
}

r_eff = requests.get(eff_url, params=eff_params)
r_eff.raise_for_status()
df_eff = pd.DataFrame(r_eff.json())

for col in ["terminales_g", "terminales_g_filles"]:
    df_eff[col] = pd.to_numeric(df_eff[col], errors="coerce")

df_eff = df_eff.dropna(subset=["terminales_g", "terminales_g_filles"])
df_eff = df_eff[df_eff["terminales_g"] > 0]
df_eff["pct_filles"] = df_eff["terminales_g_filles"] / df_eff["terminales_g"] * 100

df_eff.rename(columns={"numero_lycee": "uai"}, inplace=True)
print(f"  {len(df_eff)} lycées avec effectifs terminale G")

# ==============================================================================
# 3. JOINTURE
# ==============================================================================
df = df_bac.merge(df_eff[["uai", "pct_filles", "terminales_g"]], on="uai", how="inner")
df = df[df["taux_tb"].notna() & df["pct_filles"].notna()]

# Filtrer les lycées trop petits (< 10 élèves en terminale G)
df = df[df["terminales_g"] >= 10]

print(f"\n  Jointure : {len(df)} lycées (France)")

df_paris = df[df["libelle_academie"] == "PARIS"].copy()
print(f"  Dont Paris : {len(df_paris)} lycées")


# ==============================================================================
# 4. FONCTION DE TRACÉ
# ==============================================================================
def plot_correlation(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    color: str = "#3498db",
) -> tuple[float, float]:
    """Trace le nuage de points + régression linéaire. Renvoie (r, p)."""
    x = data["pct_filles"].values
    y = data["taux_tb"].values * 100  # en %

    ax.scatter(x, y, c=color, alpha=0.4, s=25, edgecolors="white", linewidth=0.3)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        "k--",
        linewidth=2,
        label=f"r = {r_value:.3f}  (r² = {r_value**2:.3f}, p = {p_value:.2e})",
    )

    ax.set_xlabel("Proportion de filles en terminale G (%)", fontsize=11)
    ax.set_ylabel("Taux de mentions TB au bac (%)", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)

    return r_value, p_value


# ==============================================================================
# 5. FIGURES
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

r_fr, p_fr = plot_correlation(
    axes[0],
    df,
    f"France entière — {len(df)} lycées (bac {BAC_YEAR})",
    color="#3498db",
)

r_paris, p_paris = plot_correlation(
    axes[1],
    df_paris,
    f"Paris — {len(df_paris)} lycées (bac {BAC_YEAR})",
    color="#e74c3c",
)

fig.suptitle(
    "Corrélation : proportion de filles en terminale G vs taux de mentions TB au bac",
    fontweight="bold",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()

# ==============================================================================
# 6. ANALYSE CONSOLE
# ==============================================================================
print(f"\n{'=' * 70}")
print("CORRÉLATION  % FILLES EN TERMINALE G  vs  TAUX MENTIONS TB AU BAC")
print(f"{'=' * 70}")

print(f"\n  FRANCE ENTIÈRE (n = {len(df)})")
print(f"    Pearson  r = {r_fr:.4f}  (p = {p_fr:.2e})")
print(f"    r² = {r_fr**2:.4f}")

spearman_fr, sp_p_fr = stats.spearmanr(df["pct_filles"], df["taux_tb"])
print(f"    Spearman ρ = {spearman_fr:.4f}  (p = {sp_p_fr:.2e})")

print(f"\n  PARIS (n = {len(df_paris)})")
print(f"    Pearson  r = {r_paris:.4f}  (p = {p_paris:.2e})")
print(f"    r² = {r_paris**2:.4f}")

spearman_paris, sp_p_paris = stats.spearmanr(df_paris["pct_filles"], df_paris["taux_tb"])
print(f"    Spearman ρ = {spearman_paris:.4f}  (p = {sp_p_paris:.2e})")

# Statistiques descriptives
print(f"\n  STATS DESCRIPTIVES")
print(f"    France — % filles : µ={df['pct_filles'].mean():.1f}%, "
      f"σ={df['pct_filles'].std():.1f}%, "
      f"min={df['pct_filles'].min():.1f}%, max={df['pct_filles'].max():.1f}%")
print(f"    France — taux TB  : µ={df['taux_tb'].mean()*100:.1f}%, "
      f"σ={df['taux_tb'].std()*100:.1f}%")
print(f"    Paris  — % filles : µ={df_paris['pct_filles'].mean():.1f}%, "
      f"σ={df_paris['pct_filles'].std():.1f}%")
print(f"    Paris  — taux TB  : µ={df_paris['taux_tb'].mean()*100:.1f}%, "
      f"σ={df_paris['taux_tb'].std()*100:.1f}%")

plt.show()
