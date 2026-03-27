import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
from scipy import stats

# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES IPS (JSON HuggingFace)
# ==============================================================================
ips_url = "https://huggingface.co/datasets/fgaume/affelnet-paris-bonus-ips-colleges/raw/main/affelnet-paris-bonus-ips-colleges.json"
r_ips = requests.get(ips_url)
r_ips.raise_for_status()
df_ips = pd.DataFrame(r_ips.json())

# Normalisation des colonnes
col_map = {}
for c in df_ips.columns:
    cl = c.lower()
    if cl == 'identifiant': col_map[c] = 'uai'
    elif cl == 'nom': col_map[c] = 'nom'
    elif cl == 'secteur': col_map[c] = 'secteur'
    elif cl.startswith('ips_'): col_map[c] = cl
    elif cl.startswith('bonus_ips_'): col_map[c] = cl
df_ips.rename(columns=col_map, inplace=True)

# Colonnes IPS par année
ips_cols = {2021: 'ips_2021', 2022: 'ips_2022', 2023: 'ips_2023', 2024: 'ips_2024', 2025: 'ips_2025'}
for col in ips_cols.values():
    df_ips[col] = pd.to_numeric(df_ips[col], errors='coerce')

# ==============================================================================
# 2. EXTRAPOLATION IPS 2026 PAR RÉGRESSION LINÉAIRE
# ==============================================================================
# Pour chaque collège, régression linéaire sur les IPS 2021-2025
# puis projection en 2026
years = np.array(list(ips_cols.keys()))

def extrapoler_ips_2026(row):
    """Régression linéaire sur les IPS disponibles, projection en 2026."""
    valeurs = [row[col] for col in ips_cols.values()]
    # Filtrer les NaN
    pairs = [(y, v) for y, v in zip(years, valeurs) if pd.notna(v)]
    if len(pairs) < 2:
        # Pas assez de données : on prend la dernière valeur connue
        non_na = [v for v in valeurs if pd.notna(v)]
        return non_na[-1] if non_na else np.nan
    y_arr = np.array([p[0] for p in pairs])
    v_arr = np.array([p[1] for p in pairs])
    slope, intercept, _, _, _ = stats.linregress(y_arr, v_arr)
    return slope * 2026 + intercept

df_ips['ips_2026_extrapole'] = df_ips.apply(extrapoler_ips_2026, axis=1)
df_ips = df_ips.dropna(subset=['ips_2026_extrapole'])

# Pente annuelle pour info
def pente_ips(row):
    valeurs = [row[col] for col in ips_cols.values()]
    pairs = [(y, v) for y, v in zip(years, valeurs) if pd.notna(v)]
    if len(pairs) < 2:
        return 0.0
    y_arr = np.array([p[0] for p in pairs])
    v_arr = np.array([p[1] for p in pairs])
    slope, _, _, _, _ = stats.linregress(y_arr, v_arr)
    return slope

df_ips['pente_annuelle'] = df_ips.apply(pente_ips, axis=1)

# ==============================================================================
# 3. CALCUL DES BONUS 2025 ET 2026
# ==============================================================================
# Bonus 2025 : ancien barème (0, 600, 1200) — on le prend directement du dataset
df_ips['bonus_ips_2025'] = pd.to_numeric(df_ips['bonus_ips_2025'], errors='coerce')

# Bonus 2026 : nouveau barème (0, 400, 800, 1200) appliqué à l'IPS extrapolé
def calcul_bonus_2026(ips):
    if ips < 105: return 1200
    elif ips < 117: return 800
    elif ips < 130: return 400
    else: return 0

df_ips['bonus_2026'] = df_ips['ips_2026_extrapole'].apply(calcul_bonus_2026)

# ==============================================================================
# 4. RÉCUPÉRATION DES EFFECTIFS DNB (session 2024)
# ==============================================================================
dnb_url = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-indicateurs-valeur-ajoutee-colleges/exports/json"
    "?select=uai%2Cnb_candidats_g%2Ctaux_de_reussite_g"
    "&lang=fr"
    "&refine=academie%3A%22PARIS%22"
    "&refine=session%3A%222024%22"
    "&timezone=Europe%2FBerlin"
)
r_dnb = requests.get(dnb_url)
r_dnb.raise_for_status()
df_dnb = pd.DataFrame(r_dnb.json())
df_dnb['nb_candidats_g'] = pd.to_numeric(df_dnb['nb_candidats_g'], errors='coerce')
df_dnb['taux_de_reussite_g'] = pd.to_numeric(df_dnb['taux_de_reussite_g'], errors='coerce')
df_dnb = df_dnb.dropna(subset=['nb_candidats_g', 'taux_de_reussite_g'])
df_dnb['effectif_admis'] = (df_dnb['nb_candidats_g'] * df_dnb['taux_de_reussite_g'] / 100).apply(math.ceil)

# ==============================================================================
# 5. JOINTURE + FILTRE PUBLIC
# ==============================================================================
df = df_ips.merge(df_dnb[['uai', 'effectif_admis']], on='uai', how='left')
df = df.dropna(subset=['effectif_admis'])
df['effectif_admis'] = df['effectif_admis'].astype(int)

df_pub = df[df['secteur'] == 'Public'].copy()

# ==============================================================================
# 6. STATISTIQUES EXTRAPOLATION
# ==============================================================================
print(f"\n{'='*70}")
print("EXTRAPOLATION IPS 2026 — MÉTHODE : RÉGRESSION LINÉAIRE (2021→2025)")
print(f"{'='*70}")
print(f"  Collèges publics avec effectifs : {len(df_pub)}")
print(f"  Pente annuelle médiane : {df_pub['pente_annuelle'].median():+.2f} pts/an")
print(f"  IPS 2025 moyen (public) : {df_pub['ips_2025'].mean():.1f}")
print(f"  IPS 2026 extrapolé moyen (public) : {df_pub['ips_2026_extrapole'].mean():.1f}")

# ==============================================================================
# 7. MATRICE DE TRANSITION — EFFECTIFS ÉLÈVES
# ==============================================================================
groupes_2025 = sorted(df_pub['bonus_ips_2025'].unique())
groupes_2026 = [0, 400, 800, 1200]

# Couleurs : 0=rouge, 400=orange, 600=orange, 800=bleu, 1200=vert
colors_ips = {0: '#e74c3c', 400: '#f39c12', 600: '#f39c12', 800: '#3498db', 1200: '#2ecc71'}

print(f"\n{'='*70}")
print("MATRICE DE TRANSITION — COLLÈGES PUBLICS (effectifs élèves)")
print("Bonus 2025 (lignes) → Bonus 2026 extrapolé (colonnes)")
print(f"{'='*70}")
print(f"{'':>20}", end="")
for g26 in groupes_2026:
    print(f"  → {g26:>4} pts", end="")
print("     Total 2025")
print("-" * 70)

for g25 in groupes_2025:
    mask25 = df_pub['bonus_ips_2025'] == g25
    total_g25 = df_pub[mask25]['effectif_admis'].sum()
    print(f"  Bonus {g25:>4} pts  ", end="")
    for g26 in groupes_2026:
        mask26 = df_pub['bonus_2026'] == g26
        eff = df_pub[mask25 & mask26]['effectif_admis'].sum()
        if eff > 0:
            print(f"  {eff:>6}", end="")
        else:
            print(f"  {'—':>6}", end="")
    print(f"    {total_g25:>6}")

print("-" * 70)
print(f"  Total 2026      ", end="")
total_global = 0
for g26 in groupes_2026:
    eff = df_pub[df_pub['bonus_2026'] == g26]['effectif_admis'].sum()
    print(f"  {eff:>6}", end="")
    total_global += eff
print(f"    {total_global:>6}")

# ==============================================================================
# 8. DÉTAIL DES MOUVEMENTS DU GROUPE 600 pts
# ==============================================================================
print(f"\n{'='*70}")
print("DÉTAIL : QUE DEVIENNENT LES ÉLÈVES À 600 PTS EN 2025 ?")
print(f"{'='*70}")

mask_600 = df_pub['bonus_ips_2025'] == 600
base_600 = df_pub[mask_600]

for g26 in groupes_2026:
    sub = base_600[base_600['bonus_2026'] == g26]
    if len(sub) > 0:
        eff = sub['effectif_admis'].sum()
        print(f"\n  → {g26} pts en 2026 : {len(sub)} collèges, {eff} élèves")
        for _, row in sub.sort_values('ips_2026_extrapole').iterrows():
            print(f"    {row['nom']:<40} IPS 2025={row['ips_2025']:.0f}  "
                  f"IPS 2026 extrap.={row['ips_2026_extrapole']:.1f}  "
                  f"effectif={row['effectif_admis']}")

# ==============================================================================
# 9. GRAPHIQUE DE TRANSITION (stacked bars)
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 7))
bar_width = 0.5

for idx_26, g26 in enumerate(groupes_2026):
    bottom = 0
    for g25 in groupes_2025:
        mask = (df_pub['bonus_ips_2025'] == g25) & (df_pub['bonus_2026'] == g26)
        eff = df_pub[mask]['effectif_admis'].sum()
        if eff > 0:
            bar = ax.bar(idx_26, eff, bottom=bottom, width=bar_width,
                         color=colors_ips.get(g25, '#bdc3c7'), edgecolor='white',
                         linewidth=1.5, zorder=2)
            if eff > 80:
                ax.text(idx_26, bottom + eff / 2, f"{eff}",
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if g25 in [1200, 0] else 'black')
            bottom += eff

    if bottom > 0:
        pct = bottom / df_pub['effectif_admis'].sum() * 100
        ax.text(idx_26, bottom + 50, f"{bottom}\n({pct:.0f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(groupes_2026)))
ax.set_xticklabels([f"{g} pts" for g in groupes_2026], fontsize=11)
ax.set_xlabel("Bonus IPS 2026 (nouveau barème, IPS extrapolés)", fontsize=12, labelpad=10)
ax.set_ylabel("Nombre d'élèves (admis DNB 2024)", fontsize=12)
ax.set_title("Transition bonus IPS 2025 → 2026 (public)\n"
             "IPS 2026 estimés par régression linéaire sur 2021-2025",
             fontweight='bold', fontsize=13, pad=15)

legend_patches = [mpatches.Patch(color=colors_ips[g], label=f"Ancien bonus {g} pts (2025)")
                  for g in groupes_2025 if g in colors_ips]
ax.legend(handles=legend_patches, loc='upper right', fontsize=10,
          title="Origine (bonus 2025)", title_fontsize=11)

ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

plt.tight_layout()

# ==============================================================================
# 10. GRAPHIQUE DE TRANSITION — TOUS COLLÈGES (public + privé)
# ==============================================================================
groupes_2025_all = sorted(df['bonus_ips_2025'].unique())

fig2, ax2 = plt.subplots(figsize=(11, 7))

for idx_26, g26 in enumerate(groupes_2026):
    bottom = 0
    for g25 in groupes_2025_all:
        mask = (df['bonus_ips_2025'] == g25) & (df['bonus_2026'] == g26)
        eff = df[mask]['effectif_admis'].sum()
        if eff > 0:
            bar = ax2.bar(idx_26, eff, bottom=bottom, width=bar_width,
                          color=colors_ips.get(g25, '#bdc3c7'), edgecolor='white',
                          linewidth=1.5, zorder=2)
            if eff > 80:
                ax2.text(idx_26, bottom + eff / 2, f"{eff}",
                         ha='center', va='center', fontsize=10, fontweight='bold',
                         color='white' if g25 in [1200, 0] else 'black')
            bottom += eff

    if bottom > 0:
        pct = bottom / df['effectif_admis'].sum() * 100
        ax2.text(idx_26, bottom + 50, f"{bottom}\n({pct:.0f}%)",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xticks(range(len(groupes_2026)))
ax2.set_xticklabels([f"{g} pts" for g in groupes_2026], fontsize=11)
ax2.set_xlabel("Bonus IPS 2026 (nouveau barème, IPS extrapolés)", fontsize=12, labelpad=10)
ax2.set_ylabel("Nombre d'élèves (admis DNB 2024)", fontsize=12)
ax2.set_title("Transition bonus IPS 2025 → 2026 (tous collèges : public + privé)\n"
              "IPS 2026 estimés par régression linéaire sur 2021-2025",
              fontweight='bold', fontsize=13, pad=15)

legend_patches_all = [mpatches.Patch(color=colors_ips[g], label=f"Ancien bonus {g} pts (2025)")
                      for g in groupes_2025_all if g in colors_ips]
ax2.legend(handles=legend_patches_all, loc='upper right', fontsize=10,
           title="Origine (bonus 2025)", title_fontsize=11)

ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.12)

plt.tight_layout()
plt.show()
