import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES IPS (API HuggingFace - 2 LOTS)
# ==============================================================================
base_url = "https://datasets-server.huggingface.co/rows"
params_common = {
    "dataset": "fgaume/affelnet-paris-bonus-ips-colleges",
    "config": "default",
    "split": "ips"
}

all_rows = []

# Lot 1 (0-100)
try:
    r1 = requests.get(base_url, params={**params_common, "offset": 0, "length": 100})
    r1.raise_for_status()
    all_rows.extend([item['row'] for item in r1.json()['rows']])
except Exception as e:
    print(f"Erreur Lot 1: {e}")

# Lot 2 (100-200)
try:
    r2 = requests.get(base_url, params={**params_common, "offset": 100, "length": 100})
    r2.raise_for_status()
    all_rows.extend([item['row'] for item in r2.json()['rows']])
except Exception as e:
    pass

df_ips = pd.DataFrame(all_rows)

# Normalisation des noms de colonnes (le dataset utilise des majuscules variables)
col_map = {}
for c in df_ips.columns:
    cl = c.lower()
    if cl == 'identifiant':
        col_map[c] = 'uai'
    elif cl == 'nom':
        col_map[c] = 'nom'
    elif cl == 'secteur':
        col_map[c] = 'secteur'
    elif cl == 'ips_2025':
        col_map[c] = 'ips_2025'
    elif cl == 'bonus_ips_2025':
        col_map[c] = 'bonus_ips_2025'
    elif cl == 'ips_2026':
        col_map[c] = 'ips_2026'
    elif cl == 'bonus_ips_2026':
        col_map[c] = 'bonus_ips_2026'
df_ips.rename(columns=col_map, inplace=True)

# Vérification
for col in ['uai', 'nom', 'ips_2025', 'bonus_ips_2025', 'ips_2026', 'bonus_ips_2026', 'secteur']:
    if col not in df_ips.columns:
        print(f"Erreur: Colonne '{col}' manquante dans les données IPS.")
        exit()

df_ips['ips_2025'] = pd.to_numeric(df_ips['ips_2025'], errors='coerce')
df_ips['ips_2026'] = pd.to_numeric(df_ips['ips_2026'], errors='coerce')

# ==============================================================================
# 2. RÉCUPÉRATION DES DONNÉES DNB (API data.education.gouv.fr - session 2024)
# ==============================================================================
dnb_url = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-indicateurs-valeur-ajoutee-colleges/exports/json"
    "?select=uai%2Cnb_candidats_g%2Ctaux_de_reussite_g"
    "&lang=fr"
    "&refine=academie%3A%22PARIS%22"
    "&refine=session%3A%222025%22"
    "&timezone=Europe%2FBerlin"
)

try:
    r_dnb = requests.get(dnb_url)
    r_dnb.raise_for_status()
    df_dnb = pd.DataFrame(r_dnb.json())
except Exception as e:
    print(f"Erreur récupération DNB: {e}")
    exit()

df_dnb['nb_candidats_g'] = pd.to_numeric(df_dnb['nb_candidats_g'], errors='coerce')
df_dnb['taux_de_reussite_g'] = pd.to_numeric(df_dnb['taux_de_reussite_g'], errors='coerce')
df_dnb = df_dnb.dropna(subset=['nb_candidats_g', 'taux_de_reussite_g'])

# Effectifs admis au DNB
df_dnb['effectif_admis'] = (df_dnb['nb_candidats_g'] * df_dnb['taux_de_reussite_g'] / 100).apply(math.ceil)

# ==============================================================================
# 3. JOINTURE IPS + DNB
# ==============================================================================
df = df_ips.merge(df_dnb[['uai', 'effectif_admis', 'nb_candidats_g']], on='uai', how='left')

# Rapport de couverture
n_total = len(df)
n_matched = df['effectif_admis'].notna().sum()
n_missing = n_total - n_matched
print(f"\n{'='*60}")
print(f"COUVERTURE DE LA JOINTURE IPS ↔ DNB")
print(f"{'='*60}")
print(f"  Collèges IPS total     : {n_total}")
print(f"  Avec effectifs DNB     : {n_matched} ({n_matched/n_total*100:.0f}%)")
print(f"  Sans effectifs DNB     : {n_missing}")

if n_missing > 0:
    manquants = df[df['effectif_admis'].isna()][['nom', 'uai', 'secteur']]
    print(f"\n  Collèges sans données DNB :")
    for _, row in manquants.iterrows():
        print(f"    - {row['nom']} ({row['uai']}, {row['secteur']})")

# On ne garde que les collèges avec effectifs
df = df.dropna(subset=['effectif_admis'])
df['effectif_admis'] = df['effectif_admis'].astype(int)

# ==============================================================================
# 4. LOGIQUE DE CALCUL 2026
# ==============================================================================
df['Bonus 2026'] = df['bonus_ips_2026'].fillna(0)
df.rename(columns={'bonus_ips_2025': 'Bonus 2025'}, inplace=True)

# Séparation public / tous
df_public = df[df['secteur'] == 'Public'].copy()

# ==============================================================================
# 5. FONCTION GRAPHIQUE PONDÉRÉE PAR EFFECTIFS
# ==============================================================================
def tracer_graphique(dataset, titre, ax_cible):
    cats = [0, 400, 600, 800, 1200]

    # Somme des effectifs par tranche de bonus
    e25 = dataset.groupby('Bonus 2025')['effectif_admis'].sum().reindex(cats, fill_value=0)
    e26 = dataset.groupby('Bonus 2026')['effectif_admis'].sum().reindex(cats, fill_value=0)

    # Préparation DataFrame pour Seaborn
    d_plot = pd.DataFrame({'2025': e25, '2026': e26})
    d_melt = d_plot.reset_index().melt(id_vars='index', var_name='Année', value_name='Élèves')

    # Dessin
    sns.barplot(
        data=d_melt, x='index', y='Élèves', hue='Année',
        palette={'2025': '#3498db', '2026': '#e74c3c'},
        edgecolor="black", ax=ax_cible
    )

    # Pourcentages
    total_2025 = e25.sum()
    total_2026 = e26.sum()

    for i, container in enumerate(ax_cible.containers):
        total = total_2025 if i == 0 else total_2026
        for bar in container:
            height = bar.get_height()
            if height > 0:
                pct = (height / total) * 100
                ax_cible.annotate(f'{int(height)}\n({pct:.1f}%)',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

    # Cosmétique
    ax_cible.set_title(titre, fontweight='bold', fontsize=12, pad=10)
    ax_cible.set_xlabel("Montant du Bonus (Points)", fontsize=10)
    ax_cible.set_ylabel("Nombre d'Élèves (admis DNB 2024)", fontsize=10)
    ax_cible.grid(axis='y', linestyle='--', alpha=0.6)

    ylim_max = d_melt['Élèves'].max() * 1.25
    ax_cible.set_ylim(0, ylim_max)

# ==============================================================================
# 6. GÉNÉRATION DU GRAPHIQUE
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
tracer_graphique(df_public,
    "COLLÈGES PUBLICS — Répartition des ÉLÈVES par bonus IPS\n(pondéré par effectifs admis DNB 2024)", ax1)
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(10, 6))
tracer_graphique(df,
    "TOUS COLLÈGES — Répartition des ÉLÈVES par bonus IPS\n(pondéré par effectifs admis DNB 2024)", ax2)
fig2.tight_layout()

plt.show()

# ==============================================================================
# 7. ANALYSE CONSOLE
# ==============================================================================
print(f"\n{'='*60}")
print("ANALYSE PONDÉRÉE PAR EFFECTIFS — SECTEUR PUBLIC")
print(f"{'='*60}")

total_eleves = df_public['effectif_admis'].sum()
print(f"\nTotal élèves admis DNB (public, avec données) : {total_eleves}")

for bonus in [0, 400, 600, 800, 1200]:
    n25 = df_public[df_public['Bonus 2025'] == bonus]['effectif_admis'].sum()
    n26 = df_public[df_public['Bonus 2026'] == bonus]['effectif_admis'].sum()
    delta = n26 - n25
    signe = "+" if delta >= 0 else ""
    print(f"  Bonus {bonus:>4} pts : 2025={n25:>5} élèves  →  2026={n26:>5} élèves  ({signe}{delta})")

# Analyse des changements de bonus, groupés par ancien bonus
changed = df_public[df_public['Bonus 2025'] != df_public['Bonus 2026']].copy()
changed['delta'] = changed['Bonus 2026'] - changed['Bonus 2025']

for ancien_bonus in [0, 600, 1200]:
    groupe = changed[changed['Bonus 2025'] == ancien_bonus]
    if groupe.empty:
        continue

    print(f"\n{'─'*60}")
    print(f"ANCIEN BONUS {ancien_bonus} pts → {len(groupe)} collège(s) ont changé")
    print(f"{'─'*60}")

    for nouveau_bonus, sous_groupe in groupe.groupby('Bonus 2026'):
        direction = "↑" if nouveau_bonus > ancien_bonus else "↓"
        print(f"\n  {direction} Vers {int(nouveau_bonus)} pts ({len(sous_groupe)} collèges, "
              f"{sous_groupe['effectif_admis'].sum()} élèves) :")
        for _, row in sous_groupe.sort_values('ips_2025').iterrows():
            print(f"    {row['nom']:<45} IPS={row['ips_2025']:.0f}  effectif={row['effectif_admis']}")

print(f"\n{'─'*60}")
print(f"TOTAL : {len(changed)} collèges publics ont changé de bonus "
      f"({changed['effectif_admis'].sum()} élèves)")
