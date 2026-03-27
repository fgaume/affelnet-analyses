import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from urllib.parse import quote

# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES LYCÉES (API Opendata Éducation Nationale)
# ==============================================================================
base_url = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-indicateurs-de-resultat-des-lycees-gt_v2/exports/json/"
)

select = "annee,uai,libelle_uai,secteur,nb_mentions_tb_avecf_g,nb_mentions_tb_sansf_g,eff_2nde,eff_term,presents_gnle"

# On récupère toutes les années disponibles (publics et privés)
# pour pouvoir faire la régression linéaire
all_dfs = []
for year in range(2019, 2024):
    where_clause = (
        f"((`libelle_academie` = \"PARIS\")) "
        f"AND ((`annee` >= date'{year}-01-01')) "
        f"AND ((`annee` < date'{year+1}-01-01')) "
        f"AND ((`presents_gnle` > -1)) "
        f"AND ((`eff_2nde` > 0))"
    )
    params = {
        "select": select,
        "where": where_clause,
        "timezone": "Europe/Berlin",
    }
    try:
        r = requests.get(base_url, params=params)
        r.raise_for_status()
        data = r.json()
        if data:
            df_year = pd.DataFrame(data)
            df_year['year'] = year
            all_dfs.append(df_year)
            print(f"  Année {year} : {len(df_year)} lycées récupérés")
    except Exception as e:
        print(f"  Année {year} : erreur — {e}")

df_all = pd.concat(all_dfs, ignore_index=True)

# Nettoyage numérique
for col in ['nb_mentions_tb_avecf_g', 'nb_mentions_tb_sansf_g', 'presents_gnle', 'eff_2nde']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

df_all = df_all.dropna(subset=['nb_mentions_tb_avecf_g', 'nb_mentions_tb_sansf_g', 'presents_gnle'])
df_all = df_all[df_all['presents_gnle'] > 0]

# Taux de mentions TB par année
df_all['taux_tb'] = (df_all['nb_mentions_tb_avecf_g'] + df_all['nb_mentions_tb_sansf_g']) / df_all['presents_gnle']

# ==============================================================================
# 2. EXTRAPOLATION DU TAUX TB PAR RÉGRESSION LINÉAIRE
# ==============================================================================
# Pour chaque lycée, régression linéaire sur les taux TB disponibles, projection
lycees = df_all.groupby('uai')

results = []
for uai, group in lycees:
    nom = group['libelle_uai'].iloc[0]
    secteur = group['secteur'].iloc[0]
    eff_2nde = group.sort_values('year')['eff_2nde'].iloc[-1]  # dernier effectif connu

    years_available = group['year'].values
    taux_values = group['taux_tb'].values

    if len(years_available) >= 2:
        slope, intercept, _, _, _ = stats.linregress(years_available, taux_values)
        taux_extrapole = slope * 2024 + intercept
    else:
        taux_extrapole = taux_values[0]
        slope = 0.0

    # Dernier taux connu (année la plus récente)
    last_year = group.loc[group['year'].idxmax()]
    taux_dernier = last_year['taux_tb']

    results.append({
        'uai': uai,
        'nom': nom,
        'secteur': secteur,
        'eff_2nde': eff_2nde,
        'taux_tb_dernier': taux_dernier,
        'taux_tb_extrapole': taux_extrapole,
        'pente_annuelle': slope,
        'nb_annees': len(years_available),
    })

df_lycees = pd.DataFrame(results)
df_lycees['taux_tb_extrapole'] = df_lycees['taux_tb_extrapole'].clip(lower=0, upper=1)

print(f"\n{'='*70}")
print(f"LYCÉES PARISIENS — EXTRAPOLATION TAUX MENTIONS TB")
print(f"{'='*70}")
print(f"  Lycées avec données : {len(df_lycees)}")
print(f"  Taux TB extrapolé moyen : {df_lycees['taux_tb_extrapole'].mean()*100:.1f}%")
print(f"  Pente annuelle médiane : {df_lycees['pente_annuelle'].median()*100:+.2f} pts/an")

# ==============================================================================
# 3. CALCUL DES DÉCILES
# ==============================================================================
df_lycees = df_lycees.sort_values('taux_tb_extrapole', ascending=True).reset_index(drop=True)
df_lycees['decile'] = pd.qcut(df_lycees['taux_tb_extrapole'], 10, labels=False, duplicates='drop') + 1

print(f"\n{'='*70}")
print("DÉCILES DE NIVEAU SCOLAIRE (taux mentions TB extrapolé)")
print("Décile 1 = plus faibles, Décile 10 = meilleurs taux TB")
print(f"{'='*70}")

for d in sorted(df_lycees['decile'].unique()):
    sub = df_lycees[df_lycees['decile'] == d].sort_values('taux_tb_extrapole', ascending=False)
    print(f"\n  DÉCILE {d} ({len(sub)} lycées)")
    for _, row in sub.iterrows():
        print(f"    {row['nom']:<45} TB={row['taux_tb_extrapole']*100:.1f}%  "
              f"({row['secteur']}, eff 2nde={int(row['eff_2nde'])})")

# ==============================================================================
# 4. GRAPHIQUE — DÉCILES EN BARRES VERTICALES AVEC NOMS DES LYCÉES
# ==============================================================================
fig, ax = plt.subplots(figsize=(16, 9))

# Palette de couleurs dégradée (vert → rouge)
cmap = plt.cm.RdYlGn
n_deciles = df_lycees['decile'].nunique()
deciles_sorted = sorted(df_lycees['decile'].unique())
colors = [cmap(i / (n_deciles - 1)) for i in range(n_deciles)]

for idx, d in enumerate(deciles_sorted):
    sub = df_lycees[df_lycees['decile'] == d].sort_values('taux_tb_extrapole', ascending=False)
    taux_moy = sub['taux_tb_extrapole'].mean() * 100

    # Barre verticale : taux moyen du décile
    ax.bar(idx, taux_moy, width=0.65, color=colors[idx],
           edgecolor='white', linewidth=1.5, zorder=2)

    # Taux moyen au-dessus de la barre
    ax.text(idx, taux_moy + 0.8, f"{taux_moy:.1f}%",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Noms des lycées dans la barre (empilés verticalement)
    noms = sub['nom'].str.replace('LYCEE ', '', regex=False) \
                      .str.replace('ECOLE ', '', regex=False).values
    texte = "\n".join(noms)
    # Couleur du texte adaptée à la luminosité du fond
    r, g, b, _ = colors[idx]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = '#222222' if luminance > 0.5 else 'white'
    ax.text(idx, taux_moy / 2, texte,
            ha='center', va='center', fontsize=5.5, color=text_color,
            fontweight='bold', linespacing=1.3, zorder=3)

ax.set_xticks(range(len(deciles_sorted)))
ax.set_xticklabels([f"D{d}" for d in deciles_sorted], fontsize=12, fontweight='bold')
ax.set_xlabel("Décile (D1 = plus faibles, D10 = meilleurs)", fontsize=12, labelpad=10)
ax.set_ylabel("Taux moyen de mentions Très Bien au Bac (%)", fontsize=12)
ax.set_title("Déciles de niveau scolaire des lycées de Paris\n"
             "(taux mentions TB extrapolé par régression linéaire sur 2021-2023)",
             fontweight='bold', fontsize=13, pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

plt.tight_layout()

# ==============================================================================
# 5. EXPORT JSON — DÉCILES PAR LYCÉE
# ==============================================================================
import json, os

export = df_lycees[['uai', 'nom', 'decile']].copy()
export.rename(columns={'decile': 'niveau_scolaire_decile'}, inplace=True)
export['annee'] = 2024  # année d'extrapolation
export = export.sort_values('niveau_scolaire_decile', ascending=False)

out_path = os.path.join(os.path.dirname(__file__), 'lycees_deciles_niveau.json')
export.to_json(out_path, orient='records', force_ascii=False, indent=2)
print(f"\n  Export JSON : {out_path} ({len(export)} lycées)")

plt.show()
