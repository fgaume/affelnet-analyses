import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES (API - 2 LOTS)
# ==============================================================================
base_url = "https://datasets-server.huggingface.co/rows"
params_common = {
    "dataset": "fgaume/affelnet-paris-bonus-ips-colleges",
    "config": "default",
    "split": "train"
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

df = pd.DataFrame(all_rows)

# Vérification Colonnes
required = ['nom', 'ips_2025', 'bonus_ips_2025', 'secteur']
for col in required:
    if col not in df.columns:
        print(f"Erreur: Colonne '{col}' manquante.")
        exit()

# Nettoyage
df['ips_2025'] = pd.to_numeric(df['ips_2025'], errors='coerce')
df = df.dropna(subset=['ips_2025'])

# ==============================================================================
# 2. LOGIQUE DE CALCUL 2026
# ==============================================================================
def calcul_bonus_2026(ips):
    if ips < 105: return 1200
    elif 105 <= ips < 117: return 800
    elif 117 <= ips < 130: return 400
    else: return 0

df['Bonus 2026'] = df['ips_2025'].apply(calcul_bonus_2026)
df.rename(columns={'bonus_ips_2025': 'Bonus 2025'}, inplace=True)

# Séparation des jeux de données
df_public = df[df['secteur'] == 'Public'].copy()

# ==============================================================================
# 3. FONCTION GRAPHIQUE ÉPURÉE (AVEC %)
# ==============================================================================
def tracer_graphique(dataset, titre, ax_cible):
    # On fixe les catégories pour que l'échelle X soit toujours la même
    cats = [0, 400, 600, 800, 1200]
    
    # Comptage
    c25 = dataset['Bonus 2025'].value_counts().reindex(cats, fill_value=0)
    c26 = dataset['Bonus 2026'].value_counts().reindex(cats, fill_value=0)
    
    # Préparation DataFrame pour Seaborn
    d_plot = pd.DataFrame({'2025': c25, '2026': c26})
    d_melt = d_plot.reset_index().melt(id_vars='index', var_name='Année', value_name='Nombre')
    
    # Dessin
    sns.barplot(
        data=d_melt, x='index', y='Nombre', hue='Année',
        palette={'2025': '#3498db', '2026': '#e74c3c'},
        edgecolor="black", ax=ax_cible
    )
    
    # --- AJOUT DES POURCENTAGES ---
    # On calcule les totaux pour chaque année (Total 2025 et Total 2026)
    total_2025 = c25.sum()
    total_2026 = c26.sum()

    # Seaborn place les barres par groupe 'hue'. 
    # container[0] correspond à la première valeur de hue ('2025'), container[1] à '2026'.
    for i, container in enumerate(ax_cible.containers):
        # Sélection du total approprié
        total = total_2025 if i == 0 else total_2026
        
        # Ajout des labels
        for bar in container:
            height = bar.get_height()
            if height > 0: # On n'affiche pas 0% pour la lisibilité
                pct = (height / total) * 100
                ax_cible.annotate(f'{pct:.1f}%',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),  # 3 points de décalage vertical
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    # Cosmétique
    ax_cible.set_title(titre, fontweight='bold', fontsize=12, pad=10)
    ax_cible.set_xlabel("Montant du Bonus (Points)", fontsize=10)
    ax_cible.set_ylabel("Nombre de Collèges (Volumétrie)", fontsize=10)
    
    # Grille plus marquée pour faciliter la lecture sans les chiffres
    ax_cible.grid(axis='y', linestyle='--', alpha=0.6)
    
    # On augmente un peu la limite Y max pour que le texte du haut ne soit pas coupé
    ylim_max = d_melt['Nombre'].max() * 1.15
    ax_cible.set_ylim(0, ylim_max)

# ==============================================================================
# 4. GÉNÉRATION DU GRAPHIQUE (PUBLIC EN HAUT)
# ==============================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# --- HAUT : SECTEUR PUBLIC ---
tracer_graphique(df_public, f"1. FOCUS : COLLÈGES PUBLICS UNIQUEMENT\n(Bascule de la classe moyenne vers 400/800)", axes[0])

# --- BAS : TOUS SECTEURS ---
tracer_graphique(df, f"2. VUE GLOBALE : TOUS COLLÈGES (Public + Privé)\n(Vue d'ensemble)", axes[1])

plt.tight_layout()
plt.show()

# ==============================================================================
# 5. ANALYSE CONSOLE (INDICATIVE)
# ==============================================================================
print(f"\n{'='*60}")
print("TENDANCES POUR LE SECTEUR PUBLIC (Données indicatives)")
print(f"{'='*60}")

# Analyse des mouvements du groupe "600 points"
mask_600 = df_public['Bonus 2025'] == 600
base_600 = df_public[mask_600]

gagnants = base_600[base_600['Bonus 2026'] == 800]
perdants = base_600[base_600['Bonus 2026'] == 400]

print(f"Collèges Publics actuellement à 600 pts : {len(base_600)}")
print(f" -> Tendance Hausse (Vers 800 pts) : {len(gagnants)} établissements")
print(f" -> Tendance Baisse (Vers 400 pts) : {len(perdants)} établissements")

print("\n--- LISTE DES ETABLISSEMENTS EN HAUSSE (Simulation) ---")
print(gagnants[['nom', 'ips_2025']].sort_values('ips_2025').to_string(index=False))

print("\n--- LISTE DES ETABLISSEMENTS EN BAISSE (Simulation) ---")
print(perdants[['nom', 'ips_2025']].sort_values('ips_2025').to_string(index=False))