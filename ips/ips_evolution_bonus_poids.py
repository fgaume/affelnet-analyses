import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES IPS (toutes années, HuggingFace)
# ==============================================================================
base_url = "https://datasets-server.huggingface.co/rows"
params_common = {
    "dataset": "fgaume/affelnet-paris-bonus-ips-colleges",
    "config": "default",
    "split": "ips"
}

all_rows = []
for offset in [0, 100]:
    try:
        r = requests.get(base_url, params={**params_common, "offset": offset, "length": 100})
        r.raise_for_status()
        all_rows.extend([item['row'] for item in r.json()['rows']])
    except Exception as e:
        print(f"  Erreur offset {offset}: {e}")

df_ips = pd.DataFrame(all_rows)

# Normalisation des colonnes (gestion casse variable)
col_map = {}
for c in df_ips.columns:
    cl = c.lower()
    if cl == 'identifiant':
        col_map[c] = 'uai'
    elif cl == 'nom':
        col_map[c] = 'nom'
    elif cl == 'secteur':
        col_map[c] = 'secteur'
    else:
        for year in range(2021, 2026):
            if cl == f'bonus_ips_{year}':
                col_map[c] = f'bonus_{year}'
            elif cl == f'ips_{year}':
                col_map[c] = f'ips_{year}'
df_ips.rename(columns=col_map, inplace=True)

print(f"Collèges chargés : {len(df_ips)}")
print(f"Colonnes disponibles : {[c for c in df_ips.columns if 'bonus' in c or 'ips_' in c]}")

# ==============================================================================
# 2. RÉCUPÉRATION DES EFFECTIFS DNB PAR SESSION
#    Convention : bonus de l'année Y → DNB session Y-1
# ==============================================================================
ANNEES_BONUS = [2021, 2022, 2023, 2024, 2025]
# Session DNB : correspondance directe Y → Y
SESSION_DNB = {y: y for y in ANNEES_BONUS}

def fetch_dnb_effectifs(session):
    """Retourne un dict {uai: nb_candidats_g} pour une session DNB donnée."""
    url = (
        "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
        "fr-en-indicateurs-valeur-ajoutee-colleges/exports/json"
        f"?select=uai%2Cnb_candidats_g"
        "&lang=fr"
        "&refine=academie%3A%22PARIS%22"
        f"&refine=session%3A%22{session}%22"
        "&timezone=Europe%2FBerlin"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return {}
        df = pd.DataFrame(data)
        df['nb_candidats_g'] = pd.to_numeric(df['nb_candidats_g'], errors='coerce')
        df = df.dropna(subset=['nb_candidats_g'])
        return dict(zip(df['uai'], df['nb_candidats_g'].astype(int)))
    except Exception as e:
        print(f"  ⚠ Session DNB {session} non disponible : {e}")
        return {}

print("\nRécupération des effectifs DNB...")
effectifs_par_session = {}
for year in ANNEES_BONUS:
    session = SESSION_DNB[year]
    print(f"  Bonus {year} → DNB session {session}...", end=" ")
    eff = fetch_dnb_effectifs(session)
    effectifs_par_session[year] = eff
    print(f"{len(eff)} collèges")

# ==============================================================================
# 3. CALCUL DES EFFECTIFS PAR GROUPE DE BONUS POUR CHAQUE ANNÉE
# ==============================================================================
BONUS_GROUPES = [0, 600, 1200]
COLORS = {0: '#e74c3c', 600: '#f39c12', 1200: '#2ecc71'}

# Pour chaque année : somme des effectifs par groupe, public uniquement puis tous
results_pub = []   # secteur Public
results_all = []   # tous secteurs

for year in ANNEES_BONUS:
    col_bonus = f'bonus_{year}'
    if col_bonus not in df_ips.columns:
        print(f"  ⚠ Colonne {col_bonus} manquante, année {year} ignorée")
        continue

    eff_map = effectifs_par_session[year]
    if not eff_map:
        # Fallback : chercher la session disponible la plus proche
        for fallback_year in ANNEES_BONUS:
            candidate = effectifs_par_session.get(fallback_year, {})
            if candidate:
                print(f"  ⚠ Pas d'effectifs DNB pour bonus {year} (session {SESSION_DNB[year]}), fallback → session {SESSION_DNB[fallback_year]}")
                eff_map = candidate
                break

    col_ips = f'ips_{year}'
    cols = ['uai', 'nom', 'secteur', col_bonus, col_ips] if col_ips in df_ips.columns else ['uai', 'nom', 'secteur', col_bonus]
    df_year = df_ips[cols].copy()
    df_year.rename(columns={col_bonus: 'bonus', col_ips: 'ips'}, inplace=True)
    df_year['bonus'] = pd.to_numeric(df_year['bonus'], errors='coerce')
    df_year['ips'] = pd.to_numeric(df_year.get('ips', pd.Series(dtype=float)), errors='coerce')
    df_year = df_year.dropna(subset=['bonus'])
    df_year['bonus'] = df_year['bonus'].astype(int)
    df_year['effectif'] = df_year['uai'].map(eff_map).fillna(0).astype(int)

    for secteur_label, mask in [('public', df_year['secteur'] == 'Public'),
                                  ('all',    pd.Series([True] * len(df_year), index=df_year.index))]:
        sub = df_year[mask]
        total = sub['effectif'].sum()
        # IPS moyen global pondéré par effectif
        ips_global = (sub['ips'] * sub['effectif']).sum() / total if total > 0 else np.nan
        row = {'annee': year, 'total': total, 'ips_mean': ips_global}
        for g in BONUS_GROUPES:
            sg = sub[sub['bonus'] == g]
            n = sg['effectif'].sum()
            ips_g = (sg['ips'] * sg['effectif']).sum() / n if n > 0 else np.nan
            row[f'eff_{g}'] = n
            row[f'pct_{g}'] = (n / total * 100) if total > 0 else 0
            row[f'ips_mean_{g}'] = ips_g
        if secteur_label == 'public':
            results_pub.append(row)
        else:
            results_all.append(row)

df_pub = pd.DataFrame(results_pub).set_index('annee')
df_all = pd.DataFrame(results_all).set_index('annee')

print("\n--- POIDS RELATIF (%) — PUBLIC ---")
print(df_pub[[f'pct_{g}' for g in BONUS_GROUPES]].round(1).to_string())

print("\n--- POIDS RELATIF (%) — TOUS ---")
print(df_all[[f'pct_{g}' for g in BONUS_GROUPES]].round(1).to_string())

# ==============================================================================
# 4. GRAPHIQUE : évolution du poids relatif (%) des groupes bonus
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

def plot_evolution(df_data, ax, titre):
    annees = df_data.index.tolist()
    x = np.arange(len(annees))
    width = 0.55

    # Axe secondaire pour IPS moyen global
    ax2 = ax.twinx()

    # Stacked bars par groupe de bonus
    bottom = np.zeros(len(annees))
    for g in BONUS_GROUPES:
        pcts = df_data[f'pct_{g}'].values
        ax.bar(x, pcts, bottom=bottom, width=width,
               color=COLORS[g], edgecolor='white', linewidth=1.2, zorder=2,
               label=f'{g} pts')
        # Annotations dans chaque barre : % + effectif + IPS moyen du groupe
        for i, (p, b) in enumerate(zip(pcts, bottom)):
            if p >= 5:
                eff = df_data[f'eff_{g}'].values[i]
                ips_g = df_data[f'ips_mean_{g}'].values[i]
                ips_str = f"IPS {ips_g:.0f}" if not np.isnan(ips_g) else ""
                ax.text(x[i], b + p / 2,
                        f"{p:.0f}%\n({eff:,})\n{ips_str}",
                        ha='center', va='center', fontsize=7.5,
                        fontweight='bold',
                        color='white' if g in [0, 1200] else '#222222')
        bottom += pcts

    # Ligne IPS moyen global (axe secondaire)
    ips_vals = df_data['ips_mean'].values
    ax2.plot(x, ips_vals, color='#2c3e50', marker='o', linewidth=2,
             markersize=6, zorder=5, linestyle='--', label='IPS moyen global')
    ips_min = np.nanmin(ips_vals) - 5
    ips_max = np.nanmax(ips_vals) + 12  # marge haute pour les annotations
    ax2.set_ylim(ips_min, ips_max)
    # Annotations au-dessus des points, dans la marge haute (hors zone des barres)
    for i, v in enumerate(ips_vals):
        if not np.isnan(v):
            ax2.annotate(f"{v:.1f}",
                         xy=(x[i], ips_max - 1),
                         ha='center', va='top', fontsize=8.5,
                         color='#2c3e50', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2c3e50', alpha=0.8))
    ax2.set_ylabel("IPS moyen pondéré", fontsize=10, color='#2c3e50')
    ax2.tick_params(axis='y', colors='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in annees], fontsize=12)
    ax.set_xlabel("Année Affelnet", fontsize=11, labelpad=8)
    ax.set_ylabel("Part des élèves (%)", fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_title(titre, fontweight='bold', fontsize=12, pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


    return ax2

ax2_pub = plot_evolution(df_pub, axes[0], "Collèges PUBLICS\nPoids des groupes de bonus IPS 2021–2025")
ax2_all = plot_evolution(df_all, axes[1], "TOUS collèges (public + privé)\nPoids des groupes de bonus IPS 2021–2025")

# Légende commune unique en bas de figure
from matplotlib.lines import Line2D
bar_handles = [mpatches.Patch(color=COLORS[g], label=f"Bonus {g} pts  (% élèves · effectif · IPS moy. groupe)")
               for g in BONUS_GROUPES]
line_handle = Line2D([0], [0], color='#2c3e50', linewidth=2, linestyle='--',
                     marker='o', markersize=5, label='IPS moyen global pondéré (axe droit)')
fig.legend(handles=bar_handles + [line_handle],
           loc='lower center', ncol=2, fontsize=9, frameon=True,
           title="Lecture des barres & courbe", title_fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Évolution du poids relatif des groupes de bonus IPS (Affelnet Paris)\n"
    "Pondération par effectifs candidats au DNB (session correspondante)",
    fontsize=14, fontweight='bold', y=0.97
)

plt.subplots_adjust(left=0.07, right=0.95, top=0.82, bottom=0.22, wspace=0.35)
plt.savefig("ips/ips_evolution_bonus_poids.png", dpi=150, bbox_inches='tight')
plt.show()
