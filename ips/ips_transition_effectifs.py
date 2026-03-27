import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES IPS (API HuggingFace)
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
    except Exception:
        pass

df_ips = pd.DataFrame(all_rows)

col_map = {}
for c in df_ips.columns:
    cl = c.lower()
    if cl == 'identifiant': col_map[c] = 'uai'
    elif cl == 'nom': col_map[c] = 'nom'
    elif cl == 'secteur': col_map[c] = 'secteur'
    elif cl == 'ips_2025': col_map[c] = 'ips_2025'
    elif cl == 'bonus_ips_2025': col_map[c] = 'bonus_ips_2025'
    elif cl == 'ips_2026': col_map[c] = 'ips_2026'
    elif cl == 'bonus_ips_2026': col_map[c] = 'bonus_ips_2026'
df_ips.rename(columns=col_map, inplace=True)
df_ips['ips_2025'] = pd.to_numeric(df_ips['ips_2025'], errors='coerce')
df_ips['ips_2026'] = pd.to_numeric(df_ips['ips_2026'], errors='coerce')
df_ips = df_ips.dropna(subset=['ips_2025', 'ips_2026'])

# ==============================================================================
# 2. RÉCUPÉRATION DES DONNÉES DNB (session 2024)
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
# 3. JOINTURE + FILTRE PUBLIC
# ==============================================================================
df = df_ips.merge(df_dnb[['uai', 'effectif_admis']], on='uai', how='left')
df = df.dropna(subset=['effectif_admis'])
df['effectif_admis'] = df['effectif_admis'].astype(int)

df['Bonus 2025'] = df['bonus_ips_2025']
df['Bonus 2026'] = df['bonus_ips_2026']

df_pub = df[df['secteur'] == 'Public'].copy()

# ==============================================================================
# 4. MATRICE DE TRANSITION
# ==============================================================================
groupes_2025 = sorted(df_pub['Bonus 2025'].unique())
groupes_2026 = [0, 400, 800, 1200]

# Couleurs par groupe IPS : 0=rouge, 400=orange, 800=bleu, 1200=vert
colors_ips = {0: '#e74c3c', 400: '#f39c12', 600: '#f39c12', 800: '#3498db', 1200: '#2ecc71'}

print(f"\n{'='*70}")
print("MATRICE DE TRANSITION — COLLÈGES PUBLICS (effectifs élèves)")
print(f"{'='*70}")
print(f"{'':>20}", end="")
for g26 in groupes_2026:
    print(f"  → {g26:>4} pts", end="")
print("     Total 2025")
print("-" * 70)

for g25 in groupes_2025:
    mask25 = df_pub['Bonus 2025'] == g25
    total_g25 = df_pub[mask25]['effectif_admis'].sum()
    print(f"  Bonus {g25:>4} pts  ", end="")
    for g26 in groupes_2026:
        mask26 = df_pub['Bonus 2026'] == g26
        eff = df_pub[mask25 & mask26]['effectif_admis'].sum()
        if eff > 0:
            print(f"  {eff:>6}", end="")
        else:
            print(f"  {'—':>6}", end="")
    print(f"    {total_g25:>6}")

print("-" * 70)
print(f"  Total 2026      ", end="")
for g26 in groupes_2026:
    eff = df_pub[df_pub['Bonus 2026'] == g26]['effectif_admis'].sum()
    print(f"  {eff:>6}", end="")
print()

print(f"\n{'='*70}")
print("MATRICE DE TRANSITION — COLLÈGES PUBLICS (nombre d'établissements)")
print(f"{'='*70}")
print(f"{'':>20}", end="")
for g26 in groupes_2026:
    print(f"  → {g26:>4} pts", end="")
print("     Total 2025")
print("-" * 70)

for g25 in groupes_2025:
    mask25 = df_pub['Bonus 2025'] == g25
    total_g25 = len(df_pub[mask25])
    print(f"  Bonus {g25:>4} pts  ", end="")
    for g26 in groupes_2026:
        mask26 = df_pub['Bonus 2026'] == g26
        count = len(df_pub[mask25 & mask26])
        if count > 0:
            print(f"  {count:>6}", end="")
        else:
            print(f"  {'—':>6}", end="")
    print(f"    {total_g25:>6}")

print("-" * 70)
print(f"  Total 2026      ", end="")
for g26 in groupes_2026:
    count = len(df_pub[df_pub['Bonus 2026'] == g26])
    print(f"  {count:>6}", end="")
print()

# Groupement 800 + 1200
df_pub['Bonus 2026 Groupé'] = df_pub['Bonus 2026'].replace({1200: '800-1200', 800: '800-1200'})
groupes_2026_gp = [0, 400, '800-1200']

print(f"\n{'='*70}")
print("MATRICE DE TRANSITION — PUBLIC (800 et 1200 regroupés)")
print(f"{'='*70}")
print(f"{'':>20}", end="")
for g26 in groupes_2026_gp:
    print(f"  → {str(g26):>8}", end="")
print("     Total 2025")
print("-" * 70)

for g25 in groupes_2025:
    mask25 = df_pub['Bonus 2025'] == g25
    total_g25 = len(df_pub[mask25])
    print(f"  Bonus {g25:>4} pts  ", end="")
    for g26 in groupes_2026_gp:
        mask26 = df_pub['Bonus 2026 Groupé'] == g26
        count = len(df_pub[mask25 & mask26])
        if count > 0:
            print(f"  {count:>8}", end="")
        else:
            print(f"  {'—':>8}", end="")
    print(f"    {total_g25:>6}")

print("-" * 70)
print(f"  Total 2026      ", end="")
for g26 in groupes_2026_gp:
    count = len(df_pub[df_pub['Bonus 2026 Groupé'] == g26])
    print(f"  {count:>8}", end="")
print()

print(f"\n{'='*70}")
print("MATRICE DE TRANSITION — PUBLIC (800 et 1200 regroupés, effectifs)")
print(f"{'='*70}")
print(f"{'':>20}", end="")
for g26 in groupes_2026_gp:
    print(f"  → {str(g26):>8}", end="")
print("     Total 2025")
print("-" * 70)

for g25 in groupes_2025:
    mask25 = df_pub['Bonus 2025'] == g25
    total_g25 = df_pub[mask25]['effectif_admis'].sum()
    print(f"  Bonus {g25:>4} pts  ", end="")
    for g26 in groupes_2026_gp:
        mask26 = df_pub['Bonus 2026 Groupé'] == g26
        eff = df_pub[mask25 & mask26]['effectif_admis'].sum()
        if eff > 0:
            print(f"  {eff:>8}", end="")
        else:
            print(f"  {'—':>8}", end="")
    print(f"    {total_g25:>6}")

print("-" * 70)
print(f"  Total 2026      ", end="")
for g26 in groupes_2026_gp:
    eff = df_pub[df_pub['Bonus 2026 Groupé'] == g26]['effectif_admis'].sum()
    print(f"  {eff:>8}", end="")
print()

# ==============================================================================
# 5. GRAPHIQUE 1 : TRANSITION PAR EFFECTIFS ÉLÈVES
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(8, 6))
bar_width = 0.5

for idx_26, g26 in enumerate(groupes_2026):
    bottom = 0
    for g25 in groupes_2025:
        mask = (df_pub['Bonus 2025'] == g25) & (df_pub['Bonus 2026'] == g26)
        eff = df_pub[mask]['effectif_admis'].sum()
        if eff > 0:
            bar = ax1.bar(idx_26, eff, bottom=bottom, width=bar_width,
                         color=colors_ips.get(g25, '#bdc3c7'), edgecolor='white',
                         linewidth=1.5, zorder=2)
            if eff > 80:
                ax1.text(idx_26, bottom + eff / 2, f"{eff}",
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if g25 in [1200, 0] else 'black')
            bottom += eff

    if bottom > 0:
        pct = bottom / df_pub['effectif_admis'].sum() * 100
        ax1.text(idx_26, bottom + 50, f"{bottom}\n({pct:.0f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks(range(len(groupes_2026)))
ax1.set_xticklabels([f"{g} pts" for g in groupes_2026], fontsize=11)
ax1.set_xlabel("Bonus IPS 2026 (nouveau barème)", fontsize=12, labelpad=10)
ax1.set_ylabel("Nombre d'élèves (admis DNB 2024)", fontsize=12)
ax1.set_title("Transition des ÉLÈVES du public entre bonus IPS 2025 et 2026\n"
             "(d'où viennent les élèves de chaque nouveau groupe ?)",
             fontweight='bold', fontsize=13, pad=15)

legend_patches = [mpatches.Patch(color=colors_ips[g], label=f"Ancien bonus {g} pts")
                  for g in groupes_2025 if g in colors_ips]
ax1.legend(handles=legend_patches, loc='upper left', fontsize=10,
          title="Origine (bonus 2025)", title_fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.12)

# ==============================================================================
# 6. GRAPHIQUE 2 : TRANSITION PAR NOMBRE DE COLLÈGES
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

for idx_26, g26 in enumerate(groupes_2026):
    bottom = 0
    for g25 in groupes_2025:
        mask = (df_pub['Bonus 2025'] == g25) & (df_pub['Bonus 2026'] == g26)
        nb_colleges = len(df_pub[mask])
        if nb_colleges > 0:
            bar = ax2.bar(idx_26, nb_colleges, bottom=bottom, width=bar_width,
                         color=colors_ips.get(g25, '#bdc3c7'), edgecolor='white',
                         linewidth=1.5, zorder=2)
            ax2.text(idx_26, bottom + nb_colleges / 2, f"{nb_colleges}",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if g25 in [1200, 0] else 'black')
            bottom += nb_colleges

    if bottom > 0:
        pct = bottom / len(df_pub) * 100
        ax2.text(idx_26, bottom + 0.5, f"{bottom}\n({pct:.0f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xticks(range(len(groupes_2026)))
ax2.set_xticklabels([f"{g} pts" for g in groupes_2026], fontsize=11)
ax2.set_xlabel("Bonus IPS 2026 (nouveau barème)", fontsize=12, labelpad=10)
ax2.set_ylabel("Nombre de collèges", fontsize=12)
ax2.set_title("Transition des COLLÈGES du public entre bonus IPS 2025 et 2026\n"
             "(combien de collèges dans chaque nouveau groupe ?)",
             fontweight='bold', fontsize=13, pad=15)

ax2.legend(handles=legend_patches, loc='upper left', fontsize=10,
          title="Origine (bonus 2025)", title_fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.12)

# ==============================================================================
# 7. GRAPHIQUE 3 : TRANSITION REGROUPÉE (800 + 1200)
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(8, 6))

for idx_26, g26 in enumerate(groupes_2026_gp):
    bottom = 0
    for g25 in groupes_2025:
        mask = (df_pub['Bonus 2025'] == g25) & (df_pub['Bonus 2026'] == (800 if g26 == '800-1200' else g26))
        # Pour le groupe spécial, on doit sommer les 800 et 1200
        if g26 == '800-1200':
            mask = (df_pub['Bonus 2025'] == g25) & (df_pub['Bonus 2026'].isin([800, 1200]))
        else:
            mask = (df_pub['Bonus 2025'] == g25) & (df_pub['Bonus 2026'] == g26)
            
        nb_colleges = len(df_pub[mask])
        if nb_colleges > 0:
            bar = ax3.bar(idx_26, nb_colleges, bottom=bottom, width=bar_width,
                         color=colors_ips.get(g25, '#bdc3c7'), edgecolor='white',
                         linewidth=1.5, zorder=2)
            ax3.text(idx_26, bottom + nb_colleges / 2, f"{nb_colleges}",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if g25 in [1200, 0] else 'black')
            bottom += nb_colleges

    if bottom > 0:
        pct = bottom / len(df_pub) * 100
        ax3.text(idx_26, bottom + 0.5, f"{bottom}\n({pct:.0f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_xticks(range(len(groupes_2026_gp)))
ax3.set_xticklabels([f"{g} pts" if isinstance(g, int) else g for g in groupes_2026_gp], fontsize=11)
ax3.set_xlabel("Nouveau Bonus IPS 2026 (800 et 1200 regroupés)", fontsize=12, labelpad=10)
ax3.set_ylabel("Nombre de collèges", fontsize=12)
ax3.set_title("Transition des COLLÈGES (Public) — Regroupement 800+1200\n"
             "(simplification du nouveau barème)",
             fontweight='bold', fontsize=13, pad=15)

ax3.legend(handles=legend_patches, loc='upper left', fontsize=10,
          title="Origine (bonus 2025)", title_fontsize=11)
ax3.grid(axis='y', linestyle='--', alpha=0.4)
ax3.set_ylim(0, ax3.get_ylim()[1] * 1.12)

# ==============================================================================
# 8. GRAPHIQUE 4 : TRANSITION REGROUPÉE (800 + 1200) — EFFECTIFS ÉLÈVES
# ==============================================================================
fig4, ax4 = plt.subplots(figsize=(8, 6))

for idx_26, g26 in enumerate(groupes_2026_gp):
    bottom = 0
    for g25 in groupes_2025:
        if g26 == '800-1200':
            mask = (df_pub['Bonus 2025'] == g25) & (df_pub['Bonus 2026'].isin([800, 1200]))
        else:
            mask = (df_pub['Bonus 2025'] == g25) & (df_pub['Bonus 2026'] == g26)
            
        eff = df_pub[mask]['effectif_admis'].sum()
        if eff > 0:
            bar = ax4.bar(idx_26, eff, bottom=bottom, width=bar_width,
                         color=colors_ips.get(g25, '#bdc3c7'), edgecolor='white',
                         linewidth=1.5, zorder=2)
            if eff > 80:
                ax4.text(idx_26, bottom + eff / 2, f"{eff}",
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if g25 in [1200, 0] else 'black')
            bottom += eff

    if bottom > 0:
        pct = bottom / df_pub['effectif_admis'].sum() * 100
        ax4.text(idx_26, bottom + 50, f"{bottom}\n({pct:.0f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_xticks(range(len(groupes_2026_gp)))
ax4.set_xticklabels([f"{g} pts" if isinstance(g, int) else g for g in groupes_2026_gp], fontsize=11)
ax4.set_xlabel("Nouveau Bonus IPS 2026 (800 et 1200 regroupés)", fontsize=12, labelpad=10)
ax4.set_ylabel("Nombre d'élèves (admis DNB 2024)", fontsize=12)
ax4.set_title("Transition des ÉLÈVES (Public) — Regroupement 800+1200\n"
             "(d'où viennent les élèves de chaque nouveau bloc ?)",
             fontweight='bold', fontsize=13, pad=15)

ax4.legend(handles=legend_patches, loc='upper left', fontsize=10,
          title="Origine (bonus 2025)", title_fontsize=11)
ax4.grid(axis='y', linestyle='--', alpha=0.4)
ax4.set_ylim(0, ax4.get_ylim()[1] * 1.12)

plt.tight_layout()
plt.show()
