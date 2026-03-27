import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES IPS (JSON HuggingFace)
# ==============================================================================
ips_url = "https://huggingface.co/datasets/fgaume/affelnet-paris-bonus-ips-colleges/raw/main/affelnet-paris-bonus-ips-colleges.json"

r_ips = requests.get(ips_url)
r_ips.raise_for_status()
df_ips = pd.DataFrame(r_ips.json())
df_ips.columns = [c.lower() for c in df_ips.columns]

# ==============================================================================
# 2. RÉCUPÉRATION DES DONNÉES DNB (3 sessions)
# ==============================================================================
sessions = [2022, 2023, 2024]
frames = []

for session in sessions:
    dnb_url = (
        "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
        "fr-en-indicateurs-valeur-ajoutee-colleges/exports/json"
        f"?select=uai%2Cnote_a_l_ecrit_g%2Cnb_candidats_g%2Ctaux_de_reussite_g"
        f"&lang=fr"
        f"&refine=academie%3A%22PARIS%22"
        f"&refine=session%3A%22{session}%22"
        f"&timezone=Europe%2FBerlin"
    )
    try:
        r = requests.get(dnb_url)
        r.raise_for_status()
        df_session = pd.DataFrame(r.json())
        df_session['session'] = session
        frames.append(df_session)
        print(f"  Session {session} : {len(df_session)} collèges récupérés")
    except Exception as e:
        print(f"  Session {session} : ERREUR - {e}")

df_dnb = pd.concat(frames, ignore_index=True)
df_dnb['note_a_l_ecrit_g'] = pd.to_numeric(df_dnb['note_a_l_ecrit_g'], errors='coerce')
df_dnb = df_dnb.dropna(subset=['note_a_l_ecrit_g'])

# ==============================================================================
# 3. JOINTURE IPS + DNB PAR ANNÉE
# ==============================================================================
rows = []
for session in sessions:
    ips_col = f"ips_{session}"
    if ips_col not in df_ips.columns:
        print(f"  Pas de colonne {ips_col}, session {session} ignorée")
        continue

    df_s = df_dnb[df_dnb['session'] == session].copy()
    df_s = df_s.merge(
        df_ips[['identifiant', 'nom', 'secteur', ips_col]].rename(
            columns={'identifiant': 'uai', ips_col: 'ips'}
        ),
        on='uai', how='inner'
    )
    df_s['ips'] = pd.to_numeric(df_s['ips'], errors='coerce')
    df_s = df_s.dropna(subset=['ips'])
    rows.append(df_s[['uai', 'nom', 'secteur', 'session', 'ips',
                       'note_a_l_ecrit_g']])

df = pd.concat(rows, ignore_index=True)
df.rename(columns={'note_a_l_ecrit_g': 'note_ecrit'}, inplace=True)

# ==============================================================================
# 4. STANDARDISATION DE LA NOTE PAR SESSION
# ==============================================================================
for session in sessions:
    mask = df['session'] == session
    notes = df.loc[mask, 'note_ecrit']
    mu, sigma = notes.mean(), notes.std()
    df.loc[mask, 'note_std'] = (notes - mu) / sigma + 10
    print(f"  Session {session} : µ={mu:.2f}, σ={sigma:.2f}")

# Public uniquement
df_pub = df[df['secteur'] == 'Public'].copy()

# ==============================================================================
# 5. RÉSIDUS PAR SESSION (régression IPS → note, séparée par année)
# ==============================================================================
for session in sessions:
    mask = df_pub['session'] == session
    sub = df_pub[mask]
    slope, intercept, _, _, _ = stats.linregress(sub['ips'], sub['note_std'])
    df_pub.loc[mask, 'predicted'] = slope * df_pub.loc[mask, 'ips'] + intercept
    df_pub.loc[mask, 'residual'] = df_pub.loc[mask, 'note_std'] - df_pub.loc[mask, 'predicted']

# ==============================================================================
# 6. MOYENNE PONDÉRÉE DES RÉSIDUS (poids 1/6, 2/6, 3/6)
# ==============================================================================
pivot = df_pub.pivot_table(index=['uai', 'nom'], columns='session',
                           values='residual', aggfunc='first')
pivot.columns = [f'res_{int(c)}' for c in pivot.columns]
pivot = pivot.dropna()  # collèges présents les 3 ans
pivot = pivot.reset_index()

# Moyenne pondérée
pivot['res_pond'] = (1 * pivot['res_2022'] + 2 * pivot['res_2023'] + 3 * pivot['res_2024']) / 6

n_colleges = len(pivot)
print(f"\n  Collèges publics présents les 3 sessions : {n_colleges}")

# ==============================================================================
# 7. DÉCILES
# ==============================================================================
pivot['decile'] = pd.qcut(pivot['res_pond'], q=10, labels=False) + 1  # 1 à 10

# Ajouter IPS 2025 pour référence
df_ips_2025 = df_ips[['identifiant', 'ips_2025']].rename(columns={'identifiant': 'uai'})
df_ips_2025['ips_2025'] = pd.to_numeric(df_ips_2025['ips_2025'], errors='coerce')
pivot = pivot.merge(df_ips_2025, on='uai', how='left')

# ==============================================================================
# 8. FIGURE 1 — RÉSIDU PONDÉRÉ PAR COLLÈGE, COLORÉ PAR DÉCILE
# ==============================================================================
pivot_sorted = pivot.sort_values('res_pond')

# Palette de couleurs par décile (rouge → orange → jaune → vert)
cmap = plt.cm.RdYlGn
decile_colors = {d: cmap((d - 1) / 9) for d in range(1, 11)}

fig1, ax1 = plt.subplots(figsize=(10, max(8, n_colleges * 0.18)))

y_pos = np.arange(n_colleges)
bar_colors = [decile_colors[d] for d in pivot_sorted['decile']]

ax1.barh(y_pos, pivot_sorted['res_pond'], height=0.7, color=bar_colors,
         edgecolor='white', linewidth=0.5, alpha=0.85)

ax1.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(pivot_sorted['nom'], fontsize=5.5)
ax1.set_xlabel("Résidu pondéré (sur/sous-performance vs IPS)", fontsize=11)
ax1.set_title("Classement des collèges publics par niveau scolaire relatif à l'IPS\n"
              "(résidu pondéré 1/6·2022 + 2/6·2023 + 3/6·2024, coloré par décile)",
              fontweight='bold', fontsize=12, pad=15)

# Légende déciles
legend_patches = [mpatches.Patch(color=decile_colors[d], label=f"D{d}")
                  for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
ax1.legend(handles=legend_patches, fontsize=7, loc='lower right',
           title="Décile", title_fontsize=8, ncol=2)
ax1.grid(axis='x', alpha=0.3)
fig1.tight_layout()

# ==============================================================================
# 9. FIGURE 2 — RÉPARTITION IPS PAR DÉCILE (boxplot)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

data_bp = [pivot[pivot['decile'] == d]['ips_2025'].dropna().values for d in range(1, 11)]
bp = ax2.boxplot(data_bp, tick_labels=[f"D{d}" for d in range(1, 11)],
                 patch_artist=True, widths=0.6)

for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(decile_colors[i + 1])
    patch.set_alpha(0.7)

ax2.set_xlabel("Décile de niveau scolaire (résidu pondéré)", fontsize=11)
ax2.set_ylabel("IPS 2025", fontsize=11)
ax2.set_title("Distribution de l'IPS par décile de niveau scolaire\n"
              "(collèges publics parisiens)",
              fontweight='bold', fontsize=12, pad=15)
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()

plt.show()

# ==============================================================================
# 10. ANALYSE CONSOLE
# ==============================================================================
print(f"\n{'='*70}")
print("DÉCILES DE NIVEAU SCOLAIRE — COLLÈGES PUBLICS PARIS")
print(f"{'='*70}")
print(f"\n  Méthode : résidu pondéré = (1×res_2022 + 2×res_2023 + 3×res_2024) / 6")
print(f"  Résidu = note standardisée − prédiction par IPS (régression par session)")
print(f"  Collèges classés : {n_colleges}")

for d in range(1, 11):
    sub = pivot[pivot['decile'] == d].sort_values('res_pond')
    bornes = f"[{sub['res_pond'].min():+.3f} ; {sub['res_pond'].max():+.3f}]"
    ips_moy = sub['ips_2025'].mean()
    print(f"\n  ── DÉCILE {d} {bornes}  (IPS moyen : {ips_moy:.0f}) ──")
    for _, row in sub.iterrows():
        ips_str = f"IPS={row['ips_2025']:.0f}" if pd.notna(row['ips_2025']) else "IPS=?"
        print(f"    {row['nom']:<42} {ips_str:>8}  "
              f"rés.pond={row['res_pond']:+.3f}  "
              f"({row['res_2022']:+.2f} / {row['res_2023']:+.2f} / {row['res_2024']:+.2f})")
