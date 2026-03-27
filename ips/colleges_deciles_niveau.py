import requests
import pandas as pd
import matplotlib.pyplot as plt
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

# Colonnes IPS par année pour extrapolation 2026
ips_year_cols = {y: f'ips_{y}' for y in [2021, 2022, 2023, 2024, 2025]}
for col in ips_year_cols.values():
    if col in df_ips.columns:
        df_ips[col] = pd.to_numeric(df_ips[col], errors='coerce')

# ==============================================================================
# 2. EXTRAPOLATION IPS 2026 PAR RÉGRESSION LINÉAIRE
# ==============================================================================
years = np.array(list(ips_year_cols.keys()))

def extrapoler_ips_2026(row):
    valeurs = [row.get(col, np.nan) for col in ips_year_cols.values()]
    pairs = [(y, v) for y, v in zip(years, valeurs) if pd.notna(v)]
    if len(pairs) < 2:
        non_na = [v for v in valeurs if pd.notna(v)]
        return non_na[-1] if non_na else np.nan
    y_arr = np.array([p[0] for p in pairs])
    v_arr = np.array([p[1] for p in pairs])
    slope, intercept, _, _, _ = stats.linregress(y_arr, v_arr)
    return slope * 2026 + intercept

df_ips['ips_2026'] = df_ips.apply(extrapoler_ips_2026, axis=1)

# ==============================================================================
# 3. RÉCUPÉRATION DES DONNÉES DNB (sessions 2022-2024)
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
# 4. JOINTURE IPS + DNB PAR ANNÉE
# ==============================================================================
rows = []
for session in sessions:
    ips_col = f"ips_{session}"
    if ips_col not in df_ips.columns:
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
    rows.append(df_s[['uai', 'nom', 'secteur', 'session', 'ips', 'note_a_l_ecrit_g']])

df = pd.concat(rows, ignore_index=True)
df.rename(columns={'note_a_l_ecrit_g': 'note_ecrit'}, inplace=True)

# ==============================================================================
# 5. STANDARDISATION DE LA NOTE PAR SESSION (sur l'ensemble public+privé)
# ==============================================================================
session_stats = {}
for session in sessions:
    mask = df['session'] == session
    notes = df.loc[mask, 'note_ecrit']
    mu, sigma = notes.mean(), notes.std()
    df.loc[mask, 'note_std'] = (notes - mu) / sigma + 10
    session_stats[session] = (mu, sigma)
    print(f"  Session {session} : µ={mu:.2f}, σ={sigma:.2f}")

# ==============================================================================
# 6. CALIBRATION DU MODÈLE — PUBLIC UNIQUEMENT
# ==============================================================================
df_pub = df[df['secteur'] == 'Public'].copy()

# Régression IPS baseline (train sur 2022+2023) — public uniquement
df_train = df_pub[df_pub['session'].isin([2022, 2023])].copy()

slope_ips, intercept_ips, r_ips_train, _, _ = stats.linregress(
    df_train['ips'], df_train['note_std']
)
print(f"\n  Régression baseline (public) : note_std = {slope_ips:.4f} × IPS + {intercept_ips:.2f} (r={r_ips_train:.4f})")

# Résidus par session — public
for session in sessions:
    mask = df_pub['session'] == session
    df_pub.loc[mask, 'pred_baseline'] = slope_ips * df_pub.loc[mask, 'ips'] + intercept_ips
    df_pub.loc[mask, 'res_baseline'] = df_pub.loc[mask, 'note_std'] - df_pub.loc[mask, 'pred_baseline']

# Shrinkage du résidu (calibré sur 2022→2023) — public
pivot_res = df_pub.pivot_table(index=['uai', 'nom'], columns='session',
                                values='res_baseline', aggfunc='first')
pivot_res.columns = [f'res_{int(c)}' for c in pivot_res.columns]
pivot_res = pivot_res.reset_index()

train_m1 = pivot_res.dropna(subset=['res_2022', 'res_2023'])
slope_c, intercept_c, r_c, _, _ = stats.linregress(
    train_m1['res_2022'], train_m1['res_2023']
)
print(f"  Shrinkage résidu (public) : res(t) = {slope_c:.4f} × res(t-1) + {intercept_c:.4f} (r={r_c:.4f})")

# ==============================================================================
# 7. PRÉDICTION 2026
#    - Public : modèle IPS extrapolé + résidu shrinkagé
#    - Privé  : dernière note standardisée observée (session 2024)
# ==============================================================================

# --- PUBLIC : prédiction par le modèle ---
df_pub_2024 = df_pub[df_pub['session'] == 2024][['uai', 'nom', 'secteur', 'res_baseline', 'note_std']].copy()
df_pub_2024.rename(columns={'res_baseline': 'res_2024', 'note_std': 'note_std_2024'}, inplace=True)

df_ips_2026 = df_ips[['identifiant', 'ips_2026']].rename(columns={'identifiant': 'uai'})
df_pub_2024 = df_pub_2024.merge(df_ips_2026, on='uai', how='left')
df_pub_2024 = df_pub_2024.dropna(subset=['ips_2026', 'res_2024'])

df_pub_2024['res_pred_2025'] = slope_c * df_pub_2024['res_2024'] + intercept_c
df_pub_2024['res_pred_2026'] = slope_c * df_pub_2024['res_pred_2025'] + intercept_c
df_pub_2024['pred_2026'] = slope_ips * df_pub_2024['ips_2026'] + intercept_ips + df_pub_2024['res_pred_2026']
df_pub_2024['methode'] = 'Modèle IPS+résidu'

# --- PRIVÉ : dernière note standardisée observée ---
df_priv = df[(df['secteur'] != 'Public') & (df['session'] == 2024)][['uai', 'nom', 'secteur', 'note_std']].copy()
df_priv.rename(columns={'note_std': 'note_std_2024'}, inplace=True)
df_priv = df_priv.merge(df_ips_2026, on='uai', how='left')
df_priv['pred_2026'] = df_priv['note_std_2024']  # on prend la dernière observation
df_priv['res_2024'] = np.nan
df_priv['methode'] = 'Dernière obs. DNB'

# --- FUSION ---
cols_communs = ['uai', 'nom', 'secteur', 'note_std_2024', 'ips_2026', 'pred_2026', 'methode']
df_result = pd.concat([df_pub_2024[cols_communs], df_priv[cols_communs]], ignore_index=True)

print(f"\n{'='*70}")
print("PRÉDICTION NIVEAU SCOLAIRE 2026 — COLLÈGES PARISIENS (public + privé)")
print(f"{'='*70}")
n_pub = len(df_pub_2024)
n_priv = len(df_priv)
print(f"  Public  ({n_pub} collèges) : modèle IPS extrapolé 2026 + résidu shrinkagé")
print(f"  Privé   ({n_priv} collèges) : dernière note standardisée DNB 2024")
print(f"  Total : {len(df_result)} collèges")
print(f"  Note prédite moyenne : {df_result['pred_2026'].mean():.2f}")

# ==============================================================================
# 8. CALCUL DES DÉCILES
# ==============================================================================
df_result = df_result.sort_values('pred_2026', ascending=True).reset_index(drop=True)
df_result['decile'] = pd.qcut(df_result['pred_2026'], 10, labels=False, duplicates='drop') + 1

print(f"\n{'='*70}")
print("DÉCILES DE NIVEAU SCOLAIRE 2026 (collèges public + privé)")
print("Décile 1 = plus faibles, Décile 10 = meilleurs")
print(f"{'='*70}")

for d in sorted(df_result['decile'].unique()):
    sub = df_result[df_result['decile'] == d].sort_values('pred_2026', ascending=False)
    print(f"\n  DÉCILE {d} ({len(sub)} collèges)")
    for _, row in sub.iterrows():
        ips_str = f"IPS 2026={row['ips_2026']:.0f}" if pd.notna(row['ips_2026']) else "IPS=N/A"
        print(f"    {row['nom']:<45} Préd={row['pred_2026']:.2f}  "
              f"{ips_str}  [{row['secteur']}, {row['methode']}]")

# ==============================================================================
# 9. GRAPHIQUE — DÉCILES EN BARRES VERTICALES AVEC NOMS DES COLLÈGES
# ==============================================================================
fig, ax = plt.subplots(figsize=(16, 9))

# Palette de couleurs dégradée (rouge → vert)
cmap = plt.cm.RdYlGn
n_deciles = df_result['decile'].nunique()
deciles_sorted = sorted(df_result['decile'].unique())
colors = [cmap(i / (n_deciles - 1)) for i in range(n_deciles)]

for idx, d in enumerate(deciles_sorted):
    sub = df_result[df_result['decile'] == d].sort_values('pred_2026', ascending=False)
    note_moy = sub['pred_2026'].mean()

    # Barre verticale : note moyenne prédite du décile
    ax.bar(idx, note_moy, width=0.65, color=colors[idx],
           edgecolor='white', linewidth=1.5, zorder=2)

    # Note moyenne au-dessus de la barre
    ax.text(idx, note_moy + 0.05, f"{note_moy:.2f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Noms des collèges dans la barre (empilés verticalement)
    noms = sub['nom'].str.replace('CLG ', '', regex=False).values
    texte = "\n".join(noms)
    # Couleur du texte adaptée à la luminosité du fond
    r, g, b, _ = colors[idx]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = '#222222' if luminance > 0.5 else 'white'
    ax.text(idx, note_moy / 2, texte,
            ha='center', va='center', fontsize=5, color=text_color,
            fontweight='bold', linespacing=1.3, zorder=3)

ax.set_xticks(range(len(deciles_sorted)))
ax.set_xticklabels([f"D{d}" for d in deciles_sorted], fontsize=12, fontweight='bold')
ax.set_xlabel("Décile (D1 = plus faibles, D10 = meilleurs)", fontsize=12, labelpad=10)
ax.set_ylabel("Note standardisée prédite (µ=10)", fontsize=12)
ax.set_title("Déciles de niveau scolaire 2026 — Collèges de Paris\n"
             "(public : modèle IPS+résidu / privé : dernière obs. DNB)",
             fontweight='bold', fontsize=13, pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

plt.tight_layout()

# ==============================================================================
# 10. EXPORT JSON — DÉCILES PAR COLLÈGE
# ==============================================================================
import json, os

export = df_result[['uai', 'nom', 'decile']].copy()
export.rename(columns={'decile': 'niveau_scolaire_decile'}, inplace=True)
export['annee'] = 2026  # année de prédiction
export = export.sort_values('niveau_scolaire_decile', ascending=False)

out_path = os.path.join(os.path.dirname(__file__), 'colleges_deciles_niveau.json')
export.to_json(out_path, orient='records', force_ascii=False, indent=2)
print(f"\n  Export JSON : {out_path} ({len(export)} collèges)")

plt.show()
