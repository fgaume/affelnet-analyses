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

# Normalisation colonnes
df_ips.columns = [c.lower() for c in df_ips.columns]

# ==============================================================================
# 2. RÉCUPÉRATION DES DONNÉES DNB (3 sessions)
# ==============================================================================
sessions = [2022, 2023, 2024, 2025]
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
df_dnb['nb_candidats_g'] = pd.to_numeric(df_dnb['nb_candidats_g'], errors='coerce')
df_dnb = df_dnb.dropna(subset=['note_a_l_ecrit_g'])

# ==============================================================================
# 3. JOINTURE IPS + DNB PAR ANNÉE
# ==============================================================================
rows = []
for session in sessions:
    ips_col = f"ips_{session}"
    if ips_col not in df_ips.columns:
        print(f"  Pas de colonne {ips_col} dans les données IPS, session {session} ignorée")
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
                       'note_a_l_ecrit_g', 'nb_candidats_g']])

df = pd.concat(rows, ignore_index=True)
df.rename(columns={'note_a_l_ecrit_g': 'note_ecrit'}, inplace=True)

print(f"\n  Total observations (collège × année) : {len(df)}")
print(f"  Collèges uniques : {df['uai'].nunique()}")

# ==============================================================================
# 4. STANDARDISATION DE LA NOTE PAR SESSION
# ==============================================================================
# Centre-réduit puis ramené à moyenne=10, écart-type=1
for session in sessions:
    mask = df['session'] == session
    notes = df.loc[mask, 'note_ecrit']
    mu, sigma = notes.mean(), notes.std()
    df.loc[mask, 'note_std'] = (notes - mu) / sigma + 10
    print(f"  Session {session} : µ={mu:.2f}, σ={sigma:.2f} → standardisé sur 10±1")

# Séparation
df_pub = df[df['secteur'] == 'Public'].copy()
df_priv = df[df['secteur'] == 'Privé'].copy()

# ==============================================================================
# 5. FIGURE 1 — COLLÈGES PUBLICS : NUAGE PAR SESSION + RÉGRESSION
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))

colors_session = {2022: '#3498db', 2023: '#e67e22', 2024: '#2ecc71', 2025: '#9b59b6'}

for session in sessions:
    sub = df_pub[df_pub['session'] == session]
    ax1.scatter(sub['ips'], sub['note_std'], c=colors_session[session],
                alpha=0.6, s=40, label=f"Session {session}", edgecolors='white', linewidth=0.5)

# Régression globale (public)
sl_pub, int_pub, r_pub, _, _ = stats.linregress(df_pub['ips'], df_pub['note_std'])
x_pub = np.linspace(df_pub['ips'].min(), df_pub['ips'].max(), 100)
ax1.plot(x_pub, sl_pub * x_pub + int_pub, 'k--', linewidth=2,
         label=f"Régression (r={r_pub:.3f}, r²={r_pub**2:.3f})")

ax1.set_xlabel("IPS du collège", fontsize=12)
ax1.set_ylabel("Note à l'écrit DNB (standardisée, µ=10)", fontsize=12)
ax1.set_title("Corrélation IPS vs niveau scolaire — Collèges publics parisiens\n"
              "(sessions 2022-2025, note standardisée par session)",
              fontweight='bold', fontsize=13, pad=15)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
fig1.tight_layout()

# ==============================================================================
# 6. FIGURE 2 — COLLÈGES PRIVÉS : NUAGE PAR SESSION + RÉGRESSION
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

for session in sessions:
    sub = df_priv[df_priv['session'] == session]
    ax2.scatter(sub['ips'], sub['note_std'], c=colors_session[session],
                alpha=0.6, s=40, label=f"Session {session}", edgecolors='white', linewidth=0.5)

# Régression globale (privé)
sl_priv, int_priv, r_priv = 0, 0, 0
if len(df_priv) > 5:
    sl_priv, int_priv, r_priv, _, _ = stats.linregress(df_priv['ips'], df_priv['note_std'])
    x_priv = np.linspace(df_priv['ips'].min(), df_priv['ips'].max(), 100)
    ax2.plot(x_priv, sl_priv * x_priv + int_priv, 'k--', linewidth=2,
             label=f"Régression (r={r_priv:.3f}, r²={r_priv**2:.3f})")

ax2.set_xlabel("IPS du collège", fontsize=12)
ax2.set_ylabel("Note à l'écrit DNB (standardisée, µ=10)", fontsize=12)
ax2.set_title("Corrélation IPS vs niveau scolaire — Collèges privés parisiens\n"
              "(sessions 2022-2025, note standardisée par session)",
              fontweight='bold', fontsize=13, pad=15)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
fig2.tight_layout()

plt.show()

# ==============================================================================
# 7. ANALYSE CONSOLE
# ==============================================================================
print(f"\n{'='*70}")
print("CORRÉLATION IPS vs NOTE À L'ÉCRIT DU DNB — PARIS")
print(f"{'='*70}")

# Globale
r_all, p_all = stats.pearsonr(df['ips'], df['note_std'])
print(f"\n  GLOBALE (tous secteurs, toutes sessions)")
print(f"    r = {r_all:.4f}  (p = {p_all:.2e})")
print(f"    r² = {r_all**2:.4f}")

# Par session
print(f"\n  PAR SESSION :")
for session in sessions:
    sub = df[df['session'] == session]
    r_s, p_s = stats.pearsonr(sub['ips'], sub['note_std'])
    print(f"    {session} : r = {r_s:.4f}  (n={len(sub)})")

# Par secteur
print(f"\n  PAR SECTEUR :")
for secteur in ['Public', 'Privé']:
    sub = df[df['secteur'] == secteur]
    if len(sub) > 5:
        r_s, p_s = stats.pearsonr(sub['ips'], sub['note_std'])
        sl, it, _, _, _ = stats.linregress(sub['ips'], sub['note_std'])
        print(f"    {secteur:6s} : r = {r_s:.4f}  (n={len(sub)})  "
              f"pente = {sl:.4f} σ/pt IPS")

# Régression détaillée (public)
print(f"\n  RÉGRESSION LINÉAIRE (public) :")
print(f"    note_std = {sl_pub:.4f} × IPS + {int_pub:.2f}")
print(f"    → +10 pts d'IPS ≈ {sl_pub * 10:.2f} σ de note standardisée")

# Outliers (public) — résidus > 2σ
df_pub = df_pub.copy()
df_pub['predicted'] = sl_pub * df_pub['ips'] + int_pub
df_pub['residual'] = df_pub['note_std'] - df_pub['predicted']
res_std = df_pub['residual'].std()

surperf = df_pub[df_pub['residual'] > 2 * res_std].sort_values('residual', ascending=False)
sousperf = df_pub[df_pub['residual'] < -2 * res_std].sort_values('residual')

print(f"\n  OUTLIERS PUBLIC (résidu > 2σ = {2*res_std:.2f}) :")
if len(surperf) > 0:
    print(f"\n    Sur-performants (note > prédiction) :")
    for _, row in surperf.iterrows():
        print(f"      {row['nom']:<40} IPS={row['ips']:.0f}  "
              f"note_std={row['note_std']:.2f}  résidu=+{row['residual']:.2f}  ({int(row['session'])})")

if len(sousperf) > 0:
    print(f"\n    Sous-performants (note < prédiction) :")
    for _, row in sousperf.iterrows():
        print(f"      {row['nom']:<40} IPS={row['ips']:.0f}  "
              f"note_std={row['note_std']:.2f}  résidu={row['residual']:.2f}  ({int(row['session'])})")
