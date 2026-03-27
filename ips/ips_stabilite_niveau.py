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

print(f"\n  Total observations : {len(df)}")
print(f"  Collèges uniques : {df['uai'].nunique()}")

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
# 6. PIVOT EN FORMAT LARGE (collèges présents les 3 années)
# ==============================================================================
pivot = df_pub.pivot_table(index=['uai', 'nom'], columns='session',
                           values='residual', aggfunc='first')
pivot.columns = [f'res_{int(c)}' for c in pivot.columns]
pivot = pivot.dropna()  # collèges présents les 3 ans
pivot = pivot.reset_index()

n_colleges = len(pivot)
print(f"\n  Collèges publics présents les 3 sessions : {n_colleges}")

# Statistiques par collège
pivot['res_mean'] = pivot[['res_2022', 'res_2023', 'res_2024']].mean(axis=1)
pivot['res_std'] = pivot[['res_2022', 'res_2023', 'res_2024']].std(axis=1)
pivot['trend'] = (pivot['res_2024'] - pivot['res_2022']) / 2  # pente linéaire

# ==============================================================================
# 7. FIGURE 1 — SCATTER RÉSIDU N vs N+1
# ==============================================================================
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 6))

pairs = [('res_2022', 'res_2023', '2022', '2023', ax1a),
         ('res_2023', 'res_2024', '2023', '2024', ax1b)]

for col_x, col_y, lbl_x, lbl_y, ax in pairs:
    x, y = pivot[col_x], pivot[col_y]
    ax.scatter(x, y, c='#3498db', alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

    # Régression
    sl, it, r_val, p_val, _ = stats.linregress(x, y)
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, sl * x_range + it, 'k--', linewidth=1.5,
            label=f"r = {r_val:.3f} (p = {p_val:.1e})")

    # Droite y=x
    lim = max(abs(x.min()), abs(x.max()), abs(y.min()), abs(y.max())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], ':', color='grey', alpha=0.5, label='y = x')

    ax.set_xlabel(f"Résidu {lbl_x}", fontsize=11)
    ax.set_ylabel(f"Résidu {lbl_y}", fontsize=11)
    ax.set_title(f"Stabilité {lbl_x} → {lbl_y}", fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

fig1.suptitle("Stabilité inter-annuelle des résidus (note std. − prédiction IPS)\n"
              "Collèges publics parisiens",
              fontweight='bold', fontsize=13, y=1.02)
fig1.tight_layout()

# ==============================================================================
# 8. FIGURE 2 — DOT PLOT PAR COLLÈGE (trié par résidu moyen)
# ==============================================================================
pivot_sorted = pivot.sort_values('res_mean')

fig2, ax2 = plt.subplots(figsize=(10, max(8, n_colleges * 0.18)))

y_pos = np.arange(n_colleges)
colors_session = {0: '#3498db', 1: '#e67e22', 2: '#2ecc71'}
labels_session = ['2022', '2023', '2024']

for i, col in enumerate(['res_2022', 'res_2023', 'res_2024']):
    ax2.scatter(pivot_sorted[col], y_pos, c=colors_session[i],
                s=25, alpha=0.7, label=labels_session[i], zorder=3)

# Lignes reliant min-max par collège
for j, (_, row) in enumerate(pivot_sorted.iterrows()):
    vals = [row['res_2022'], row['res_2023'], row['res_2024']]
    ax2.plot([min(vals), max(vals)], [j, j], '-', color='grey', alpha=0.3, linewidth=0.8)

ax2.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(pivot_sorted['nom'], fontsize=6)
ax2.set_xlabel("Résidu (note std. − prédiction IPS)", fontsize=11)
ax2.set_title("Résidus par collège et par session (trié par résidu moyen)\n"
              "Collèges publics parisiens",
              fontweight='bold', fontsize=13, pad=15)
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(axis='x', alpha=0.3)
fig2.tight_layout()

# ==============================================================================
# 9. FIGURE 3 — DISTRIBUTION DE LA VOLATILITÉ
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.hist(pivot['res_std'], bins=20, color='#3498db', edgecolor='white',
         alpha=0.7, linewidth=1.2)

median_vol = pivot['res_std'].median()
ax3.axvline(median_vol, color='#e74c3c', linestyle='--', linewidth=1.5,
            label=f"Médiane = {median_vol:.3f}")

# Annotation des plus volatils
top_vol = pivot.nlargest(5, 'res_std')
for _, row in top_vol.iterrows():
    ax3.annotate(row['nom'], xy=(row['res_std'], 0),
                 xytext=(row['res_std'], ax3.get_ylim()[1] * 0.7),
                 fontsize=7, ha='center', rotation=45,
                 arrowprops=dict(arrowstyle='->', color='grey', alpha=0.5))

ax3.set_xlabel("Écart-type des résidus sur 3 sessions (volatilité)", fontsize=11)
ax3.set_ylabel("Nombre de collèges", fontsize=11)
ax3.set_title("Distribution de la volatilité inter-annuelle\n"
              "Collèges publics parisiens",
              fontweight='bold', fontsize=13, pad=15)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)
fig3.tight_layout()

plt.show()

# ==============================================================================
# 10. ANALYSE CONSOLE
# ==============================================================================
print(f"\n{'='*70}")
print("STABILITÉ INTER-ANNUELLE DES RÉSIDUS — COLLÈGES PUBLICS PARIS")
print(f"{'='*70}")
print(f"\n  Collèges publics présents 3 sessions : {n_colleges}")

# Corrélations inter-sessions
print(f"\n  CORRÉLATIONS INTER-SESSIONS (résidus) :")
for col_x, col_y, lbl in [('res_2022', 'res_2023', '2022 → 2023'),
                            ('res_2023', 'res_2024', '2023 → 2024'),
                            ('res_2022', 'res_2024', '2022 → 2024')]:
    r_val, p_val = stats.pearsonr(pivot[col_x], pivot[col_y])
    print(f"    {lbl} : r = {r_val:.4f}  (p = {p_val:.2e})")

# ICC (one-way random, absolute agreement)
# ICC(1,1) = (BMS - WMS) / (BMS + (k-1)*WMS)
k = 3  # nombre de sessions
residuals_matrix = pivot[['res_2022', 'res_2023', 'res_2024']].values
n = residuals_matrix.shape[0]
grand_mean = residuals_matrix.mean()
subject_means = residuals_matrix.mean(axis=1)
BMS = k * np.sum((subject_means - grand_mean) ** 2) / (n - 1)
WMS = np.sum((residuals_matrix - subject_means[:, np.newaxis]) ** 2) / (n * (k - 1))
ICC = (BMS - WMS) / (BMS + (k - 1) * WMS)
print(f"\n  ICC(1,1) = {ICC:.4f}  (fiabilité test-retest sur 3 sessions)")
if ICC > 0.75:
    print(f"    → Excellente stabilité")
elif ICC > 0.5:
    print(f"    → Bonne stabilité")
elif ICC > 0.25:
    print(f"    → Stabilité modérée")
else:
    print(f"    → Faible stabilité")

# Volatilité
print(f"\n  VOLATILITÉ (σ des résidus par collège sur 3 ans) :")
print(f"    Médiane  : {pivot['res_std'].median():.4f}")
print(f"    Moyenne  : {pivot['res_std'].mean():.4f}")
print(f"    Min      : {pivot['res_std'].min():.4f}")
print(f"    Max      : {pivot['res_std'].max():.4f}")

# Top stables
print(f"\n  TOP 10 — COLLÈGES LES PLUS STABLES (faible volatilité) :")
stable = pivot.nsmallest(10, 'res_std')
for _, row in stable.iterrows():
    print(f"    {row['nom']:<40} σ={row['res_std']:.4f}  "
          f"résidu moyen={row['res_mean']:+.3f}")

# Top volatils
print(f"\n  TOP 10 — COLLÈGES LES PLUS VOLATILS :")
volatile = pivot.nlargest(10, 'res_std')
for _, row in volatile.iterrows():
    print(f"    {row['nom']:<40} σ={row['res_std']:.4f}  "
          f"résidus: {row['res_2022']:+.3f} / {row['res_2023']:+.3f} / {row['res_2024']:+.3f}")

# Trajectoires claires (tendance forte)
print(f"\n  TRAJECTOIRES CLAIRES (tendance linéaire > 0.2 σ/an) :")
trajectoires = pivot[pivot['trend'].abs() > 0.2].sort_values('trend', ascending=False)

if len(trajectoires) > 0:
    en_hausse = trajectoires[trajectoires['trend'] > 0]
    en_baisse = trajectoires[trajectoires['trend'] < 0]

    if len(en_hausse) > 0:
        print(f"\n    En amélioration :")
        for _, row in en_hausse.iterrows():
            print(f"      {row['nom']:<40} "
                  f"résidus: {row['res_2022']:+.3f} → {row['res_2023']:+.3f} → {row['res_2024']:+.3f}  "
                  f"(pente={row['trend']:+.3f}/an)")

    if len(en_baisse) > 0:
        print(f"\n    En dégradation :")
        for _, row in en_baisse.iterrows():
            print(f"      {row['nom']:<40} "
                  f"résidus: {row['res_2022']:+.3f} → {row['res_2023']:+.3f} → {row['res_2024']:+.3f}  "
                  f"(pente={row['trend']:+.3f}/an)")
else:
    print(f"    Aucune trajectoire claire détectée.")

# Conclusion sur la prédictibilité
print(f"\n  CONCLUSION SUR LA PRÉDICTIBILITÉ :")
r_23_24, _ = stats.pearsonr(pivot['res_2023'], pivot['res_2024'])
print(f"    r(2023→2024) = {r_23_24:.3f}")
print(f"    → {r_23_24**2*100:.0f}% de la variance du résidu 2024 est expliquée par le résidu 2023")
print(f"    → Un modèle IPS + résidu année précédente devrait bien prédire")
print(f"       le niveau relatif d'un collège à la session suivante.")
