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

# ==============================================================================
# 4. STANDARDISATION DE LA NOTE PAR SESSION
# ==============================================================================
session_stats = {}
for session in sessions:
    mask = df['session'] == session
    notes = df.loc[mask, 'note_ecrit']
    mu, sigma = notes.mean(), notes.std()
    df.loc[mask, 'note_std'] = (notes - mu) / sigma + 10
    session_stats[session] = (mu, sigma)
    print(f"  Session {session} : µ={mu:.2f}, σ={sigma:.2f}")

# Public uniquement
df_pub = df[df['secteur'] == 'Public'].copy()

# ==============================================================================
# 5. CALIBRATION DES MODÈLES (train sur 2022-2023)
# ==============================================================================
print(f"\n{'='*70}")
print("CALIBRATION DES MODÈLES (train: 2022-2023, test: 2024)")
print(f"{'='*70}")

# --- Régression IPS sur données d'entraînement (2022+2023) ---
df_train = df_pub[df_pub['session'].isin([2022, 2023])].copy()
df_test = df_pub[df_pub['session'] == 2024].copy()

slope_ips, intercept_ips, r_ips_train, _, _ = stats.linregress(
    df_train['ips'], df_train['note_std']
)
print(f"\n  Régression IPS (train 2022+2023) :")
print(f"    note_std = {slope_ips:.4f} × IPS + {intercept_ips:.2f}")
print(f"    r = {r_ips_train:.4f}")

# --- Résidus par session (en utilisant la régression d'entraînement) ---
for session in sessions:
    mask = df_pub['session'] == session
    df_pub.loc[mask, 'pred_baseline'] = slope_ips * df_pub.loc[mask, 'ips'] + intercept_ips
    df_pub.loc[mask, 'res_baseline'] = df_pub.loc[mask, 'note_std'] - df_pub.loc[mask, 'pred_baseline']

# --- Pivot des résidus ---
pivot_res = df_pub.pivot_table(index=['uai', 'nom'], columns='session',
                                values='res_baseline', aggfunc='first')
pivot_res.columns = [f'res_{int(c)}' for c in pivot_res.columns]
pivot_res = pivot_res.reset_index()

# Pivot des notes std
pivot_note = df_pub.pivot_table(index='uai', columns='session',
                                 values='note_std', aggfunc='first')
pivot_note.columns = [f'note_{int(c)}' for c in pivot_note.columns]
pivot_note = pivot_note.reset_index()

# Pivot des IPS
pivot_ips = df_pub.pivot_table(index='uai', columns='session',
                                values='ips', aggfunc='first')
pivot_ips.columns = [f'ips_{int(c)}' for c in pivot_ips.columns]
pivot_ips = pivot_ips.reset_index()

# Merge
pivot = pivot_res.merge(pivot_note, on='uai').merge(pivot_ips, on='uai')

# --- Modèle 1 : IPS + résidu t-1 ---
# Calibrer c sur : note_2023 = a × IPS_2023 + b + c × res_2022
# Comme a × IPS + b est déjà la prédiction baseline, on a :
# note_2023 - pred_baseline_2023 = c × res_2022
# Donc : res_2023 = c × res_2022
train_m1 = pivot.dropna(subset=['res_2022', 'res_2023'])
slope_c, intercept_c, r_c, _, _ = stats.linregress(
    train_m1['res_2022'], train_m1['res_2023']
)
print(f"\n  Modèle 1 — Shrinkage du résidu (calibré sur 2022→2023) :")
print(f"    res(t) = {slope_c:.4f} × res(t-1) + {intercept_c:.4f}")
print(f"    coefficient de shrinkage c = {slope_c:.4f}")
print(f"    r(res_2022, res_2023) = {r_c:.4f}")

# --- Modèle 2 : IPS + moyenne pondérée res(t-1) et res(t-2) ---
# Besoin des 3 années → calibrer sur la prédiction de 2024 en utilisant 2022+2023
# Mais pour une vraie validation out-of-sample, on calibre c sur 2022→2023
# et on applique les poids fixes (0.67/0.33)
w1, w2 = 0.67, 0.33

# ==============================================================================
# 6. PRÉDICTIONS 2024 (TEST OUT-OF-SAMPLE)
# ==============================================================================
test = pivot.dropna(subset=['res_2022', 'res_2023', 'note_2024', 'ips_2024']).copy()
n_test = len(test)
print(f"\n  Collèges testables (présents 2022-2024) : {n_test}")

# Baseline
test['pred_baseline_2024'] = slope_ips * test['ips_2024'] + intercept_ips

# Modèle 1 : baseline + shrinkage × résidu 2023
test['pred_m1_2024'] = test['pred_baseline_2024'] + slope_c * test['res_2023'] + intercept_c

# Modèle 2 : baseline + shrinkage × moyenne pondérée résidus 2022+2023
test['res_weighted'] = w1 * test['res_2023'] + w2 * test['res_2022']
test['pred_m2_2024'] = test['pred_baseline_2024'] + slope_c * test['res_weighted'] + intercept_c

# Erreurs
for model, pred_col in [('baseline', 'pred_baseline_2024'),
                          ('m1', 'pred_m1_2024'),
                          ('m2', 'pred_m2_2024')]:
    err = test['note_2024'] - test[pred_col]
    test[f'err_{model}'] = err

# ==============================================================================
# 7. MÉTRIQUES DE PERFORMANCE
# ==============================================================================
print(f"\n{'='*70}")
print("PERFORMANCE DES MODÈLES — PRÉDICTION 2024 (OUT-OF-SAMPLE)")
print(f"{'='*70}")

results = {}
for label, pred_col in [('Baseline (IPS seul)', 'pred_baseline_2024'),
                          ('Modèle 1 (IPS + res t-1)', 'pred_m1_2024'),
                          ('Modèle 2 (IPS + res pondéré)', 'pred_m2_2024')]:
    err = test['note_2024'] - test[pred_col]
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    ss_res = (err ** 2).sum()
    ss_tot = ((test['note_2024'] - test['note_2024'].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    r_val = np.corrcoef(test['note_2024'], test[pred_col])[0, 1]
    results[label] = {'RMSE': rmse, 'MAE': mae, 'r²': r2, 'r': r_val}
    print(f"\n  {label} :")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    MAE  = {mae:.4f}")
    print(f"    r²   = {r2:.4f}")
    print(f"    r    = {r_val:.4f}")

# Amélioration relative
rmse_base = results['Baseline (IPS seul)']['RMSE']
rmse_m1 = results['Modèle 1 (IPS + res t-1)']['RMSE']
rmse_m2 = results['Modèle 2 (IPS + res pondéré)']['RMSE']
print(f"\n  Amélioration RMSE :")
print(f"    Modèle 1 vs baseline : {(1 - rmse_m1/rmse_base)*100:+.1f}%")
print(f"    Modèle 2 vs baseline : {(1 - rmse_m2/rmse_base)*100:+.1f}%")

# Top améliorations
test['gain_m1'] = test['err_baseline'].abs() - test['err_m1'].abs()
top_gain = test.nlargest(10, 'gain_m1')
print(f"\n  TOP 10 — Plus forte amélioration avec le modèle 1 :")
for _, row in top_gain.iterrows():
    print(f"    {row['nom']:<40} "
          f"err baseline={row['err_baseline']:+.3f}  "
          f"err modèle 1={row['err_m1']:+.3f}  "
          f"gain={row['gain_m1']:+.3f}")

# ==============================================================================
# 8. FIGURE 1 — PRÉDIT vs OBSERVÉ 2024 (3 modèles)
# ==============================================================================
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))

models_fig = [
    ('Baseline (IPS seul)', 'pred_baseline_2024'),
    ('Modèle 1 (IPS + rés. t-1)', 'pred_m1_2024'),
    ('Modèle 2 (IPS + rés. pondéré)', 'pred_m2_2024'),
]

for ax, (label, pred_col) in zip(axes, models_fig):
    err = test['note_2024'] - test[pred_col]
    rmse = np.sqrt((err ** 2).mean())
    ss_res = (err ** 2).sum()
    ss_tot = ((test['note_2024'] - test['note_2024'].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    ax.scatter(test[pred_col], test['note_2024'], c='#3498db',
               alpha=0.6, s=35, edgecolors='white', linewidth=0.5)

    lim_min = min(test[pred_col].min(), test['note_2024'].min()) - 0.2
    lim_max = max(test[pred_col].max(), test['note_2024'].max()) + 0.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel("Note prédite (std.)", fontsize=10)
    ax.set_ylabel("Note observée 2024 (std.)", fontsize=10)
    ax.set_title(f"{label}\nRMSE={rmse:.3f}  r²={r2:.3f}",
                 fontweight='bold', fontsize=10)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

fig1.suptitle("Prédiction vs réalité — Session 2024 (out-of-sample)\n"
              "Collèges publics parisiens",
              fontweight='bold', fontsize=13, y=1.05)
fig1.tight_layout()

# ==============================================================================
# 9. FIGURE 2 — AMÉLIORATION PAR COLLÈGE
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(8, 8))

ax2.scatter(test['err_baseline'].abs(), test['err_m1'].abs(),
            c='#3498db', alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

lim = max(test['err_baseline'].abs().max(), test['err_m1'].abs().max()) * 1.1
ax2.plot([0, lim], [0, lim], 'k--', alpha=0.5, linewidth=1, label='Pas d\'amélioration')
ax2.fill_between([0, lim], [0, lim], [0, 0], alpha=0.05, color='green')
ax2.fill_between([0, lim], [lim, lim], [0, lim], alpha=0.05, color='red')

# Annoter les plus gros gains
for _, row in top_gain.head(5).iterrows():
    ax2.annotate(row['nom'].replace('CLG ', ''),
                 xy=(abs(row['err_baseline']), abs(row['err_m1'])),
                 fontsize=7, ha='left', va='bottom',
                 xytext=(3, 3), textcoords='offset points')

ax2.set_xlabel("|Erreur baseline (IPS seul)|", fontsize=11)
ax2.set_ylabel("|Erreur modèle 1 (IPS + résidu)|", fontsize=11)
ax2.set_title("Amélioration par collège — Modèle 1 vs baseline\n"
              "(sous la diagonale = amélioration)",
              fontweight='bold', fontsize=12, pad=15)
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_aspect('equal')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig2.tight_layout()

# ==============================================================================
# 10. PRÉDICTIONS 2025 (projection)
# ==============================================================================
# On utilise IPS_2025 + résidu 2024 pour prédire la session 2025

# Collèges avec IPS 2025 et résidu 2024
df_2024 = df_pub[df_pub['session'] == 2024][['uai', 'nom', 'ips', 'note_std', 'res_baseline']].copy()
df_2024.rename(columns={'ips': 'ips_2024', 'res_baseline': 'res_2024'}, inplace=True)

# IPS 2025
ips_2025_cols = ['identifiant', 'nom', 'ips_2025']
if 'ips_2025' in df_ips.columns:
    df_ips_2025 = df_ips[['identifiant', 'ips_2025']].rename(columns={'identifiant': 'uai'})
    df_ips_2025['ips_2025'] = pd.to_numeric(df_ips_2025['ips_2025'], errors='coerce')
    df_2024 = df_2024.merge(df_ips_2025, on='uai', how='left')
    df_2024 = df_2024.dropna(subset=['ips_2025'])
else:
    # Fallback : utiliser IPS 2024 comme proxy
    df_2024['ips_2025'] = df_2024['ips_2024']
    print("  (IPS 2025 non disponible, IPS 2024 utilisé comme proxy)")

# Prédictions
df_2024['pred_2025_baseline'] = slope_ips * df_2024['ips_2025'] + intercept_ips
df_2024['pred_2025_m1'] = df_2024['pred_2025_baseline'] + slope_c * df_2024['res_2024'] + intercept_c

# Couleurs par tranche IPS
colors_ips = {0: '#e74c3c', 400: '#f39c12', 800: '#3498db', 1200: '#2ecc71'}

def bonus_from_ips(ips):
    if ips < 105: return 1200
    elif ips < 117: return 800
    elif ips < 130: return 400
    else: return 0

df_2024['bonus'] = df_2024['ips_2025'].apply(bonus_from_ips)
df_2024 = df_2024.sort_values('pred_2025_m1')

# ==============================================================================
# 11. FIGURE 3 — PRÉDICTIONS 2025
# ==============================================================================
n_pred = len(df_2024)
fig3, ax3 = plt.subplots(figsize=(10, max(8, n_pred * 0.18)))

y_pos = np.arange(n_pred)
bar_colors = [colors_ips[b] for b in df_2024['bonus']]

ax3.barh(y_pos, df_2024['pred_2025_m1'], height=0.7, color=bar_colors,
         alpha=0.7, edgecolor='white', linewidth=0.5)

# Note observée 2024 comme point de référence
ax3.scatter(df_2024['note_std'], y_pos, c='black', s=15, zorder=3,
            marker='|', linewidths=1.5, label='Observé 2024')

ax3.axvline(10, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(df_2024['nom'], fontsize=5.5)
ax3.set_xlabel("Note standardisée (µ=10, σ=1)", fontsize=11)
ax3.set_title("Prédiction du niveau scolaire — Session 2025\n"
              "(modèle IPS + résidu 2024, collèges publics parisiens)",
              fontweight='bold', fontsize=13, pad=15)

# Légende couleurs IPS
import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color=colors_ips[b], label=f"Bonus {b} pts")
                  for b in [1200, 800, 400, 0]]
legend_patches.append(plt.Line2D([0], [0], marker='|', color='black',
                                  linestyle='None', markersize=8,
                                  label='Observé 2024'))
ax3.legend(handles=legend_patches, fontsize=8, loc='lower right')
ax3.grid(axis='x', alpha=0.3)
fig3.tight_layout()

plt.show()

# ==============================================================================
# 12. PRÉDICTIONS 2025 — CONSOLE
# ==============================================================================
print(f"\n{'='*70}")
print("PRÉDICTIONS 2025 — COLLÈGES PUBLICS PARISIENS")
print(f"{'='*70}")
print(f"\n  Modèle : note_std = {slope_ips:.4f} × IPS + {intercept_ips:.2f}"
      f" + {slope_c:.4f} × résidu(t-1) + {intercept_c:.4f}")
print(f"  IPS utilisé : IPS 2025")
print(f"  Résidu utilisé : session 2024")
print(f"\n  {'Collège':<42} {'IPS':>5}  {'Rés.2024':>8}  {'Préd.2025':>9}  {'Obs.2024':>8}")
print(f"  {'-'*78}")

for _, row in df_2024.sort_values('pred_2025_m1', ascending=False).iterrows():
    print(f"  {row['nom']:<42} {row['ips_2025']:>5.0f}  "
          f"{row['res_2024']:>+8.3f}  "
          f"{row['pred_2025_m1']:>9.3f}  "
          f"{row['note_std']:>8.3f}")
