import numpy as np
import pandas as pd

# Paramètres
N_ELEVES = 10_000
COEF_RECTORAT = 2.5

# Simulation de moyennes générales (Distribution normale)
moyennes = np.clip(np.random.normal(12.5, 3.8, N_ELEVES), 0, 20)

def score_ancien(m):
    # Simulation simplifiée du score discret (paliers)
    if m >= 15: base = 3300 + 4800
    elif m >= 10: base = 2700 + 4200
    else: base = 1500 + 3600
    return base

def score_nouveau(m):
    # Score continu
    # On arrondit à 2 décimales car les notes scolaires sont rarement plus précises
    # Mais le calcul pondéré crée de la décimale
    return round(m * 100 * COEF_RECTORAT, 2) 

# Génération des scores
scores_old = [score_ancien(m) for m in moyennes]
scores_new = [score_nouveau(m) for m in moyennes]

# Analyse des collisions (Ex-aequos)
df = pd.DataFrame({'Old': scores_old, 'New': scores_new})

collision_old = df.duplicated(subset=['Old']).sum()
collision_new = df.duplicated(subset=['New']).sum()

print(f"--- ANALYSE DE L'EFFET LOTERIE (Sur {N_ELEVES} élèves) ---")
print(f"ANCIEN SYSTÈME :")
print(f"  - Nombre d'élèves ex-aequo : {collision_old}")
print(f"  - Taux de saturation : {collision_old / N_ELEVES:.1%} (Le hasard décide pour eux)")

print(f"\nNOUVEAU SYSTÈME :")
print(f"  - Nombre d'élèves ex-aequo : {collision_new}")
print(f"  - Taux de saturation : {collision_new / N_ELEVES:.2%} (Presque nul)")