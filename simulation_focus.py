import numpy as np
import pandas as pd

# 1. PARAMÈTRES (Hardcodés pour la reproductibilité de ton article)
# Paramètres OBSERVES (utilisés pour le calcul de l'ANCIEN score)
OBS_PARAMS = {
    "MATHEMATIQUES":      {"mu": 11.8787, "sigma": 3.95099, "coef": 50},
    "FRANCAIS":           {"mu": 12.3112, "sigma": 3.26782, "coef": 50},
    "HISTOIRE-GEO":       {"mu": 12.6060, "sigma": 3.17345, "coef": 40},
    "ARTS":               {"mu": 14.5552, "sigma": 1.96101, "coef": 40},
    "SCIENCES-TECHNO-DP": {"mu": 12.9214, "sigma": 2.80484, "coef": 40},
    "EPS":                {"mu": 14.5939, "sigma": 1.96253, "coef": 40},
    "LANGUES VIVANTES":   {"mu": 13.1834, "sigma": 3.01142, "coef": 40}
}

# Paramètres REELS ESTIMÉS (issus de l'optimisation précédente, utilisés pour générer les élèves et le NOUVEAU score)
REAL_PARAMS = {
    "MATHEMATIQUES":      {"mu": 12.33, "sigma": 5.20},
    "FRANCAIS":           {"mu": 12.47, "sigma": 3.94},
    "HISTOIRE-GEO":       {"mu": 12.88, "sigma": 3.95},
    "ARTS":               {"mu": 15.68, "sigma": 3.22},
    "SCIENCES-TECHNO-DP": {"mu": 13.12, "sigma": 3.44},
    "EPS":                {"mu": 15.81, "sigma": 3.30},
    "LANGUES VIVANTES":   {"mu": 13.79, "sigma": 4.10}
}

SUBJECTS = list(OBS_PARAMS.keys())
N_STUDENTS = 500_000  # Taille de l'échantillon

def run_simulation():
    print(f"--- GÉNÉRATION DE {N_STUDENTS} ÉLÈVES VIRTUELS ---")
    
    # Création des DataFrames pour stocker les notes
    # On génère les notes selon la distribution RÉELLE (Latente)
    data = {}
    for subj in SUBJECTS:
        # Génération aléatoire normale
        raw_grades = np.random.normal(REAL_PARAMS[subj]["mu"], REAL_PARAMS[subj]["sigma"], N_STUDENTS)
        # On borne à 0-20 (réalité physique de la note)
        data[subj] = np.clip(raw_grades, 0, 20)
    
    df = pd.DataFrame(data)

    # --- FILTRAGE : POPULATION CIBLE ---
    # Condition : Avoir au moins 10.0 dans TOUTES les matières
    condition = (df >= 10).all(axis=1)
    df_focus = df[condition].copy()
    
    count_qualified = len(df_focus)
    print(f"Élèves qualifiés (>=10 partout) : {count_qualified} / {N_STUDENTS} ({count_qualified/N_STUDENTS:.1%})")

    if count_qualified == 0:
        print("Aucun élève ne correspond aux critères (vérifiez les paramètres).")
        return

    # --- CALCUL SCORES ---

    # 1. ANCIEN SYSTÈME (Tranchage + Params Obs)
    old_scores = np.zeros(count_qualified)
    
    for subj in SUBJECTS:
        grades = df_focus[subj].values
        
        # Application du Tranchage
        # Note : Ici on sait que grades >= 10, donc seulement deux cas possibles :
        # Si < 15 -> 13 points
        # Si >= 15 -> 16 points
        tranchees = np.where(grades >= 15, 16, 13)
        
        # Formule de lissage avec les paramètres OBSERVES de l'époque
        mu_obs = OBS_PARAMS[subj]["mu"]
        sigma_obs = OBS_PARAMS[subj]["sigma"]
        coef = OBS_PARAMS[subj]["coef"]
        
        z_score = (tranchees - mu_obs) / sigma_obs
        score_matiere = coef * (10 + z_score)
        old_scores += score_matiere

    # 2. NOUVEAU SYSTÈME (Note brute + Params Réels)
    new_scores = np.zeros(count_qualified)
    
    for subj in SUBJECTS:
        grades = df_focus[subj].values # Notes brutes
        
        # Formule de lissage avec les paramètres REELS (car calculés sur notes brutes)
        mu_real = REAL_PARAMS[subj]["mu"]
        sigma_real = REAL_PARAMS[subj]["sigma"]
        coef = OBS_PARAMS[subj]["coef"]
        
        z_score = (grades - mu_real) / sigma_real
        score_matiere = coef * (10 + z_score)
        new_scores += score_matiere

    # --- ANALYSE COMPARATIVE ---
    
    stats_old = {
        "Min": old_scores.min(),
        "Max": old_scores.max(),
        "Moyenne": old_scores.mean(),
        "Écart-type": old_scores.std(),
        "Étendue": old_scores.max() - old_scores.min()
    }
    
    stats_new = {
        "Min": new_scores.min(),
        "Max": new_scores.max(),
        "Moyenne": new_scores.mean(),
        "Écart-type": new_scores.std(),
        "Étendue": new_scores.max() - new_scores.min()
    }

    print("\n--- RÉSULTATS POUR LA POPULATION 'BON ÉLÈVE' (>=10 PARTOUT) ---")
    results = pd.DataFrame([stats_old, stats_new], index=["ANCIEN SYSTÈME", "NOUVEAU SYSTÈME"])
    # Arrondi pour affichage propre
    print(results.round(2).T)
    
    print(f"\nFacteur d'augmentation de l'étendue : x {stats_new['Étendue'] / stats_old['Étendue']:.2f}")

if __name__ == "__main__":
    run_simulation()