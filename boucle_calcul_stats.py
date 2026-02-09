import numpy as np
import pandas as pd

# --- 1. PARAMÈTRES RÉELS (Distribution Latente) ---
REAL_PARAMS = {
    "MATHEMATIQUES":      {"mu": 12.33, "sigma": 5.20, "coef": 50},
    "FRANCAIS":           {"mu": 12.47, "sigma": 3.94, "coef": 50},
    "HISTOIRE-GEO":       {"mu": 12.88, "sigma": 3.95, "coef": 40},
    "ARTS":               {"mu": 15.68, "sigma": 3.22, "coef": 40},
    "SCIENCES-TECHNO-DP": {"mu": 13.12, "sigma": 3.44, "coef": 40},
    "EPS":                {"mu": 15.81, "sigma": 3.30, "coef": 40},
    "LANGUES VIVANTES":   {"mu": 13.79, "sigma": 4.10, "coef": 40}
}

# Nombre d'élèves à simuler pour avoir une stat robuste
N_STUDENTS = 100_000

def main():
    print(f"--- SIMULATION STATISTIQUE ACADÉMIQUE (Moyenne & Écart-Type) ---")
    print(f"Population : {N_STUDENTS} élèves virtuels générés selon les lois normales réelles.")
    print("-" * 85)

    # --- ÉTAPE 1 : GÉNÉRATION DE LA POPULATION (Une seule fois) ---
    # On calcule d'abord le "Score Base" (Coef K=1) pour chaque élève
    # Cela permet de voir l'effet pur du coefficient ensuite
    
    # Initialisation des scores à 0
    scores_base = np.zeros(N_STUDENTS)
    
    for matiere, p in REAL_PARAMS.items():
        # Génération des notes (Loi Normale)
        notes_brutes = np.random.normal(p["mu"], p["sigma"], N_STUDENTS)
        
        # Application des bornes réelles (0-20)
        # C'est important car cela impacte la variance réelle
        notes_bornees = np.clip(notes_brutes, 0, 20)
        
        # Calcul du Z-score
        z = (notes_bornees - p["mu"]) / p["sigma"]
        
        # Score Matière (Base 10 + Z) * Poids Matière
        points_matiere = (10 + z) * p["coef"]
        
        scores_base += points_matiere

    # --- ÉTAPE 2 : BOUCLE SUR LES COEFFICIENTS (2.0 à 2.5) ---
    coefficients = np.arange(2.0, 2.51, 0.1)
    results = []

    for coef in coefficients:
        # Application du multiplicateur
        scores_finaux = scores_base * coef
        
        # Calcul des stats
        moyenne_acad = np.mean(scores_finaux)
        ecart_type_acad = np.std(scores_finaux)
        
        # Min et Max observés dans la population simulée
        min_obs = np.min(scores_finaux)
        max_obs = np.max(scores_finaux)
        
        results.append({
            "Coef K": f"x {coef:.1f}",
            "Moyenne Score": int(moyenne_acad),
            "Écart-Type Score": int(ecart_type_acad),
            "Min Observé": int(min_obs),
            "Max Observé": int(max_obs)
        })

    # --- ÉTAPE 3 : AFFICHAGE ---
    df = pd.DataFrame(results)
    
    # Formatage propre
    print(df.to_string(index=False, justify='center', col_space=14))
    print("-" * 85)
    
    # Petit calcul pour vérifier la théorie
    ratio_sigma = results[-1]["Écart-Type Score"] / results[0]["Écart-Type Score"]
    print(f"Note : L'écart-type augmente proportionnellement au coefficient.")
    print(f"Ratio Sigma (2.5 / 2.0) = {ratio_sigma:.2f} (Théorie : 1.25)")

if __name__ == "__main__":
    main()