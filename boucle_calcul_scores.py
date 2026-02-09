import numpy as np
import pandas as pd

# --- 1. PARAMÈTRES STATISTIQUES RÉELS (Estimés) ---
# Distribution LATENTE (réelle) des notes 2025
REAL_PARAMS = {
    "MATHEMATIQUES":      {"mu": 12.33, "sigma": 5.20, "coef": 50},
    "FRANCAIS":           {"mu": 12.47, "sigma": 3.94, "coef": 50},
    "HISTOIRE-GEO":       {"mu": 12.88, "sigma": 3.95, "coef": 40},
    "ARTS":               {"mu": 15.68, "sigma": 3.22, "coef": 40},
    "SCIENCES-TECHNO-DP": {"mu": 13.12, "sigma": 3.44, "coef": 40},
    "EPS":                {"mu": 15.81, "sigma": 3.30, "coef": 40},
    "LANGUES VIVANTES":   {"mu": 13.79, "sigma": 4.10, "coef": 40}
}

def calculer_score_total(moyenne, coef_multiplicateur):
    """Calcule le score Affelnet total pour une moyenne donnée."""
    total = 0
    for _, p in REAL_PARAMS.items():
        z = (moyenne - p["mu"]) / p["sigma"]
        total += (10 + z) * p["coef"] * coef_multiplicateur
    return total

def main():
    print(f"--- ANALYSE DE SENSIBILITÉ : IMPACT DU COEFFICIENT (2.0 à 2.5) ---")
    print(f"Hypothèse : Bonus Social fixe à 600 points")
    print("-" * 110)

    # Plage de coefficients de 2.0 à 2.5
    coefficients = np.arange(2.0, 2.51, 0.1)
    
    data = []

    for coef in coefficients:
        # Calcul des bornes
        score_20 = calculer_score_total(20, coef)
        score_10 = calculer_score_total(10, coef)
        
        # Indicateurs clés
        etendue = score_20 - score_10
        valeur_point = etendue / 10.0  # Gain moyen pour +1 point de moyenne
        
        # Le "Rattrapage" : Combien de pts de moyenne pour compenser 600 pts de bonus ?
        rattrapage = 600 / valeur_point if valeur_point > 0 else 999
        
        data.append({
            "Coef K": f"x {coef:.1f}",
            "Score Max (20/20)": int(score_20),
            "Score Min (10/20)": int(score_10),
            "Étendue (Max-Min)": int(etendue),
            "Valeur 1 pt Moy.": int(valeur_point),
            "Rattrapage Bonus 600": f"{rattrapage:.2f} pts"
        })

    # Création et affichage du tableau
    df = pd.DataFrame(data)
    
    # Affichage propre
    print(df.to_string(index=False, justify='center', col_space=12))
    print("-" * 110)
    print("LÉGENDE :")
    print("* Étendue : Écart total de points entre le meilleur dossier possible (20/20) et le dossier moyen (10/20).")
    print("* Rattrapage : Points de moyenne générale qu'un élève sans bonus doit avoir en plus pour égaler un élève avec bonus.")

if __name__ == "__main__":
    main()