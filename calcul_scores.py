import argparse
import sys

# --- 1. PARAMÈTRES STATISTIQUES RÉELS (Estimés précédemment) ---
# Ces paramètres correspondent à la distribution LATENTE (réelle) des notes,
# et non aux statistiques "écrasées" par l'ancien système de tranchage.
REAL_PARAMS = {
    "MATHEMATIQUES":      {"mu": 12.33, "sigma": 5.20, "coef": 50},
    "FRANCAIS":           {"mu": 12.47, "sigma": 3.94, "coef": 50},
    "HISTOIRE-GEO":       {"mu": 12.88, "sigma": 3.95, "coef": 40},
    "ARTS":               {"mu": 15.68, "sigma": 3.22, "coef": 40},
    "SCIENCES-TECHNO-DP": {"mu": 13.12, "sigma": 3.44, "coef": 40},
    "EPS":                {"mu": 15.81, "sigma": 3.30, "coef": 40},
    "LANGUES VIVANTES":   {"mu": 13.79, "sigma": 4.10, "coef": 40}
}

def calculer_score(moyenne_eleve, coef_multiplicateur):
    """
    Calcule le score Affelnet total pour une moyenne donnée.
    Formule : Somme [ Poids_Matiere * Multiplicateur * (10 + Z_score) ]
    """
    total = 0
    for matiere, p in REAL_PARAMS.items():
        # Z-score : Position de l'élève par rapport à la moyenne réelle
        z = (moyenne_eleve - p["mu"]) / p["sigma"]
        
        # Note lissée (base 10 + écart)
        note_lissee = 10 + z
        
        # Score matière pondéré
        score_matiere = note_lissee * p["coef"] * coef_multiplicateur
        total += score_matiere
        
    return total

def main():
    # Gestion des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Simulation des scores Affelnet par moyenne.")
    parser.add_argument(
        "coef", 
        type=float, 
        nargs='?', 
        default=2.3, 
        help="Le coefficient multiplicateur du rectorat (défaut: 2.3)"
    )
    
    args = parser.parse_args()
    coef_rectorat = args.coef

    print(f"\n--- SIMULATION AFFELNET 2025 ---")
    print(f"Paramètre : Coefficient Multiplicateur = x{coef_rectorat}")
    print("-" * 60)
    print(f"{'MOYENNE':<10} | {'SCORE TOTAL':<15} | {'GAIN MARGINAL':<15}")
    print("-" * 60)

    score_precedent = None
    
    # On simule de 20/20 jusqu'à 10/20
    for note in range(20, 9, -1):
        score = calculer_score(note, coef_rectorat)
        
        if score_precedent is not None:
            # Le gain marginal est la différence avec la note inférieure
            # (Ici on affiche la perte en descendant d'un point)
            diff = score - score_precedent
            gain_txt = f"{diff:.0f} pts"
        else:
            gain_txt = "-" # Pas de référence pour le 20/20
            
        print(f"{note:<10} | {score:,.0f} pts      | {gain_txt:<15}")
        score_precedent = score

    print("-" * 60)
    
    # Calculs récapitulatifs
    score_20 = calculer_score(20, coef_rectorat)
    score_10 = calculer_score(10, coef_rectorat)
    etendue = score_20 - score_10
    
    # Valeur moyenne d'un point
    valeur_point = etendue / 10
    
    print(f"RÉSUMÉ RAPIDE :")
    print(f"> Étendue (10-20) : {etendue:,.0f} points")
    print(f"> 1 point de moyenne générale rapporte environ : {valeur_point:,.0f} points Affelnet")
    print(f"> Pour rattraper un Bonus de 600 pts, il faut : +{600/valeur_point:.1f} points de moyenne.")

if __name__ == "__main__":
    main()