import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. PARAMÈTRES RÉELS (Estimés précédemment) ---
REAL_PARAMS = {
    "MATHEMATIQUES":      {"mu": 12.33, "sigma": 5.20, "coef": 50},
    "FRANCAIS":           {"mu": 12.47, "sigma": 3.94, "coef": 50},
    "HISTOIRE-GEO":       {"mu": 12.88, "sigma": 3.95, "coef": 40},
    "ARTS":               {"mu": 15.68, "sigma": 3.22, "coef": 40},
    "SCIENCES-TECHNO-DP": {"mu": 13.12, "sigma": 3.44, "coef": 40},
    "EPS":                {"mu": 15.81, "sigma": 3.30, "coef": 40},
    "LANGUES VIVANTES":   {"mu": 13.79, "sigma": 4.10, "coef": 40}
}

# Coefficient envisagé par le rectorat
COEF_RECTORAT = 2.3 

def calculate_new_score(average_grade, coef_multiplicateur=1.0):
    """Calcule le score scolaire NOUVEAU (Note Z * Coef Rectorat)"""
    score = 0
    for subj, p in REAL_PARAMS.items():
        z = (average_grade - p["mu"]) / p["sigma"]
        points = 10 + z
        score += points * p["coef"]
    return score * coef_multiplicateur

def calculate_old_score(average_grade):
    """
    Estime l'ANCIEN score (Tranchage + Socle).
    Hypothèse Socle :
    - Si moy >= 14 : 4800 pts (Tout Très Satisfaisant)
    - Si 12 <= moy < 14 : ~4200 pts (Mélange TS/S)
    - Si 10 <= moy < 12 : ~3600 pts (Majorité Satisfaisant)
    """
    # 1. Partie Notes (Tranchées)
    score_notes = 0
    for subj, p in REAL_PARAMS.items():
        # Simulation grossière du tranchage sur la moyenne
        # (Pour être précis il faudrait le faire matière par matière, mais ça suffit pour l'ordre de grandeur)
        if average_grade >= 15: palier = 16
        elif average_grade >= 10: palier = 13
        else: palier = 8
        
        # On utilise les params observés (approximatifs ici pour la démo)
        # On simplifie : note tranchée brute * coef * facteur d'échelle
        # Dans l'ancien système, 16/20 donnait le max. 
        # On va utiliser une approximation linéaire des points "Notes" (max ~3300)
        
        # Note: Cette partie est complexe à simuler parfaitement sans les params 'OBS'
        # On va utiliser une approximation : Max possible = 3300. Min (10/20) = 2700.
        if average_grade >= 15: score_notes = 3300
        elif average_grade >= 10: 
            # Interpolation entre 2700 et 3300
            score_notes = 2700 + (average_grade - 10)/(15-10) * (3300 - 2700)
        else: score_notes = 1500

    # 2. Partie Socle
    if average_grade >= 14: socle = 4800
    elif average_grade >= 12: socle = 4200
    else: socle = 3600 # Simplification
    
    return score_notes + socle

# --- SIMULATION ---
grades = np.linspace(10, 20, 100)
bonus = 600

# Calcul des courbes
y_old_nobonus = [calculate_old_score(g) for g in grades]
y_old_bonus = [s + bonus for s in y_old_nobonus]

y_new_nobonus = [calculate_new_score(g, COEF_RECTORAT) for g in grades]
y_new_bonus = [s + bonus for s in y_new_nobonus]

# --- VISUALISATION ---
plt.figure(figsize=(12, 10))

# GRAPHIQUE 1 : ANCIEN SYSTÈME
plt.subplot(2, 1, 1)
plt.plot(grades, y_old_nobonus, label='Sans Bonus', color='blue')
plt.plot(grades, y_old_bonus, label='Avec Bonus (600)', color='orange', linestyle='--')
plt.title("ANCIEN Système (Socle + Tranchage) : L'Effet Palier", fontsize=12)
plt.ylabel("Score Total")
plt.grid(True, alpha=0.3)
plt.legend()

# GRAPHIQUE 2 : NOUVEAU SYSTÈME (avec Coef)
plt.subplot(2, 1, 2)
plt.plot(grades, y_new_nobonus, label=f'Sans Bonus (Coef x{COEF_RECTORAT})', color='green')
plt.plot(grades, y_new_bonus, label='Avec Bonus (600)', color='red', linestyle='--')
plt.title(f"NOUVEAU Système (Continu x {COEF_RECTORAT}) : La Pente du Mérite", fontsize=12)
plt.xlabel("Moyenne Générale")
plt.ylabel("Score Total")
plt.grid(True, alpha=0.3)
plt.legend()

# Calcul du point de croisement (Rattrapage)
# À quel niveau un élève Sans Bonus dépasse un élève à 12/20 Avec Bonus ?
target_grade = 12.0

# Ancien
score_target_old = calculate_old_score(target_grade) + bonus
# On cherche qui bat ça sans bonus
catchup_old = np.interp(score_target_old, y_old_nobonus, grades)

# Nouveau
score_target_new = calculate_new_score(target_grade, COEF_RECTORAT) + bonus
catchup_new = np.interp(score_target_new, y_new_nobonus, grades)

print(f"--- RATTRAPAGE DU BONUS (Comparatif) ---")
print(f"Scénario : L'élève 'Social' a 12/20 + 600 pts.")
print(f"ANCIEN Système : Il fallait avoir {catchup_old:.2f}/20 pour le dépasser.")
print(f"NOUVEAU Système (x{COEF_RECTORAT}) : Il faut avoir {catchup_new:.2f}/20 pour le dépasser.")

plt.tight_layout()
plt.savefig("comparaison_socle_coef.png")
plt.show()