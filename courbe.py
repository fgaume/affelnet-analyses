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

def calculate_academic_score(average_grade):
    """Calcule le score scolaire total pour une moyenne donnée"""
    score = 0
    for subj, p in REAL_PARAMS.items():
        # Simulation d'un élève homogène (ex: 12 partout)
        z = (average_grade - p["mu"]) / p["sigma"]
        points = 10 + z
        score += points * p["coef"]
    return score

# --- 2. GÉNÉRATION DES DONNÉES ---
grades = np.linspace(10, 20, 100) # Axe X : Moyenne de 10 à 20
scores_base = np.array([calculate_academic_score(g) for g in grades])

# Création des scénarios
y_sans_bonus = scores_base
y_bonus_600 = scores_base + 600
y_bonus_1200 = scores_base + 1200

# --- 3. TRACÉ DU GRAPHIQUE ---
plt.figure(figsize=(12, 8))
plt.style.use('ggplot') # Style graphique propre pour l'article

# Courbe 1 : Sans Bonus
plt.plot(grades, y_sans_bonus, label='Collège Standard (Bonus 0)', color='#3498db', linewidth=3)

# Courbe 2 : Bonus IPS (600)
plt.plot(grades, y_bonus_600, label='Collège IPS (Bonus 600)', color='#e67e22', linewidth=3, linestyle='--')

# Courbe 3 : Bonus REP+ (1200) - Optionnel
plt.plot(grades, y_bonus_1200, label='Collège REP+ (Bonus 1200)', color='#2ecc71', linewidth=2, linestyle=':')

# --- 4. ANNOTATIONS PÉDAGOGIQUES ---

# Exemple : Un élève à 12/20 avec Bonus 600
x_ref = 12
y_ref = calculate_academic_score(x_ref) + 600
plt.scatter([x_ref], [y_ref], color='black', zorder=5)
plt.annotate(f'Élève A (12/20 + Bonus)', 
             xy=(x_ref, y_ref), xytext=(x_ref-1, y_ref+200),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Trouver l'équivalent sans bonus (Projection horizontale)
# On cherche quel X donne ce Y sur la courbe bleue
idx = (np.abs(y_sans_bonus - y_ref)).argmin()
x_target = grades[idx]

plt.scatter([x_target], [y_sans_bonus[idx]], color='red', zorder=5)
plt.hlines(y=y_ref, xmin=x_ref, xmax=x_target, colors='gray', linestyles='dotted')
plt.annotate(f'Rattrapage nécessaire : {x_target:.1f}/20', 
             xy=(x_target, y_ref), xytext=(x_target, y_ref-400),
             arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontweight='bold')


# Mise en forme
plt.title("Nouveau Système Affelnet : Le 'Coût Scolaire' du Bonus Social", fontsize=16)
plt.xlabel("Moyenne Générale de l'élève (/20)", fontsize=12)
plt.ylabel("Score Affelnet Total", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(10, 20)

# Sauvegarde et Affichage
print("Génération du graphique en cours...")
plt.savefig("analyse_affelnet.png", dpi=300)
print("Graphique sauvegardé sous 'analyse_affelnet.png'")
plt.show()