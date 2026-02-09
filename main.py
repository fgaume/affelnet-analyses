import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# --- 1. DONNÉES OBSERVÉES (Source: Académie 2025) ---
# Ce sont les stats issues des notes "tranchées" (3, 8, 13, 16)
data_obs = {
    "MATHEMATIQUES":      {"mu": 11.8787, "sigma": 3.95099, "coef": 50},
    "FRANCAIS":           {"mu": 12.3112, "sigma": 3.26782, "coef": 50},
    "HISTOIRE-GEO":       {"mu": 12.6060, "sigma": 3.17345, "coef": 40},
    "ARTS":               {"mu": 14.5552, "sigma": 1.96101, "coef": 40},
    "SCIENCES-TECHNO-DP": {"mu": 12.9214, "sigma": 2.80484, "coef": 40},
    "EPS":                {"mu": 14.5939, "sigma": 1.96253, "coef": 40},
    "LANGUES VIVANTES":   {"mu": 13.1834, "sigma": 3.01142, "coef": 40}
}

# --- 2. FONCTIONS DU MODÈLE ---

def simulate_tranchage_moments(mu_real, sigma_real):
    """
    Simule le processus de notation Affelnet actuel.
    Calcule la moyenne et l'écart-type théoriques qu'on obtiendrait 
    si les élèves suivaient N(mu_real, sigma_real).
    """
    if sigma_real <= 0.01: return 999, 999 # Sécurité

    # Calcul des probabilités d'appartenance à chaque palier (Area under curve)
    # P(X < 5)
    p1 = norm.cdf(5, loc=mu_real, scale=sigma_real)
    # P(5 <= X < 10)
    p2 = norm.cdf(10, loc=mu_real, scale=sigma_real) - p1
    # P(10 <= X < 15)
    p3 = norm.cdf(15, loc=mu_real, scale=sigma_real) - (p1 + p2)
    # P(X >= 15)
    p4 = 1 - (p1 + p2 + p3)

    # Valeurs discrètes du système actuel
    values = np.array([3, 8, 13, 16])
    probs = np.array([p1, p2, p3, p4])

    # Calcul des moments simulés
    mu_sim = np.sum(values * probs)
    var_sim = np.sum(probs * (values - mu_sim)**2)
    sigma_sim = np.sqrt(var_sim)

    return mu_sim, sigma_sim

def loss_function(params, target_mu, target_sigma):
    """Fonction de coût à minimiser (Moindres Carrés sur les moments)"""
    mu_r, sigma_r = params
    sim_mu, sim_sigma = simulate_tranchage_moments(mu_r, sigma_r)
    
    # On pondère un peu plus l'écart-type pour forcer le modèle à bien capturer la dispersion
    return (sim_mu - target_mu)**2 + 2 * (sim_sigma - target_sigma)**2

# --- 3. REVERSE ENGINEERING (Optimisation) ---
reconstructed_params = {}

print("--- ESTIMATION DES PARAMÈTRES RÉELS (Loi Normale Latente) ---")
print(f"{'Matière':<20} | {'Mu Obs':<6} -> {'Mu Réel':<6} | {'Sig Obs':<6} -> {'Sig Réel':<6}")

for subject, stats in data_obs.items():
    # Estimation initiale (guess)
    x0 = [stats['mu'], stats['sigma'] * 1.5] 
    
    res = minimize(loss_function, x0, args=(stats['mu'], stats['sigma']),
                   method='Nelder-Mead', bounds=[(0, 20), (0.1, 10)])
    
    reconstructed_params[subject] = {
        "mu": res.x[0],
        "sigma": res.x[1],
        "coef": stats['coef']
    }
    
    print(f"{subject:<20} | {stats['mu']:<6.2f} -> {res.x[0]:<6.2f} | {stats['sigma']:<6.2f} -> {res.x[1]:<6.2f}")

# --- 4. SIMULATION DES SCORES ---

def calculate_score_new_system(grades_dict, params_dict):
    """
    Calcule le score avec la nouvelle formule : 
    Score = Coef * (10 + (Note - Mu_Reel) / Sigma_Reel)
    """
    total_score = 0
    for subj, p in params_dict.items():
        note = grades_dict.get(subj, 0)
        # Formule Z-score standardisée
        z = (note - p["mu"]) / p["sigma"]
        # Formule Affelnet lissée
        points = 10 + z
        total_score += points * p["coef"]
    return total_score

def calculate_score_old_system(grades_dict):
    """
    Reconstitution approximative du score max ancien système
    (Borné à 16 points max = note plafond, et 3 points min)
    """
    # Note: Ici on simplifie en prenant les bornes théoriques max/min du système à paliers
    # Max possible partout = 16 points * coef
    # Min possible partout = 3 points * coef
    total_max = 0
    total_min = 0
    for subj, stats in data_obs.items():
        # Système lissé sur note tranchée : 10 + (16 - mu_obs)/sigma_obs
        z_max = (16 - stats["mu"]) / stats["sigma"]
        total_max += (10 + z_max) * stats["coef"]
        
        z_min = (3 - stats["mu"]) / stats["sigma"]
        total_min += (10 + z_min) * stats["coef"]
        
    return total_max, total_min

# --- 5. COMPARAISON DES RÉSULTATS ---

# Scénarios
student_20 = {s: 20 for s in data_obs}
student_10 = {s: 10 for s in data_obs}

new_score_max = calculate_score_new_system(student_20, reconstructed_params)
new_score_min = calculate_score_new_system(student_10, reconstructed_params)

# Anciens scores théoriques (bornes)
old_score_max, old_score_min = calculate_score_old_system({}) 

print("\n--- COMPARAISON D'IMPACT (SUR L'ÉTENDUE) ---")
print(f"ANCIEN Système (Borné [3;16]) :")
print(f"  - Plafond (tout à 16+) : {old_score_max:.2f}")
print(f"  - Plancher (tout à 3)  : {old_score_min:.2f}")
print(f"  - Étendue théorique    : {old_score_max - old_score_min:.2f}")

print(f"\nNOUVEAU Système (Continu [0;20]) :")
print(f"  - Plafond (tout à 20)  : {new_score_max:.2f}")
print(f"  - Plancher (tout à 10) : {new_score_min:.2f}")
print(f"  - Étendue [10-20]      : {new_score_max - new_score_min:.2f}")

print(f"\nGain net pour l'excellence (20 vs Ancien Max) : +{new_score_max - old_score_max:.2f} pts")