import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# --- 1. DONNÉES OBSERVÉES (Source : Académie 2025 - Notes Tranchées) ---
# Ce sont les cibles que l'algorithme doit atteindre via simulation.
DATA_OBSERVED = {
    "MATHEMATIQUES":      {"mu": 11.8787, "sigma": 3.95099},
    "FRANCAIS":           {"mu": 12.3112, "sigma": 3.26782},
    "HISTOIRE-GEO":       {"mu": 12.6060, "sigma": 3.17345},
    "ARTS":               {"mu": 14.5552, "sigma": 1.96101},
    "SCIENCES-TECHNO-DP": {"mu": 12.9214, "sigma": 2.80484},
    "EPS":                {"mu": 14.5939, "sigma": 1.96253},
    "LANGUES VIVANTES":   {"mu": 13.1834, "sigma": 3.01142}
}

def simulate_tranchage_process(mu_real, sigma_real):
    """
    Simule le processus de notation Affelnet 'Ancien Système'.
    Transforme une distribution continue N(mu, sigma) en distribution discrète {3, 8, 13, 16}.
    Retourne la moyenne et l'écart-type de cette distribution discrète.
    """
    if sigma_real <= 0.01: return 999, 999 # Sécurité anti-div0

    # Calcul des masses de probabilité pour chaque palier (Aire sous la courbe)
    # P(X < 5) -> Score 3
    p1 = norm.cdf(5, loc=mu_real, scale=sigma_real)
    
    # P(5 <= X < 10) -> Score 8
    p2 = norm.cdf(10, loc=mu_real, scale=sigma_real) - p1
    
    # P(10 <= X < 15) -> Score 13
    p3 = norm.cdf(15, loc=mu_real, scale=sigma_real) - (p1 + p2)
    
    # P(X >= 15) -> Score 16
    # Note: On prend tout ce qui reste au-dessus de 15
    p4 = 1 - (p1 + p2 + p3)

    # Valeurs imposées par le système
    values = np.array([3.0, 8.0, 13.0, 16.0])
    probs = np.array([p1, p2, p3, p4])

    # Calcul des moments statistiques simulés
    mu_sim = np.sum(values * probs)
    var_sim = np.sum(probs * (values - mu_sim)**2)
    sigma_sim = np.sqrt(var_sim)

    return mu_sim, sigma_sim

def loss_function(params, target_mu, target_sigma):
    """
    Fonction de coût à minimiser.
    Calcule la distance entre les stats simulées et les stats officielles.
    """
    mu_r, sigma_r = params
    
    # Pénalité pour les paramètres aberrants (hors 0-20 ou sigma négatif)
    if not (0 <= mu_r <= 20) or sigma_r < 0.1:
        return 1e6

    sim_mu, sim_sigma = simulate_tranchage_process(mu_r, sigma_r)
    
    # On pondère l'erreur sur le sigma (x2) car c'est la variable la plus sensible
    error = (sim_mu - target_mu)**2 + 2.0 * (sim_sigma - target_sigma)**2
    return error

def estimate_real_parameters():
    print(f"{'DISCIPLINE':<20} | {'MU RÉEL (Est.)':<15} | {'SIGMA RÉEL (Est.)':<15}")
    print("-" * 60)
    
    estimated_params = {}

    for subject, stats in DATA_OBSERVED.items():
        # Point de départ de l'optimisation (Guess initial)
        # On part de la moyenne observée et d'un sigma boosté (car on sait qu'il est sous-estimé)
        initial_guess = [stats['mu'], stats['sigma'] * 1.5]
        
        # Lancement de l'optimiseur (Nelder-Mead est robuste pour ce type de surface non-convexe)
        result = minimize(
            loss_function, 
            initial_guess, 
            args=(stats['mu'], stats['sigma']),
            method='Nelder-Mead',
            bounds=[(0, 20), (0.1, 10)] # Bornes physiques
        )
        
        mu_est, sigma_est = result.x
        estimated_params[subject] = {"mu": round(mu_est, 2), "sigma": round(sigma_est, 2)}
        
        print(f"{subject:<20} | {mu_est:<15.2f} | {sigma_est:<15.2f}")

    print("-" * 60)
    print("\nDICTIONNAIRE POUR COPIER-COLLER DANS VOS SCRIPTS :")
    print("ESTIMATED_REAL_PARAMS = " + str(estimated_params))

if __name__ == "__main__":
    estimate_real_parameters()