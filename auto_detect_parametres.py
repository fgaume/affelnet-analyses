import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import minimize
import pandas as pd

# --- 1. DONNÉES OBSERVÉES ---
DATA_OBSERVED = {
    "MATHEMATIQUES":      {"mu": 11.8787, "sigma": 3.95099},
    "FRANCAIS":           {"mu": 12.3112, "sigma": 3.26782},
    "HISTOIRE-GEO":       {"mu": 12.6060, "sigma": 3.17345},
    "ARTS":               {"mu": 14.5552, "sigma": 1.96101},
    "SCIENCES-TECHNO-DP": {"mu": 12.9214, "sigma": 2.80484},
    "EPS":                {"mu": 14.5939, "sigma": 1.96253},
    "LANGUES VIVANTES":   {"mu": 13.1834, "sigma": 3.01142}
}

# --- 2. FONCTIONS DE SIMULATION ---

def simulate_tranchage_skew(params):
    """
    Simule le système Affelnet avec les 3 paramètres variables : loc, scale, alpha
    """
    loc, scale, alpha = params
    
    # Contraintes physiques pour aider l'optimiseur
    if scale < 0.1: return 999, 999 
    
    # Calcul des probabilités (Loi Skew-Normal)
    p1 = skewnorm.cdf(5, alpha, loc=loc, scale=scale)
    p2 = skewnorm.cdf(10, alpha, loc=loc, scale=scale) - p1
    p3 = skewnorm.cdf(15, alpha, loc=loc, scale=scale) - (p1 + p2)
    p4 = 1 - (p1 + p2 + p3)
    
    # Normalisation
    probs = np.array([p1, p2, p3, p4])
    probs = probs / np.sum(probs)
    
    values = np.array([3.0, 8.0, 13.0, 16.0])
    
    mu_sim = np.sum(values * probs)
    var_sim = np.sum(probs * (values - mu_sim)**2)
    sigma_sim = np.sqrt(var_sim)
    
    return mu_sim, sigma_sim

def loss_function(params, target_mu, target_sigma):
    """
    Fonction de coût globale.
    On cherche à matcher mu et sigma, tout en gardant un alpha 'raisonnable'.
    """
    loc, scale, alpha = params
    
    sim_mu, sim_sigma = simulate_tranchage_skew(params)
    
    # Erreur de reconstruction (Poids fort sur Sigma)
    mse_error = (sim_mu - target_mu)**2 + 5.0 * (sim_sigma - target_sigma)**2
    
    # Régularisation (Rasoir d'Occam) : 
    # On préfère une solution symétrique (alpha=0) si elle marche aussi bien.
    # On ajoute une petite pénalité proportionnelle à alpha^2.
    # Cela évite que l'algo parte vers alpha = -50 si alpha = -2 suffit.
    regularization = 0.01 * (alpha**2)
    
    return mse_error + regularization

# --- 3. DÉTECTION AUTOMATIQUE ---

def main():
    print(f"--- DÉTECTION AUTOMATIQUE DES PARAMÈTRES (SANS A PRIORI) ---")
    print(f"{'MATIERE':<20} | {'ALPHA (Déduit)':<15} | {'STYLE DÉDUIT':<20} | {'LOC':<8} {'SCALE':<8}")
    print("-" * 85)
    
    auto_params = {}

    for subject, obs in DATA_OBSERVED.items():
        # Point de départ neutre (Loi Normale standard)
        # [loc=mu_obs, scale=sigma_obs*1.5, alpha=0]
        x0 = [obs['mu'], obs['sigma'] * 1.5, 0.0]
        
        # Optimisation sur les 3 paramètres
        res = minimize(
            loss_function, 
            x0, 
            args=(obs['mu'], obs['sigma']),
            method='Nelder-Mead',
            bounds=[(0, 25), (0.1, 10), (-10, 10)] # On borne alpha entre -10 et 10
        )
        
        best_loc, best_scale, best_alpha = res.x
        
        # Interprétation du Style
        if best_alpha < -2: style = "Très Généreux (Queue Gauche)"
        elif best_alpha < -0.5: style = "Bienveillant"
        elif best_alpha > 2: style = "Sévère (Queue Droite)"
        elif best_alpha > 0.5: style = "Exigeant"
        else: style = "Neutre / Symétrique"
        
        print(f"{subject:<20} | {best_alpha:<15.2f} | {style:<20} | {best_loc:<8.2f} {best_scale:<8.2f}")
        
        auto_params[subject] = {
            "loc": round(best_loc, 2),
            "scale": round(best_scale, 2),
            "alpha": round(best_alpha, 2)
        }

    print("-" * 85)
    print("PARAMÈTRES CALCULÉS :")
    print(auto_params)

if __name__ == "__main__":
    main()