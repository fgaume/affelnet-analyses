#!/usr/bin/env python3
"""
Intervalles de confiance sur les statistiques brutes (sans tranchage)
=====================================================================
On explore la grille (ρ, σ_within) × K seeds pour borner les µ et σ bruts
de chaque champ disciplinaire, en recalibrant à chaque point de la grille
pour toujours coller aux cibles tranchées 2025.

Modèle génératif : loi Beta + copule gaussienne (cohérent avec la simulation
principale affelnet_simulation_final.py).

Parallélisation via ProcessPoolExecutor sur les cœurs physiques.

Sortie : pour chaque champ, un triplet (min, central, max) + couverture.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, beta as beta_dist
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os, warnings, time

warnings.filterwarnings("ignore")

N_ELEVES = 11_000
N_CPU = max(1, os.cpu_count() // 2)  # cœurs physiques

# ═══════════════════════════════════════════════════════════════════════════
# CIBLES ACADÉMIQUES 2025 (après tranchage + regroupement)
# ═══════════════════════════════════════════════════════════════════════════
STATS_ACAD = {
    "ARTS":               {"mu": 14.555,     "sigma": 1.96117},
    "EPS":                {"mu": 14.6405,    "sigma": 1.89741},
    "FRANCAIS":           {"mu": 12.3602,    "sigma": 3.22447},
    "HISTOIRE-GEO":       {"mu": 12.60584,   "sigma": 3.17359568},
    "LANGUES VIVANTES":   {"mu": 13.0314,    "sigma": 3.17393},
    "MATHEMATIQUES":      {"mu": 11.7792,    "sigma": 4.0464},
    "SCIENCES-TECHNO-DP": {"mu": 12.9213,    "sigma": 2.80492},
}

CHAMPS = {
    "ARTS":               2,  # nb de matières
    "EPS":                1,
    "FRANCAIS":           1,
    "HISTOIRE-GEO":       2,
    "LANGUES VIVANTES":   2,
    "MATHEMATIQUES":      1,
    "SCIENCES-TECHNO-DP": 3,
}

CHAMPS_ORDER = list(CHAMPS.keys())

POIDS = {"ARTS": 4, "EPS": 4, "FRANCAIS": 5, "HISTOIRE-GEO": 4,
         "LANGUES VIVANTES": 4, "MATHEMATIQUES": 5, "SCIENCES-TECHNO-DP": 4}

# Champs nécessitant une haute précision de calibration (σ_cible < 2.0)
_HIGH_PREC_CHAMPS = {"ARTS", "EPS"}

# ═══════════════════════════════════════════════════════════════════════════
# MODÈLE GÉNÉRATIF — LOI BETA + COPULE GAUSSIENNE
# ═══════════════════════════════════════════════════════════════════════════

def beta_params_from_mu_sigma(mu_01, sigma_01):
    """Paramètres (a, b) d'une Beta de moyenne µ et écart-type σ dans [0,1]."""
    mu, sig = mu_01, sigma_01
    var = min(sig ** 2, mu * (1 - mu) * 0.99)
    common = mu * (1 - mu) / var - 1
    a = mu * common
    b = (1 - mu) * common
    return max(a, 0.5), max(b, 0.5)


def tranchage(notes):
    result = np.full_like(notes, 3, dtype=float)
    result[notes >= 5]  = 8
    result[notes >= 10] = 13
    result[notes >= 15] = 16
    return result


def generer_et_calculer(mu_raw, sigma_between, n_mat, rho, sigma_within,
                        n=N_ELEVES, seed=None):
    """
    Génère les notes brutes via Beta + copule gaussienne et retourne
    (valeurs_tranchées, valeurs_brutes) pour un champ. Chaque vecteur : (n,).

    rho et sigma_within sont des paramètres de la grille d'exploration.
    """
    if seed is not None:
        np.random.seed(seed)

    # Paramètres Beta sur [0, 1]
    mu_01 = np.clip(mu_raw / 20.0, 0.01, 0.99)
    sig_01 = sigma_between / 20.0
    a, b = beta_params_from_mu_sigma(mu_01, sig_01)

    # Copule gaussienne corrélée
    z_common = np.random.randn(n, 1)
    z_indiv = np.random.randn(n, n_mat)
    z = np.sqrt(rho) * z_common + np.sqrt(1 - rho) * z_indiv

    # Transformation uniforme → Beta
    u = norm.cdf(z)
    aptitudes = beta_dist.ppf(u, a, b) * 20.0  # [0, 20]

    # 3 trimestres bruités
    bruit = np.random.randn(n, n_mat, 3) * sigma_within
    notes = np.clip(aptitudes[:, :, np.newaxis] + bruit, 0, 20)

    # Voie tranchée (étapes 1+2)
    tranches = tranchage(notes)
    val_tranchee = tranches.mean(axis=2).mean(axis=1)

    # Voie brute
    val_brute = notes.mean(axis=2).mean(axis=1)

    return val_tranchee, val_brute


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION POUR UN JEU (rho, sigma_within) DONNÉ
# ═══════════════════════════════════════════════════════════════════════════

# Seeds FIXES pour la calibration — indépendantes des paramètres
# → surface d'objectif lisse, essentiel pour Nelder-Mead
_CALIB_SEEDS_STD = [1000 * i + 42 for i in range(3)]
_CALIB_SEEDS_HI  = [1000 * i + 42 for i in range(6)]
N_CALIB_STD = 3000
N_CALIB_HI  = 6000


def _objectif_calib(params, cible_mu, cible_sig, n_mat, rho, sigma_within,
                    high_precision=False):
    """Fonction objectif pour calibrer (mu_raw, sigma_between)."""
    mu_raw, sig_b = params
    if sig_b < 0.1:
        return 1e6
    if high_precision:
        n_calib, seeds = N_CALIB_HI, _CALIB_SEEDS_HI
    else:
        n_calib, seeds = N_CALIB_STD, _CALIB_SEEDS_STD

    errs_mu, errs_sig = [], []
    for seed in seeds:
        vt, _ = generer_et_calculer(mu_raw, max(sig_b, 0.1), n_mat,
                                     rho, sigma_within, n=n_calib, seed=seed)
        errs_mu.append(vt.mean())
        errs_sig.append(vt.std())
    sm, ss = np.mean(errs_mu), np.mean(errs_sig)

    w_sigma = 2.0 if high_precision else 1.0
    return ((sm - cible_mu) / cible_mu) ** 2 + \
           w_sigma * ((ss - cible_sig) / cible_sig) ** 2


def calibrer_champ(champ, rho, sigma_within):
    """
    Trouve (mu_raw, sigma_between) pour que les stats tranchées
    collent aux cibles, pour un (rho, sigma_within) donné.
    Retourne (mu_raw, sigma_between, erreur_relative_max).
    """
    cible_mu = STATS_ACAD[champ]["mu"]
    cible_sig = STATS_ACAD[champ]["sigma"]
    n_mat = CHAMPS[champ]
    hp = champ in _HIGH_PREC_CHAMPS

    # Grille de points de départ
    if hp:
        mus = [cible_mu, cible_mu + 0.5, cible_mu + 1, cible_mu + 2]
        sigs = [2.0, 3.5, 5.0, 6.5]
    else:
        mus = [cible_mu - 1, cible_mu, cible_mu + 1, cible_mu + 2]
        sigs = [2.0, 3.5, 5.0, 7.0]

    best_cost, best_x = 1e10, None
    for mu0 in mus:
        for s0 in sigs:
            try:
                res = minimize(_objectif_calib, x0=[mu0, s0],
                               args=(cible_mu, cible_sig, n_mat,
                                     rho, sigma_within, hp),
                               method="Nelder-Mead",
                               options={"maxiter": 200, "xatol": 0.01,
                                        "fatol": 1e-7})
                if res.fun < best_cost:
                    best_cost, best_x = res.fun, res.x
            except Exception:
                pass

    # Raffinement
    if best_x is not None:
        try:
            res2 = minimize(_objectif_calib, x0=best_x,
                            args=(cible_mu, cible_sig, n_mat,
                                  rho, sigma_within, hp),
                            method="Nelder-Mead",
                            options={"maxiter": 400 if hp else 300,
                                     "xatol": 0.001 if hp else 0.002,
                                     "fatol": 1e-10 if hp else 1e-9})
            if res2.fun < best_cost:
                best_cost, best_x = res2.fun, res2.x
        except Exception:
            pass

    # Passe 3 haute précision : grille fine 2D (3×3)
    if hp and best_x is not None:
        mu_c, sig_c = best_x
        for dmu in np.linspace(-0.15, 0.15, 3):
            for dsig in np.linspace(-0.2, 0.2, 3):
                try:
                    res3 = minimize(_objectif_calib,
                                    x0=[mu_c + dmu, max(sig_c + dsig, 0.2)],
                                    args=(cible_mu, cible_sig, n_mat,
                                          rho, sigma_within, True),
                                    method="Nelder-Mead",
                                    options={"maxiter": 100, "xatol": 0.001,
                                             "fatol": 1e-10})
                    if res3.fun < best_cost:
                        best_cost, best_x = res3.fun, res3.x
                except Exception:
                    pass

    mu_opt, sig_opt = best_x
    sig_opt = max(sig_opt, 0.1)

    # Erreur relative max pour contrôle qualité
    vt, _ = generer_et_calculer(mu_opt, sig_opt, n_mat, rho, sigma_within,
                                 seed=99999)
    err_mu = abs(vt.mean() - cible_mu) / cible_mu
    err_sig = abs(vt.std() - cible_sig) / cible_sig
    return mu_opt, sig_opt, max(err_mu, err_sig)


# ═══════════════════════════════════════════════════════════════════════════
# WORKER PARALLÈLE — traite un point de grille complet
# ═══════════════════════════════════════════════════════════════════════════

# Grille d'exploration des hyperparamètres structurels
# (réduite : les valeurs extrêmes apportent peu d'info, les rejets sont fréquents)
RHOS = [0.60, 0.70, 0.80, 0.90]
SIGMA_WITHINS = [1.5, 2.0, 2.5]

# Variabilité Monte Carlo par point de grille
K_SEEDS = 20

# Seuil de qualité
SEUIL_QUALITE = 0.05


def _process_grid_point(args):
    """
    Worker : pour un point (rho, sigma_within), calibre tous les champs
    puis simule K seeds. Retourne un dict de résultats ou None si rejeté.
    """
    idx, rho, sw = args
    import time as _t
    _t0 = _t.time()
    pid = os.getpid()

    # Calibrer chaque champ
    calibrations = {}
    for ch in CHAMPS_ORDER:
        _tc = _t.time()
        mu_opt, sig_opt, err_max = calibrer_champ(ch, rho, sw)
        hp_tag = " [HP]" if ch in _HIGH_PREC_CHAMPS else ""
        elapsed_ch = _t.time() - _tc
        if err_max > SEUIL_QUALITE:
            print(f"  [PID {pid}] ρ={rho:.2f} σw={sw:.1f} | "
                  f"{ch:<20s} REJETÉ (err={err_max:.3f}) "
                  f"[{elapsed_ch:.1f}s]", flush=True)
            return None
        calibrations[ch] = (mu_opt, sig_opt)
        print(f"  [PID {pid}] ρ={rho:.2f} σw={sw:.1f} | "
              f"{ch:<20s} ✓ err={err_max:.3f} "
              f"[{elapsed_ch:.1f}s]{hp_tag}", flush=True)

    # Simuler K seeds et collecter les stats brutes
    results = {ch: {"mu_brut": [], "sigma_brut": []} for ch in CHAMPS_ORDER}
    for seed_k in range(K_SEEDS):
        base_seed = seed_k * 10000 + idx
        for ch in CHAMPS_ORDER:
            mu_opt, sig_opt = calibrations[ch]
            _, vb = generer_et_calculer(mu_opt, sig_opt, CHAMPS[ch],
                                         rho, sw,
                                         seed=base_seed + hash(ch) % 9999)
            results[ch]["mu_brut"].append(vb.mean())
            results[ch]["sigma_brut"].append(vb.std())

    elapsed_total = _t.time() - _t0
    print(f"  [PID {pid}] ρ={rho:.2f} σw={sw:.1f} | "
          f"══ TERMINÉ ({K_SEEDS} seeds) [{elapsed_total:.0f}s total] ══",
          flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    grid_points = list(product(RHOS, SIGMA_WITHINS))
    n_grid = len(grid_points)

    print("=" * 72)
    print("  ESTIMATION DES INTERVALLES DE CONFIANCE")
    print(f"  Modèle : Beta + copule gaussienne (cohérent simulation finale)")
    print(f"  Grille : {len(RHOS)} ρ × {len(SIGMA_WITHINS)} σ_within = "
          f"{n_grid} points × {K_SEEDS} seeds")
    print(f"  Parallélisation sur {N_CPU} cœurs physiques")
    print(f"  Haute précision pour : {', '.join(sorted(_HIGH_PREC_CHAMPS))}")
    print("=" * 72)

    # Préparer les tâches
    tasks = [(idx, rho, sw) for idx, (rho, sw) in enumerate(grid_points)]

    # Lancer en parallèle
    t0 = time.time()
    all_results = {ch: {"mu_brut": [], "sigma_brut": []} for ch in CHAMPS_ORDER}
    n_rejet = 0
    n_done = 0

    with ProcessPoolExecutor(max_workers=N_CPU) as pool:
        for result in pool.map(_process_grid_point, tasks):
            n_done += 1
            elapsed = time.time() - t0
            eta = elapsed / n_done * (n_grid - n_done)
            print(f"\r  [{100*n_done/n_grid:5.1f}%] "
                  f"{n_done}/{n_grid} points  "
                  f"({elapsed:.0f}s écoulé, ~{eta:.0f}s restant)   ",
                  end="", flush=True)

            if result is None:
                n_rejet += 1
                continue

            for ch in CHAMPS_ORDER:
                all_results[ch]["mu_brut"].extend(result[ch]["mu_brut"])
                all_results[ch]["sigma_brut"].extend(result[ch]["sigma_brut"])

    elapsed_total = time.time() - t0
    print(f"\n\n  Terminé en {elapsed_total:.0f}s — "
          f"{n_rejet}/{n_grid} points de grille rejetés "
          f"(err > {SEUIL_QUALITE*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSE DES RÉSULTATS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  RÉSULTATS : INTERVALLES DE CONFIANCE À 95%")
    print("=" * 72)

    print(f"\n  {'Champ':<22s} │ {'µ_min':>7s}  {'µ_central':>9s}  "
          f"{'µ_max':>7s} │ "
          f"{'σ_min':>7s}  {'σ_central':>9s}  {'σ_max':>7s} │ N_obs")
    print("  " + "─" * 85)

    summary = {}
    for ch in CHAMPS_ORDER:
        mus = np.array(all_results[ch]["mu_brut"])
        sigs = np.array(all_results[ch]["sigma_brut"])

        mu_lo, mu_med, mu_hi = np.percentile(mus, [2.5, 50, 97.5])
        sig_lo, sig_med, sig_hi = np.percentile(sigs, [2.5, 50, 97.5])

        summary[ch] = {
            "mu":    {"min": mu_lo,  "central": mu_med,  "max": mu_hi},
            "sigma": {"min": sig_lo, "central": sig_med, "max": sig_hi},
            "n_obs": len(mus),
        }

        print(f"  {ch:<22s} │ {mu_lo:>7.3f}  {mu_med:>9.3f}  "
              f"{mu_hi:>7.3f} │ "
              f"{sig_lo:>7.3f}  {sig_med:>9.3f}  {sig_hi:>7.3f} │ {len(mus)}")

    # ═══════════════════════════════════════════════════════════════════════
    # INTERPRÉTATION PROBABILISTE
    # ═══════════════════════════════════════════════════════════════════════
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATION                                                    │
  │                                                                    │
  │  Les intervalles ci-dessus sont des percentiles 2.5%–97.5% sur     │
  │  l'ensemble des observations (grille × seeds).                     │
  │                                                                    │
  │  Ils intègrent DEUX sources d'incertitude :                        │
  │   • Variabilité Monte Carlo (N={N_ELEVES}, {K_SEEDS} seeds/point)          │
  │   • Incertitude de modélisation (ρ ∈ [{min(RHOS)},{max(RHOS)}],              │
  │     σ_within ∈ [{min(SIGMA_WITHINS)},{max(SIGMA_WITHINS)}])                            │
  │                                                                    │
  │  Modèle : Beta + copule gaussienne (bornée [0,20], pas de         │
  │  clipping artificiel). Calibration haute précision pour ARTS/EPS.  │
  │                                                                    │
  │  Les intervalles sont une enveloppe crédible plutôt qu'un IC       │
  │  fréquentiste formel.                                              │
  └─────────────────────────────────────────────────────────────────────┘
""")

    # ═══════════════════════════════════════════════════════════════════════
    # LARGEUR DES INTERVALLES
    # ═══════════════════════════════════════════════════════════════════════
    print("  Largeur des intervalles :")
    print(f"  {'Champ':<22s} │ {'Δµ':>7s}  {'Δσ':>7s}")
    print("  " + "─" * 40)
    for ch in CHAMPS_ORDER:
        s = summary[ch]
        dmu = s["mu"]["max"] - s["mu"]["min"]
        dsig = s["sigma"]["max"] - s["sigma"]["min"]
        print(f"  {ch:<22s} │ {dmu:>7.3f}  {dsig:>7.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # IMPACT SUR LE SCORE AFFELNET
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  Impact sur le score Affelnet (élève à 19/20 partout, coef 2.5) :")
    print(f"  {'Scénario':<22s} │ {'Score':>8s}")
    print("  " + "─" * 35)

    for label, mu_key, sig_key in [("Intervalle bas",  "max", "min"),
                                    ("Central",         "central", "central"),
                                    ("Intervalle haut", "min", "max")]:
        score = 0.0
        for ch in CHAMPS_ORDER:
            T = 19.0
            mu_b = summary[ch]["mu"][mu_key]
            sig_b = summary[ch]["sigma"][sig_key]
            H = 10 * (10 + (T - mu_b) / sig_b)
            score += H * POIDS[ch]
        score *= 2.5
        print(f"  {label:<22s} │ {score:>8.0f}")

    print("\n" + "=" * 72)
    print("  ✅ Analyse terminée")
    print("=" * 72)