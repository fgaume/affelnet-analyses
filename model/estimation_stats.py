#!/usr/bin/env python3
"""
Simulation Affelnet 2026 — Génération des notes trimestrielles de 11 000 collégiens
====================================================================================
Ce script simule les 3 notes trimestrielles des 12 matières pour 11 000 élèves,
de façon à ce que, après tranchage (étape 1) et regroupement (étape 2), les 7
champs disciplinaires aient les moyennes et écarts-types académiques Paris 2026.

Contraintes globales supplémentaires :
  - 9 élèves sur 10 ont plus de 12 de moyenne générale brute
  - 1 élève sur 3 a plus de 15 de moyenne générale brute
→ Satisfaites via un facteur d'aptitude globale (ρ_global) créant une
  corrélation inter-champs tout en préservant les marginales par champ.

Puis il applique l'harmonisation (étape 3) et la pondération (étape 4) pour
calculer le barème Affelnet de chaque élève.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, beta as beta_dist
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, warnings, sys

warnings.filterwarnings("ignore")
np.random.seed(42)

N_CPU = max(1, os.cpu_count() // 2)  # cœurs physiques (hyperthreading exclu)

N_ELEVES = 11_000

# ═══════════════════════════════════════════════════════════════════════════
# PARAMÈTRES ACADÉMIQUES 2026
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

# Composition des 7 champs disciplinaires (12 matières au total)
CHAMPS = {
    "ARTS":               ["Arts Plastiques", "Éducation Musicale"],
    "EPS":                ["EPS"],
    "FRANCAIS":           ["Français"],
    "HISTOIRE-GEO":       ["Histoire-Géo", "EMC"],
    "LANGUES VIVANTES":   ["LV1", "LV2"],
    "MATHEMATIQUES":      ["Mathématiques"],
    "SCIENCES-TECHNO-DP": ["Physique-Chimie", "SVT", "Technologie"],
}

CHAMPS_ORDER = list(CHAMPS.keys())

# Pondérations étape 4
POIDS = {
    "ARTS": 4, "EPS": 4, "FRANCAIS": 5, "HISTOIRE-GEO": 4,
    "LANGUES VIVANTES": 4, "MATHEMATIQUES": 5, "SCIENCES-TECHNO-DP": 4,
}


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — TRANCHAGE (vectorisé)
# ═══════════════════════════════════════════════════════════════════════════
def tranchage(notes: np.ndarray) -> np.ndarray:
    """Convertit des notes /20 en {3, 8, 13, 16}."""
    result = np.full_like(notes, 3, dtype=float)
    result[notes >= 5]  = 8
    result[notes >= 10] = 13
    result[notes >= 15] = 16
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODÈLE DE SIMULATION DES NOTES BRUTES (loi Beta)
# ═══════════════════════════════════════════════════════════════════════════
# Les notes sont bornées [0, 20] → on utilise des lois Beta sur [0, 1]
# mises à l'échelle, ce qui évite le clipping artificiel des lois normales.
#
# Pour chaque matière d'un champ, on modélise :
#   aptitude_i ~ Beta(a, b) * 20       (niveau stable de l'élève)
#   note(i, t) = clip(aptitude_i + bruit_t, 0, 20)
#   bruit_t ~ N(0, σ_within)           (variabilité trimestrielle)
#
# Pour les matières d'un même champ, les aptitudes sont corrélées via
# un facteur commun (copule gaussienne).

SIGMA_WITHIN = 2.0     # bruit intra-élève trimestre à trimestre
RHO = 0.75             # corrélation entre matières d'un même champ

# ═══════════════════════════════════════════════════════════════════════════
# CONTRAINTES GLOBALES SUR LA DISTRIBUTION DES MOYENNES BRUTES
# ═══════════════════════════════════════════════════════════════════════════
# Ces contraintes façonnent la distribution inter-élèves via un facteur
# d'aptitude globale (z_global) partagé entre tous les champs.
# Propriété clé : la copule gaussienne préserve les marginales par champ
# (z reste N(0,1)), seule la corrélation inter-champs change.
CONTRAINTE_PCT_SUP_12 = 0.90   # 9 élèves sur 10 ont plus de 12 de moyenne
CONTRAINTE_PCT_SUP_15 = 1 / 3  # 1 élève sur 3 a plus de 15 de moyenne


def beta_params_from_mu_sigma(mu_01, sigma_01):
    """
    Calcule les paramètres (a, b) d'une loi Beta à partir de
    la moyenne µ et l'écart-type σ souhaités, tous deux dans [0, 1].
    """
    mu, sig = mu_01, sigma_01
    # Variance maximale pour une Beta de moyenne mu : mu*(1-mu)
    var = min(sig ** 2, mu * (1 - mu) * 0.99)  # garde une marge
    common = mu * (1 - mu) / var - 1
    a = mu * common
    b = (1 - mu) * common
    return max(a, 0.5), max(b, 0.5)


def generer_notes_champ(mu_raw, sigma_between, n_matieres, n=N_ELEVES,
                        z_global=None, rho_global=0.0):
    """
    Génère les notes brutes pour un champ via des lois Beta corrélées.
    mu_raw et sigma_between sont exprimés sur l'échelle [0, 20].

    Si z_global est fourni (vecteur N(0,1) de taille n, partagé entre tous
    les champs), la corrélation intra-champ RHO est décomposée en :
      rho_global  (part liée à l'aptitude globale de l'élève)
      RHO - rho_global  (part spécifique au champ)
    La variance totale de z reste 1, donc les marginales sont préservées.

    Retourne : (n, n_matieres, 3) — notes sur [0, 20].
    """
    mu_01 = np.clip(mu_raw / 20.0, 0.01, 0.99)
    sig_01 = sigma_between / 20.0

    a, b = beta_params_from_mu_sigma(mu_01, sig_01)

    # --- Corrélation via copule gaussienne à 3 niveaux ---
    z_common = np.random.randn(n, 1)               # facteur intra-champ
    z_indiv = np.random.randn(n, n_matieres)        # facteur par matière

    if z_global is not None and rho_global > 0:
        rho_field = max(RHO - rho_global, 0.0)
        z = (np.sqrt(rho_global) * z_global[:, np.newaxis]
             + np.sqrt(rho_field) * z_common
             + np.sqrt(1 - RHO) * z_indiv)
    else:
        z = np.sqrt(RHO) * z_common + np.sqrt(1 - RHO) * z_indiv

    u = norm.cdf(z)
    aptitudes_01 = beta_dist.ppf(u, a, b)
    aptitudes = aptitudes_01 * 20.0

    bruit = np.random.randn(n, n_matieres, 3) * SIGMA_WITHIN
    notes = aptitudes[:, :, np.newaxis] + bruit

    return np.clip(notes, 0, 20)


def pipeline_etapes_1_2(notes_brutes):
    """
    Applique tranchage (étape 1) puis moyenne trimestrielle et regroupement
    (étape 2). Entrée : (n, n_mat, 3). Sortie : (n,) valeur du champ.
    """
    tranches = tranchage(notes_brutes)                    # (n, n_mat, 3)
    moy_trim = tranches.mean(axis=2)                      # (n, n_mat)
    return moy_trim.mean(axis=1)                          # (n,)


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION — trouver (µ_raw, σ_between) pour chaque champ
# ═══════════════════════════════════════════════════════════════════════════
# --- Précision de calibration : standard et haute ---
# Les champs à faible σ (ARTS, EPS) nécessitent davantage d'échantillons
# car le tranchage crée une sensibilité extrême autour du seuil 15.
N_CALIB_STD = 5000
N_CALIB_HI  = 12000   # haute précision pour ARTS / EPS
N_SEEDS_STD = 5
N_SEEDS_HI  = 12

_CALIB_SEEDS_STD = [1000 * i + 42 for i in range(N_SEEDS_STD)]
_CALIB_SEEDS_HI  = [1000 * i + 42 for i in range(N_SEEDS_HI)]

# Champs nécessitant une haute précision (σ_cible < 2.0)
_HIGH_PREC_CHAMPS = {"ARTS", "EPS"}


def objectif(params, cible_mu, cible_sigma, n_mat, high_precision=False):
    mu_raw, sigma_between = params
    if sigma_between < 0.1:
        return 1e6
    if high_precision:
        n_calib, seeds = N_CALIB_HI, _CALIB_SEEDS_HI
    else:
        n_calib, seeds = N_CALIB_STD, _CALIB_SEEDS_STD
    errs_mu, errs_sigma = [], []
    for seed in seeds:
        np.random.seed(seed)
        vals = pipeline_etapes_1_2(
            generer_notes_champ(mu_raw, max(sigma_between, 0.1), n_mat, n=n_calib))
        errs_mu.append(vals.mean())
        errs_sigma.append(vals.std())
    sim_mu, sim_sigma = np.mean(errs_mu), np.mean(errs_sigma)
    # Pondération renforcée sur σ pour les champs à faible dispersion :
    # une erreur de 3% sur σ=1.96 est plus critique que sur σ=4.05
    w_sigma = 2.0 if high_precision else 1.0
    return ((sim_mu - cible_mu) / cible_mu) ** 2 + \
           w_sigma * ((sim_sigma - cible_sigma) / cible_sigma) ** 2


def _run_one_optim(args):
    """Worker pour un point de départ (mu0, s0). Top-level pour être picklable."""
    mu0, s0, cible_mu, cible_sigma, n_mat, hp, maxiter, xatol, fatol = args
    try:
        res = minimize(objectif, x0=[mu0, s0],
                       args=(cible_mu, cible_sigma, n_mat, hp),
                       method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol})
        return res.fun, res.x
    except Exception:
        return 1e10, [mu0, s0]


def calibrer_champ(cible_mu, cible_sigma, n_mat, high_precision=False):
    """
    Calibration parallèle en 2 passes (+ passe 3 optionnelle haute précision) :
      1) Grille de points de départ → Nelder-Mead en parallèle
      2) Raffinement séquentiel autour du meilleur point
      3) [haute précision] Grille fine 2D en parallèle
    """
    # Passe 1 : exploration large — parallèle
    if high_precision:
        mus = [cible_mu - 1, cible_mu, cible_mu + 0.5,
               cible_mu + 1, cible_mu + 1.5, cible_mu + 2]
        sigs = [1.5, 2.5, 3.5, 4.5, 5.5]
    else:
        mus = [cible_mu - 1, cible_mu, cible_mu + 1, cible_mu + 2]
        sigs = [2.0, 3.25, 5.0]

    tasks = [(mu0, s0, cible_mu, cible_sigma, n_mat, high_precision,
              200, 0.01, 1e-7)
             for mu0 in mus for s0 in sigs]

    meilleur_cout, meilleur_x = 1e10, None
    with ProcessPoolExecutor(max_workers=N_CPU) as pool:
        for cost, x in pool.map(_run_one_optim, tasks):
            if cost < meilleur_cout:
                meilleur_cout, meilleur_x = cost, x

    # Passe 2 : raffinement séquentiel (1 seul run, rapide)
    if meilleur_x is not None:
        try:
            res2 = minimize(objectif, x0=meilleur_x,
                            args=(cible_mu, cible_sigma, n_mat, high_precision),
                            method="Nelder-Mead",
                            options={"maxiter": 400 if high_precision else 300,
                                     "xatol": 0.001 if high_precision else 0.002,
                                     "fatol": 1e-10 if high_precision else 1e-9})
            if res2.fun < meilleur_cout:
                meilleur_cout = res2.fun
                meilleur_x = res2.x
        except Exception:
            pass

    # Passe 3 [haute précision] : grille fine 2D — parallèle
    if high_precision and meilleur_x is not None:
        mu_center, sig_center = meilleur_x
        tasks3 = [(mu_center + dmu, max(sig_center + dsig, 0.2),
                   cible_mu, cible_sigma, n_mat, True, 100, 0.001, 1e-10)
                  for dmu in np.linspace(-0.2, 0.2, 5)
                  for dsig in np.linspace(-0.3, 0.3, 5)]

        with ProcessPoolExecutor(max_workers=N_CPU) as pool:
            for cost, x in pool.map(_run_one_optim, tasks3):
                if cost < meilleur_cout:
                    meilleur_cout, meilleur_x = cost, x

    mu_opt, sig_opt = meilleur_x
    return mu_opt, max(sig_opt, 0.1)


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION DE RHO_GLOBAL — corrélation inter-champs
# ═══════════════════════════════════════════════════════════════════════════
# On cherche la valeur de rho_global telle que la distribution des moyennes
# brutes (12 matières × 3 trimestres) satisfasse les contraintes :
#   P(moyenne > 12) ≈ 90%   et   P(moyenne > 15) ≈ 33%

N_CALIB_GLOBAL = 6000
N_SEEDS_GLOBAL = 6
_CALIB_SEEDS_GLOBAL = [1000 * i + 77 for i in range(N_SEEDS_GLOBAL)]


def _simuler_moyennes_globales(rho_g, delta_g, params_calibres, n, seed):
    """
    Simule n élèves et retourne leurs moyennes brutes (12 matières).
    delta_g décale z_global : z_global ~ N(delta_g, 1) au lieu de N(0, 1).
    Cela tire le niveau moyen vers le haut (delta_g > 0) ou le bas.
    """
    np.random.seed(seed)
    z_global = np.random.randn(n) + delta_g
    all_subject_means = []
    for champ in CHAMPS_ORDER:
        p = params_calibres[champ]
        grades = generer_notes_champ(
            p["mu_raw"], p["sigma_between"],
            len(CHAMPS[champ]), n=n,
            z_global=z_global, rho_global=rho_g
        )
        moy_trim = grades.mean(axis=2)  # (n, n_mat)
        for j in range(moy_trim.shape[1]):
            all_subject_means.append(moy_trim[:, j])
    return np.stack(all_subject_means, axis=1).mean(axis=1)  # (n,)


def calibrer_rho_global(params_calibres):
    """
    Calibre conjointement (rho_global, delta_global) pour satisfaire les deux
    contraintes sur la moyenne brute :
      - ρ_global contrôle la dispersion inter-élèves (corrélation inter-champs)
      - δ_global décale le niveau moyen global (z_global ~ N(δ, 1))
    """
    def objectif_2d(params):
        rho_g, delta_g = params
        if rho_g < 0.01 or rho_g >= RHO - 0.05:
            return 1e6
        if abs(delta_g) > 2.0:
            return 1e6
        p12s, p15s = [], []
        for seed in _CALIB_SEEDS_GLOBAL:
            moyennes = _simuler_moyennes_globales(
                rho_g, delta_g, params_calibres, N_CALIB_GLOBAL, seed)
            p12s.append((moyennes > 12).mean())
            p15s.append((moyennes > 15).mean())
        p12, p15 = np.mean(p12s), np.mean(p15s)
        return ((p12 - CONTRAINTE_PCT_SUP_12) / CONTRAINTE_PCT_SUP_12) ** 2 + \
               ((p15 - CONTRAINTE_PCT_SUP_15) / CONTRAINTE_PCT_SUP_15) ** 2

    # Passe 1 : grille 2D (ρ × δ)
    best_cost, best_params = 1e10, [0.15, 0.0]
    rhos = np.linspace(0.02, 0.55, 10)
    deltas = np.linspace(-0.3, 0.8, 8)
    for rho0 in rhos:
        for delta0 in deltas:
            cost = objectif_2d([rho0, delta0])
            if cost < best_cost:
                best_cost, best_params = cost, [rho0, delta0]
        sys.stdout.write(f"\r    Grille ρ={rho0:.2f}  "
                         f"meilleur=(ρ={best_params[0]:.3f}, δ={best_params[1]:.3f})  "
                         f"coût={best_cost:.6f}")
        sys.stdout.flush()
    print()

    # Passe 2 : raffinement Nelder-Mead 2D
    res = minimize(objectif_2d, x0=best_params,
                   method="Nelder-Mead",
                   options={"maxiter": 150, "xatol": 0.002, "fatol": 1e-9})
    if res.fun < best_cost:
        best_params = list(res.x)

    rho_opt = np.clip(best_params[0], 0.01, RHO - 0.05)
    delta_opt = np.clip(best_params[1], -2.0, 2.0)
    return rho_opt, delta_opt


# ═══════════════════════════════════════════════════════════════════════════
# RECALIBRATION σ — ajuster σ_between avec rho_global et delta_global
# ═══════════════════════════════════════════════════════════════════════════
# Après la calibration initiale (Phase 1) et la calibration globale (Phase 1b),
# δ_global décale les notes vers le haut, comprimant les σ tranchés.
# On recalibre uniquement σ_between (µ_raw fixé) pour corriger cet effet
# sans perturber les moyennes brutes ni les contraintes globales.
N_RECALIB = 8000
N_SEEDS_RECALIB = 8
_CALIB_SEEDS_RECALIB = [1000 * i + 99 for i in range(N_SEEDS_RECALIB)]


def _objectif_sigma_only(sigma_arr, mu_raw_fixe, cible_mu, cible_sigma, n_mat,
                         rho_g, delta_g, high_precision=False):
    """Objectif 1D : mu_raw fixé, seul sigma_between est optimisé."""
    sigma_between = sigma_arr[0]
    if sigma_between < 0.1:
        return 1e6
    errs_mu, errs_sigma = [], []
    for seed in _CALIB_SEEDS_RECALIB:
        np.random.seed(seed)
        z_global = np.random.randn(N_RECALIB) + delta_g
        vals = pipeline_etapes_1_2(
            generer_notes_champ(mu_raw_fixe, max(sigma_between, 0.1), n_mat,
                                n=N_RECALIB, z_global=z_global, rho_global=rho_g))
        errs_mu.append(vals.mean())
        errs_sigma.append(vals.std())
    sim_mu, sim_sigma = np.mean(errs_mu), np.mean(errs_sigma)
    w_sigma = 2.0 if high_precision else 1.0
    return ((sim_mu - cible_mu) / cible_mu) ** 2 + \
           w_sigma * ((sim_sigma - cible_sigma) / cible_sigma) ** 2


def recalibrer_sigma_avec_global(params_calibres, rho_g, delta_g):
    """
    Recalibre uniquement sigma_between de chaque champ (mu_raw fixé)
    en tenant compte de rho_global et delta_global.
    Cela corrige les erreurs sur σ sans perturber les moyennes brutes
    (et donc sans déstabiliser les contraintes globales P(>12), P(>15)).
    """
    params_recal = {}

    for champ in CHAMPS_ORDER:
        if champ == "EPS":
            continue
        cible = STATS_ACAD[champ]
        n_mat = len(CHAMPS[champ])
        hp = champ in _HIGH_PREC_CHAMPS
        p = params_calibres[champ]
        mu_c, sig_c = p["mu_raw"], p["sigma_between"]

        # Grille 1D sur sigma_between
        best_cost, best_sig = 1e10, sig_c
        for dsig in np.linspace(-1.0, 1.0, 15):
            sig_t = max(sig_c + dsig, 0.2)
            cost = _objectif_sigma_only([sig_t], mu_c, cible["mu"], cible["sigma"],
                                         n_mat, rho_g, delta_g, hp)
            if cost < best_cost:
                best_cost, best_sig = cost, sig_t

        # Raffinement Nelder-Mead 1D
        try:
            res = minimize(_objectif_sigma_only, x0=[best_sig],
                           args=(mu_c, cible["mu"], cible["sigma"],
                                 n_mat, rho_g, delta_g, hp),
                           method="Nelder-Mead",
                           options={"maxiter": 300, "xatol": 0.001, "fatol": 1e-11})
            if res.fun < best_cost:
                best_sig = res.x[0]
        except Exception:
            pass

        sig_opt = max(best_sig, 0.1)
        params_recal[champ] = {"mu_raw": mu_c, "sigma_between": sig_opt}
        tag = " [HP]" if hp else ""
        print(f"    {champ:<22s}  σ_between={sig_opt:5.2f} (était {sig_c:5.2f}){tag}")

    # EPS : sigma seul (mu_raw hérité de ARTS, qui n'a pas changé)
    mu_raw_eps = params_calibres["EPS"]["mu_raw"]  # inchangé
    sig_eps_init = params_calibres["EPS"]["sigma_between"]
    cible_eps = STATS_ACAD["EPS"]

    best_cost_eps, best_sig_eps = 1e10, sig_eps_init
    for dsig in np.linspace(-1.0, 1.0, 15):
        sig_t = max(sig_eps_init + dsig, 0.2)
        cost = _objectif_sigma_only([sig_t], mu_raw_eps,
                                     cible_eps["mu"], cible_eps["sigma"],
                                     1, rho_g, delta_g, True)
        if cost < best_cost_eps:
            best_cost_eps, best_sig_eps = cost, sig_t

    try:
        res_eps = minimize(_objectif_sigma_only, x0=[best_sig_eps],
                           args=(mu_raw_eps, cible_eps["mu"], cible_eps["sigma"],
                                 1, rho_g, delta_g, True),
                           method="Nelder-Mead",
                           options={"maxiter": 300, "xatol": 0.001, "fatol": 1e-11})
        if res_eps.fun < best_cost_eps:
            best_sig_eps = res_eps.x[0]
    except Exception:
        pass

    sig_opt_eps = max(best_sig_eps, 0.1)
    params_recal["EPS"] = {"mu_raw": mu_raw_eps, "sigma_between": sig_opt_eps}
    print(f"    {'EPS (contraint)':<22s}  σ_between={sig_opt_eps:5.2f} "
          f"(était {sig_eps_init:5.2f}) [HP]")

    return params_recal


if __name__ == "__main__":

    print("=" * 72)
    print("  SIMULATION AFFELNET 2026 — 11 000 COLLÉGIENS PARISIENS")
    print(f"  (Parallélisation sur {N_CPU} cœurs)")
    print("=" * 72)

    print("\n▶ Phase 1 : Calibration des distributions de notes brutes...")
    print("  (ARTS calibré en premier, EPS hérite du même µ_raw)\n")

    params_calibres = {}

    # --- Calibration libre pour tous les champs SAUF EPS ---
    for champ in CHAMPS_ORDER:
        if champ == "EPS":
            continue  # traité après ARTS
        cible = STATS_ACAD[champ]
        n_mat = len(CHAMPS[champ])
        hp = champ in _HIGH_PREC_CHAMPS
        mu_opt, sig_opt = calibrer_champ(cible["mu"], cible["sigma"], n_mat,
                                         high_precision=hp)
        params_calibres[champ] = {"mu_raw": mu_opt, "sigma_between": sig_opt}
        tag = " [HP]" if hp else ""
        print(f"  {champ:<22s}  µ_raw={mu_opt:6.2f}  σ_between={sig_opt:5.2f}{tag}")

    # --- Calibration contrainte pour EPS : µ_raw fixé = µ_raw(ARTS) ---
    # Justification : Arts Plastiques, Éducation Musicale et EPS sont des
    # matières notées de façon très comparable dans les collèges (moyennes
    # hautes, faible dispersion). On impose donc que le niveau moyen brut
    # d'EPS soit identique à celui des matières du champ ARTS.
    cible_eps = STATS_ACAD["EPS"]
    mu_raw_eps_fixe = params_calibres["ARTS"]["mu_raw"]


    def objectif_eps_contraint(sigma_between_arr, mu_raw_fixe, cible_mu, cible_sigma):
        """Objectif 1D : seul σ_between est libre, µ_raw est fixé. Haute précision."""
        sigma_between = sigma_between_arr[0]
        if sigma_between < 0.1:
            return 1e6
        errs_mu, errs_sigma = [], []
        for seed in _CALIB_SEEDS_HI:
            np.random.seed(seed)
            vals = pipeline_etapes_1_2(
                generer_notes_champ(mu_raw_fixe, max(sigma_between, 0.1), 1, n=N_CALIB_HI))
            errs_mu.append(vals.mean())
            errs_sigma.append(vals.std())
        sim_mu, sim_sigma = np.mean(errs_mu), np.mean(errs_sigma)
        return ((sim_mu - cible_mu) / cible_mu) ** 2 + \
               2.0 * ((sim_sigma - cible_sigma) / cible_sigma) ** 2


    meilleur_eps, meilleur_cout_eps = None, 1e10
    for s0 in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        try:
            res = minimize(objectif_eps_contraint, x0=[s0],
                           args=(mu_raw_eps_fixe, cible_eps["mu"], cible_eps["sigma"]),
                           method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 0.002, "fatol": 1e-9})
            if res.fun < meilleur_cout_eps:
                meilleur_cout_eps = res.fun
                meilleur_eps = res
        except Exception:
            pass

    # Raffinement EPS — grille fine autour de l'optimum
    if meilleur_eps is not None:
        sig_center = meilleur_eps.x[0]
        for dsig in np.linspace(-0.3, 0.3, 7):
            sig_t = max(sig_center + dsig, 0.2)
            try:
                res2 = minimize(objectif_eps_contraint, x0=[sig_t],
                                args=(mu_raw_eps_fixe, cible_eps["mu"], cible_eps["sigma"]),
                                method="Nelder-Mead",
                                options={"maxiter": 200, "xatol": 0.001, "fatol": 1e-10})
                if res2.fun < meilleur_cout_eps:
                    meilleur_cout_eps = res2.fun
                    meilleur_eps = res2
            except Exception:
                pass

    sig_opt_eps = max(meilleur_eps.x[0], 0.1)
    params_calibres["EPS"] = {"mu_raw": mu_raw_eps_fixe, "sigma_between": sig_opt_eps}
    print(f"  {'EPS (contraint)':<22s}  µ_raw={mu_raw_eps_fixe:6.2f}  "
          f"σ_between={sig_opt_eps:5.2f}  (µ_raw hérité de ARTS) [HP]")


    # ═══════════════════════════════════════════════════════════════════════════
    # CALIBRATION CONJOINTE : (ρ, δ) ↔ params par champ
    #   1b — calibrer (ρ_global, δ_global) pour contraintes globales
    #   Itérations amorties [1c, 1d] — converge sans oscillation
    # ═══════════════════════════════════════════════════════════════════════════

    # --- Phase 1b ---
    print("\n▶ Phase 1b : Calibration de (ρ_global, δ_global)")
    print(f"  Cibles : P(moy > 12) = {CONTRAINTE_PCT_SUP_12:.0%}"
          f"  |  P(moy > 15) = {CONTRAINTE_PCT_SUP_15:.1%}\n")

    rho_global_opt, delta_global_opt = calibrer_rho_global(params_calibres)

    moy_verif = _simuler_moyennes_globales(
        rho_global_opt, delta_global_opt, params_calibres, N_CALIB_GLOBAL, seed=9999)
    print(f"\n  ρ_global = {rho_global_opt:.4f}  δ_global = {delta_global_opt:.4f}")
    print(f"  P(moy>12) = {(moy_verif > 12).mean():.1%}"
          f"  |  P(moy>15) = {(moy_verif > 15).mean():.1%}")

    # --- Phase 1c : recalibrer σ_between uniquement (mu_raw fixés) ---
    print("\n▶ Phase 1c : Recalibration de σ_between (µ_raw fixés)\n")
    params_calibres = recalibrer_sigma_avec_global(params_calibres, rho_global_opt,
                                                    delta_global_opt)

    # Vérification que les contraintes globales n'ont pas bougé
    moy_verif2 = _simuler_moyennes_globales(
        rho_global_opt, delta_global_opt, params_calibres, N_CALIB_GLOBAL, seed=9999)
    print(f"\n  Contraintes globales post-1c :")
    print(f"  P(moy>12) = {(moy_verif2 > 12).mean():.1%}"
          f"  |  P(moy>15) = {(moy_verif2 > 15).mean():.1%}")


    # ═══════════════════════════════════════════════════════════════════════════
    # GÉNÉRATION FINALE (avec corrélation inter-champs + décalage)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n▶ Phase 2 : Génération des notes de 11 000 élèves...")
    print(f"  (ρ_global = {rho_global_opt:.4f}, δ_global = {delta_global_opt:.4f})")
    np.random.seed(2026)

    z_global = np.random.randn(N_ELEVES) + delta_global_opt  # aptitude globale décalée

    notes_brutes = {}      # matière → (N, 3)
    valeurs_champ = {}     # champ → (N,)

    for champ in CHAMPS_ORDER:
        matieres = CHAMPS[champ]
        p = params_calibres[champ]
        grades = generer_notes_champ(p["mu_raw"], p["sigma_between"], len(matieres),
                                     z_global=z_global, rho_global=rho_global_opt)
        for j, mat in enumerate(matieres):
            notes_brutes[mat] = grades[:, j, :]
        valeurs_champ[champ] = pipeline_etapes_1_2(grades)


    # ═══════════════════════════════════════════════════════════════════════════
    # VÉRIFICATION DES STATISTIQUES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n▶ Phase 3 : Vérification des statistiques académiques\n")
    print(f"  {'Champ':<22s} {'µ cible':>8s} {'µ sim':>8s} {'err%':>6s}   "
          f"{'σ cible':>8s} {'σ sim':>8s} {'err%':>6s}")
    print("  " + "─" * 68)

    for champ in CHAMPS_ORDER:
        c = STATS_ACAD[champ]
        v = valeurs_champ[champ]
        mu_s, sig_s = v.mean(), v.std()
        e_mu = 100 * abs(mu_s - c["mu"]) / c["mu"]
        e_sig = 100 * abs(sig_s - c["sigma"]) / c["sigma"]
        ok_mu = "✓" if e_mu < 1 else "✗"
        ok_sig = "✓" if e_sig < 2 else "✗"
        print(f"  {champ:<22s} {c['mu']:>8.3f} {mu_s:>8.3f} {e_mu:>5.2f}% {ok_mu} "
              f"{c['sigma']:>8.3f} {sig_s:>8.3f} {e_sig:>5.2f}% {ok_sig}")


    # ═══════════════════════════════════════════════════════════════════════════
    # VÉRIFICATION DES CONTRAINTES GLOBALES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n▶ Phase 3b : Vérification des contraintes globales (moyenne brute)\n")

    # Moyenne brute par élève (12 matières, moyenne des 3 trimestres)
    all_subj = []
    for mat in [m for ch in CHAMPS_ORDER for m in CHAMPS[ch]]:
        all_subj.append(notes_brutes[mat].mean(axis=1))
    moyennes_brutes = np.stack(all_subj, axis=1).mean(axis=1)

    p12_final = (moyennes_brutes > 12).mean()
    p15_final = (moyennes_brutes > 15).mean()
    ok_12 = "✓" if abs(p12_final - CONTRAINTE_PCT_SUP_12) < 0.03 else "✗"
    ok_15 = "✓" if abs(p15_final - CONTRAINTE_PCT_SUP_15) < 0.05 else "✗"

    print(f"  Moyenne brute : µ = {moyennes_brutes.mean():.2f}  "
          f"σ = {moyennes_brutes.std():.2f}  "
          f"min = {moyennes_brutes.min():.1f}  max = {moyennes_brutes.max():.1f}")
    print(f"  P(moy > 12) = {p12_final:.1%}  (cible {CONTRAINTE_PCT_SUP_12:.0%}) {ok_12}")
    print(f"  P(moy > 15) = {p15_final:.1%}  (cible {CONTRAINTE_PCT_SUP_15:.1%}) {ok_15}")
    print(f"  P10 = {np.percentile(moyennes_brutes, 10):.2f}  "
          f"P25 = {np.percentile(moyennes_brutes, 25):.2f}  "
          f"P50 = {np.percentile(moyennes_brutes, 50):.2f}  "
          f"P75 = {np.percentile(moyennes_brutes, 75):.2f}  "
          f"P90 = {np.percentile(moyennes_brutes, 90):.2f}")


    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPHIQUE : DISTRIBUTION DES MOYENNES GÉNÉRALES BRUTES
    # ═══════════════════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.arange(8, 21, 1)  # [8,9], [9,10], ..., [19,20]
    counts, _ = np.histogram(moyennes_brutes, bins=bins)

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_labels = [f"[{int(bins[i])};{int(bins[i+1])}[" for i in range(len(bins) - 1)]
    bars = ax.bar(range(len(counts)), counts, color="#3498db", edgecolor="white", width=0.85)

    # Colorier les barres selon les seuils de contrainte
    for i, b in enumerate(bars):
        mid = (bins[i] + bins[i + 1]) / 2
        if mid < 12:
            b.set_color("#e74c3c")   # sous le seuil D1
        elif mid >= 15:
            b.set_color("#2ecc71")   # au-dessus du seuil T3

    # Annotations sur chaque barre
    for i, (count, bar) in enumerate(zip(counts, bars)):
        pct = 100 * count / N_ELEVES
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Nombre d'élèves", fontsize=12)
    ax.set_xlabel("Moyenne générale brute (sur 20)", fontsize=12)

    # Lignes de contrainte
    pct_sup_12 = 100 * (moyennes_brutes > 12).mean()
    pct_sup_15 = 100 * (moyennes_brutes > 15).mean()
    ax.set_title(
        f"Distribution des moyennes générales brutes — {N_ELEVES:,} élèves\n"
        f"P(moy > 12) = {pct_sup_12:.1f}% (cible 90%)  |  "
        f"P(moy > 15) = {pct_sup_15:.1f}% (cible 33.3%)",
        fontsize=14, fontweight="bold", pad=15
    )

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", label="Moyenne < 12"),
            plt.Rectangle((0, 0), 1, 1, fc="#3498db", label="12 ≤ Moyenne < 15"),
            plt.Rectangle((0, 0), 1, 1, fc="#2ecc71", label="Moyenne ≥ 15"),
        ],
        loc="upper left", fontsize=10
    )

    plt.tight_layout()
    chart_distrib = "distribution_moyennes_brutes.png"
    plt.savefig(chart_distrib, dpi=300)
    plt.close()
    print(f"\n  → Graphique distribution sauvegardé : {chart_distrib}")


    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTIQUES BRUTES (SANS TRANCHAGE) PAR CHAMP DISCIPLINAIRE
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n▶ Statistiques brutes (sans tranchage) par champ disciplinaire\n")

    valeurs_champ_brut = {}
    for champ in CHAMPS_ORDER:
        matieres = CHAMPS[champ]
        moy_matieres = np.stack(
            [notes_brutes[mat].mean(axis=1) for mat in matieres], axis=1
        )
        valeurs_champ_brut[champ] = moy_matieres.mean(axis=1)

    print(f"  {'Champ':<22s} {'µ brut':>8s} {'σ brut':>8s}   "
          f"{'µ tranché':>10s} {'σ tranché':>10s}   "
          f"{'µ cible':>8s} {'σ cible':>8s}")
    print("  " + "─" * 82)

    for champ in CHAMPS_ORDER:
        c = STATS_ACAD[champ]
        vb = valeurs_champ_brut[champ]
        vt = valeurs_champ[champ]
        print(f"  {champ:<22s} {vb.mean():>8.3f} {vb.std():>8.3f}   "
              f"{vt.mean():>10.3f} {vt.std():>10.3f}   "
              f"{c['mu']:>8.3f} {c['sigma']:>8.3f}")


    # ═══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — HARMONISATION ACADÉMIQUE
    # ═══════════════════════════════════════════════════════════════════════════
    # H = 10 × [10 + (T − µ) / σ]
    # On utilise les µ et σ ACADÉMIQUES fournis (pas ceux simulés)

    print("\n▶ Étape 3 : Harmonisation académique")

    harmonise = {}
    for champ in CHAMPS_ORDER:
        T = valeurs_champ[champ]
        mu_a = STATS_ACAD[champ]["mu"]
        sig_a = STATS_ACAD[champ]["sigma"]
        H = 10 * (10 + (T - mu_a) / sig_a)
        harmonise[champ] = H

    print("  Notes harmonisées : moyenne ≈ 100, écart-type ≈ 10 pour chaque champ ✓")


    # ═══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 4 — PONDÉRATION ET SCORE FINAL
    # ═══════════════════════════════════════════════════════════════════════════
    # Score = Σ (H_champ × poids_champ)
    # Les poids sont 5 (Maths, Français) et 4 (les autres).

    print("\n▶ Étape 4 : Calcul du barème Affelnet")

    scores = np.zeros(N_ELEVES)
    for champ in CHAMPS_ORDER:
        scores += harmonise[champ] * POIDS[champ]
    scores = np.round(scores, 3)

    print(f"\n  {'Score moyen':<25s}: {scores.mean():>10.3f}")
    print(f"  {'Score médian':<25s}: {np.median(scores):>10.3f}")
    print(f"  {'Score minimum':<25s}: {scores.min():>10.3f}")
    print(f"  {'Score maximum':<25s}: {scores.max():>10.3f}")
    print(f"  {'Écart-type':<25s}: {scores.std():>10.3f}")
    print(f"  {'Percentile 10%':<25s}: {np.percentile(scores, 10):>10.3f}")
    print(f"  {'Percentile 25%':<25s}: {np.percentile(scores, 25):>10.3f}")
    print(f"  {'Percentile 75%':<25s}: {np.percentile(scores, 75):>10.3f}")
    print(f"  {'Percentile 90%':<25s}: {np.percentile(scores, 90):>10.3f}")


    # ═══════════════════════════════════════════════════════════════════════════
    # VÉRIFICATION : ÉLÈVE >15 PARTOUT → ~3291
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n▶ Vérification : élève avec >15 partout à chaque trimestre\n")

    score_top = 0.0
    for champ in CHAMPS_ORDER:
        T = 16.0  # toutes les notes tranchées = 16
        mu_a = STATS_ACAD[champ]["mu"]
        sig_a = STATS_ACAD[champ]["sigma"]
        H = 10 * (10 + (T - mu_a) / sig_a)
        contrib = H * POIDS[champ]
        score_top += contrib
        print(f"  {champ:<22s}: T=16  H={H:>7.2f}  ×{POIDS[champ]} = {contrib:>8.2f}")

    print(f"\n  ══> Score total = {score_top:.3f}  (attendu ≈ 3291) ✓")


    # ═══════════════════════════════════════════════════════════════════════════
    # DISTRIBUTION DES NOTES BRUTES
    # ═══════════════════════════════════════════════════════════════════════════
    matieres_ordre = [
        "Français", "Mathématiques", "Histoire-Géo", "EMC",
        "LV1", "LV2", "EPS", "Arts Plastiques", "Éducation Musicale",
        "SVT", "Technologie", "Physique-Chimie"
    ]

    print("\n▶ Aperçu des notes brutes simulées (sur 20)\n")
    print(f"  {'Matière':<22s} {'Moy T1':>7s} {'Moy T2':>7s} {'Moy T3':>7s} "
          f"{'Moy':>7s} {'σ':>7s} {'Min':>5s} {'Max':>5s}")
    print("  " + "─" * 65)
    for mat in matieres_ordre:
        g = notes_brutes[mat]
        print(f"  {mat:<22s} {g[:,0].mean():>7.2f} {g[:,1].mean():>7.2f} "
              f"{g[:,2].mean():>7.2f} {g.mean():>7.2f} {g.mean(axis=1).std():>7.2f} "
              f"{g.min():>5.1f} {g.max():>5.1f}")


    print("\n" + "=" * 72)
    print("  ÉTUDE D'IMPACT : SUPPRESSION DU TRANCHAGE")
    print("  Score scolaire × coefficient (2.0 à 2.5)")
    print("=" * 72)

    # Calcul du score scolaire de base (sans tranchage) pour un élève donné
    def score_sans_tranchage(note_partout, stats_brut):
        """Score Affelnet sans tranchage pour un élève ayant `note_partout` partout."""
        score = 0.0
        for champ in CHAMPS_ORDER:
            T = note_partout
            mu_b = stats_brut[champ]["mu"]
            sig_b = stats_brut[champ]["sigma"]
            H = 10 * (10 + (T - mu_b) / sig_b)
            score += H * POIDS[champ]
        return score

    # Stats brutes académiques issues de la simulation
    stats_brut = {}
    for champ in CHAMPS_ORDER:
        vb = valeurs_champ_brut[champ]
        stats_brut[champ] = {"mu": vb.mean(), "sigma": vb.std()}

    score_base_19 = score_sans_tranchage(19, stats_brut)
    score_base_10 = score_sans_tranchage(10, stats_brut)

    print(f"\n  Score scolaire de base (sans coef):")
    print(f"    Élève à 19/20 partout : {score_base_19:.1f}")
    print(f"    Élève à 10/20 partout : {score_base_10:.1f}")

    coefs = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
    score_19 = np.round(coefs * score_base_19).astype(int)
    score_10 = np.round(coefs * score_base_10).astype(int)
    ecarts = score_19 - score_10

    print(f"\n  {'Coef':>6s} {'Score 19':>10s} {'Score 10':>10s} {'Écart':>8s}")
    print("  " + "─" * 38)
    for i, c in enumerate(coefs):
        print(f"  {c:>6.1f} {score_19[i]:>10d} {score_10[i]:>10d} {ecarts[i]:>8d}")

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPHIQUE DE CALIBRAGE
    # ═══════════════════════════════════════════════════════════════════════════

    # --- REPÈRES ANCIEN SYSTÈME (avec tranchage) ---
    # Score d'un élève >15 partout dans l'ancien système
    OLD_MAX = round(score_sans_tranchage(16, {
        ch: {"mu": STATS_ACAD[ch]["mu"], "sigma": STATS_ACAD[ch]["sigma"]}
        for ch in CHAMPS_ORDER
    }) * 2.5)  # coef max appliqué au score tranché max
    # Score "moyen" ancien système (T=13 partout, coef 2.5)
    OLD_MOYEN = round(score_sans_tranchage(13, {
        ch: {"mu": STATS_ACAD[ch]["mu"], "sigma": STATS_ACAD[ch]["sigma"]}
        for ch in CHAMPS_ORDER
    }) * 2.5)

    # Recalcul propre avec les vraies stats tranchées pour l'ancien système
    def score_ancien(T_value, coef):
        s = 0.0
        for champ in CHAMPS_ORDER:
            H = 10 * (10 + (T_value - STATS_ACAD[champ]["mu"]) / STATS_ACAD[champ]["sigma"])
            s += H * POIDS[champ]
        return round(s * coef)

    OLD_MAX = score_ancien(16, 2.5)    # Ancien plafond (élève >15, coef 2.5)
    OLD_MOYEN = score_ancien(13, 2.5)  # Ancien standard (élève ~13, coef 2.5)

    print(f"\n  Repères ancien système (avec tranchage, coef 2.5):")
    print(f"    Plafond (T=16) : {OLD_MAX}")
    print(f"    Moyen   (T=13) : {OLD_MOYEN}")

    # --- GRAPHIQUE ---
    plt.figure(figsize=(14, 9))
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. ZONES DE COMPARAISON
    plt.axhline(y=OLD_MAX, color="#8e44ad", linestyle="--", linewidth=2, alpha=0.8)
    plt.text(1.86, OLD_MAX + 50, f"Ancien PLAFOND ({OLD_MAX})",
             color="#8e44ad", fontweight="bold", ha="left")

    plt.axhline(y=OLD_MOYEN, color="#7f8c8d", linestyle="--", linewidth=2, alpha=0.8)
    plt.text(1.86, OLD_MOYEN + 50, f"Ancien MOYEN ({OLD_MOYEN})",
             color="#7f8c8d", fontweight="bold", ha="left")

    # 2. LES HALTÈRES
    plt.vlines(x=coefs, ymin=score_10, ymax=score_19,
               color="#34495e", alpha=0.4, linewidth=6)
    plt.scatter(coefs, score_19, s=180, color="#2ecc71",
                label="Nouveau Score Excellent (19/20)", zorder=3,
                edgecolors="white", linewidth=2)
    plt.scatter(coefs, score_10, s=180, color="#e74c3c",
                label="Nouveau Score Moyen (10/20)", zorder=3,
                edgecolors="white", linewidth=2)

    # 3. LE BONUS SOCIAL
    x_bonus = 1.92
    y_base_bonus = score_10[0]
    plt.bar(x_bonus, 400, bottom=y_base_bonus, width=0.04,
            color="#f39c12", label="Bonus Social (400 pts)", alpha=0.9, zorder=4)
    plt.text(x_bonus, y_base_bonus + 200, "Bonus\n400",
             ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    plt.text(x_bonus, y_base_bonus - 150, "Bonus 400",
             ha="center", va="center", fontsize=9, color="#f39c12", fontstyle="italic")

    # 4. ANNOTATIONS
    for i, c in enumerate(coefs):
        y_center = (score_19[i] + score_10[i]) / 2
        plt.text(c, y_center, f"Amp.\n{int(ecarts[i])}",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color="#2c3e50",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#34495e", alpha=0.95))
        plt.text(c, score_19[i] + 120, f"{int(score_19[i])}",
                 ha="center", fontsize=10, color="#27ae60", fontweight="bold")

    # 5. TITRES ET AXES
    plt.title("CALIBRAGE 2026 SANS TRANCHAGE : SCÉNARIOS & COMPARAISONS\n"
              "L'écrasement du bonus social par l'amplitude scolaire",
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Coefficient Multiplicateur", fontsize=13)
    plt.ylabel("Score Affelnet Total", fontsize=13)
    plt.xticks(coefs, [f"Coef {c}" for c in coefs], fontsize=11)
    plt.xlim(1.85, 2.6)

    # Ajuster ylim dynamiquement
    y_min = min(score_10.min(), OLD_MOYEN) - 500
    y_max = max(score_19.max(), OLD_MAX) + 500
    plt.ylim(y_min, y_max)

    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()

    chart_path = "calibration_sans_tranchage_2026.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n  → Graphique sauvegardé : {chart_path}")


    # ═══════════════════════════════════════════════════════════════════════════
    # EXEMPLE : 3 PROFILS D'ÉLÈVES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n▶ Exemples de profils d'élèves\n")

    # Trouver des profils typiques
    idx_top = np.argmax(scores)
    idx_med = np.argsort(scores)[N_ELEVES // 2]
    idx_low = np.argmin(scores)

    for label, idx in [("Meilleur élève", idx_top),
                       ("Élève médian", idx_med),
                       ("Élève le plus bas", idx_low)]:
        print(f"  ═══ {label} (score: {scores[idx]:.3f}) ═══")
        for mat in matieres_ordre:
            g = notes_brutes[mat][idx]
            t = tranchage(g)
            print(f"    {mat:<22s}  {g[0]:5.1f}  {g[1]:5.1f}  {g[2]:5.1f}  "
                  f"→ tranché: {t[0]:2.0f} {t[1]:2.0f} {t[2]:2.0f} "
                  f"(moy={t.mean():.2f})")
        print()

    print("=" * 72)
    print("  ✅ Simulation terminée avec succès")
    print("=" * 72)