#!/usr/bin/env python3
"""
Écarts d'IPS entre deux sources :
  1. data.education.gouv.fr — IPS Collèges Paris Public 2024-2025
  2. HuggingFace fgaume/affelnet-paris-bonus-ips-colleges (IPS_2025)

Jointure par UAI (= Identifiant dans le dataset HF).
"""

import json
import urllib.request

# ═══════════════════════════════════════════════════════════════════════════
# URLs
# ═══════════════════════════════════════════════════════════════════════════
URL_OFFICIEL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-ips-colleges-ap2023/exports/json/"
    "?lang=fr"
    "&select=ips%2C+uai%2C+nom_de_l_etablissement"
    "&timezone=Europe%2FParis"
    "&where=%28%28%60code_academie%60+%3D+%2201%22%29%29"
    "+AND+%28%28%60secteur%60+%3D+%22public%22%29%29"
    "+AND+%28%28%60rentree_scolaire%60+%3D+%222024-2025%22%29%29"
)

URL_HF = (
    "https://huggingface.co/datasets/fgaume/affelnet-paris-bonus-ips-colleges"
    "/raw/main/affelnet-paris-bonus-ips-colleges.json"
)

# ═══════════════════════════════════════════════════════════════════════════
# Téléchargement
# ═══════════════════════════════════════════════════════════════════════════
def fetch_json(url, label):
    print(f"Téléchargement {label}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Python/3"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"  → {len(data)} lignes")
    return data

officiel = fetch_json(URL_OFFICIEL, "data.education.gouv.fr (collèges Paris public 2024-2025)")
hf = fetch_json(URL_HF, "HuggingFace (affelnet-paris-bonus-ips-colleges)")

# ═══════════════════════════════════════════════════════════════════════════
# Indexer par UAI
# ═══════════════════════════════════════════════════════════════════════════
# Officiel : champs = uai, ips, nom_de_l_etablissement
dict_off = {}
for r in officiel:
    uai = r.get("uai", "").strip().upper()
    ips = r.get("ips")
    nom = r.get("nom_de_l_etablissement", "")
    if uai and ips is not None:
        dict_off[uai] = {"ips": float(ips), "nom": nom}

# HuggingFace : champs = Identifiant, IPS_2025, Nom, Secteur
dict_hf = {}
for r in hf:
    uai = r.get("Identifiant", "").strip().upper()
    ips = r.get("IPS_2025")
    nom = r.get("Nom", "")
    secteur = r.get("Secteur", "")
    if uai and ips is not None:
        dict_hf[uai] = {"ips": float(ips), "nom": nom, "secteur": secteur}

print(f"\nOfficial indexés : {len(dict_off)}")
print(f"HF indexés       : {len(dict_hf)}")

# ═══════════════════════════════════════════════════════════════════════════
# Jointure et écarts
# ═══════════════════════════════════════════════════════════════════════════
uai_communs = set(dict_off.keys()) & set(dict_hf.keys())
print(f"UAI communs      : {len(uai_communs)}")

ecarts = []
for uai in sorted(uai_communs):
    ips_off = dict_off[uai]["ips"]
    ips_hf = dict_hf[uai]["ips"]
    nom = dict_hf[uai]["nom"] or dict_off[uai]["nom"]
    delta = ips_hf - ips_off
    ecarts.append({
        "uai": uai,
        "nom": nom,
        "ips_officiel": ips_off,
        "ips_hf": ips_hf,
        "ecart": delta,
    })

ecarts.sort(key=lambda x: abs(x["ecart"]), reverse=True)

# ═══════════════════════════════════════════════════════════════════════════
# Affichage
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*95}")
print(f" ÉCARTS D'IPS : data.education.gouv.fr vs HuggingFace (IPS_2025)")
print(f"{'='*95}")
print(f" {'UAI':<12s} {'Nom':<32s} {'IPS Officiel':>12s} {'IPS HF':>8s} {'Δ':>7s}")
print(f" {'─'*12} {'─'*32} {'─'*12} {'─'*8} {'─'*7}")

for e in ecarts:
    nom_short = e["nom"][:32]
    sign = "+" if e["ecart"] > 0 else ""
    marker = " ⚠️" if abs(e["ecart"]) > 3 else ""
    print(f" {e['uai']:<12s} {nom_short:<32s} {e['ips_officiel']:>12.1f} "
          f"{e['ips_hf']:>8.1f} {sign}{e['ecart']:>6.1f}{marker}")

# Stats
deltas = [e["ecart"] for e in ecarts]
abs_deltas = [abs(d) for d in deltas]
n = len(ecarts)

print(f"\n{'─'*60}")
print(f" Statistiques sur {n} établissements communs :")
print(f"   Écart moyen       : {sum(deltas)/n:+.2f}")
print(f"   Écart moyen (abs) : {sum(abs_deltas)/n:.2f}")
print(f"   Écart médian      : {sorted(deltas)[n//2]:+.1f}")
print(f"   Écart max (abs)   : {max(abs_deltas):.1f}")
print(f"   Identiques (Δ=0)  : {sum(1 for d in deltas if d == 0)}")
print(f"   |Δ| ≤ 1           : {sum(1 for d in abs_deltas if d <= 1)}")
print(f"   |Δ| > 1           : {sum(1 for d in abs_deltas if d > 1)}")
print(f"   |Δ| > 3           : {sum(1 for d in abs_deltas if d > 3)}")
print(f"   |Δ| > 5           : {sum(1 for d in abs_deltas if d > 5)}")

# ═══════════════════════════════════════════════════════════════════════════
# Établissements présents dans une seule source
# ═══════════════════════════════════════════════════════════════════════════
only_off = set(dict_off.keys()) - set(dict_hf.keys())
only_hf = set(dict_hf.keys()) - set(dict_off.keys())

if only_off:
    print(f"\n--- Uniquement dans data.education.gouv.fr : {len(only_off)} ---")
    for uai in sorted(only_off):
        print(f"  {uai} - {dict_off[uai]['nom'][:45]} (IPS={dict_off[uai]['ips']:.1f})")

if only_hf:
    # Filtrer sur Public seulement car la source officielle ne contient que le public
    only_hf_pub = [u for u in only_hf if dict_hf[u].get("secteur") == "Public"]
    only_hf_priv = [u for u in only_hf if dict_hf[u].get("secteur") != "Public"]
    if only_hf_pub:
        print(f"\n--- Uniquement dans HF (Public) : {len(only_hf_pub)} ---")
        for uai in sorted(only_hf_pub):
            print(f"  {uai} - {dict_hf[uai]['nom'][:45]} (IPS={dict_hf[uai]['ips']:.1f})")
    print(f"\n--- Uniquement dans HF (Privé) : {len(only_hf_priv)} ---")
    print(f"  (normal : la source officielle ne contient que le public)")