#!/usr/bin/env python3
"""
Écarts d'IPS entre deux sources :
  1. HuggingFace fgaume/affelnet-paris-bonus-ips-colleges (IPS_2025 + IPS_2026)
  2. Jeu de données fourni (UAI / IPS)

Jointure par UAI = Identifiant.
"""

import json
import urllib.request

# ═══════════════════════════════════════════════════════════════════════════
# Source 1 : HuggingFace
# ═══════════════════════════════════════════════════════════════════════════
URL_HF = (
    "https://huggingface.co/datasets/fgaume/affelnet-paris-bonus-ips-colleges"
    "/raw/main/affelnet-paris-bonus-ips-colleges.json"
)

# ═══════════════════════════════════════════════════════════════════════════
# Source 2 : Données fournies
# ═══════════════════════════════════════════════════════════════════════════
DATA_FOURNI = {
    "0750360J": 126.5, "0750362L": 128.6, "0750387N": 138.3, "0750407K": 155.9,
    "0750429J": 124.3, "0750444A": 123.7, "0750445B": 99.8, "0750465Y": 113.7,
    "0750478M": 106.8, "0750484U": 94.4, "0750507U": 111.1, "0750525N": 120.0,
    "0750546L": 84.3, "0750550R": 113.3, "0750552T": 115.8, "0750575T": 90.4,
    "0750584C": 111.5, "0750591K": 115.1, "0750607C": 113.8, "0750608D": 121.9,
    "0750609E": 123.2, "0750610F": 106.0, "0750611G": 112.6, "0751563S": 108.9,
    "0751703U": 127.4, "0751705W": 100.7, "0751706X": 118.5, "0751707Y": 100.0,
    "0751790N": 144.3, "0751791P": 141.3, "0751793S": 86.4, "0752107H": 118.9,
    "0752108J": 131.9, "0752186U": 134.5, "0752187V": 129.4, "0752189X": 113.0,
    "0752190Y": 92.1, "0752192A": 118.1, "0752195D": 105.0, "0752196E": 94.5,
    "0752198G": 93.6, "0752248L": 130.8, "0752249M": 122.3, "0752250N": 136.1,
    "0752251P": 124.4, "0752252R": 101.9, "0752316K": 112.9, "0752317L": 128.3,
    "0752318M": 103.8, "0752319N": 108.2, "0752385K": 96.3, "0752387M": 126.8,
    "0752523K": 121.3, "0752524L": 135.8, "0752525M": 131.1, "0752526N": 143.6,
    "0752527P": 141.8, "0752528R": 144.2, "0752529S": 138.0, "0752530T": 142.0,
    "0752531U": 147.0, "0752532V": 125.5, "0752533W": 131.3, "0752534X": 141.8,
    "0752536Z": 131.8, "0752537A": 104.3, "0752538B": 127.5, "0752539C": 119.5,
    "0752540D": 108.5, "0752542F": 118.5, "0752543G": 126.2, "0752544H": 98.2,
    "0752545J": 127.0, "0752546K": 129.9, "0752693V": 125.0, "0752548M": 130.8,
    "0752549N": 142.1, "0752550P": 132.1, "0752551R": 117.4, "0752552S": 138.4,
    "0752553T": 128.0, "0752554U": 117.9, "0752555V": 94.3, "0752556W": 119.2,
    "0752557X": 120.5, "0752606A": 103.4, "0752547L": 98.3, "0752694W": 90.0,
    "0752695X": 101.6, "0752696Y": 106.9, "0752829T": 123.9, "0752957G": 120.3,
    "0752958H": 93.6, "0753046D": 97.4, "0753047E": 111.4, "0753345D": 121.8,
    "0753518S": 109.6, "0753936W": 97.2, "0753937X": 91.8, "0753938Y": 100.0,
    "0753939Z": 86.4, "0754253R": 122.7, "0754305X": 112.4, "0754355B": 100.8,
    "0754528P": 117.2, "0754706H": 104.0, "0755000C": 103.2, "0755030K": 74.4,
    "0755241P": 103.5, "0755433Y": 82.9, "0755747P": 89.1, "0755778Y": 100.6,
    "0755779Z": 105.2,
}
# Note : 0752606A apparaît 2× dans la source (84.6 et 103.4) — on garde 103.4

print(f"Données fournies : {len(DATA_FOURNI)} UAI")

# ═══════════════════════════════════════════════════════════════════════════
# Téléchargement HuggingFace
# ═══════════════════════════════════════════════════════════════════════════
def fetch_json(url, label):
    print(f"Téléchargement {label}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Python/3"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"  → {len(data)} lignes")
    return data

hf = fetch_json(URL_HF, "HuggingFace (affelnet-paris-bonus-ips-colleges)")

# Indexer HF par UAI — on prend IPS_2025 et IPS_2026
dict_hf = {}
for r in hf:
    uai = r.get("Identifiant", "").strip().upper()
    ips_25 = r.get("IPS_2025")
    ips_26 = r.get("IPS_2026")
    nom = r.get("Nom", "")
    secteur = r.get("Secteur", "")
    if uai:
        dict_hf[uai] = {"ips_25": ips_25, "ips_26": ips_26, "nom": nom, "secteur": secteur}

print(f"HF indexés : {len(dict_hf)}")

# ═══════════════════════════════════════════════════════════════════════════
# Jointure et écarts
# ═══════════════════════════════════════════════════════════════════════════
uai_communs = sorted(set(DATA_FOURNI.keys()) & set(dict_hf.keys()))
print(f"UAI communs : {len(uai_communs)}")

ecarts = []
for uai in uai_communs:
    ips_fourni = DATA_FOURNI[uai]
    ips_hf_25 = dict_hf[uai]["ips_25"]
    ips_hf_26 = dict_hf[uai]["ips_26"]
    nom = dict_hf[uai]["nom"]

    delta_25 = (ips_fourni - ips_hf_25) if ips_hf_25 is not None else None
    delta_26 = (ips_fourni - ips_hf_26) if ips_hf_26 is not None else None

    ecarts.append({
        "uai": uai, "nom": nom,
        "ips_fourni": ips_fourni,
        "ips_hf_25": ips_hf_25, "delta_25": delta_25,
        "ips_hf_26": ips_hf_26, "delta_26": delta_26,
    })

# Trier par |Δ vs HF_2026| puis |Δ vs HF_2025|
ecarts.sort(key=lambda x: abs(x["delta_26"] or 0), reverse=True)

# ═══════════════════════════════════════════════════════════════════════════
# Affichage
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f" ÉCARTS D'IPS : Données fournies vs HuggingFace (IPS_2025 et IPS_2026)")
print(f"{'='*110}")
print(f" {'UAI':<11s} {'Nom':<28s} {'Fourni':>7s} │ {'HF 2025':>8s} {'Δ 2025':>7s} │ {'HF 2026':>8s} {'Δ 2026':>7s}")
print(f" {'─'*11} {'─'*28} {'─'*7} │ {'─'*8} {'─'*7} │ {'─'*8} {'─'*7}")

for e in ecarts:
    nom_short = e["nom"][:28]
    d25 = f"{e['delta_25']:+.1f}" if e["delta_25"] is not None else "  n/a"
    d26 = f"{e['delta_26']:+.1f}" if e["delta_26"] is not None else "  n/a"
    hf25 = f"{e['ips_hf_25']:.1f}" if e["ips_hf_25"] is not None else "  n/a"
    hf26 = f"{e['ips_hf_26']:.1f}" if e["ips_hf_26"] is not None else "  n/a"

    marker = ""
    if e["delta_26"] is not None and abs(e["delta_26"]) > 3:
        marker = " ⚠️"
    elif e["delta_26"] is None and e["delta_25"] is not None and abs(e["delta_25"]) > 3:
        marker = " ⚠️"

    print(f" {e['uai']:<11s} {nom_short:<28s} {e['ips_fourni']:>7.1f} │ "
          f"{hf25:>8s} {d25:>7s} │ {hf26:>8s} {d26:>7s}{marker}")

# ═══════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════
for label, key in [("Δ vs HF_2025", "delta_25"), ("Δ vs HF_2026", "delta_26")]:
    deltas = [e[key] for e in ecarts if e[key] is not None]
    if not deltas:
        print(f"\n  {label} : aucune donnée")
        continue
    abs_d = [abs(d) for d in deltas]
    n = len(deltas)
    print(f"\n{'─'*60}")
    print(f" {label} — {n} établissements :")
    print(f"   Écart moyen       : {sum(deltas)/n:+.2f}")
    print(f"   Écart moyen (abs) : {sum(abs_d)/n:.2f}")
    print(f"   Écart médian      : {sorted(deltas)[n//2]:+.1f}")
    print(f"   Écart max (abs)   : {max(abs_d):.1f}")
    print(f"   Identiques (Δ=0)  : {sum(1 for d in deltas if d == 0.0)}")
    print(f"   |Δ| ≤ 1           : {sum(1 for d in abs_d if d <= 1)}")
    print(f"   1 < |Δ| ≤ 3       : {sum(1 for d in abs_d if 1 < d <= 3)}")
    print(f"   |Δ| > 3           : {sum(1 for d in abs_d if d > 3)}")
    print(f"   |Δ| > 5           : {sum(1 for d in abs_d if d > 5)}")

# ═══════════════════════════════════════════════════════════════════════════
# UAI non trouvés
# ═══════════════════════════════════════════════════════════════════════════
only_fourni = set(DATA_FOURNI.keys()) - set(dict_hf.keys())
if only_fourni:
    print(f"\n--- UAI dans données fournies mais absents de HF : {len(only_fourni)} ---")
    for uai in sorted(only_fourni):
        print(f"  {uai} (IPS={DATA_FOURNI[uai]:.1f})")
        