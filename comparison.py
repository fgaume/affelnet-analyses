import numpy as np
import matplotlib.pyplot as plt

# --- PARAMÈTRES ---
COEF_RECTORAT = 2.3
BONUS_SOCIAL = 600

# Paramètres stats réels (Moyenne pondérée des matières pour simplifier la visualisation)
# On prend une moyenne des Sigmas pondérés ~3.8 pour l'académie
SIGMA_MOYEN = 3.8
MU_MOYEN = 12.5
TOTAL_COEFS = 300 # Somme des coefs matières (50+50+40...)

def get_new_score_gain(note_depart, note_arrivee):
    """Calcule le gain de points Affelnet en passant de note_depart à note_arrivee"""
    # Formule : (Delta_Z * 10 + 10) * Coefs * Multiplicateur
    # Simplifié ici au gain marginal
    z1 = (note_depart - MU_MOYEN) / SIGMA_MOYEN
    z2 = (note_arrivee - MU_MOYEN) / SIGMA_MOYEN
    
    score1 = (10 + z1) * TOTAL_COEFS * COEF_RECTORAT
    score2 = (10 + z2) * TOTAL_COEFS * COEF_RECTORAT
    return score2 - score1

def get_old_score_gain(note_depart, note_arrivee):
    """Gain dans l'ancien système (Tranchage + Socle)"""
    # Palier 15+ -> Note 16. Socle 4800.
    # Palier 10-15 -> Note 13. Socle ~4200 (varie).
    
    def eval_old(n):
        if n >= 15:
            # Plafond atteint : Note=16, Socle=Max
            # Note 16 (Z approx +1) -> Score matière ~3300
            return 3300 + 4800 
        elif n >= 14:
             return 2700 + 4800 # Note 13 mais bon socle
        else:
            return 2700 + 4200 # Note 13 et socle moyen
            
    return eval_old(note_arrivee) - eval_old(note_depart)

# --- VISUALISATION 1 : LE TURBO 15-20 ---
notes_x = [15, 16, 17, 18, 19]
gains_new = [get_new_score_gain(n, n+1) for n in notes_x]
gains_old = [get_old_score_gain(n, n+1) for n in notes_x]

plt.figure(figsize=(14, 12))

# --- GRAPHIQUE 1 : Bar Chart des Gains Marginaux ---
plt.subplot(2, 1, 1)
x_pos = np.arange(len(notes_x))
width = 0.35

plt.bar(x_pos - width/2, gains_old, width, label='Ancien Système (Plafond)', color='#95a5a6', alpha=0.7)
bars_new = plt.bar(x_pos + width/2, gains_new, width, label=f'Nouveau Système (Coef {COEF_RECTORAT})', color='#e74c3c')

# Annotations
for i, v in enumerate(gains_new):
    plt.text(i + width/2, v + 10, f'+{int(v)} pts', ha='center', fontweight='bold', color='#e74c3c')
    
plt.xticks(x_pos, [f'{n}->{n+1}' for n in notes_x])
plt.title(f"LE 'TURBO' SCOLAIRE : Combien rapporte 1 point de moyenne supplémentaire ?\n(Zone d'excellence 15-20)", fontsize=14)
plt.ylabel("Points Affelnet Gagnés")
plt.legend()
plt.grid(axis='y', alpha=0.3)

# --- VISUALISATION 2 : LA COURSE POURSUITE ---
plt.subplot(2, 1, 2)

# Scénario : Cible à battre = Élève avec 12/20 + BONUS 600
# Ancien système : Socle 4200 + Notes(13) ~2700 + 600 = ~7500 pts
# Nouveau système : Score(12) + 600

grades = np.linspace(10, 20, 100)

# Calculs Ancien
score_cible_old = 7500 # Approx pour 12/20 + Bonus
courbe_old = []
for g in grades:
    if g >= 15: val = 3300 + 4800 # Max possible
    elif g >= 14: val = 2700 + 4800
    else: val = 2700 + 4200
    courbe_old.append(val)

# Calculs Nouveau
# Score de la cible (12/20)
z_cible = (12 - MU_MOYEN) / SIGMA_MOYEN
score_base_cible = (10 + z_cible) * TOTAL_COEFS * COEF_RECTORAT
score_total_cible = score_base_cible + 600

# Courbe du challenger (Sans Bonus)
courbe_new = []
for g in grades:
    z = (g - MU_MOYEN) / SIGMA_MOYEN
    val = (10 + z) * TOTAL_COEFS * COEF_RECTORAT
    courbe_new.append(val)

plt.plot(grades, courbe_old, label='Ancien Score (Sans Bonus)', color='gray', linestyle='--')
plt.axhline(y=score_cible_old, color='gray', linestyle=':', label='Cible ANCIENNE (12/20 + Bonus)')

plt.plot(grades, courbe_new, label='Nouveau Score (Sans Bonus)', color='#e74c3c', linewidth=3)
plt.axhline(y=score_total_cible, color='#c0392b', linestyle=':', label='Cible NOUVELLE (12/20 + Bonus)')

# Zone de Rattrapage
plt.fill_between(grades, score_total_cible, courbe_new, where=(np.array(courbe_new) > score_total_cible), 
                 color='#2ecc71', alpha=0.3, label='Zone de Victoire (Rattrapage réussi)')

plt.title("RATTRAPER LE BONUS SOCIAL : Mission Impossible vs Mission Possible", fontsize=14)
plt.xlabel("Moyenne Générale de l'élève Sans Bonus")
plt.ylabel("Score Total")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("excellence_paye.png")
plt.show()