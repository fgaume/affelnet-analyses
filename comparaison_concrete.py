import matplotlib.pyplot as plt
import numpy as np

# --- PARAMÈTRES ---
COEF_RECTORAT = 2.3
# Paramètres moyens pondérés (Estimés via l'optimisation précédente)
MU_MOYEN = 12.5
SIGMA_MOYEN = 3.8
TOTAL_COEFS = 300 

def calcul_score_ancien(note):
    """
    Simule le score ANCIEN (Notes Tranchées + Socle).
    Rappel : 
    - Note < 10 -> 8 pts (Score ~1500)
    - 10 <= Note < 15 -> 13 pts (Score ~2700)
    - Note >= 15 -> 16 pts (Score ~3300)
    
    Socle (Simplifié pour l'exemple) :
    - 18/20 -> 4800 (Expert)
    - 15/20 -> 4800 (Expert)
    - 14/20 -> 4200 (Bon) - Souvent le cas
    - 11/20 -> 3600 (Moyen)
    """
    # Partie Notes
    if note >= 15: pts_note = 3300
    elif note >= 10: pts_note = 2700
    else: pts_note = 1500
    
    # Partie Socle (Estimation réaliste)
    if note >= 15: socle = 4800
    elif note >= 13: socle = 4200
    else: socle = 3600
    
    return pts_note + socle

def calcul_score_nouveau(note):
    """Simule le score NOUVEAU (Note Z * Coef 2.3)"""
    z = (note - MU_MOYEN) / SIGMA_MOYEN
    score_base = (10 + z) * TOTAL_COEFS
    return score_base * COEF_RECTORAT

# --- DONNÉES ---
# Duel 1 : Excellence
notes_d1 = [15, 18]
labels_d1 = ["Élève A (15/20)", "Élève B (18/20)"]

# Duel 2 : Classe Moyenne
notes_d2 = [11, 14]
labels_d2 = ["Élève C (11/20)", "Élève D (14/20)"]

# Calculs
old_d1 = [calcul_score_ancien(n) for n in notes_d1]
new_d1 = [calcul_score_nouveau(n) for n in notes_d1]

old_d2 = [calcul_score_ancien(n) for n in notes_d2]
new_d2 = [calcul_score_nouveau(n) for n in notes_d2]

# --- GRAPHIQUE ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Fonction helper pour tracer
def plot_duel(ax, old_scores, new_scores, labels, title):
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, old_scores, width, label='Ancien Système', color='#95a5a6')
    rects2 = ax.bar(x + width/2, new_scores, width, label='Nouveau Système (Coef 2.3)', color='#e74c3c')
    
    ax.set_ylabel('Points Affelnet Totaux')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Annotations des écarts
    ecart_old = old_scores[1] - old_scores[0]
    ecart_new = new_scores[1] - new_scores[0]
    
    # Ligne de l'écart Ancien
    ax.annotate(f'Écart Ancien : {ecart_old} pts', 
                xy=(0.5, max(old_scores) + 200), ha='center', color='#7f8c8d', fontweight='bold')
    
    # Ligne de l'écart Nouveau
    ax.annotate(f'Écart Nouveau : +{int(ecart_new)} pts', 
                xy=(0.5, max(new_scores) + 200), ha='center', color='#c0392b', fontweight='bold', fontsize=12)

# Tracé
plot_duel(ax1, old_d1, new_d1, labels_d1, "DUEL 1 : L'Excellence (15 vs 18)\nLa fin du plafond de verre")
plot_duel(ax2, old_d2, new_d2, labels_d2, "DUEL 2 : Le Ventre Mou (11 vs 14)\nLa fin de la rente du palier 13")

plt.tight_layout()
plt.savefig("comparaison_duels_concrets.png")
plt.show()