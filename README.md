Suite à une modélisation statistique, on prend comme hypothèse les moyennes et écarts-types académiques suivants pour simuler 2026 :

|            | Maths  | Français | Histoire-Géo | Langues | Sciences | Arts   | EPS    |
|------------|--------|----------|--------------|---------|----------|--------|--------|
| Moyenne    | 12,152 | 12,631   | 13,119       | 13,822  | 13,458   | 15,930 | 15,883 |
| Écart-type | 4,989  | 3,945    | 4,076        | 4,211   | 3,660    | 2,960  | 2,889  |

## Méthode d'estimation des statistiques brutes (sans tranchage)

Les seules données publiques disponibles sont les statistiques académiques **après tranchage** (le système Affelnet convertit les notes en paliers {3, 8, 13, 16}). Or, pour simuler un scénario sans tranchage, il faut connaître les distributions brutes des notes.

Le problème est donc inverse : à partir des moyennes et écarts-types observés après tranchage, retrouver les distributions de notes brutes sous-jacentes.

### Modèle génératif

Pour chaque champ disciplinaire, on simule 11 000 élèves :

1. **Notes brutes** : les aptitudes de chaque élève suivent une loi **Beta** sur [0, 20]. Le choix d'une loi Beta plutôt qu'une simple gaussienne est important : les notes scolaires sont bornées entre 0 et 20, et leur distribution est souvent asymétrique (concentrée vers le haut pour des matières comme Arts ou EPS). Une loi normale tronquée à [0, 20] introduirait des artefacts aux bornes (accumulation artificielle d'élèves à 0 ou 20 par clipping), fausserait la moyenne et l'écart-type effectifs, et ne pourrait pas capturer l'asymétrie naturelle des distributions. La loi Beta, définie nativement sur un intervalle borné, respecte ces contraintes sans manipulation artificielle et peut prendre des formes très variées (symétrique, asymétrique à gauche ou à droite, uniforme) selon ses paramètres (a, b), déduits ici d'une moyenne `µ_raw` et d'un écart-type `σ_between` à calibrer.

2. **Corrélation intra-champ** : pour les champs composés de plusieurs matières (ex. Sciences = Physique-Chimie + SVT + Technologie), les aptitudes des matières d'un même champ sont corrélées via une **copule gaussienne** (ρ = 0.75). Une copule est un outil statistique qui permet de coupler des distributions marginales quelconques (ici des lois Beta) tout en contrôlant leur dépendance. La copule gaussienne fonctionne en trois étapes : on génère d'abord des variables normales corrélées (via un facteur commun partagé et un facteur individuel par matière), puis on les transforme en variables uniformes sur [0, 1] via la CDF normale, et enfin on les convertit en loi Beta via la CDF inverse. Ainsi, chaque matière suit bien une loi Beta marginale, mais un élève bon en SVT a de fortes chances d'être aussi bon en Physique-Chimie.

3. **Variabilité trimestrielle** : chaque élève a 3 notes par matière. Un bruit gaussien (σ = 2 points) est ajouté à l'aptitude stable pour simuler la variabilité d'un trimestre à l'autre.

4. **Pipeline Affelnet** : les notes brutes passent par le tranchage ({3, 8, 13, 16}), puis par la moyenne trimestrielle et le regroupement par champ, reproduisant le calcul officiel.

### Calibration par optimisation

Pour chaque champ, on cherche le couple (µ_raw, σ_between) tel que, après passage dans le pipeline complet (tranchage → moyenne → regroupement), les statistiques simulées correspondent aux statistiques académiques cibles.

L'optimisation utilise **Nelder-Mead** en plusieurs passes. Nelder-Mead est un algorithme d'optimisation sans gradient : il explore l'espace des paramètres en déformant un simplexe (un triangle en 2D) qui se contracte progressivement vers le minimum. Il est adapté ici car la fonction objectif est bruitée (basée sur des simulations Monte-Carlo) et non différentiable (le tranchage introduit des discontinuités), ce qui exclut les méthodes à gradient classiques.
- **Passe 1** : exploration large sur une grille de points de départ, parallélisée sur plusieurs cœurs.
- **Passe 2** : raffinement séquentiel autour du meilleur point.
- **Passe 3** (champs sensibles Arts/EPS) : grille fine 2D supplémentaire, car le tranchage crée une sensibilité extrême autour du seuil 15 pour ces champs à moyennes élevées.

La fonction objectif compare, sur plusieurs seeds (5 en standard, 12 en haute précision), les moyennes et écarts-types simulés aux cibles, avec une pondération renforcée sur σ pour les champs à faible dispersion.

**Contrainte EPS** : le µ_raw d'EPS est contraint à être identique à celui d'Arts (matières notées de façon très comparable), seul σ_between est optimisé librement.

### Résultat

Une fois calibrés, les paramètres (µ_raw, σ_between) permettent de générer des distributions de notes brutes réalistes. Les statistiques de ces distributions **avant tranchage** donnent les moyennes et écarts-types du tableau ci-dessus, utilisés dans les simulations de scénarios sans tranchage.

### Sortie du programme

```
========================================================================
  SIMULATION AFFELNET 2025 — 11 000 COLLÉGIENS PARISIENS
  (Parallélisation sur 6 cœurs)
========================================================================

▶ Phase 1 : Calibration des distributions de notes brutes...
  (ARTS calibré en premier, EPS hérite du même µ_raw)

  ARTS                    µ_raw= 16.09  σ_between= 3.27 [HP]
  FRANCAIS                µ_raw= 12.68  σ_between= 3.83
  HISTOIRE-GEO            µ_raw= 13.22  σ_between= 4.34
  LANGUES VIVANTES        µ_raw= 13.96  σ_between= 4.52
  MATHEMATIQUES           µ_raw= 12.25  σ_between= 4.97
  SCIENCES-TECHNO-DP      µ_raw= 13.54  σ_between= 4.01
  EPS (contraint)         µ_raw= 16.09  σ_between= 2.88  (µ_raw hérité de ARTS) [HP]

▶ Phase 2 : Génération des notes de 11 000 élèves...

▶ Phase 3 : Vérification des statistiques académiques

  Champ                   µ cible    µ sim   err%    σ cible    σ sim   err%
  ────────────────────────────────────────────────────────────────────
  ARTS                     14.555   14.538  0.12% ✓    1.961    1.977  0.81% ✓
  EPS                      14.640   14.656  0.11% ✓    1.897    1.901  0.21% ✓
  FRANCAIS                 12.360   12.362  0.01% ✓    3.224    3.221  0.11% ✓
  HISTOIRE-GEO             12.606   12.622  0.13% ✓    3.174    3.143  0.96% ✓
  LANGUES VIVANTES         13.031   13.063  0.24% ✓    3.174    3.136  1.20% ✓
  MATHEMATIQUES            11.779   11.804  0.21% ✓    4.046    4.061  0.35% ✓
  SCIENCES-TECHNO-DP       12.921   12.935  0.11% ✓    2.805    2.785  0.72% ✓

▶ Statistiques brutes (sans tranchage) par champ disciplinaire

  Champ                    µ brut   σ brut    µ tranché  σ tranché    µ cible  σ cible
  ──────────────────────────────────────────────────────────────────────────────────
  ARTS                     15.885    2.999       14.538      1.977     14.555    1.961
  EPS                      15.920    2.948       14.656      1.901     14.640    1.897
  FRANCAIS                 12.621    3.949       12.362      3.221     12.360    3.224
  HISTOIRE-GEO             13.118    4.025       12.622      3.143     12.606    3.174
  LANGUES VIVANTES         13.840    4.143       13.063      3.136     13.031    3.174
  MATHEMATIQUES            12.201    5.020       11.804      4.061     11.779    4.046
  SCIENCES-TECHNO-DP       13.462    3.620       12.935      2.785     12.921    2.805

▶ Étape 3 : Harmonisation académique
  Notes harmonisées : moyenne ≈ 100, écart-type ≈ 10 pour chaque champ ✓

▶ Étape 4 : Calcul du barème Affelnet

  Score moyen              :   3001.129
  Score médian             :   3009.848
  Score minimum            :   2388.801
  Score maximum            :   3286.071
  Écart-type               :    113.822
  Percentile 10%           :   2847.734
  Percentile 25%           :   2927.581
  Percentile 75%           :   3083.327
  Percentile 90%           :   3142.809

▶ Vérification : élève avec >15 partout à chaque trimestre

  ARTS                  : T=16  H= 107.37  ×4 =   429.47
  EPS                   : T=16  H= 107.17  ×4 =   428.66
  FRANCAIS              : T=16  H= 111.29  ×5 =   556.44
  HISTOIRE-GEO          : T=16  H= 110.69  ×4 =   442.78
  LANGUES VIVANTES      : T=16  H= 109.35  ×4 =   437.41
  MATHEMATIQUES         : T=16  H= 110.43  ×5 =   552.16
  SCIENCES-TECHNO-DP    : T=16  H= 110.98  ×4 =   443.90

  ══> Score total = 3290.824  (attendu ≈ 3291) ✓

▶ Aperçu des notes brutes simulées (sur 20)

  Matière                 Moy T1  Moy T2  Moy T3     Moy       σ   Min   Max
  ─────────────────────────────────────────────────────────────────
  Français                 12.63   12.60   12.64   12.62    3.95   0.0  20.0
  Mathématiques            12.19   12.23   12.18   12.20    5.02   0.0  20.0
  Histoire-Géo             13.08   13.13   13.11   13.11    4.38   0.0  20.0
  EMC                      13.13   13.15   13.11   13.13    4.36   0.0  20.0
  LV1                      13.85   13.84   13.85   13.85    4.51   0.0  20.0
  LV2                      13.84   13.82   13.84   13.83    4.51   0.0  20.0
  EPS                      15.92   15.91   15.93   15.92    2.95   0.0  20.0
  Arts Plastiques          15.90   15.89   15.90   15.90    3.28   0.0  20.0
  Éducation Musicale       15.85   15.87   15.89   15.87    3.30   0.0  20.0
  SVT                      13.46   13.46   13.46   13.46    4.09   0.0  20.0
  Technologie              13.47   13.45   13.45   13.46    4.06   0.0  20.0
  Physique-Chimie          13.48   13.45   13.49   13.47    4.08   0.0  20.0

========================================================================
  ÉTUDE D'IMPACT : SUPPRESSION DU TRANCHAGE
  Score scolaire × coefficient (2.0 à 2.5)
========================================================================

  Score scolaire de base (sans coef):
    Élève à 19/20 partout : 3401.3
    Élève à 10/20 partout : 2679.8

    Coef   Score 19   Score 10    Écart
  ──────────────────────────────────────
     2.0       6803       5360     1443
     2.1       7143       5627     1516
     2.2       7483       5895     1588
     2.3       7823       6163     1660
     2.4       8163       6431     1732
     2.5       8503       6699     1804

  Repères ancien système (avec tranchage, coef 2.5):
    Plafond (T=16) : 8227
    Moyen   (T=13) : 7411

  → Graphique sauvegardé : calibration_sans_tranchage_2025.png

▶ Exemples de profils d'élèves

  ═══ Meilleur élève (score: 3286.071) ═══
    Français                 19.2   19.4   15.2  → tranché: 16 16 16 (moy=16.00)
    Mathématiques            18.3   18.6   19.2  → tranché: 16 16 16 (moy=16.00)
    Histoire-Géo             20.0   18.9   16.2  → tranché: 16 16 16 (moy=16.00)
    EMC                      19.4   18.3   20.0  → tranché: 16 16 16 (moy=16.00)
    LV1                      19.2   17.4   18.3  → tranché: 16 16 16 (moy=16.00)
    LV2                      20.0   17.2   20.0  → tranché: 16 16 16 (moy=16.00)
    EPS                      19.5   19.9   18.3  → tranché: 16 16 16 (moy=16.00)
    Arts Plastiques          17.4   18.6   19.1  → tranché: 16 16 16 (moy=16.00)
    Éducation Musicale       20.0   16.2   20.0  → tranché: 16 16 16 (moy=16.00)
    SVT                      18.7   16.8   15.0  → tranché: 16 16 16 (moy=16.00)
    Technologie              15.3   16.2   15.0  → tranché: 16 16 13 (moy=15.00)
    Physique-Chimie          18.2   20.0   20.0  → tranché: 16 16 16 (moy=16.00)

  ═══ Élève médian (score: 3009.850) ═══
    Français                 12.9   11.9   16.3  → tranché: 13 13 16 (moy=14.00)
    Mathématiques            14.6   20.0   16.9  → tranché: 13 16 16 (moy=15.00)
    Histoire-Géo             12.2   11.2   12.3  → tranché: 13 13 13 (moy=13.00)
    EMC                       7.5    8.5   10.4  → tranché:  8  8 13 (moy=9.67)
    LV1                       2.8    3.4    4.5  → tranché:  3  3  3 (moy=3.00)
    LV2                       2.8    0.0    2.0  → tranché:  3  3  3 (moy=3.00)
    EPS                      16.3   17.7   20.0  → tranché: 16 16 16 (moy=16.00)
    Arts Plastiques          14.2   15.7   15.8  → tranché: 13 16 16 (moy=15.00)
    Éducation Musicale       18.3   16.6   19.3  → tranché: 16 16 16 (moy=16.00)
    SVT                      13.7   20.0   18.0  → tranché: 13 16 16 (moy=15.00)
    Technologie              20.0   18.6   16.8  → tranché: 16 16 16 (moy=16.00)
    Physique-Chimie          16.7   18.5   19.1  → tranché: 16 16 16 (moy=16.00)

  ═══ Élève le plus bas (score: 2388.801) ═══
    Français                  8.5    4.8    6.0  → tranché:  8  3  8 (moy=6.33)
    Mathématiques             2.6    1.2    2.5  → tranché:  3  3  3 (moy=3.00)
    Histoire-Géo             13.3   11.1   16.1  → tranché: 13 13 16 (moy=14.00)
    EMC                      17.2   13.8   12.4  → tranché: 16 13 13 (moy=14.00)
    LV1                       7.4    9.8    7.9  → tranché:  8  8  8 (moy=8.00)
    LV2                       6.4    5.2    3.0  → tranché:  8  8  3 (moy=6.33)
    EPS                       5.0    4.5    8.3  → tranché:  3  3  8 (moy=4.67)
    Arts Plastiques          10.8   11.9    9.9  → tranché: 13 13  8 (moy=11.33)
    Éducation Musicale       14.6   14.1   13.0  → tranché: 13 13 13 (moy=13.00)
    SVT                       3.2    0.0    0.2  → tranché:  3  3  3 (moy=3.00)
    Technologie               7.4    9.6    6.3  → tranché:  8  8  8 (moy=8.00)
    Physique-Chimie           7.8    6.4    5.1  → tranché:  8  8  8 (moy=8.00)

========================================================================
  ✅ Simulation terminée avec succès
========================================================================

```
