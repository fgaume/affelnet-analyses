# Calcul du score scolaire Affelnet

## Description
Il s'agit ici de calculer le score scolaire Affelnet à partir des moyennes trimestrielles ou semestrielles d'un élève.

## Principe
Les matières à prendre en compte sont uniquement celles-ci: Français, Mathématiques, Histoire-Géo, LV1, LV2, EPS, Arts Plastiques, Education Musicale, SVT, Technologie, Physique-Chimie et EMC
Si la note d'EMC est absente, tu la remplaceras par la moyenne de Histoire-Géo.
Pour chaque matière la note à prendre en compte est celle de l'élève, pas de la classe, donc choisis la bonne colonne.
Parfois les 3 trimestres sont dans le même tableau, fais bien attention.

## Étape 1: moyenne des matières
On part des 3 moyennes trimestrielles (ou des 2 pour les collèges qui fonctionnent en semestres) de chacune des matières et on effectue la moyenne annuelle, pour chaque matière.

## Étape 2 : regroupement des matières en “champs disciplinaires”
On regroupe  certaines matières, afin de ne se retrouver qu’avec 7 “champs disciplinaires”. Les Mathématiques, Français et EPS restent quant à eux séparés et constituent chacun un champ disciplinaire à part entière. Concernant les matières restantes, on effectue les regroupements suivants:
ARTS = moyenne(Arts Plastiques, Education Musicale)
HISTOIRE-GEO = moyenne(Histoire-Géo, EMC)
LANGUES VIVANTES = moyenne(LV1, LV2)
SCIENCES-TECHNO-DP = moyenne(Physique-Chimie, SVT, Technologie)

## Étape 3: harmonisation des champs disciplinaires sur l’Académie
On va maintenant harmoniser la note de chaque champ disciplinaire sur l’Académie.
Affelnet va en premier lieu calculer la moyenne µ et l’écart-type σ des notes intermédiaires issues de l’étape 2 de chacun des 7 champs disciplinaires pour tous les élèves de l’académie.
Si on appelle T la variable représentant la note intermédiaire (de l'étape 2 donc) d’un élève relative à un champ disciplinaire donné, Affelnet va réaliser la « cuisine » statistique suivante sur cette variable, il va:
- Centrer T pour avoir une moyenne nulle en lui soustrayant la moyenne académique µ
- Réduire la variable centrée obtenue en divisant par l’écart-type académique σ afin que tous les champs disciplinaires aient la même dispersion (à savoir un écart-type de 1)
- Translater enfin cette variable centrée réduite pour avoir une moyenne de 10 (en ajoutant simplement 10)
- Multiplier par 10
Mise en équation, la nouvelle note harmonisée H s’écrit donc comme suit : H = 10 x [10 + (T-µ)/σ]
Les moyennes et écart-types académiques à prendre en compte sont à récupérer via la requête :
https://datasets-server.huggingface.co/rows?dataset=fgaume%2Faffelnet-paris-statistiques-champs-disciplinaires&config=default&split=train&offset=0&length=100
Il faut filtrer les rows avec l'année la plus récente.

## Étape 4 : pondération des champs disciplinaires
Fort de nos 7 notes “harmonisées” autour de 100, on applique pour terminer des pondérations différentes selon le champ disciplinaire (5 pour les Mathématiques et le Français, 4 pour les autres).
Le score total pour le bilan périodique est la somme de ces 7 valeurs pondérées.

## Étape 5 : multiplication par le coefficent de pondération scolaire
On multiplie le total précédent de l'étape 4 par le coefficient de pondération scolaire, qui vaut 2.
