# Niveau scolaire et poids des collèges et lycées

## Description
Il s'agit ici de déterminer :
* le niveau scolaire d'un collège,
* le poids d'un collège en termes d'effectif
* le niveau scolaire d'un lycée
* l'effectif des classes de seconde d'un lycée (utile pour affectation au lycée via Affelnet)

### Collège
Un indicateur du niveau scolaire d'un collège est la note moyenne à l'écrit du DNB pour les élèves de 3e de ce collège.
Un indicateur du niveau scolaire d'un collège s'obtient directement avec cette requête HTTP, qui retourne également le nombre de candidats et le taux de réussite (qui permet de déduire le nombre d'élèves admis au DNB) par exemple pour l'académie de PARIS pour la session 2024 :

```
https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-indicateurs-valeur-ajoutee-colleges/exports/json?select=uai%2Cnote_a_l_ecrit_g%2Cnb_candidats_g%2Ctaux_de_reussite_g&lang=fr&refine=academie%3A%22PARIS%22&refine=session%3A%222024%22&facet=facet(name%3D%22academie%22%2C%20disjunctive%3Dtrue)&timezone=Europe%2FBerlin
```
Il est possible de filtrer public/privé avec le paramètre 'secteur' (secteur = 'PU' ou 'PR'), par exemple pour les collèges publiques parisiens, la requête devient :

```
https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-indicateurs-valeur-ajoutee-colleges/exports/json?select=uai%2Cnote_a_l_ecrit_g%2Cnb_candidats_g%2Ctaux_de_reussite_g&lang=fr&refine=academie%3A%22PARIS%22&refine=session%3A%222024%22&facet=facet(name%3D%22academie%22%2C%20disjunctive%3Dtrue)&timezone=Europe%2FBerlin&refine=secteur%3A%22PU%22
```
### Lycée
Un indicateur du niveau scolaire d'un lycée est le taux de mentions très bien obtenues au bacalauréat
Le poids d'un lycée en terme d'effectif est l'effectif des classes de seconde.
La requête à effectuer est l'API Opendata de l'Education Nationale :

```
https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-indicateurs-de-resultat-des-lycees-gt_v2/exports/json/
```

On filtre via une clause where passée en paramètre (qu'il faudra donc URL encoder):
Par exemple ici uniquement les lycées publics généraux de l'année 2023 :

```
where=((`libelle_academie`+=+"PARIS"))+AND+((`secteur`+=+"public"))+AND+((`annee`+>=+date'2023-01-01'))+AND+((`annee`+<+date'2024-01-01'))+AND+((`presents_gnle`+>+-1))+AND+((`eff_2nde`+>+0))
```

Pour les publics et privés, il suffit d'enlever le filtre sur le secteur.
On ne s'interesse qu'aux champs suivants donc ajoutera à la query string un autre paramètre :
```
select=annee,uai,libelle_uai,secteur,nb_mentions_tb_avecf_g,nb_mentions_tb_sansf_g,eff_2nde,eff_term,presents_gnle
```

* annee : année du bac
* uai : identifiant UAI du lycée
* libelle_uai : nom du lycée
* secteur : vaut soit 'public' soit 'privé sous contrat'
* nb_mentions_tb_sansf_g : nombre de mentions très bien sans félicitations
* nb_mentions_tb_avecf_g : nombre de mentions très bien avec félicitations
* eff_2nde : effectif des classes de seconde
* eff_term : effectif des classes de terminale
* presents_gnle : nombre d'élèves ayant passé le bac

Pour obtenir de le niveau scolaire, on calculera le rapport : (nb_mentions_tb_sansf_g + nb_mentions_tb_avecf_g) / presents_gnle
