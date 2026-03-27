# Capacité de déterminer les collèges et les lycées de secteurs Affelnet parisiens

## Description

Ce skill permet de :

- déterminer son collège de secteur à partir de son adresse de domicile parisien.
- retrouver les lycées de secteurs d'un collège donné.
- et à l'inverse, déterminer tous les collèges ayant tel lycée en secteur 1, 2 ou 3.
  Il n'est valable que pour l'Académie de Paris.

## Déterminer son collège de secteur à partir de son adresse de domicile parisien

On détermine le collège de secteur relatif à l'adresse fournie en entrée.
Pas besoin de préciser la ville, car c'est toujours Paris.
Cela passe en 3 étapes :

### Step 1 : adresse normalisée vers coordonnées EPSG:2154 RGF93 v1 / Lambert-93 :

Supposons par exemple que l'adresse est "10 rue Servan, 75011", alors on exécute la requête suivante, après avoir URL-encodée l'adresse :

```
https://capgeo2.paris.fr/datastore/rest/services/GEOCODAGE/Localisateur_Capgeo_Paris/GeocodeServer/findAddressCandidates?outFields=Score&outSR=%7B%22latestWkid%22%3A2154%2C%22wkid%22%3A102110%7D&f=json&SingleLine=10%20rue%20servan%2075011
```

On insere l'adresse (URL encodée) dans le paramètre d'URL 'SingleLine'.

La requete renvoie par exemple ce JSON :

```
{
    "candidates": [
        {
            "address": "10 rue servan, 75011, PARIS",
            "attributes": {
                "Score": 100
            },
            "extent": {
                "xmax": 654877.7010534635,
                "xmin": 654729.2311531635,
                "ymax": 6862445.336981761,
                "ymin": 6862224.097908684
            },
            "location": {
                "x": 654803.4675074597,
                "y": 6862334.716919544
            },
            "score": 100
        }
    ],
    "spatialReference": {
        "latestWkid": 2154,
        "wkid": 102110
    }
}
```

Les coordonnées sont dans la section "candidates/location" (on choisit le score le plus élevé si il y en a plusieurs).

### Step 2 : convertir les coordonnées en systeme web mercator

La carte scolaire fonctionne en système web mercator (EPSG:3857), il faut donc effectuer la conversion.
Par exemple en Python:

```
from pyproj import Transformer, CRS

# 1. Définir les systèmes de coordonnées source et cible via leur code EPSG
crs_lambert93 = CRS("EPSG:2154")
crs_web_mercator = CRS("EPSG:3857")

# 2. Créer un transformateur pour passer d'un système à l'autre
#    always_xy=True garantit que l'ordre des coordonnées est (x, y)
transformer = Transformer.from_crs(crs_lambert93, crs_web_mercator, always_xy=True)
```

### Step 3 : la carte scolaire avec les nouvelles coordonnées converties

On peut maintenant appeler l'API de carte scolaire officielle.
Il faut d'abord (URL-)encoder les coordonnées, par exemple :
geometry: {"x":265383.7736198207,"y":6251048.42305689}
->
geometry=%7B%22x%22%3A265383.7736198207%2C%22y%22%3A6251048.423215006%7D

Puis l'inserer dans l'url, à la fin, par exemple :

```
https://capgeo2.paris.fr/public/rest/services/DASCO/DASCO_Carte_scolaire/MapServer/0/query?f=json&resultOffset=0&resultRecordCount=2000&where=annee_scol%3D%272026-2027%27%20AND%20type_etabl%3D%27COL%27&outFields=libelle%2Ctype_etabl&returnZ=true&spatialRel=esriSpatialRelIntersects&geometryType=esriGeometryPoint&inSR=102100&outSR=102100&returnGeometry=false&geometry=%7B%22x%22%3A265383.7736198207%2C%22y%22%3A6251048.423215006%7D
```

Cette requete va retourner une structure JSON de ce type:

```
{
    "displayFieldName": "id_projet",
    "features": [
        {
            "attributes": {
                "libelle": "ALAIN FOURNIER",
                "type_etabl": "COL"
            }
        }
    ],
    "fieldAliases": {
        "libelle": "LIBELLE",
        "type_etabl": "TYPE_ETABL"
    },
    "fields": [
        {
            "alias": "LIBELLE",
            "length": 60,
            "name": "libelle",
            "type": "esriFieldTypeString"
        },
        {
            "alias": "TYPE_ETABL",
            "length": 5,
            "name": "type_etabl",
            "type": "esriFieldTypeString"
        }
    ]
}
```

La réponse se trouve dans features/attributes, on cherche l'item pour lequel "type_etabl" == "COL".
Dans l"exemple fourni, la réponse est ALAIN FOURNIER.

## Liste des lycées de secteurs à partir d'un collège donné

La source de vérité est l'API ARCGIS publique du Rectorat de Paris.

Cette liste s'obtient par une simple requete HTTP :

```
method: "GET",
url: "https://services9.arcgis.com/ekT8MJFiVh8nvlV5/arcgis/rest/services/Affectation_Lyc%C3%A9es/FeatureServer/0/query",
headers: {},
params: {
    outFields: "UAI,Nom,secteur",
    returnGeometry: "false",
    f: "pjson",
    orderByFields: "secteur",
    where: `secteur<>'Tête' and Réseau='${collegeSecteur.uai}'`,
}
```

où ${collegeSecteur.uai} représente l'identifiant (UAI) du collège recherché (par exemple : 0752523K)
L'UAI retourné est l'UAI du lycée. Le champs secteur vaut 1, 2 ou 3.
La liste sera triée par secteur croissant.

## Liste des collèges ayant un lycée donné en secteur 1

La source de vérité est l'API ARCGIS publique du Rectorat de Paris.

Cette liste s'obtient par une seimple requete HTTP :

```
method: "GET",
url: "https://services9.arcgis.com/ekT8MJFiVh8nvlV5/arcgis/rest/services/Affectation_Lyc%C3%A9es/FeatureServer/0/query",
headers: {},
params: {
    outFields: "Réseau,Nom_Tete",
    returnGeometry: "false",
    f: "pjson",
    orderByFields: "Nom_Tete",
    where: `secteur='1' and UAI='${lyceeSecteur.uai}'`,
}
```

où ${lyceeSecteur.uai} représente l'identifiant (UAI) du lycée recherché, et le champ "Réseau" contient l'UAI du collège.
