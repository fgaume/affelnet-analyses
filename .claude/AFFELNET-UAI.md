# Lien entre les UAI et les nom des collèges et lycées Paris

## Description

Ce skill permet, à partir de l'API ArcGIS du Rectorat de :

- Déterminer l'UAI d'un collège à partir de son nom
- Déterminer le nom d'un collège à partir de son UAI

## Détail

Attention la signification des champs pour ArcGIS est la suivante:

- UAI: UAI d'un lycée
- Nom: Nom du lycée
- Réseau: UAI du collège
- Nom_Tete: Nom du collège

## Déterminer l'UAI d'un collège à partir de son nom

```
method: "GET",
url: "https://services9.arcgis.com/ekT8MJFiVh8nvlV5/arcgis/rest/services/Affectation_Lyc%C3%A9es/FeatureServer/0/query",
headers: {},
params: {
    outFields: "Réseau, Nom_Tete",
    returnGeometry: "false",
    f: "pjson",
    returnDistinctValues: "true",
    where: `Nom_Tete like '%${college.nom}'`,
}
```

où ${college.nom} représente le nom du collège recherché.

## Déterminer le nom d'un collège à partir de son UAI

```
method: "GET",
url: "https://services9.arcgis.com/ekT8MJFiVh8nvlV5/arcgis/rest/services/Affectation_Lyc%C3%A9es/FeatureServer/0/query",
headers: {},
params: {
    outFields: "Nom_Tete",
    returnGeometry: "false",
    f: "pjson",
    returnDistinctValues: "true",
    where: `Réseau='${college.uai}'`,
}
```

où ${college.uai} représente l'UAI du collège recherché.
