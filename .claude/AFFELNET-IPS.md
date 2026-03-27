# IPS et bonus IPS des collèges de Paris

## Description

Ce skill explique comment déterminer l'IPS moyen et le bonus IPS d'un collège parisien.

## Détail

Les IPS (et le bonus Affelnet associé) de chaque année depuis 2021 sont récupérables via la requête sur un dataset huggingface publique :
https://huggingface.co/datasets/fgaume/affelnet-paris-bonus-ips-colleges/raw/main/affelnet-paris-bonus-ips-colleges.json

Sans précision sur l'année, prends toujours celui de l'année la plus récente du dataset.

Pour 2026, lorsque les IPS des établissements seront connus, le bonus IPS sera déterminé comme suit :
Les tranches d’IPS évolueraient de [0, 600, 1200] à [0, 400, 800 et 1200]. Les nouveaux seuils seraient les suivants :

- sous la moyenne nationale public/privé (105) : 1200
- sous la moyenne académique publique (117) : 800
- sous la moyenne académique public/privé (130) : 400
- 0 sinon
