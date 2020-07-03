# Data Science Biocéanor

**Installation de l'environnement avec conda pour exécuter les scripts**

-- Installer les packages requis   
`conda env update --name nom_env --file environment.yml`

-- Activer l'environnement
`conda activate nom_env`

-- Tester la version de python (python 3.6 requis) :     
`python --version`

-- Ajouter l'environnement à jupyter notebook :
`pip install ipykernel`
`python -m ipykernel install --user --name=nom_env`

**Organisation du git**

-- Le notebook notebook_time_series_forecasting_method contient l'exploration des données et explique la méthodologie utilisée           
-- Le script python create_model_for_each_device.py crée un modèle pour chaque capteur avec les paramètres retenus dans le notebook          
-- Le dossier models contient les modèles entrainés pour chaque capteur                   
-- Le dossier loss contient les courbes d'entrainement et de validation de la fonction objectif ou "loss function" pour chaque capteur              
-- Le dossier prediction_example contient les graphiques d'exemples de prédiction pour chaque capteur               
