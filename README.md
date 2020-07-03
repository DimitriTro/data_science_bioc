<span style="color:red">Please scroll down bellow for an explaination of my work </span>

## Initial information

### But: 

Vous avez accès à un dataset contenant des données issues de sondes liées à la qualité de l'eau et de données météorologiques liées à la quailité de l'air. 
Le but de l'exercice est de construire un modèle de prédiction pour l'oxygène dissous. 
L'idée est d'évaluer la qualité du code sur la pipeline de mise en forme de la data jusqu'à la modélisation ainsi que l'utilisation des différents modèles que vous utiliserez. Nous n'attendons pas un résultat parfait,  le but est de surtout voir la maitrise des outils existants et la compréhension des différents modèles.
Pensez bien à citer les différents tests que vous avez fait pour parvenir au résultat que vous pusherez. 

L'idée est de prendre des slots de 2 jours (48 points) pour prédire les 6 heures suivantes (6 points). 
A partir des données bruts, mettez les en forme pour l'entrainement du modèle en time series, construisez vos train et test sets et appliquez votre modèle.

Pour la restitution du travail, il faudra forker le repo et pusher votre code. 
Si vous avez la moindre question concernant les données ou les paramètres n'hésitez pas à me contacter. 
### Données: 

Vous possédez les données issues de 5 sondes sur 2 sites différents, les données météo peuvent donc être identiques pour plusieurs device. 
Les données sont regroupées dans un fichier csv avec les heures de prélèvements et les différent paramètres. 

#### Librairies:
N'oubliez pas de mettre au mieux un moyen de packager (venv, poetry, pipenv..) ou à minima un fichier requirements pour pouvoir tester les modèles. 
Pour les librairies vous pouvez utiliser celles avec lesquelles vous êtes le plus à l'aise (Keras, TF, Pytorch). 

---

I am quite used to pipenv (much lighter version than using conda) therefore you will find a Pipfile to run my jupyter notebook
Before this I had never installed jupyter, I always used pycharm, much more convenient when you start to work with a team
However, I used jupyter notebook when I was working with Databricks which is how to work by default on Azure.
I think jupiter 

# installation
prerequisite pip and python3
>pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
>pip install pipenv
>git clone https://github.com/DimitriTro/data_science_bioc.git
>cd data_science_bioc
>pipenv install

# info
Please find below the various notebook used to analyse, train and test the data
I started with pytorch and then found a keras lstm

# Notebooks

## Checking the data shape per site
Fist Analysis of the data with this notebook
[here](Checking_the_data_value_per_site.ipynb)


## Checking the data relation
[here](Checking_the_data_relation.ipynb)


## Training using Keras - LSTM with 1 previous samples
[here](Training_-_Keras_-_LSTM_with_1_previous_sample.ipynb)
- Purpose was to get familiar with Keras LSTM
- Input : current data
- No data alignement 
- Normalisation between 0 and 1
- Data are splitted I train on one probe, tried the various probe with 80% and verify with 20%
- training with 10 Neurons, 12 training epoch with batchsize=50
- result rmse = 2.062
- Output : current Dissolved Oxygen

## Training using Keras - LSTM with 4 previous samples
[here](Training_-_Keras_-_LSTM_4_previous_samples.ipynb)
- Purpose was to add 4 samples history
- No data alignement 
- Normalisation between 0 and 1
- Data are splitted I train on one probe (B3), tried the various probe with 80% and verify with 20%
- training with 10 Neurons, 6 training epoch with batchsize=50
- result rmse = 4.753
- Output : current Dissolved Oxygen

<B> Since only one device was used to train and therefore the LSTM would work only for that device I decided to average the dataset per site to have an LSTM per site</B>

## Training - Keras - LSTM 4s - mean per site
[here](Training_-_Keras_-_LSTM_4s_-_mean_per_site.ipynb)
- input : 4 samples history 
- <B>Data alignement : mean device site A</B>
- Normalisation between 0 and 1 
- Data are splitted I train on one site, tried the various probe with 80% and verify with 20% 
- training with 4 Neurons, 12 training epoch with batchsize=50 
- result rmse = 4.577 
- Output : current Dissolved Oxygen
    
## Training - Keras - LSTM 4s - mean interpolated per site B
[here](Training_-_Keras_-_LSTM_-_site_B_interpolated.ipynb)
- input : 4 samples history 
- Data interpolated : mean site B: Mean per site</B>
- Normalisation between 0 and 1 
- Data are splitted I train on one site, tried the various probe with 80% and verify with 20% 
- training with 4 Neurons, 12 training epoch with batchsize=50 
- result rmse = 4.577 
- Output : current Dissolved Oxygen

## Training - Keras - LSTM - site A and site B
[here](Training_-_Keras_-_LSTM_-_site_A_site_B.ipynb)
- input : 24 samples history
- data from sites are mean over all device and sensor
- data from site B is interpolated (added missing hours)
- calculation is done to find the floor for mean sites Dissolved Oxygen and the floor is set to the middle of both sites
- data from site B is shifted so it is starting when data from site A is finishing
- Normalisation between 0 and 1 
- training with 2 Neurons, 5 training epoch with batchsize=100 
- Test RMSE: 2.709
- Output : current Dissolved Oxygen
