#!/bin/bash

#####################################################################
#
#                         create_dset.sh
#
#  Script che lancia la generazione del dataset, della matrice delle
#  features e del vettore dei target, lo split in training e test set
#  per ciascun radar, operata dallo script python create_dset.py,
#  in scripts/.
#  I settori dei volumi radar da cui sono estratti i bin del dataset
#  sono settati al path relativo, all'interno del repo locale, di
#  "data/<RADAR>/Settori".
#  Analogamente i volumi radar da cui sono estratti i bin sono
#  settati al path relativo "data/<RADAR>/Volumi"
#  Infine il path dove vengono salvati i dataset, la matrice delle
#  features, vettore dei target, training set, test set, è settato
#  a "data/<RADAR>/Dataset".
#  Per San Pietro Capofiume (SPC) il dataset è generato usando come
#  features le grandezze radar alla prima e alla quarta elevazione
#  (corrispondenti agli angoli 0.5° e 3.2°).
#  Per il radar di Gattatico (GAT) il dataset è generato usando
#  come features le grandezze radar alla prima elevazione (0.5°).
#
####################################################################

LOCAL_REPO=$(git rev-parse --show-toplevel)
PYTHON=$LOCAL_REPO/.venv/bin/python3

echo "Generazione dataset per San Pietro Capofiume.."
$PYTHON $LOCAL_REPO/scripts/create_dset.py -r spc -s $LOCAL_REPO/data/SPC/Settori -v $LOCAL_REPO/data/SPC/Volumi -o $LOCAL_REPO/data/SPC/Dataset -e 0.5 3.2

echo "Generazione dataset per Gattatico.."
$PYTHON $LOCAL_REPO/scripts/create_dset.py -r gat -s $LOCAL_REPO/data/GAT/Settori -v $LOCAL_REPO/data/GAT/Volumi -o $LOCAL_REPO/data/GAT/Dataset -e 0.5

