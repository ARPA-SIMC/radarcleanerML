from utils_dset import BinDSet
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)

"""
                                create_dset.py

Script che genera o legge (se già generati) il dataset di bins, la matrice delle features
ed il vettore dei target per il radar (San Pietro Capofiume o Gattatico), passato da
riga di comando alla flag -r.
Viene infine generato il set di training e test.

PARAMETRI da riga di comando:

  FLAG    FLAG_ESTESA       DEST                         DESCRIZIONE
  ______________________________________________________________________________________
  -r      --radar           radar                        radar per cui genero il dataset:
                                                         'spc' <--> San Pietro Capofiume
                                                         'gat' <--> Gattatico

  -s      --sectors_path    sectors_path                 path dei settori di volumi radar
                                                         da cui sono estratti i bin.

  -v      --volumes_path    vol_path                     path dei volumi radar da cui sono
                                                         estratti i bin.

  -o      --output_path     out_path                     path in cui sono salvati il
                                                         dataset, la matrice delle features
                                                         e il vettore dei target, nonchè
                                                         il set di training e test.

  -e      --elevs           sel_elevs                    angoli di elevazione a cui sono
                                                         estratte le grandezze radar usate
                                                         come features.
"""

def get_args():
    
    parser = argparse.ArgumentParser(description='Realization BinDset for more elevations',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-r', '--radar', dest="radar", required=True,
                        help="radar 'spc' or 'gat', required")
    parser.add_argument('-s', '--sectors_path', dest="sectors_path", required=False,
                        help="path where sectors txtfiles are stored")
    parser.add_argument('-v', '--volumes_path', dest="vol_path", required=False,
                        help="path where volumes odimfiles are stored")
    parser.add_argument('-o', '--output_path', dest="out_path", required=False,
                        help="path where to save features,target,dataset preprocessed")
    parser.add_argument('-e', '--elevs', nargs='+', dest="sel_elevs",
                        required=False, default=[0.5], type=float,
                        help="Elevation angles where to extract radar variables values.\n \
                        Default=0.5, the first one.")
    
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = get_args()
    print()
    radar=args.radar
    data_path=args.sectors_path
    volume_path=args.vol_path
    save_path=args.out_path
    selected_elevs=args.sel_elevs

    used_features_elevs=[['DBZH',
            'DBZH_SD2D',
            'DBZH_SDAz',
            'DBZH_SDRay_21',
            'DBZV',
            'RHOHV',
            'SNR',
            'SQI',
            'VRAD',
            'WRAD',
            'ZDR',
            'ZDR_SD2D',
            'Z_VD'],
            ['DBZH',
            'DBZH_SD2D',
            'DBZH_SDAz',
            'DBZH_SDRay_21',
            'DBZV',
            'RHOHV',
            'SNR',
            'SQI',
            'VRAD',
            'WRAD',
            'ZDR',
            'ZDR_SD2D',
            'Z_VD']
             ]
    used_features_elevs = [used_features_elevs[i] for i in range(len(selected_elevs))]
    if radar=='gat':
        for i in range(len(selected_elevs)):
            used_features_elevs[i].remove('SQI')
        
    logging.debug("preparing dataset..")
    Bset = BinDSet(radar=radar,
                   selected_features_elevs=used_features_elevs,
                   selected_elevs=selected_elevs,
                   store_dtype=np.float64,
                   save_path=save_path,
                   data_path=data_path,
                   volume_path=volume_path,
                   verbose=True
                   )
    logging.debug("train-test split..")
    Bset.split_train_test_xy()

    logging.debug("End data preparation")
