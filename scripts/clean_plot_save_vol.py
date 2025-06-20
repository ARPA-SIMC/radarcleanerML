"""
                                     clean_plot_save_vol.py

Applica il modello di machine learning passato in input ad un volume, per la ripulitura.
Il campo di riflettività viene sovrascritto con quello ripulito in questo modo.
Viene plottato il PPI per l'elevazione passata in input o per tutte le elevazioni se viene
passata la flag -a.
Se viene passata la flag -b, vengono plottati i PPI dei volumi non ripuliti.

-------
INPUT
-------
  -d --dataora  <str> : istante di acquisizione del volume da processare in
                        formato YYYYmmddHHMM (es: 202401010000)
  -r --radar    <str> : nome del radar che ha acquisito il volume da processare
                        (es: spc)
  -i --indir    <str> : path in cui si trova il volume radar ODIM da processare.
  -o --outdir   <str> : path in cui plottare i PPI del volume ripulito.
  -m --modelname <str> : nome del modello di machine learning da usare per la
                         ripulitura, comprensivo di path.
                         (es: gat_best_rf_train.pkl)
  -l --label    <str> : nickname del modello di ML per la ripulitura.
                        Le figure dei PPI vengono salvate in outdir/<radar>/label/.
  -e --elev     <float> : angolo di elevazione di cui plotto il PPI.
  -b --base <store_true>: questa flag non prende in input argomenti.
                          Se passata, vengono plottati i PPI dei volumi non ripuliti
                          (salvati in outdir/<radar>/orig).
  -a --all      <store_true> : questa flag non prende in input argomenti.
                          Se passata, plotto il PPI a tutti gli angoli
                          di elevazione del volume radar.

-------------------
Esempio di lancio
-------------------
(prima attivare virtualenv con
    source /autofs/scratch-rad/ccardinali/TEST_ML/.repvenv/bin/activate
)

python3 clean_plot_save_vol.py -d 202401010000 -r gat
        -i /autofs/scratch-rad/ccardinali/TEST_ML/echo_classification/Predict/Volumi
        -o /autofs/scratch-rad/ccardinali/TEST_ML/echo_classification/Predict/
        -m /autofs/scratch-rad/ccardinali/TEST_ML/for_github/models/GAT/test1/gat_best_rf_train.pkl
        -l RF
        -e 0.5

In questo caso viene generato il plot del PPI all'elevazione di 0.5° per il volume di
Gattatico delle 00:00 del 2024/01/01 in /autofs/scratch-rad/ccardinali/TEST_ML/echo_classification/Predict/GAT/RF/.
Il volume da processare deve essere presente in /autofs/scratch-rad/ccardinali/TEST_ML/echo_classification/Predict/Volumi.
"""

import argparse
import logging
import sys
import numpy as np
import h5py
import joblib
from datetime import datetime, timedelta
import os
#sys.path.insert(0,"/autofs/scratch-rad/ccardinali/simcradarlib")
from simcradarlib.odim.odim_pvol import OdimHierarchyPvol
from simcradarlib import visualization
from simcradarlib.visualization.plot_ppi import plot_ppi_from_vol
from predict import pred

def get_args():
    
    parser = argparse.ArgumentParser(description='Predict del modello ML sul volume, salvataggio,\
                                     genero plot PPI e salvo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dataora', dest="dataora", required=True, type=str,
                        help="Dataora YYYYmmddHHMM, required")
    parser.add_argument('-r', '--radar', dest="radar", required=True, type=str,
                        help="radar :\n -spc\n -gat\n -compo\nrequired")
    parser.add_argument('-i', '--indir', dest="indir", required=True, type=str,
                        help="directory where volumes to predict are")
    parser.add_argument('-o', '--outdir', dest="outdir", required=True, type=str,
                        help="directory where predicted volumes and PPI-plots are saved.")
    parser.add_argument('-m', '--modelname', dest="model_name", required=False,
                        default="none", type=str, help="model filename, optional")
    parser.add_argument('-l', '--label', dest="label", required=False, type=str,
                        default="tmp",help="label model: GAT->(RF, RF_08, XGB0, XGB1)")
    parser.add_argument('-e', '--elev', dest="elev", required=False, type=float,
                        default=0.5, help="elevation angle of PPI to plot")
    parser.add_argument('-b', '--base', dest="base", required=False, type=str,
                        help="The directory where raw volumes are stored. If passed, PPI of uncleaned data are plotted.")
    parser.add_argument('-a', '--all', dest="all", required=False, 
                        action='store_true', help="if True, plot all elevations")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    logging.basicConfig(level=logging.WARNING)
    args = get_args()
    print(f"args : {args}")
    dataora=args.dataora
    time_run = datetime.strptime(dataora,"%Y%m%d%H%M")
    radar=args.radar
    indir=args.indir
    outdir=args.outdir
    f_pred = os.path.join(indir,f'{time_run.strftime("%Y-%m-%d-%H-%M-%S")}.it{radar}.PVOL.0.h5')
    label=args.label
    elev_angle=args.elev

    if radar == "gat":
        used_features_elevs=[['DBZH',
                'DBZH_SD2D',
                'DBZH_SDAz',
                'DBZH_SDRay_21',
                'DBZV',
                'RHOHV',
                'SNR',
                'VRAD',
                'WRAD',
                'ZDR',
                'ZDR_SD2D',
                'Z_VD']
                ]
    elif radar == "spc":
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
            'Z_VD']  ]

    model_name=args.model_name
    if model_name!="none":
        model_basename=os.path.splitext(os.path.basename(model_name))[0]
        rf_model = joblib.load(model_name)
        vol = OdimHierarchyPvol()
        vol.read_odim(f_pred)

        pred(rf_model, f_pred, vol, used_features_elevs, label)
        
    if args.all :
        vol = OdimHierarchyPvol()
        vol.read_odim(f_pred)
        elangles = vol.elangles
        elangles.sort()
    else:
        elangles = [elev_angle]

    for el in elangles:        

        if args.base is not None:
            logging.info("Plotto i dati non ripuliti")
            outname=f"E{el}_orig_{radar}_{dataora}.png"
            sub_outdir=f"{outdir}/{radar}/orig/"
            if not os.path.exists(sub_outdir):
                os.makedirs(sub_outdir,exist_ok=True)
            outname = os.path.join(sub_outdir,outname)
            f_orig = os.path.join(args.base,f'{time_run.strftime("%Y-%m-%d-%H-%M-%S")}.it{radar}.PVOL.0.h5')
            plot_ppi_from_vol( f_vol=f_orig, elangle=el, intitle=f"ORIG",rad_qty="TH",outname=outname,save=True )
       
        outname=f"E{el}_{label}_{radar}_{dataora}.png"
        sub_outdir=f"{outdir}/{radar}/{label}/"
        if not os.path.exists(sub_outdir):
            os.makedirs(sub_outdir,exist_ok=True)
        outname = os.path.join(sub_outdir,outname)
        plot_ppi_from_vol( f_vol=f_pred, elangle=el, intitle=label,rad_qty="DBZH", outname=outname,save=True )
