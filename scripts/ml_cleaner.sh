#!/bin/bash

"""
                         ml_cleaner.sh

Script principale che lancia il cleaning tramite machine learning (ML) di volumi
radar nel periodo specificato in input e del sistema radar passato in input.
Il file di ciascun volume viene sovrascritto, con il volume ripulito usando la
classificazione dell'eco prevista dal modello di ML passato in input.
Se viene passata la flag -c, i plot dei PPI ripuliti di ogni elevazione vengono
salvati in <outdir/RADAR/modlabel/>, definiti negli Input.

Input:
  Name      Flag   Type   Description
  ---------------------------------------------------------------------------------
  start     -s   --str : inizio del periodo da processare <YYYYmmddZHHMM>
  end       -e   --str : fine del periodo da processare <YYYYmmddZHHMM>
  radar     -e   --str : sistema radar da cui i volumi sono stati acquisiti: 'spc' o 'gat'
  indir     -i   --str : path dei volumi da ripulire
  outdir    -o   --str : path dove vengono storati eventualmente i plot
  modelname -m   --str : filename del modello ML da usare
  modlabel  -l 	 --str : label del modello ML usato. Sarà anche il nome della sub
  	    	       	 directory all'interno di outdir in cui verranno salvati i plot.
  cleanplot -c   --store_true :
  	    	 	 se passata in input la flag -c, i PPI vengono salvati in
                         outdir/RADAR/modlabel/.
Processing:
  Itera sugli istanti di acquisizione dei volumi radar in un periodo in input,
  definito tra start e end, passati in input, e per ogni istante
 - estrae il volume da <indir> e lo copia in <outdir>/<RADAR>/<modelname>/.
 - il campo di riflettività viene sovrascritto con quello ripulito post
   classificazione dell'eco del modello di ML passato in input <modelname>.
 - se la flag '-c' è passata in input, vengono anche plottati i PPI di z del
   volume ripulito usando lo script clean_plot_save_vol.py.
   I PPI vengono generati in <outdir>/<RADAR>/<modelname>/

Esempio di lancio:
./ml_cleaner.sh -s 20250127Z0225 -e 20250127Z0225 -r spc -i ../testcase/ -o . -m ../models/spc/rftest/spc_best_rf_train.pkl -l XGB -c

in questo modo si processa il volume acquisito da spc alle 2:25 UTC del 2025/1/27 con il 
modello di ML spc_best_rf_train.pkl al path passato in input sotto la flag -m; 
Grazie alla flag -c, i PPI ripuliti vengono plottati e salvati, insieme al volume ripulito nella
directory corrente. Il volume viene preso dalla directory in input ../testcase/ e copiato nella
directory di output dove viene sovrascritto con i campi ripuliti. 
"""
while getopts s:e:r:i:o:m:l:cb flag
do
    case "${flag}" in
        s) start=${OPTARG};;
        e) end=${OPTARG};;
        r) radar=${OPTARG};;
	i) indir=${OPTARG};;
	o) outdir=${OPTARG};;
	m) modelname=${OPTARG};;
	l) modlabel=${OPTARG};;
	c) cleanplot=true;;
	b) baseplot=true;;
    esac
done

WORKDIR=$PWD
PYTHON=$WORKDIR/../.venv/bin/python3
pred_rf_script=$WORKDIR/pred_save.py
plot_ppi_script=$WORKDIR/clean_plot_save_vol.py

start_t=$(date -ud "$start" '+%Y-%m-%d %H:%M')
end_t=$(date -ud "$end" '+%Y-%m-%d %H:%M')
dt=$start_t

#-----------------------------------------------------------------------
# Inizio ciclo sul periodo da start a end
#-----------------------------------------------------------------------

while ((1))
do
  fname_prefix=$(date -ud "$dt" '+%Y%m%d%H%M')
  vol_name=$(date -ud "$dt" '+%Y-%m-%d-%H-%M-%S.it')$radar.PVOL.0.h5
  
  if [ ! -f $indir/$vol_name ]; then
      echo "$vol_name non trovato in $indir, passo al prossimo dt."
  fi  

  if [ -f $indir/$vol_name ]; then

      #copio volume in outdir/RADAR/modlabel
      out_mod_dir=$outdir/$radar/$modlabel
      if [ ! -d $out_mod_dir ]; then
	  mkdir -p $out_mod_dir
      fi
      cp $indir/$vol_name $outdir/$radar/$modlabel/

      # plotto PPI del volume ripulito se flag cleanplot attiva, altrimenti ripulisco e basta
      if $cleanplot; then
	  echo "plotto PPI post cleaning"
	  if $baseplot; then
	      $PYTHON $plot_ppi_script -d $fname_prefix -r $radar -i $out_mod_dir -o $outdir -m $modelname -l $modlabel -b $indir -a
	  else
	      $PYTHON $plot_ppi_script -d $fname_prefix -r $radar -i $out_mod_dir -o $outdir -m $modelname -l $modlabel -a
	  fi
      else
	  # ripulisco con random forest senza plot
	  echo "faccio cleaning del volume con $modelname"
	  $PYTHON $pred_rf_script -d $fname_prefix -r $radar -i $out_mod_dir -m $modelname
      fi	  
  else
      echo "Non esiste "$indir/$vol_name
  fi
 
  if [[ $dt == $end_t ]]; then break; fi;
  dt=$(date -ud "5 minutes $dt" '+%Y-%m-%d %H:%M')
  
done
