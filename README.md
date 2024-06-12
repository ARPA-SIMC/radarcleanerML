# radarcleanerML

## Introduzione

Questo progetto riguarda lo sviluppo di un modello di machine learning per la ripulitura dei volumi radar dei sistemi regionali, nell'ambito del task 4.1 della convenzione Modmet tra Arpae Emilia-Romagna SIMC e il DPC, per il biennio 2024-2025.

## Descrizione

Il repository è organizzato nel seguente modo:

* `setup_env.sh`: script per il setup (vedi sezione "Setup").
* `requirements.txt`: contiene l'elenco dei moduli Python necessari.
* `scripts`: contiene gli script Python e il dataloader.

Inoltre, dopo il setup, sono generate le seguenti directory:

* `data`: directory dei dati e dei dataset dei due radar, San Pietro Capofiume
  (SPC) e Gattatico (GAT).
  * `SPC`/
    * `Settori` : directory dei settori di volumi radar associati univocamente ad una classe target di eco
    * `Volumi` : directory dei volumi radar
    * `Dataset`/
      * `SPCsamples.npy` : dataset
      * `SPCfeatures.npy` : matrice delle features
      * `SPCtarget.npy` : vettore dei target
      * `SPC_xtrain.npy`: matrice delle features di training
      * `SPC_xtest.npy` : matrice delle features di test
      * `SPC_ytrain.npy` : vettore dei target di training
      * `SPC_ytest.npy` : vettore dei target di test
  * `GAT`/
    * `Settori` : directory dei settori di volumi radar associati univocamente ad una classe target di eco
    * `Volumi` : directory dei volumi radar
    * `Dataset`/
      * `GATsamples.npy` : dataset
      * `GATfeatures.npy` : matrice delle features
      * `GATtarget.npy` : vettore dei target
      * `GAT_xtrain.npy`: matrice delle features di training
      * `GAT_xtest.npy` : matrice delle features di test
      * `GAT_ytrain.npy` : vettore dei target di training
      * `GAT_ytest.npy` : vettore dei target di test
* `.venv`: virtual environment Python contenente le dipendenze necessarie
  agli script.

## Setup

Per preparare l'ambiente di lavoro e generare i dataset è sufficiente lanciare lo script `setup_env.sh`:

```
$ ./setup_env.sh
```

Questo script crea l'ambiente virtuale Python e scarica i dati dal server opendata di Arpae-SIMC.
Poi genera, per ciascun radar, il dataset, la matrice delle features e il vettore dei target e opera lo split in training set e test set.

## Licenza

`radarcleanerML` è rilasciato sotto licenza GPLv3. I dati scaricati dal server opendata e salvati nella
directory `data` sono rilasciati sotto licenza CC BY 4.0.
