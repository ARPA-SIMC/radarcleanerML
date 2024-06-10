import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Optional,Union
from tqdm import tqdm
from pandas.errors import EmptyDataError
import h5py
import logging
import copy
from sklearn.model_selection import train_test_split

module_logger = logging.getLogger(__name__)

"""
                       utils_dset.py

Questo modulo contiene utilities per implementare il dataset per allenare un
modello machine learning capace di classificare tra echi radar associati a
meteo, clutter, interferenze multiple, interferenze medie, interferenze deboli.
Ogni sample del dataset è un oggetto 'bin', caratterizzato da features date da
alcune grandezze radar e radar polarimetriche, storate nei volumi radar.
Il valore delle features per un dato bin corrisponde al valore di quelle
grandezze in quel bin.
I bin usati sono estratti da settori di volumi pre-classificati dallo
sviluppatore (manualmente).
La label per ogni bin è quella del settore radar di appartenenza e può essere:

1- classe eco : meteo
2- classe eco : clutter
3- classe eco : interferenze multiple
4- classe eco : interferenze medie
5- classe eco : interferenze deboli
"""

features_names_all = [
                'azimuth',
                'range',
                'DBZH',
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

class NGO:

    """
    Classe di metadati: da usare per implementare
    l'insieme di metadati relativi a nodata, gain
    e offset di una grandezza radar.

    L'istanza ha attributi:
    nodata  --Optional[float] (default=None) : valore assegnato alla grandezza se il
                                               segnale non è rilevato e il dato non è
                                               disponibile.
    gain    --Optional[float] (default=None) : parametro moltiplicativo per trasformare
                                               i dati di una grandezza da uint16 (raw_data)
                                               a float in unità fisiche (data), nella 
                                               relazione:
                                               data = raw_data*gain + offset
    offset  --Optional[float] (default=None) : parametro additivo per trasformare
                                               i dati di una grandezza da uint16 (raw_data)
                                               a float in unità fisiche (data), nella 
                                               relazione:
                                               data = raw_data*gain + offset

    esempio: per la riflettività (DBZH) nei volumi radar di SPC :
             nodata=65535
             gain=0.0019455531146377325
             offset=-31.50194549560547
    """

    def __init__(self,
                 nodata : Optional[float] = None,
                 gain : Optional[float] = None,
                 offset : Optional[float] = None
                 ):
        self.nodata = nodata
        self.gain = gain
        self.offset = offset
    
class CustomBin:

    """
    Un'istanza di questa classe implementa un bin del volume radar.
    
    L'istanza è inizializzata, passando al costruttore i parametri:
    time    --datetime : oggetto datetime.datetime che rappresenta
                         la dataora di acquisizione del volume radar
                         da cui è estratto il bin.
    Range   --float    : range [m] del bin.
    radar_var_elevs   --Optional[dict] (default=None):
                         dizionario aventi chiavi le grandezze radar
                         acquisite ad un'elevazione e valori i valori
                         di quelle grandezze storati per il bin.
    elangles    --Optional[list] (default=None) : 
                         lista degli angoli di elevazione a cui
                         vengono estratte le grandezze radar usate
                         come features.
                         (es per SPC: [0.5] se uso come features solo le
                          grandezze radar alla prima elevazione, [0.5,3.2]
                          se uso anche le grandezze radar alla quarta
                          elevazione).

    Attributi dell'istanza:
    time    --datetime : (passato in input al costruttore) dataora di acquisizione
                          del volume radar.
    Range   --float    : (passato in input al costruttore) range [m].
    <radar_var_elevs.keys()[i]>   --float : 
                         valore di ciascuna delle grandezze radar usate come features
                         (valore della chiave i-esima di radar_var_elevs corrispondente alla
                         grandezza radar i-esima usata come feature).
    elangles    --Optional[list] (default=None) : (passato in input al costruttore)
                         lista degli angoli di elevazione a cui
                         vengono estratte le grandezze radar usate
                         come features.

    Metodi di istanza:
    
    hasattrbin : prende in input il nome dell'attributo (es "DBZH") e restituisce
                 True se l'istanza ha quell'attributo altrimenti False.
    """
    
    def __init__(
        self,
        time : datetime,
        Range : np.float64,
        radar_var_elevs : Optional[dict] = None,
        elangles : Optional[list] = None
    ):
        self.time=time
        self.Range=Range
        if radar_var_elevs is not None:
            if radar_var_elevs.__len__()>0:
                for i, elev in enumerate(elangles):
                    for varname in radar_var_elevs.keys():
                        if self.hasattrbin(varname):
                            continue
                        else:
                            setattr(self, varname, radar_var_elevs[varname])
        self.elangles = elangles
                        
    def hasattrbin(self, attr : str) -> bool:
        """
        verifica l'esistenza di un attributo per l'istanza.
        
        INPUT:
        attr --str : nome dell'attributo corrispondente ad una grandezza radar
                     usata come feature (es: DBZH)
        OUTPUT:
        --bool : True se l'istanza ha l'attributo attr, False altrimenti.
        """
            if attr in self.__dict__.keys():
                return True
            return False

class BinDSet:

    """
    Dataloader del progetto:
    Genera, salva e carica
    - il dataset di oggetti bins, come istanze di CustomBin (<RADAR>samples.npy)
    - la matrice delle features  (<RADAR>features.npy)
    - il vettore dei target (<RADAR>target.npy )
    iterando sui settori dei volumi nei path data_path e volume_path
    rispettivamente (passati in input al costruttore di BinDSet).

    -------------------------------------------------------------------------------
    Parametri in input:
    -------------------------------------------------------------------------------
    radar  --str           : nome del radar 
                             ('spc':San Pietro Capofiume, 'gat':Gattatico)
    selected_features_elevs  --list :
                             lista di lunghezza pari al numero di elevazioni a cui
                             sono estratte le grandezze radar usate come features.
                             Ogni membro i-esimo è la lista delle grandezze usate
                             come features all'elevazione i-esima.
                             Es: se le features sono "DBZH" e "VRAD" alla prima e
                                 quarta elevazione, allora:
                                 selected_features_elevs = [
                                                             ["DBZH","VRAD"],
                                                             ["DBZH","VRAD"]
                                                           ]
    selected_elevs  --list : lista degli angoli di elevazione a cui vengono
                             estratte le grandezze radar usate come features.
    store_dtype  --dtype   : dtype con cui salvare i dati del dataset, della
                             matrice delle features e dei target.
    save_path    --str     : path dove salvare (o caricare, se già salvato)
                             il dataset, la matrice delle features e il vettore
                             dei target.
    data_path    --str     : path dei settori da cui generare i samples come
                             istanze di CustomBin.
    volume_path  --str     : path dei volumi radar da cui sono stati estratti
                             i settori.
    verbose  --bool (default=False):
                             se True il livello dei log stampati viene settato
                             a DEBUG.
    -------------------------------------------------------------------------------
    Attributi dell'istanza BinDSet:
    -------------------------------------------------------------------------------
    radar  --str           : nome del radar 
                             ('spc':San Pietro Capofiume, 'gat':Gattatico)
    data_path    --str     : path dei settori da cui generare i samples come
                             istanze di CustomBin.
    volume_path  --str     : path dei volumi radar da cui sono stati estratti
                             i settori.
    save_path    --str     : path dove salvare (o caricare, se già salvato)
                             il dataset, la matrice delle features e il vettore
                             dei target.
    sel_elangles  --list   : lista degli angoli di elevazione a cui vengono
                             estratte le grandezze radar usate come features
                             (data dal parametro in input selected_elevs).
    az_scale_factor  --float = 100 : 
                             se store_dtype=uint allora moltiplica gli azimuth per 100
                             per non perdere l'informazione.
    store_dtype  --dtype   : dtype con cui salvare i dati del dataset, della
                             matrice delle features e dei target.
    
    selected_features_elevs  --list :
                             lista di lunghezza pari al numero di elevazioni a cui
                             sono estratte le grandezze radar usate come features.
                             Ogni membro i-esimo è la lista delle grandezze usate
                             come features all'elevazione i-esima.
                             E' la lista passata come parametro in input all'argomento
                             selected_features_elevs.
    required_features_elevs  --list :
                             lista data dall'unione di ['azimuth','Range'] e la lista
                             delle features che sono grandezze radar ad ogni elevazione.
                             Per ogni elevazione, rappresenta gli attributi di un sample
                             del dataset creato come istanza di CustomBin.
                             In questo modo, per ogni sample (bin) abbiamo l'informazione
                             non solo sulle grandezze radar che sono le features, ma
                             anche sull'azimuth, il Range (,dataora, angoli di elevazione).
    ngo_dict_elevs  --dict[elev]['key':NGO] :
                             dizionario avente come chiavi gli angoli di elevazione a cui sono
                             estratte le grandezze radar usate come features e come values
                             un dizionario. Ciascun dizionario per ciascuna elevazione ha come
                             chiavi i nomi delle grandezze radar usate come features e come
                             values istanze della classe NGO, le quali sono oggetti con attributi
                             dati da nodata, gain e offset per la grandezza radar.
                             Rappresenta il dizionario con le informazioni di nodata, gain e offset
                             per ogni grandezza radar usata come feature, per ogni elevazione.
    
    features  --np.array   : matrice delle features, salvata come oggetto numpy.
                             Il numero di colonne è pari al numero di grandezze radar usate come features,
                             il numero di righe è pari al numero di samples del dataset.
    target  --np.array     : vettore dei labels (target), salvato come oggetto numpy.
                             La lunghezza è pari al numero di samples del dataset e ogni elemento è
                             l'intero rappresentativo della classe
                             ( 0=meteo,1=clutter,2=int.multiple,3=int.medie,4=int.deboli).
    samples  --list[CustomBin] : lista avente elementi dati da istanze di CustomBin, le quali
                                 rappresentano gli oggetti bin, aventi attributi di Range, azimuth
                                 e del valore delle grandezze radar usate come features nel dato
                                 bin del volume radar. Ogni elemento ha anche l'attributo 'time' della
                                 dataora del volume acquisito da cui è stato estratto e l'attributo
                                 'elangles', cioè la lista degli angoli di elevazione a cui sono state
                                 estratte le features.
    features_names  --list : lista dei nomi degli attributi di ciascun sample come istanza di CustomBin
                             (quindi contiene anche 'time', 'azimuth','Range' oltre le features).
    N_features  --int      : lunghezza della lista features_names. Rappresenta il numero di attributi
                             cioè informazioni disponibili per ogni sample.                            
    -------------------------------------------------------------------------------
    Metodi dell'istanza BinDSet:
    -------------------------------------------------------------------------------
    add_required_features:
    per ogni elevazione aggiunge alla lista delle grandezze radar usate come features
    la lista di attributi 'time','azimuth','Range' e assegna la nuova lista all'
    attributo d'istanza required_features_elevs.

    set_attrs_elevs:
    assegna all'attributo di istanza ngo_dict_elevs il dizionario avente chiavi
    gli angoli di elevazione a cui sono estratte le grandezze radar usate come features
    e come values un dizionario. Ciascun dizionario per ciascuna elevazione ha come
    chiavi i nomi delle grandezze radar usate come features e come values istanze della
    classe NGO, le quali sono oggetti con attributi dati da nodata, gain e offset per la
    grandezza radar.
    Ad esempio se uso come features le grandezze radar solo alla prima elevazione:
    --> istanza.ngo_dict_elevs = { "0.5" : {
                                            "DBZH" : NGO(nodata=65535,
                                                         gain=0.0019455531146377325,
                                                         offset=-31.50194549560547),
                                            ...<altre features analogamente>
                                           }
                                 }

    fill_long_ranges:
    metodo che crea un dataframe di 1 riga e colonne pari alle grandezze radar usate come
    features ad una data elevazione. I valori sono i nodata per la grandezza radar corrispondente.
    Questo metodo viene invocato durante la generazione del dataset qualora siano usate come features
    anche le grandezze ad un'elevazione superiore alla prima, a cui il dato non è disponibile
    per bin a grandi range, perchè fuori dalla portata radar a quell'elevazione.
    Per avere il dataset completo, le features corrispondenti alle grandezze radar ad
    un'elevazione grande in cui il range del bin è fuori portata radar, si riempe il valore
    delle features a tale elevazione con nodata di quella feature.
    Esempio:
    per un bin con 200km di range (1elev), il vettore delle features sarà tipo
    features[bin] = np.array([ <valore feature1 1elev>,
                               ...,
                               <valore featureN 1elev>,
                               <nodata feature1 4elev>,
                                ...,
                               <nodata featureN 4elev>,
    ])
    dove nodata=nodata*gain+offset se store_dtype non è di tipo unsigned int.

    select4elev:
    raccorda il dataframe delle features alla prima elevazione con quello delle
    features all'elevazione superiore nel caso siano usate features a più elevazioni
    e non solo alla prima.

    parse_data:
    metodo che itera sui settori in data_path, estratti dai volumi in volume_path,
    e per ogni bin del settore, crea un'istanza CustomBin con attributi sulla dataora
    di acquisizione del volume radar di appartenenza, Range, azimuth ad ogni elevazione
    a cui sono acquisite le grandezze radar usate come features e le grandezze radar
    usate come features a ogni elevazione considerata.
    (NB: DBZH alla prima elevazione è una feature diversa da DBZH ad un'altra elevazione)
    Il metodo crea una lista 'samples' di questi oggetti bin come istanze di CustomBin.
    Dalla lista samples viene estratta la matrice delle features, 'features'.
    Il vettore dei target, 'target', viene estratto iterando sui settori.
    Il metodo ritorna in output features, target, samples.

    save_dset:
    Metodo che salva la matrice delle features, il vettore dei target e il dataset
    (ottenuti dal metodo parse_data) come oggetti numpy in save_path, se non sono
    già salvati in save_path.

    load_dset:
    Metodo che carica la matrice delle features, il vettore dei target e il dataset
    salvati come oggetti numpy in save_path.

    load_train_xy:
    Metodo che carica la matrice delle features e il vettore di target del subset di
    training, salvati in save_path con il metodo split_train_test_xy.
    Se non salvati in save_path, viene invocato split_train_test_xy e vengono poi
    caricati.

    load_test_xy:
    Metodo che carica la matrice delle features e il vettore di target del subset di
    test, salvati in save_path con il metodo split_train_test_xy .
    Se non salvati in save_path, viene invocato split_train_test_xy e vengono poi
    caricati.

    split_train_test_xy:
    Metodo che fa lo split della matrice delle features e del vettore dei target in
    training e test, riservando al test una percentuale data dal parametro in input
    test_size (default=0.25 come in scikit-learn).
    
    """

    global features_names_all
    
    def __init__(self,
                 radar : str,
                 selected_features_elevs : list,
                 selected_elevs : list,
                 store_dtype : type,
                 save_path : str,
                 data_path : str, # path dei settori
                 volume_path : str,
                 verbose : bool = False,
                ):
        
        if verbose:
            module_logger.setLevel(logging.DEBUG)
        
        self.data_path = data_path
        self.save_path = save_path
        self.volume_path = volume_path
        self.radar=radar
        self.sel_elangles = selected_elevs
        self.az_scale_factor=100 # moltiplico gli azimuth per 100 per non perdere informazione nel passaggio a uint
        self.store_dtype=store_dtype
        
        self.selected_features_elevs = copy.deepcopy(selected_features_elevs)
        self.required_features_elevs = self.add_required_features()
        
        self.ngo_dict_elevs = self.set_attrs_elevs()
        self.fill_long_ranges()

        fname_features = os.path.join(self.save_path,f'{self.radar.upper()}features.npy')
        fname_target = os.path.join(self.save_path,f'{self.radar.upper()}target.npy')
        fname_samples = os.path.join(self.save_path,f'{self.radar.upper()}samples.npy')
        if( os.path.exists(fname_features)&os.path.exists(fname_target)&os.path.exists(fname_samples) ):
            self.features, self.target, self.samples = self.load_dset()
        else:
            if self.data_path is not None and self.volume_path is not None:
                self.features, self.target, self.samples = self.parse_data()
                self.save_dset()
            else:
                raise Exception("Specificare volume_path e data_path per generare il dataset!")
                exit()
        try:
            self.features_names=list(self.samples[0].__dict__.keys())[:-1]
        except:
            logging.exception(f"self.samples[0]={self.samples[0]}")
            self.features_names=[]

        try:
            self.Nfeatures = self.features.shape[1]
        except:
            logging.exception(f"shape features={self.features.shape}")
            self.Nfeatures = 0
     
    def add_required_features(self):

        """
        per ogni elevazione aggiunge alla lista delle grandezze radar usate come features
        la lista di attributi 'azimuth','Range' e assegna la nuova lista all'
        attributo d'istanza required_features_elevs.

        INPUT:
         -
        OUTPUT:
        required_features_elevs  --list[list] : lista di liste, ognuna data dall'unione di
                                                ['azimuth','Range'] e le features per un'elevazione.
        """
        required_bin_fields=['azimuth','Range'] 
        required_features_elevs=copy.deepcopy(self.selected_features_elevs)
        for req in required_bin_fields:
            for i in range(len(self.sel_elangles)):
                if req not in required_features_elevs[i]:
                    required_features_elevs[i].append(req)
                    
        return required_features_elevs

    def set_attrs_elevs(self):

        """
        Crea un dizionario contenente a sua volta un dizionario per ogni elevazione.
        Il dizionario per ciascuna elevazione ha come chiavi i nomi delle grandezze radar
        usate come features e come values istanze della classe NGO, le quali sono oggetti
        con attributi dati da nodata, gain e offset per la grandezza radar.
        Ad esempio se uso come features le grandezze radar solo alla prima elevazione:
        --> istanza.ngo_dict_elevs = { "0.5" : {
                                            "DBZH" : NGO(nodata=65535,
                                                         gain=0.0019455531146377325,
                                                         offset=-31.50194549560547),
                                            ...<altre features analogamente>
                                                }
                                      }

        INPUT:
         -
        OUTPUT:
        attrs_dict  --dict['elevation_angle':dict['feature':NGO]] :
                                   dizionario avente chiavi gli angoli di elevazione a
                                   cui vengono prese le grandezze radar come features e
                                   come values un dizionario. Questo dizionario secondario
                                   contiene l'istanza di NGO con attributi di nodata,gain
                                   e offset la grandezza radar corrispondente alla chiave.
        """

        f = os.listdir(self.volume_path)[0]
        h = h5py.File(os.path.join(self.volume_path,f),'r')
        nodata_dict = {}
        attrs_dict = {}
        hangles = [h[f'{d}/where'].attrs['elangle'] for d in h.keys() if 'dataset' in d]
        for el in self.sel_elangles:
            attrs_dict[el] = {}
            ih = hangles.index(el)+1
            for ds in h[f'dataset{ih}'].keys():
                if 'data' in ds:
                    quantity = h[f'dataset{ih}/{ds}/what'].attrs['quantity'].decode('utf-8')
                    attrs_dict[el][quantity] = NGO(nodata=h[f'dataset{ih}/{ds}/what'].attrs['nodata'],
                                                   gain=h[f'dataset{ih}/{ds}/what'].attrs['gain'],
                                                   offset= h[f'dataset{ih}/{ds}/what'].attrs['offset']
                                                   )

        h.close()
        return attrs_dict

    def fill_long_ranges( self):
        
        """
        Metodo che crea un dataframe di 1 riga e colonne pari alle required_features_elevs per
        una data elevazione. I valori sono i nodata per la grandezza radar corrispondente.
        Il dataframe così ottenuto per ciascuna elevazione usata è appeso all'attributo
        di istanza ( di tipo lista) fill_dfs.
        Questo metodo viene invocato durante la generazione del dataset qualora siano usate come
        features anche le grandezze ad un'elevazione superiore alla prima, a cui il dato non è
        disponibile per bin a grandi range, perchè fuori dalla portata radar a quell'elevazione.
        Per avere il dataset completo, le features corrispondenti alle grandezze radar ad
        un'elevazione grande in cui il range del bin è fuori portata radar, si riempe il valore
        delle features a tale elevazione con nodata di quella feature.
        Esempio:
        per un bin con 200km di range (1elev), il vettore delle features sarà tipo
        features[bin] = np.array([ <valore feature1 1elev>,
                               ...,
                               <valore featureN 1elev>,
                               <nodata feature1 4elev>,
                                ...,
                               <nodata featureN 4elev>,
                                 ])
        dove nodata=nodata*gain+offset se store_dtype non è di tipo unsigned int.

        INPUT:
         -
        OUTPUT:
         -
        """
        self.fill_dfs = []
        for i, elev in enumerate(self.sel_elangles):
            nodatalist = []
            for fe in self.selected_features_elevs[i]:
                curnodata=self.ngo_dict_elevs[elev][fe].nodata*self.ngo_dict_elevs[elev][fe].gain + self.ngo_dict_elevs[elev][fe].offset
                nodatalist.append(curnodata)
            nodatalist.extend([-9999.,-9999.])
            d = {}
            for name, val in zip(self.required_features_elevs[i], nodatalist):
                d[name]=val
            df = pd.DataFrame(
                data=d,
                index=[0],
                columns=self.required_features_elevs[i]
            )
            self.fill_dfs.append(df)

        return

    def select4elev( self, df1elev : pd.DataFrame , df2elev: pd.DataFrame)-> pd.DataFrame:

        """
        Raccorda il dataframe delle features alla prima elevazione con quello delle
        features all'elevazione superiore nel caso siano usate features a più elevazioni
        e non solo alla prima, in modo da associare i valori corretti alle features di un bin
        per un dato settore quando l'estrazione su elevazioni diverse ha associato allo stesso bin
        più records perchè cade a cavallo tra più azimuth.

        INPUT:
        df1elev  --pd.DataFrame : dataframe delle required_features_elevs alla prima elevazione
        df2elev  --pd.DataFrame : dataframe delle required_features_elev all'elevazione superiore

        OUTPUT:
        df2      --pd.DataFrame : estensione del dataframe più corto tra quelli in input, dopo il
                                  matching con il dataframe originario più lungo.
        """
         
        if len(df1elev)>len(df2elev):
            print("df_tot>df")
            df0 = df1elev.copy()
            df1 = df2elev.copy()
            retain_keys = list(df1elev.keys())
        elif len(df1elev)<len(df2elev):
            #print("df_tot<df")
            #print(f"df1elev.keys={df1elev.keys()}")
            #print(f"df2elev.keys={df2elev.keys()}")
            df1 = df1elev.copy()
            df0 = df2elev.copy()
            retain_keys = list(df2elev.keys())
        #print(f"retain keys() = {retain_keys}")
        df0.rename( {list(df0.keys())[i]: list(df0.keys())[i][:-2] for i in range(len(df0.keys())) if df0.keys()[i]!='Range' }, axis='columns', inplace=True)
        df1.rename( {list(df1.keys())[i]: list(df1.keys())[i][:-2] for i in range(len(df1.keys())) if df1.keys()[i]!='Range'}, axis='columns', inplace=True)
        df2 = df1.copy()

        #for r0 in unique_range0:
        for ix in df1.index:
            r1 = df1.loc[ix].Range
            az1 = df1.loc[ix].azimuth
            match0_r1 = df0.loc[df0['Range']==r1]
            #module_logger.warning(f"r1:{r1},az1={az1},\nabs(match0_r1['azimuth'].values-az1)={abs(match0_r1['azimuth'].values-az1)}")
            az0 = df0.loc[np.argmin(abs(match0_r1['azimuth'].values-az1))].azimuth
            index_match0 = np.where(((df0['azimuth']==az0)&(df0['Range']==r1)))[0]
            df2.loc[ix] = df0.loc[index_match0].values.copy()
        df2.rename({list(df2.keys())[i]: retain_keys[i] for i in range(len(df2.keys())) }, axis='columns', inplace=True)
        return df2
        
    def parse_data( self):
        """
        Metodo che itera sui settori in data_path, estratti dai volumi in volume_path,
        e per ogni bin del settore, crea un'istanza CustomBin con attributi sulla dataora
        di acquisizione del volume radar di appartenenza, Range, azimuth ad ogni elevazione
        a cui sono acquisite le grandezze radar usate come features e le grandezze radar
        usate come features a ogni elevazione considerata.
        (NB: DBZH alla prima elevazione è una feature diversa da DBZH ad un'altra elevazione)
        Il metodo crea una lista 'samples' di questi oggetti bin come istanze di CustomBin.
        Dalla lista samples viene estratta la matrice delle features, 'features'.
        Il vettore dei target, 'target', viene estratto iterando sui settori.
        Il metodo ritorna in output features, target, samples.

        INPUT:
         -
        OUTPUT:
        features           --np.array : matrice delle features
        np.array(targets)  --np.array : vettore dei target
        samples            --list     : dataset come lista di istanze CustomBin.
        """
        features = []
        targets = []
        samples = []
        datetimes=[]
        first_elev = self.sel_elangles[0]
        ibreak=0
        #parsed_fnames=[]
        elenco_classe=[]
        ibreak=0
        df_all=None
        for f in tqdm(os.scandir(self.data_path), desc='raccolta campioni dataset', leave=True ):
            
            #if ibreak==100:
             #   break
            #ibreak+=1
            
            if os.path.isfile(f):
                module_logger.debug(f'scan f={f.name}')
            else:
                continue
            fname_splits = f.name.split('_')
            elev = fname_splits[-1].strip('.txt')

            #almeno la prima elevazione ci deve essere
            if float(elev)!=self.sel_elangles[0]:
                continue
            basename=os.path.splitext(f.name)[0].split(f'_{elev}')[0]

            classe = int(fname_splits[0].strip('classid'))
            
            df_tot=None
            for i,elev in enumerate(self.sel_elangles):
                try:
                    fin = os.path.join(self.data_path,f'{basename}_{elev}.txt')
                    if not os.path.exists(fin):
                        if i>0:
                            df = pd.concat([self.fill_dfs[i]]*len(df_tot), ignore_index=True)
                            df.rename(columns={k:k+f'_{i}' for k in df.keys() if k!='Range'}, inplace=True)
                            df[f'azimuth_{i}']=df_tot[f'azimuth_0'].values.copy()
                            #print(df['azimuth'])
                            print(df.keys())
                            df['Range'] = df_tot['Range'].values.copy()
                        else:
                            break       
                    else:
                        if os.stat(fin).st_size>0:
                            module_logger.debug(f"Apro {fin}")
                            df = pd.read_csv(fin,header=0)
                            if len(df)==0:
                                if i==0:
                                    break
                                else:
                                    #module_logger.debug(f"setto df a nodata")
                                    df = pd.concat([self.fill_dfs[i]]*len(df_tot), ignore_index=True)
                                    print(df.keys())
                                    #df.rename(columns={k:k+f'_{i}' for k in df.keys() if k!='Range'}, inplace=True)
                                    df[f'azimuth_{i}']=df_tot[f'azimuth_0'].values.copy()
                                    df['Range'] = df_tot['Range'].values.copy()
                                    #module_logger.debug(f"df:{df}")
                            # azimuth    Range      WRAD       SQI  DBZH_SD2D  DBZH_SDRay_9  DBZH_SDRay_21  DBZH_SDAz  ZDR_SD2D       Z_VD      VRAD       DBZH       ZDR     RHOHV        SNR       DBZV
                            #df.rename(columns={k:k.strip() for k in df.keys()}, inplace=True)
                            try:
                                df = df[self.required_features_elevs[i]]
                            except KeyError:
                                module_logger.exception(f"Il volume del settore non aveva tutte le features d'interesse, skippo")
                                break
                            if self.store_dtype==np.uint16:
                                for fe in self.selected_features_elevs[i]:
                                    df[fe] = ((df[fe].values - self.ngo_dict_elevs[elev][fe].offset)/self.ngo_dict_elevs[elev][fe].gain).astype(self.store_dtype)
                            df.rename(columns={k:k+f'_{i}' for k in df.keys() if k!='Range'}, inplace=True)
                                    #df.rename(columns={f'Az._{i}':f'azimuth_{i}'},inplace=True)
                            if self.store_dtype==np.uint16:
                                try:
                                    df[f'azimuth_{i}'] = (df[f'azimuth_{i}'].values*self.az_scale_factor).astype(self.store_dtype)
                                    df['Range'] = df['Range'].values.astype(self.store_dtype)
                                except:
                                    module_logger.warning("Non ho trovato keys azimuth e Range!!")
                        else:
                            #se il file ha size 0
                            module_logger.warning(f'{fin} corrisponde a file vuoto, lo riempo con nodata')
                            df = self.fill_dfs[i]
                        
                    if df_tot is not None:
                            #module_logger.debug(f"aggiungo a df_tot campi a elev={elev}")
                            first_suff = "" 
                            if 'Range' in df.keys():
                                #df_tot=pd.merge(df_tot, df, on='Range', suffixes=("",""))
                                print(f"df.keys {df.keys()}\ndf_tot.keys() {df_tot.keys()}")
                                if len(df)!=len(df_tot):
                                    try:
                                        df_ = self.select4elev(df_tot,df)
                                    except Exception:
                                        module_logger.exception("Fallita Select4elev!")
                                        exit()

                                if len(df)>len(df_tot):
                                    df_tot = pd.concat([df_tot,df_],axis=1)
                                elif len(df)<len(df_tot):
                                    df_tot = pd.concat([df_,df],axis=1)
                                else:
                                    df_tot=pd.concat([df_tot,df],axis=1)
                                #df_tot=pd.concat([df_tot,df],axis=1)
                                module_logger.debug(f"prima del drop:{df_tot.keys()}")
                                df_tot.drop_duplicates(inplace=True)
                                module_logger.debug(f"dopo del drop:{df_tot.keys()}")
                                df_tot = df_tot.loc[:,~df_tot.columns.duplicated()]
                                #module_logger.debug(f"{df_tot}")
                                
                                if True in np.isnan(df_tot.values):
                                    module_logger.debug(f"df_tot:{df_tot}")
                                    exit()
                            else:
                                break
                    else:
                        df_tot = df.copy()
                        #module_logger.debug(f'inizializzato df_tot  {df_tot}')
                except Exception:
                    module_logger.exception(f'{fin}: non estratto df')
                    # il file dei settori a quella elevazione non esiste
                    break

            if df_tot is None:
                module_logger.debug(f"skip to next f")
                continue
            if df_all is None:
                df_all = df_tot.copy()
            else:
                df_all = pd.concat([df_all,df_tot])
                print(f"df_all check nan:")
                if True in np.isnan(df_all.values):
                    #print(df_all)
                    exit()
            time = datetime.strptime(basename.split('_')[-1],"%Y%m%d%H%M")
            datetimes.extend([time]*len(df_tot))
            elenco_classe.extend([classe]*len(df_tot))
            #parsed_fnames.extend([basename]*len(df_tot))
            

            del df_tot
            df_all.reset_index(drop=True, inplace=True)

            if True in np.isnan(df_all.values):
                #print(df_all)
                exit()
            
        for i in df_all.index:               
            try:                
                sample = CustomBin(
                    time = datetimes[i],
                    Range= df_all.loc[i,'Range'],
                    radar_var_elevs=df_all.iloc[i].to_dict(),
                    elangles =self.sel_elangles
                )
            except KeyError as kerr:
                logging.warning(f"df_all.keys={df_all.keys()}")
                break

            samples.append(sample)
            features.append(list(sample.__dict__.values())[1:-1]) # aggiungo -1 perchè non ci metto elangles
            targets.append( int(elenco_classe[i]) )
        try:
            features = np.array(features, dtype=self.store_dtype)
        except Exception:
            features = np.array(features, dtype=object)
        #try:
            #features = np.array(features, dtype=np.uint16)
        #except:
            #features = np.array(features, dtype=np.float16)
        #features[np.isnan(features)] = np.uint16(65535)# = np.uint16(-1) #np.float64(-9999.)
                
        #self.datetimes = datetimes
        #self.parsed_fnames = parsed_fnames
        return features,np.array(targets),samples

    def save_dset(self):
        """
        Metodo che salva la matrice delle features, il vettore dei target e il dataset
        (ottenuti dal metodo parse_data) come oggetti numpy in save_path, se non sono
        già salvati in save_path.
        """
        fname_features = os.path.join(self.save_path,f'{self.radar.upper()}features')
        fname_target = os.path.join(self.save_path,f'{self.radar.upper()}target')
        fname_samples = os.path.join(self.save_path,f'{self.radar.upper()}samples')
        if not os.path.exists(fname_features):
            np.save(fname_features, self.features, allow_pickle=True)
        if not os.path.exists(fname_target):
            np.save(fname_target, self.target, allow_pickle=True)
        if not os.path.exists(fname_samples):
            np.save(fname_samples, self.samples, allow_pickle=True)

    def load_dset(self):
        """
        Metodo che carica la matrice delle features, il vettore dei target e il dataset
        salvati come oggetti numpy in save_path.
        """
        features = np.load(os.path.join(self.save_path,f'{self.radar.upper()}features.npy'),allow_pickle=True)
        target = np.load(os.path.join(self.save_path,f'{self.radar.upper()}target.npy'),allow_pickle=True)
        samples = np.load(os.path.join(self.save_path,f'{self.radar.upper()}samples.npy'),allow_pickle=True)

        return features, target, samples

    def load_train_xy(self):
        """
        Metodo che carica la matrice delle features e il vettore di target del subset di
        training, salvati in save_path con il metodo split_train_test_xy.
        Se non salvati in save_path, viene invocato split_train_test_xy e vengono poi
        caricati.
        """
        xtrain_fname=os.path.join(self.save_path,f'{self.radar.upper()}_xtrain.npy')
        ytrain_fname=os.path.join(self.save_path,f'{self.radar.upper()}_ytrain.npy')
        if not (os.path.exists(xtrain_fname) and os.path.exists(ytrain_fname)):
            self.split_train_test_xy()
        x_train = np.load(xtrain_fname,allow_pickle=True)
        y_train = np.load(ytrain_fname,allow_pickle=True)
            
        return x_train, y_train

    def load_test_xy(self):
        """
        Metodo che carica la matrice delle features e il vettore di target del subset di
        test, salvati in save_path con il metodo split_train_test_xy.
        Se non salvati in save_path, viene invocato split_train_test_xy e vengono poi
        caricati.
        """
        xtest_fname=os.path.join(self.save_path,f'{self.radar.upper()}_xtest.npy')
        ytest_fname=os.path.join(self.save_path,f'{self.radar.upper()}_ytest.npy')
        if not (os.path.exists(xtest_fname) and os.path.exists(ytest_fname)):
            self.split_train_test_xy()

        x_test = np.load(xtest_fname,allow_pickle=True)
        y_test = np.load(ytest_fname,allow_pickle=True)
        return x_test, y_test

    def split_train_test_xy(self, test_size : float = 0.25):
        """
        Metodo che fa lo split della matrice delle features e del vettore dei target in
        training e test, riservando al test una percentuale data dal parametro in input
        test_size (default=0.25 come in scikit-learn).

        INPUT:
        test_size  --float (default=0.25) : percentuale (normalizzata tra 0 e 1) del
                                            dataset da usare per il test set.                                              
        """
        geo_fe_index=[ c for c in range(len(self.features_names[1:])) if 'Range' in self.features_names[1:][c] or 'azimuth' in self.features_names[1:][c] ]
        radar_features = np.delete(self.features,geo_fe_index,axis=1)
        test_size=np.clip(test_size,0.,1.)
        x_train, x_test, y_train, y_test = train_test_split(radar_features, self.target, test_size=test_size, random_state=0)

        xtrain_fname=os.path.join(self.save_path,f'{self.radar.upper()}_xtrain.npy')
        ytrain_fname=os.path.join(self.save_path,f'{self.radar.upper()}_ytrain.npy')
        xtest_fname=os.path.join(self.save_path,f'{self.radar.upper()}_xtest.npy')
        ytest_fname=os.path.join(self.save_path,f'{self.radar.upper()}_ytest.npy')
        np.save(xtrain_fname, x_train, allow_pickle=True)
        np.save(ytrain_fname, y_train, allow_pickle=True)
        np.save(xtest_fname, x_test, allow_pickle=True)
        np.save(ytest_fname, y_test, allow_pickle=True)
                                  
        
    
