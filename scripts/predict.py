import h5py
import numpy as np
import logging
import os
from simcradarlib.visualization.plot_ppi import plot_ppi_curvilinear
from simcradarlib.odim.odim_pvol import OdimHierarchyPvol
from simcradarlib.geo_utils.georef import spherical_to_xyz
from matplotlib import colors
from datetime import datetime
import matplotlib.pyplot as plt

def get_Xdsvars_fromh5_elevs4(fname, features_names_elevs, elangles, el_to_pred, raw: bool = False):
    """
    crea matrice delle features per il volume da predire con filename 'fname' (parametro in input).
    INPUT:
    fname  --str : volume da predire
    features_names_elevs  --list : lista in cui ogni componente è una lista 
                                   delle grandezze radar da usare come features.
                                   Il numero di componenti della lista dipende da quante
                                   elevazioni usiamo.
                                   Per SPC avrà due componenti (la prima per l'elevazione da predire,
                                   la seconda corrisponde sempre alla quarta elevazione).
                                   Per GAT ha una sola componente.
    elangles --list : lista degli angoli di elevazione a cui estraggo le features per creare la matrice 
                       delle features, usando come features per ciascun angolo quelle della lista 
                       features_names_elevs.
    el_to_pred --list : lista contenente l'angolo di elevazione a cui voglio prevedere la ripulitura.
    raw --bool (default=False) : se True restituisco i dati in formato unsigned np.uint16, altrimenti
                                    moltiplico per gain e sommo offset per ciascuna grandezza.
                                    
    OUTPUT: 
    actualds --np.array : matrice delle features per il volume da predire.
    """
    
    h = h5py.File(fname,'r')
    actualdf = {}
    Offset_list = []
    actualds = np.empty([])
    shape0=0
    elangles[0] = el_to_pred
    elangles_dsets=[]
    elangles_in_f = []
    for el in elangles:
        dsetname=[d for d in h.keys() if ('dataset' in d and h[f"{d}/where"].attrs['elangle']==el)]
        if dsetname.__len__()>0:
            elangles_dsets.append(dsetname[0])
            elangles_in_f.append(el)
    elangles_dsets_in_f = [d for d in h.keys() if 'dataset' in d]
        
    for i,de in enumerate(elangles_dsets):
        actualdf[i]={}
        nrays_el = h[f'{de}/where'].attrs['nrays']
        nbins_el = h[f'{de}/where'].attrs['nbins']
        elangle_val = h[f'{de}/where'].attrs['elangle']
        print(f"estraggo dati a elangle={elangle_val}")
        for fe in features_names_elevs[i]:
            try:
                denow=de
                ke = [k for k in h[f'{denow}'].keys() if('data' in k and h[f'{denow}/{k}/what'].attrs['quantity'].decode('utf-8')==fe)][0]
            except Exception:
                #print([h[f'{denow}/{k}/what'].attrs['quantity'] for k in h[f'{denow}'].keys()])
                if elangle_val==max(elangles_in_f) and fe=="Z_VD":
                    print(f'caso Z_VD alla massima elevazione {elangle_val}')
                    denow='dataset1'
                    ke = [k for k in h[f'{denow}'].keys() if('data' in k and h[f'{denow}/{k}/what'].attrs['quantity'].decode('utf-8')==fe)][0]
                    #print(f'uso ke={ke}')
                    denow=None
            #print(f'denow={denow}')
            if denow is None:
                data_ = np.ones((nrays_el,nbins_el))*h[f'dataset1/{ke}/what'].attrs['nodata']
                #data_ = np.ones((nrays_el,nbins_el))
                data_ = data_*h[f'dataset1/{ke}/what'].attrs['gain']+h[f'dataset1/{ke}/what'].attrs['offset']
                #print(f'denowNOne--> data_.shape={data_.shape}')
                denow='dataset1'
            else:
                try:
                    #print(ke, h[f'{denow}/{ke}/what'].attrs['quantity'].decode('utf-8'), fe)
                    data_ = h[f'{denow}/{ke}/data'][:]
                    #print(f"min={data_.min()}, max={data_.max()}")
                    if raw:
                        pass
                    else:
                        data_ = data_*h[f'{denow}/{ke}/what'].attrs['gain']+h[f'{denow}/{ke}/what'].attrs['offset']
                        #print(f"min={data_.min()}, max={data_.max()}")
                except:
                    logging.exception(f'{fe} non trovato')
                    continue
            if shape0==0:
                shape0=data_.shape[1]
            else:
                diff = int(shape0-data_.shape[1])
                if diff>0:
                    ext = np.ones((data_.shape[0],diff))*h[f'{denow}/{ke}/what'].attrs['nodata']
                    ext = ext*h[f'{denow}/{ke}/what'].attrs['gain'] + h[f'{denow}/{ke}/what'].attrs['offset']
                    #ext = np.ones((data_.shape[0],diff))*h[f'{de}/{ke}/what'].attrs['offset']
                    #print(f"ext shape: {ext.shape}, data_.shape={data_.shape}")
                    data_ = np.concatenate([data_.T,ext.T]).T
    #data_ = np.ma.masked_array(np.ma.masked_where(data_==offset,data_),fill_value=offset)
            actualdf[i][fe] = data_.ravel()
            
        if i==0:
            actualds = np.array(list(actualdf[i].values())).T
        else:
            actualdf_to_np = np.array(list(actualdf[i].values())).T
            if actualdf_to_np.shape[0]!=actualds.shape[0]:
                #print('modifico shape')
                temp_ds = actualds.reshape((nrays_el,-1,actualds.shape[1]))
                bin_lim = temp_ds.shape[1]
                actualdf_to_np=actualdf_to_np.reshape((nrays_el,-1,actualdf_to_np.shape[1]))
                varn =  actualdf_to_np.shape[2]
                actualdf_to_np = actualdf_to_np[:,:bin_lim,:]
                actualdf_to_np = actualdf_to_np.reshape((-1,varn))
            #print(actualds.shape,actualdf_to_np.shape)
            actualds = np.hstack([actualds,actualdf_to_np ])

    h.close()

    return actualds

def pred(model_, f_topred, origvol, used_features_elevs, label)->None:
    """
    sovrascrive riflettività corretta col random forest al
    dataset con quantity "DBZH" a ciascuna elevazione 
    nel file da prevedere f_topred.
    
    INPUT:
    model_ --model : istanza di sklearn.ensemble._forest.RandomForestClassifier
                     o di xgboost.sklearn.XGBClassifier,
                     preallenata sul dataset del radar corrispondente.
    f_topred --str : nome del volume da ripulire/prevedere.
    origvol --simcradarlib.odim.odim_pvol.OdimHierarchyPvol :
                     istanza di OdimHierarchyPvol in cui è storato il contenuto del
                     volume originale da ripulire, già letto.
    used_features_elevs --list:
                     lista di liste, ciasuna contenente le grandezze radar usate come features
                     per le elevazioni usate. Dipende dal modello.
    label --str : stringa label del modello utilizzato per il cleaning (GAT: RF08, RF09, XGB0, XGB1\nSPC:
                     RF, XGB).
                     
    OUTPUT:
    None : sovrascrive la riflettività con il campo corretto con model_ e non restituisce niente.
    """
    sourcevol = origvol.root_what.source.split(',')
    radar=[w.split(':')[-1] for w in sourcevol if 'NOD' in w][0].replace('it','')
    out_vol = h5py.File(f_topred,'r+')

    sortelangles=origvol.elangles.copy()
    sortelangles.sort()
    
    indexz = origvol.varsnames.index("DBZH")
    undetect= origvol.group_datasets_data_what[0][indexz].undetect
    nodata = origvol.group_datasets_data_what[0][indexz].nodata
    nodata = nodata*origvol.group_datasets_data_what[0][indexz].gain + origvol.group_datasets_data_what[0][indexz].offset

    for iel, elev_angle in enumerate(origvol.elangles):
        zraw = origvol.get_data_by_elangle(elev_angle,"DBZH")
        #if radar=='gat':
        if len(used_features_elevs)==1:
            actualds = get_Xdsvars_fromh5_elevs4(f_topred, used_features_elevs, [elev_angle], [elev_angle])
        else:
            actualds = get_Xdsvars_fromh5_elevs4(f_topred, used_features_elevs, [elev_angle,sortelangles[3]], [elev_angle])
        retain_indx = np.where((actualds[:,0]!=nodata)&(actualds[:,0]!=undetect))[0]
        ret_ds_ = actualds[retain_indx,:]
        RFlabs_ = model_.predict_proba( ret_ds_)
        out_ = np.zeros((RFlabs_[:,0].shape))
        for i,b in enumerate(RFlabs_):
            maxclass = list(b).index(b.max())
            if radar=='gat':
                if "XGB" in label:
                    if(maxclass == 0 and b.max()<=0.95 and (b[0]-b[1])<0.98):
                        maxclass = 1
                elif label=="RF09" :
                    if maxclass==0 and b.max<0.9:
                        maxclass = 1
                elif( label=="RF08" and maxclass==0 and b.max<0.8):
                    maxclass = 1
                else:
                    pass
            if radar=="spc":
                if label=="RF":
                    if(maxclass==0 and b.max()<=0.8 and (b[0]-b[4])<0.98):
                        maxclass=4
                elif label=="XGB":
                    if(maxclass == 0 and ((b[0]-b[3])<0.98 or (b[0]-b[1])<0.98 or (b[0]-b[4])<0.98 or (b[0]-b[2])<0.98)):
                        maxclass = 1
                else:
                     pass   
            out_[i] = maxclass

        mask_RF_ = np.ones(zraw.ravel().shape)
        mask_RF_[retain_indx] = out_.astype(bool)

        rstart = origvol.group_datasets_where[iel].rstart
        nbins = origvol.group_datasets_where[iel].nbins
        rscale = origvol.group_datasets_where[iel].rscale
        r = np.arange(rstart,rstart + (nbins+1) * rscale, rscale)

        nrays = origvol.group_datasets_where[iel].nrays
        az_binsize = 360. / nrays
        az = np.arange(0., 360.+az_binsize, 360. / nrays)

        site=(origvol.root_where.lon,origvol.root_where.lat,origvol.root_where.height)

        #myxyz, myproj = spherical_to_xyz(r, az, elev_angle, site)

        varsnames_elangle = origvol.varsnames.copy()
        if elev_angle==max(origvol.elangles) and "Z_VD" in varsnames_elangle:
            varsnames_elangle.remove("Z_VD")
        indexq = varsnames_elangle.index("DBZH")
        offsetx = origvol.group_datasets_data_what[iel][indexq].offset
        gainx = origvol.group_datasets_data_what[iel][indexq].gain
        zraw = (zraw- offsetx )/ gainx
        outRF_ = np.ma.masked_array(zraw, mask_RF_.reshape((nrays,nbins)))
        outRF_ = np.ma.filled(outRF_, fill_value=nodata)
        #outRF_ = np.ma.filled(outRF_, fill_value=undetect-offsetx)
        pyel = [k for k in out_vol.keys() if 'dataset' in k and out_vol[f'{k}/where'].attrs['elangle']==elev_angle][0]
        pyq = [q for q in out_vol[f'{pyel}'].keys() if 'data' in q and out_vol[f'{pyel}/{q}/what'].attrs['quantity'].decode()=='DBZH' ][0]
        out_vol[f'{pyel}/{pyq}/data'][:] = outRF_.astype(np.uint16)
        del zraw, actualds, retain_indx, ret_ds_, RFlabs_, out_, outRF_, indexq, varsnames_elangle

    out_vol.close()

def plot_classpred(model_, outdir, f_vol, used_features_elevs)->None:
    """
    Plotta la probabilità di ciascuna classe target prevista dal modello model_
    per il PPI della prima elevazione.
    
    INPUT:
    model_ --model : istanza di sklearn.ensemble._forest.RandomForestClassifier
                     preallenata sul dataset del radar corrispondente.
    outdir --str : path dove salvare le probabilità delle classi plottate.
    f_vol --str : pathname del volume da classificare.
    used_features_elevs --list:
                     lista di liste, ciasuna contenente le grandezze radar usate come features
                     per le elevazioni usate. Dipende dal modello.
                     
    OUTPUT:
    None : sovrascrive la riflettività con il campo corretto con model_ e non restituisce niente.
    """

    livelli_prob = [0.,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]
    colori_prob  = ['#3288bd','#66c2a5','#abdda4','#e6f598',
                    '#fee08b','#fdae61','#f46d43','#d53e4f', '#f05b6c']
    cmap_prob,norm_prob=colors.from_levels_and_colors(livelli_prob,colori_prob,extend='neither')

    livelli_class = [0.,1.,2.,3,4]
    coloriclass  = ['#3288bd','#abdda4',
               '#fee08b','#d53e4f']
    cmap_class,norm_class=colors.from_levels_and_colors(livelli_class,coloriclass,extend='neither')

    origvol=OdimHierarchyPvol()
    origvol.read_odim(f_vol)
    sourcevol = origvol.root_what.source.split(',')
    radar=[w.split(':')[-1] for w in sourcevol if 'NOD' in w][0].replace('it','')
    voldate=datetime.strptime(origvol.root_what.date+origvol.root_what.time, "%Y%m%d%H%M%S")
    
    sortelangles=origvol.elangles.copy()
    sortelangles.sort()
    
    indexz = origvol.varsnames.index("DBZH")
    undetect= origvol.group_datasets_data_what[0][indexz].undetect
    nodata = origvol.group_datasets_data_what[0][indexz].nodata
    nodata = nodata*origvol.group_datasets_data_what[0][indexz].gain + origvol.group_datasets_data_what[0][indexz].offset

    #for iel, elev_angle in enumerate(origvol.elangles):
    for iel, elev_angle in enumerate([0.5]):
        zraw = origvol.get_data_by_elangle(elev_angle,"DBZH")
        #if radar=='gat':
        if len(used_features_elevs)==1:
            actualds = get_Xdsvars_fromh5_elevs4(f_vol, used_features_elevs, [elev_angle], [elev_angle])
        else:
            actualds = get_Xdsvars_fromh5_elevs4(f_vol, used_features_elevs, [elev_angle,sortelangles[3]], [elev_angle])
        retain_indx = np.where((actualds[:,0]!=nodata)&(actualds[:,0]!=undetect))[0]
        ret_ds_ = actualds[retain_indx,:]
        RFlabs_ = model_.predict_proba( ret_ds_)

        rstart = origvol.group_datasets_where[iel].rstart
        nbins = origvol.group_datasets_where[iel].nbins
        rscale = origvol.group_datasets_where[iel].rscale
        r = np.arange(rstart,rstart + (nbins+1) * rscale, rscale)

        nrays = origvol.group_datasets_where[iel].nrays
        az_binsize = 360. / nrays
        az = np.arange(0., 360.+az_binsize, 360. / nrays)

        site=(origvol.root_where.lon,origvol.root_where.lat,origvol.root_where.height)

        myxyz, myproj = spherical_to_xyz(r, az, elev_angle, site)

        target_names=['meteo','clutter','intmul','intmed','intdeb']
        #ciclo sui target
        for i in range(5):
            class_pred = np.zeros(zraw.ravel().shape)
            class_pred[retain_indx] = RFlabs_[...,i]
            fig = plt.figure(figsize=(10,10))
            plot_ppi_curvilinear(fig,cmap_prob,norm_prob,livelli_prob,"dBZ",livelli_prob,class_pred.reshape((nrays,nbins)),4,myxyz,r)
            outname=os.path.join(outdir,f"{radar}prob_{target_names[i]}_{voldate.strftime('%Y%m%d%H%M')}.png")
            plt.savefig(outname, dpi=300, bbox_inches="tight")

        #plotto classe + probabile
        out_ = np.zeros((RFlabs_[:,0].shape))
        for i,b in enumerate(RFlabs_):
            maxclass = list(b).index(b.max())
            out_[i] = maxclass
        base_ppi=np.zeros(zraw.ravel().shape)
        base_ppi[retain_indx] = out_
        fig = plt.figure(figsize=(10,10))
        plot_ppi_curvilinear(fig,cmap_class,norm_class,livelli_class,"class",livelli_class,base_ppi.reshape((nrays,nbins)),4,myxyz,r)
        outname=os.path.join(outdir,f"{radar}class_{voldate.strftime('%Y%m%d%H%M')}.png")
        plt.savefig(outname, dpi=300, bbox_inches="tight")
        
        del zraw, actualds, retain_indx, ret_ds_, RFlabs_
