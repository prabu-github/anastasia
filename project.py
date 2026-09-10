from __future__ import annotations
import sys
from typing import List, Dict, Tuple, Union, Callable
from pathlib import Path
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import json
from pprint import pprint
from itertools import product
from sklearn.model_selection import KFold
import fnmatch

hytraits_path = Path(__file__).resolve().parent.parent/'hytraits'
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
from hytraits import (TabularSpectraDataset,
                      Splits,
                      get_train_valid_test_splits,
                      WavelengthResampling5nm,
                      KeepWavelengths,
                      UnitVectorize,
                      fit_fastdplsr,
                      analyze_univariate_models,
                      postfit_org)


def get_dataset(comp_dir: Path,
                ds_key: str,
                extension: str,
                trait_col: str,
                xtransforms: List[H.BaseTransform] = [],
                ytransforms: List[H.BaseTransform] = [],
                seed: int = 42) -> H.TabularSpectraDataset:
    if ds_key not in ['sophia260424']:
         raise Exception(f'Invalid {ds_key = }')
        
    s_files = [comp_dir/f'{ds_key}_spectra.{extension}']
    t_files = [comp_dir/f'{ds_key}_traits.{extension}']
    
    with open(comp_dir/f'{ds_key}_waveranges.json', 'r') as f:
        wave_ranges_ = json.load(f)
    wave_ranges = [tuple(e) for e in wave_ranges_]

    trait_clip = (0.0, None)
    
    return TabularSpectraDataset(spectra_files=s_files,
                                 wave_ranges=wave_ranges,
                                 trait_files=t_files,
                                 trait_column=trait_col,
                                 xtransforms=xtransforms,
                                 ytransforms=ytransforms,
                                 spectra_averaging=False,
                                 spectra_sampling=False,
                                 sample_selector=None,
                                 trait_sampling=False,
                                 trait_clip=trait_clip,
                                 seed=seed)


def get_splits(dataset: TabularSpectraDataset,
               ds_key: str,
               seed: int = 42) -> Splits:
    repeats = (100, 30)
    types = ('montecarlo', 'montecarlo')
    params = (15, 15)
    idxs = dataset.sid2idx(dataset.i2s)
    return get_train_valid_test_splits(idxs=idxs,
                                       repeats=repeats,
                                       types=types,
                                       params=params,
                                       seed=seed)
    

######################################################


def get_xtransforms(ds_key: str) -> Dict:
    '''
    xtransform strategies. 

    Return: Dict
            {xtransform_key: List[BaseTransform]}        
    '''  
    xtransforms = {}

    wr = WavelengthResampling5nm(wave_range=(410, 800))
    kw = KeepWavelengths(keep_ranges=[(410, 800)])
    uv = UnitVectorize()
    xtransforms['wr5-vsbl-uv'] = [wr, 
                                  kw, 
                                  uv]
    return xtransforms
    

def get_ytransforms(ds_key: str) -> Dict:
    '''
    ytransform strategies. 

    Return: Dict
            {ytransform_key: List[BaseTransform]}        
    '''
    ytransforms = {'id': []}

    return ytransforms 


def get_modelnames(comp_dir: Path,
                   patterns: List[str] = ['*']) -> List[str]:
    model_names = []
    
    ds_keys = ['sophia260424']
    
    for ds_key in ds_keys:
        with open(comp_dir/f'{ds_key}_traitcols.json') as reader:
            traitcols = json.load(reader)
        traits = [f'{ds_key}-{tcol}' for tcol in traitcols]
        
        xts = get_xtransforms(ds_key=ds_key).keys()
        yts = get_ytransforms(ds_key=ds_key).keys()
        pieces = [['univar', 'dplsr'],
                  traits,
                  xts,
                  yts]
        for e in product(*pieces):
            model_names.append('__'.join(list(e)))

    filtered = []
    for pattern in patterns:
        filtered += fnmatch.filter(model_names, pattern)
    return filtered 


######################################################


def get_data(model_name: str,
             comp_dir: Path,
             seed: int = 42) -> Tuple[TabularSpectraDataset, Splits]:
    tkns = model_name.split('__')
    k_mt = tkns[0]
    k_ds, k_tt = tkns[1].split('-')
    k_xt = tkns[2]
    k_yt = tkns[3]

    xts = get_xtransforms(ds_key=k_ds)
    yts = get_ytransforms(ds_key=k_ds)
    ds = get_dataset(comp_dir=comp_dir,
                     ds_key=k_ds,
                     extension='parquet',
                     trait_col=k_tt,
                     xtransforms=xts[k_xt],
                     ytransforms=yts[k_yt],
                     seed=seed)

    splits = get_splits(dataset=ds,
                        ds_key=k_ds,
                        seed=seed)

    return (ds, splits) 


def project_fit_model(model_name: str,
                      comp_dir: Path,
                      model_dir: Path,
                      deploy_dir: Path,
                      oi_start: int,
                      oi_stop: int,
                      seed: int = 42) -> None:
    print(f'Fitting: {model_name} ...')
    
    dataset, splits = get_data(comp_dir=comp_dir,
                               model_name=model_name,
                               seed=seed)
    
    hyps = {'n_components': 60,
            'sweep': True,
            'model_type': 'dplsr'}

    dataset.set_seed(seed=seed)
    fit_fastdplsr(hyps=hyps,
                  dataset=dataset,
                  splits=splits,
                  model_dir=model_dir/model_name,
                  deploy_dir=deploy_dir/model_name,
                  press_reduce='median',
                  oi_start=oi_start,
                  oi_stop=oi_stop)


######################################################


def project_analyze_univariate(model_name: str,
                               comp_dir: Path,
                               model_dir: Path,
                               deploy_dir: Path,
                               seed: int = 42) -> None:
    print(f'Analyzing: {model_name} ...')
    
    dataset, splits = get_data(comp_dir=comp_dir,
                               model_name=model_name,
                               seed=seed)
    analyze_univariate_models(dataset=dataset,
                              splits=splits,
                              model_dir=model_dir/model_name,
                              deploy_dir=deploy_dir/model_name)


###################################################### 


def project_postfit_org(model_name: str,
                        model_dir: Path,
                        deploy_dir: Path) -> None:
    postfit_org(model_name=model_name,
                model_dir=model_dir,
                deploy_dir=deploy_dir,
                targz_models=True,
                targz_deploys=True)
    

###################################################### 

