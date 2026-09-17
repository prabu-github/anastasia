from __future__ import annotations
import sys
from typing import List, Dict, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import argparse
import json
import requests
import time
from pprint import pprint
from paths import PATHS

hytraits_path = Path(__file__).resolve().parent.parent/'hytraits'
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
from hytraits import (TabularSpectraDataset,
                      Splits,
                      CommunityWeightingPercent)


def save_df(df: DataFrame,
            save_dir: Path,
            stem: str,
            ext: str) -> None:
    '''
    Helper to save DataFrames.
    '''
    if ext == 'parquet':
        df.to_parquet(save_dir/f'{stem}.parquet',
                      index=False)
    if ext == 'csv':
        df.to_csv(save_dir/f'{stem}.csv',
                  index=False)


def compatible(orig_dir: Path,
               comp_dir: Path,
               prefix: str) -> None:
    '''
    Read the original data from orig_dir and write the compatible-format
    spectra/traits files into comp_dir as CSV and parquet.
    '''

    comp_dir.mkdir(parents=True, 
                   exist_ok=True)

    # load
    xlsx_file = list(orig_dir.glob('*.xlsx'))[0].name
    odf = pd.read_excel(orig_dir/xlsx_file)
    # print(f'odf shape: {odf.shape}')

    # rename: 'PlotID'              -> 'sample_id',
    #         '2492.92_wave_in_nm'  -> 'X_2492.920',
    #         't_mean_nitrogen'     -> 'nitrogen',
    #         't_std_nitrogen'      -> 'nitrogen_sdev'
    wave_suffix = '_wave_in_nm'
    mean_prefix = 't_mean_'
    sdev_prefix = 't_std_'

    col_remap = {'PlotID':        'sample_id',
                 'SubplotID':     'subsample_id',
                 'SubplotWeight': 'subsample_wt',
                 'CoverID':       'cover_id',
                 'CoverWeight':   'cover_wt'}

    for c in odf.columns:
        s = str(c)
        if s.endswith(wave_suffix):
            f = float(s.removesuffix(wave_suffix))
            col_remap[c] = f'X_{f:4.3f}'
        elif s.startswith(mean_prefix):
            col_remap[c] = s.removeprefix(mean_prefix)
        elif s.startswith(sdev_prefix):
            col_remap[c] = f'{s.removeprefix(sdev_prefix)}_sdev'

    odf = odf.rename(columns=col_remap)
    # print(f'renamed {len(col_remap)} columns')
    # pprint({k: v for k, v in col_remap.items()
    #         if not v.startswith('X_')})

    # sample_id is always 2/1/2 digits, so it needs no padding
    odf['sample_id'] = odf['sample_id'].astype(str)

    # number the rows within each sample: 01, 02, 03, ...
    # zero-padded so that a lexicographic sort matches numeric order
    spec_id = odf.groupby('sample_id').cumcount() + 1
    max_pad = len(str(spec_id.max()))
    odf['spectrum_id'] = spec_id.astype(str).str.zfill(max_pad)

    odf['unique_id'] = (odf['sample_id'] + 
                        '__' + 
                        odf['spectrum_id'])

    odf = odf.sort_values(by=['unique_id'])


    # save spectra
    wave_cols = sorted([c for c in odf.columns if str(c).startswith('X_')],
                       key=lambda c: float(str(c).removeprefix('X_')))
    sdf_cols = ['sample_id',
                'spectrum_id',
                'unique_id'] + wave_cols
    # the spectrum is identical for every row of a sample, so keep the first
    first_id = '1'.zfill(max_pad)
    sdf = odf.loc[odf['spectrum_id'] == first_id, sdf_cols].copy()
    
    save_df(df=sdf,
            save_dir=comp_dir,
            stem=f'{prefix}_spectra',
            ext='csv')
    save_df(df=sdf,
            save_dir=comp_dir,
            stem=f'{prefix}_spectra',
            ext='parquet')

    # group the wavelengths into contiguous ranges, breaking wherever the
    # spacing jumps past nominal (the dropped water-vapour bands)
    waves = [float(str(c).removeprefix('X_')) for c in wave_cols]
    diffs = np.diff(waves)
    nominal = np.median(diffs)

    wave_ranges = []
    start = waves[0]
    for w0, w1, d in zip(waves, waves[1:], diffs):
        if d > 1.5*nominal:
            wave_ranges.append((start, w0))
            start = w1
    wave_ranges.append((start, waves[-1]))
    with open(comp_dir/f'{prefix}_waveranges.json', 'w') as fp:
        json.dump(wave_ranges, fp, indent=4)


    # save traits: each trait sits next to its own sdev
    traits = [v for k, v in col_remap.items()
              if str(k).startswith(mean_prefix)]
    tdf_cols = ['sample_id',
                'subsample_id',
                'subsample_wt',
                'cover_id',
                'cover_wt']
    for t in traits:
        tdf_cols += [t, f'{t}_sdev']
    tdf = odf[tdf_cols].copy()

    save_df(df=tdf,
            save_dir=comp_dir,
            stem=f'{prefix}_traits',
            ext='csv')
    save_df(df=tdf,
            save_dir=comp_dir,
            stem=f'{prefix}_traits',
            ext='parquet')

    # save trait columns list
    with open(comp_dir/f'{prefix}_traitcols.json', 'w') as fp:
        json.dump(traits, fp, indent=4)


def community_weighting(comp_dir: Path,
                        prefix: str) -> None:
    # create a community weighting "strategy" file.
    # Percent cover has sdev = 1% around specified!
    # We have only one bin.
    strat_df = DataFrame({'bin_index': [1],
                          'bin_sdev': [1.0],
                          'bin_left': [0.0],
                          'bin_right': [100.01]})
    strat_df.to_csv(comp_dir/f'{prefix}_cwparams.csv',
                    index=False)
    
    # actually do the community weighting
    comwgt = CommunityWeightingPercent(coverid_averaging=False,
                                       coverwt_sampling=True,
                                       n_repeats=100000)
    t_pqts = [comp_dir/f'{prefix}_traits.parquet']
    cwt_csv = comp_dir/f'{prefix}_cwparams.csv'
    with open(comp_dir/f'{prefix}_traitcols.json', 'r') as fp:
        traits = json.load(fp)

    dfs = []
    for trait in traits:
        print(f'{trait}: community weighting ...')
        df = comwgt(trait_files=t_pqts,
                    trait_column=trait,
                    coverwt_params=cwt_csv)
        dfs.append(df.set_index('sample_id'))
    mdf = pd.concat(dfs, axis=1).reset_index()
    save_df(df=mdf,
            save_dir=comp_dir,
            stem=f'{prefix}_cwtraits',
            ext='csv')
    save_df(df=mdf,
            save_dir=comp_dir,
            stem=f'{prefix}_cwtraits',
            ext='parquet')

    # save cwtrait columns list
    cwtrait_cols = [f'mc_{t}' for t in traits]
    with open(comp_dir/f'{prefix}_cwtraitcols.json', 'w') as fp:
        json.dump(cwtrait_cols, fp, indent=4)
    
    
if __name__ == '__main__':
    
    # See README.

    compatible(orig_dir=PATHS['origdata'],
               comp_dir=PATHS['compdata'],
               prefix='anastasia')
    community_weighting(comp_dir=PATHS['compdata'],
                        prefix='anastasia')