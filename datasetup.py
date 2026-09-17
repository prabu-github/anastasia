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
               comp_dir: Path) -> None:
    '''
    Read the original data from orig_dir and write the compatible-format
    spectra/traits files into comp_dir as CSV and parquet.
    '''

    prefix = 'anastasia'
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
    # print(f'sdf shape: {sdf.shape}')

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
    # print(f'nominal spacing: {nominal:.2f} nm')
    # pprint(wave_ranges)

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
    # print(f'tdf shape: {tdf.shape}')

    save_df(df=tdf,
            save_dir=comp_dir,
            stem=f'{prefix}_traits',
            ext='csv')
    save_df(df=tdf,
            save_dir=comp_dir,
            stem=f'{prefix}_traits',
            ext='parquet')

    with open(comp_dir/f'{prefix}_traitcols.json', 'w') as fp:
        json.dump(traits, fp, indent=4)


if __name__ == '__main__':
    
    # See README.

    compatible(orig_dir=PATHS['origdata'],
               comp_dir=PATHS['compdata'])
