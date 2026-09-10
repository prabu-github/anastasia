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
               comp_dir: Path,
               ext: str) -> None:

    prefix = 'sophia260424'
    comp_dir.mkdir(parents=True, 
                   exist_ok=True)

    # load
    odf = pd.read_csv(orig_dir/'LakeView_hytraitsinput_20260424.csv')
    odf['spectrum_id'] = str(1)
    odf['unique_id'] = odf.apply(lambda r: (r['sample_id'] + 
                                            '__' + 
                                            r['spectrum_id']),
                                 axis=1)

    col_remap = {}
    sdf_cols = ['sample_id',
                'spectrum_id',
                'unique_id']
    min_wave, max_wave = 3000.0, 0.0
    
    for c in odf.columns:
        try:
            f = float(c)
            col_remap[c] = f'X_{f:4.3f}'
            sdf_cols.append(col_remap[c])
            if f < min_wave:
                min_wave = f
            if f > max_wave:
                max_wave = f
        except:
            col_remap[c] = c
    
    tdf_cols = ['sample_id',
                'TP', 
                'TN', 
                'IC',
                'SRP', 
                'nitrite_nitrate', 
                'ammonium',
                'chl_CF', 
                'temp', 
                'phyco_CF',
                'mmHg', 
                'SPcond', 
                'sulfate',
                'Silica', 
                'DO_pct', 
                'TSS',
                'secchi_avg', 
                'NPOC', 
                'chloride']
    
    odf = odf.rename(columns=col_remap)
    sdf = odf[sdf_cols].copy()
    tdf = odf[tdf_cols].copy()


    # save
    # saving renamed versions of trait file! 
    # change this if file size gets larger.
    sdf = sdf.sort_values(by=['sample_id', 'spectrum_id'])
    save_df(df=sdf,
            save_dir=comp_dir,
            stem=f'{prefix}_spectra',
            ext=ext)
    wave_ranges = [(min_wave, max_wave)]
    with open(comp_dir/f'{prefix}_waveranges.json', 'w') as fp:
        json.dump(wave_ranges, fp, indent=4)

    tdf = tdf.sort_values(by=['sample_id'])
    save_df(df=tdf,
            save_dir=comp_dir,
            stem=f'{prefix}_traits',
            ext=ext)
    with open(comp_dir/f'{prefix}_traitcols.json', 'w') as fp:
        json.dump(tdf_cols[1:], fp, indent=4)

        
if __name__ == '__main__':
    
    # See README.
    
    parser = argparse.ArgumentParser('datasetup.')    
    parser.add_argument('--extension',
                        action='store',
                        default='parquet',
                        choices=['csv', 'parquet'])

    args = parser.parse_args()

    compatible(orig_dir=PATHS['origdata'],
               comp_dir=PATHS['compdata'],
               ext=args.extension)