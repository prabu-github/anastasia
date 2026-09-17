from pathlib import Path
import json
import argparse
import time
import datetime
from pprint import pprint

from paths import PATHS
from project import (get_dataset,
                     get_splits,
                     get_xtransforms,
                     get_ytransforms,
                     get_modelnames,
                     get_data,
                     project_analyze_univariate,
                     project_fit_model,
                     project_postfit_org)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Tests for project.py.')

    parser.add_argument('--dskey',
                        choices=['anastasia'],
                        action='store',
                        default='anastasia')
    parser.add_argument('--get_xtransforms',
                        action='store_true',
                        default=False)
    parser.add_argument('--get_ytransforms',
                        action='store_true',
                        default=False)
    parser.add_argument('--get_modelnames',
                        action='store_true',
                        default=False)
    parser.add_argument('--pattern',
                        action='store',
                        nargs='*',
                        default=['*'])
    parser.add_argument('--get_data',
                        action='store_true',
                        default=False)
    parser.add_argument('--modelname',
                        action='store',
                        default='dplsr__anastasia-TSS__wr5-vsbl-uv__id')
    parser.add_argument('--train',
                        action='store_true',
                        default=False)
    parser.add_argument('--extension',
                        action='store',
                        default='parquet',
                        choices=['csv', 
                                 'parquet'])

    args = parser.parse_args()

    if args.get_xtransforms:
        xts = get_xtransforms(comp_dir=PATHS['compdata'],
                              ds_key=args.dskey)
        for k in xts:
            print(k)
            for t in xts[k]:
                print(t.get_name_params())

    if args.get_ytransforms:
        yts = get_ytransforms(ds_key=args.dskey)
        for k in yts:
            print(k)
            for t in yts[k]:
                print(t.get_name_params())

    if args.get_modelnames:
        modelnames = get_modelnames(comp_dir=PATHS['compdata'],
                                    patterns=args.pattern)
        pprint(f'{len(modelnames) = }')
        pprint(modelnames)

    if args.get_data:
        dataset, splits = get_data(model_name=args.modelname,
                                   comp_dir=PATHS['compdata'],
                                   seed=42)
        print(f'{args.modelname = }')
        print(f'{len(dataset) = }')
        print(f'{splits.n_outers = }')
        print(f'{splits.n_inners = }')
        print(f'{dataset.wave_ranges = }')
        d = dataset[0]
        print(f'{d["wave_ranges"] = }')
        print(f'{d["wavelengths"] = }')

    if args.train:
        if 'univar' in args.modelname:
            before = time.perf_counter()
            project_analyze_univariate(model_name=args.modelname,
                                       comp_dir=PATHS['compdata'],
                                       model_dir=PATHS['model'],
                                       deploy_dir=PATHS['deploy'],
                                       seed=42)
            after = time.perf_counter()
            elapsed = after - before
            print(str(datetime.timedelta(seconds=round(elapsed))))
        else:
            project_fit_model(model_name=args.modelname,
                              comp_dir=PATHS['compdata'],
                              model_dir=PATHS['model'],
                              deploy_dir=PATHS['deploy'],
                              oi_start=None,
                              oi_stop=None,
                              seed=42) 
            project_postfit_org(model_name=args.modelname,
                                model_dir=PATHS['model'],
                                deploy_dir=PATHS['deploy'])
        