from pathlib import Path
import argparse
from pprint import pprint

from paths import PATHS
from project import (project_analyze_univariate,
                     project_fit_model,
                     project_postfit_org)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('run.')
    parser.add_argument('--train_model',
                        action='store',
                        type=str,
                        default='',
                        help='Specify as MODELNAME or MODELNAME1:MODELNAME2:..')
    parser.add_argument('--oi_start',
                        type=int,
                        default=None)
    parser.add_argument('--oi_stop',
                        type=int,
                        default=None)
    
    args = parser.parse_args()

    
    ##########################################################
    # train model(s)
    if len(args.train_model) > 0:
        model_names = [mn.strip() for mn in args.train_model.split(':')]
        for model_name in model_names:
            if 'univar' in model_name:
                project_analyze_univariate(model_name=model_name,
                                           comp_dir=PATHS['compdata'],
                                           model_dir=PATHS['model'],
                                           deploy_dir=PATHS['deploy'],
                                           seed=42)
            else:
                project_fit_model(model_name=model_name,
                                  comp_dir=PATHS['compdata'],
                                  model_dir=PATHS['model'],
                                  deploy_dir=PATHS['deploy'],
                                  oi_start=args.oi_start,
                                  oi_stop=args.oi_stop,
                                  seed=42) 
                project_postfit_org(model_name=model_name,
                                    model_dir=PATHS['model'],
                                    deploy_dir=PATHS['deploy'])
    
            