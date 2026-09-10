from __future__ import annotations
from typing import List, Dict, Tuple, Union, Callable
from pathlib import Path
import argparse
from pprint import pprint

from paths import PATHS
from project import get_modelnames


if __name__ == '__main__':
    # See README.
    
    parser = argparse.ArgumentParser('chtcsetup.')

    parser.add_argument('--username',
                        type=str,
                        default='pravindran',
                        help='Username on CHTC')
    parser.add_argument('--projname',
                        type=str,
                        default=Path(__file__).parent.stem,
                        help='Current project name')
    parser.add_argument('--models_per_submit',
                        type=int,
                        action='store',
                        default=20)
    parser.add_argument('--patterns',
                        nargs='+',
                        default=['*'],
                        help='List of patterns')
    args = parser.parse_args()

    PATHS['chtc'].mkdir(parents=True, exist_ok=True)
    
    # create cpujob.sh
    with open(PATHS['assets']/'chtc_cpujob.sh.template', 'r') as fp:
        contents = ''.join(fp.readlines())
    contents = contents.replace('INITIAL', f'{args.username[0]}')
    contents = contents.replace('USERNAME', f'{args.username}')
    contents = contents.replace('PROJNAME', f'{args.projname}')
    with open(PATHS['chtc']/'cpujob.sh', 'w') as fp:
        fp.write(contents)
    
    # create cpujobs.sub
    with open(PATHS['assets']/'chtc_cpujobs.sub.template', 'r') as fp:
        contents = ''.join(fp.readlines())
    contents = contents.replace('INITIAL', f'{args.username[0]}')
    contents = contents.replace('USERNAME', f'{args.username}')
    contents = contents.replace('PROJNAME', f'{args.projname}')
    contents = contents.replace('PARAMSFILE', 'cpujobs.params')
    with open(PATHS['chtc']/'cpujobs.submit', 'w') as fp:
        fp.write(contents)

    # create cpujobs.params
    model_names = get_modelnames(comp_dir=PATHS['compdata'],
                                 patterns=args.patterns)
    model_patts = []
    for i in range(0, len(model_names), args.models_per_submit):
        mp = ':'.join(model_names[i:(i + args.models_per_submit)])
        model_patts.append(mp)
    with open(PATHS['chtc']/'cpujobs.params', 'w') as fp:
        fp.write('\n'.join(model_patts))


    print(f'{len(model_names)} models, {len(model_patts)} jobs.')

        