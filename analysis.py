import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

fromchtc_dir = Path('/media/prabu/ssd3/anastasia/09_22_2026')

hytraits_path = Path(__file__).resolve().parent.parent/'hytraits'
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
from hytraits import plot_pred_vs_true as hytraits_plot_pred_vs_true
from hytraits import plot_values_vs_waves as hytraits_plot_values_vs_waves
from hytraits import split_by_wave_ranges as hytraits_split_by_wave_ranges


def plot_pred_vs_true(deploy_dir: Path):
    '''
    For every dplsr__* job under deploy_dir, reads its preds.csv and plots
    mean prediction vs y_true using hytraits.plot_pred_vs_true.

    Yields (job_dir, fig, ax) for each job.
    '''
    for job_dir in sorted(deploy_dir.glob('dplsr__*')):
        preds_df = pd.read_csv(job_dir/'preds.csv')
        pred_cols = [c for c in preds_df.columns if c.endswith('_pred')]
        pred_vals = preds_df[pred_cols].to_numpy(dtype=float)

        pred_df = pd.DataFrame({'sample_id': preds_df['unique_id'],
                                'y_true': preds_df['y_true'],
                                'mean_pred': np.nanmean(pred_vals, axis=1),
                                'sdev_pred': np.nanstd(pred_vals, axis=1)})

        color_df = pd.DataFrame({'sample_id': pred_df['sample_id'],
                                 'color': '#179EE6'})

        fig, ax = plt.subplots()
        hytraits_plot_pred_vs_true(ax=ax,
                                   pred_df=pred_df,
                                   color_df=color_df)
        ax.set_title(job_dir.name)
        yield job_dir, fig, ax


def _trait_from_job_name(job_name: str) -> str:
    # e.g. 'dplsr__anastasia-mc_aluminum__full-uv__id' -> 'aluminum'
    return job_name.split('__')[1].split('_')[-1]


def metrics_summary(deploy_dir: Path):
    '''
    For each dplsr__* job under deploy_dir, reads metrics.csv and computes
    the mean/sdev of range_normalized_rmse and fitted_r2 across repeats.
    Writes the per-job summary, sorted by TRAIT, to
    deploy_dir.parent/'analysis'/'metrics_summary.csv'.

    Returns the summary as a DataFrame.
    '''
    rows = []
    for job_dir in sorted(deploy_dir.glob('dplsr__*')):
        metrics_df = pd.read_csv(job_dir/'metrics.csv')
        rows.append({'trait': _trait_from_job_name(job_dir.name),
                     'range_normalized_rmse_mean': metrics_df['range_normalized_rmse'].mean(),
                     'range_normalized_rmse_sdev': metrics_df['range_normalized_rmse'].std(),
                     'fitted_r2_mean': metrics_df['fitted_r2'].mean(),
                     'fitted_r2_sdev': metrics_df['fitted_r2'].std(),
                     'job_dir': job_dir.name})

    summary_df = pd.DataFrame(rows).sort_values('trait').reset_index(drop=True)

    analysis_dir = deploy_dir.parent/'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(analysis_dir/'metrics_summary.csv', index=False)

    return summary_df


def plot_metrics_bars(deploy_dir: Path):
    '''
    Reads metrics_summary.csv (from deploy_dir.parent/'analysis') and makes
    a bar plot per metric (range_normalized_rmse, fitted_r2), one bar per
    trait, with an error bar for that metric's sdev. Saves both as PNGs in
    the analysis directory.
    '''
    analysis_dir = deploy_dir.parent/'analysis'
    summary_df = pd.read_csv(analysis_dir/'metrics_summary.csv')

    for metric in ['range_normalized_rmse', 'fitted_r2']:
        fig, ax = plt.subplots(figsize=(max(6, len(summary_df)*0.4), 5))
        ax.bar(summary_df['trait'],
              summary_df[f'{metric}_mean'],
              yerr=summary_df[f'{metric}_sdev'],
              capsize=3)
        ax.set_xlabel('trait')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=90)
        fig.tight_layout()
        fig.savefig(analysis_dir/f'{metric}_bar.png')
        plt.close(fig)


def plot_vips(model_dir: Path):
    '''
    For every dplsr__* job under model_dir, reads vips, wavelengths and
    wave_ranges from its model.npz and plots mean VIP vs wavelength
    (show_sdev=False, mean only) using hytraits.plot_values_vs_waves, which
    splits the wavelength axis into wave_ranges' regions via
    hytraits.split_by_wave_ranges internally.

    Yields (job_dir, fig, ax) for each job.
    '''
    for job_dir in sorted(model_dir.glob('dplsr__*')):
        data = np.load(job_dir/'model.npz', allow_pickle=True)
        vips = data['vips']
        wavelengths = data['wavelengths']
        wave_ranges = [tuple(r) for r in data['wave_ranges']]

        fig, ax = plt.subplots()
        hytraits_plot_values_vs_waves(ax=ax,
                                      values=vips,
                                      waves=wavelengths,
                                      ranges=wave_ranges,
                                      show_sdev=False)
        ax.set_ylabel('VIP')
        ax.set_title(_trait_from_job_name(job_dir.name))
        yield job_dir, fig, ax


def plot_scoefs(model_dir: Path):
    '''
    For every dplsr__* job under model_dir, reads scoefs, wavelengths and
    wave_ranges from its model.npz and plots mean scoefs vs wavelength
    (show_sdev=False, mean only) using hytraits.plot_values_vs_waves, which
    splits the wavelength axis into wave_ranges' regions via
    hytraits.split_by_wave_ranges internally.

    Yields (job_dir, fig, ax) for each job.
    '''
    for job_dir in sorted(model_dir.glob('dplsr__*')):
        data = np.load(job_dir/'model.npz', allow_pickle=True)
        scoefs = data['scoefs']
        wavelengths = data['wavelengths']
        wave_ranges = [tuple(r) for r in data['wave_ranges']]

        fig, ax = plt.subplots()
        hytraits_plot_values_vs_waves(ax=ax,
                                      values=scoefs,
                                      waves=wavelengths,
                                      ranges=wave_ranges,
                                      show_sdev=False)
        ax.set_ylabel('scoefs')
        ax.set_title(_trait_from_job_name(job_dir.name))
        yield job_dir, fig, ax


def plot_heatmaps(deploy_dir: Path, model_dir: Path, metric: str = 'rnrmse_mean'):
    '''
    For every univar__* job under deploy_dir, reads metric (one of
    'rmse_mean', 'rnrmse_mean', 'r2_mean') from its metrics.npz (a
    wavelength-pair matrix) and plots it as a heatmap. wavelengths and
    wave_ranges come from the corresponding dplsr__* job's model.npz (same
    TRAIT, 'univar__' swapped for 'dplsr__'). The heatmap is displayed in
    pieces, split along both axes by wave_ranges via
    hytraits.split_by_wave_ranges, so gaps between regions are real empty
    space rather than a continuous image with lines drawn on top.

    Yields (job_dir, fig, ax) for each job.
    '''
    for job_dir in sorted(deploy_dir.glob('univar__*')):
        metrics_data = np.load(job_dir/'metrics.npz', allow_pickle=True)
        values = metrics_data[metric]
        # (-1, 0) is a fill value, not a real fitted result; it fills the
        # whole lower triangle (the matrix is only fit upper-triangular).
        values = np.ma.masked_equal(values, values[-1, 0])

        model_job_dir = model_dir/job_dir.name.replace('univar__', 'dplsr__', 1)
        model_data = np.load(model_job_dir/'model.npz', allow_pickle=True)
        wavelengths = model_data['wavelengths']
        wave_ranges = [tuple(r) for r in model_data['wave_ranges']]

        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='white')
        vmin, vmax = values.min(), values.max()

        fig, ax = plt.subplots()

        col_pieces = hytraits_split_by_wave_ranges(waves=wavelengths,
                                                   ranges=wave_ranges,
                                                   values=values)
        for col_waves, col_vals in col_pieces:
            row_pieces = hytraits_split_by_wave_ranges(waves=wavelengths,
                                                       ranges=wave_ranges,
                                                       values=col_vals.T)
            for row_waves, row_vals in row_pieces:
                piece = row_vals.T
                im = ax.imshow(piece,
                              origin='lower',
                              extent=[col_waves.min(), col_waves.max(),
                                     row_waves.min(), row_waves.max()],
                              aspect='auto',
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)

        ax.set_xlim(wavelengths.min(), wavelengths.max())
        ax.set_ylim(wavelengths.min(), wavelengths.max())
        fig.colorbar(im, ax=ax, label=metric)

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('wavelength (nm)')
        ax.set_title(_trait_from_job_name(job_dir.name))
        yield job_dir, fig, ax


def main():
    parser = argparse.ArgumentParser('analysis.')
    parser.add_argument('--pred_vs_true',
                        action='store_true',
                        help='Generate true-vs-pred plots')
    parser.add_argument('--metrics_summary',
                        action='store_true',
                        help='Make metrics summary and metrics bar plots')
    parser.add_argument('--vips',
                        action='store_true',
                        help='Make VIP plots')
    parser.add_argument('--scoefs',
                        action='store_true',
                        help='Make scoefs plots')
    parser.add_argument('--heatmap',
                        action='store_true',
                        help='Make heatmap plots')
    parser.add_argument('--rnrmse',
                        action='store_true',
                        help='With --heatmap: plot rnrmse_mean')
    parser.add_argument('--r2',
                        action='store_true',
                        help='With --heatmap: plot r2_mean')
    args = parser.parse_args()

    # No flags given: run everything, as before.
    run_all = not (args.pred_vs_true or args.metrics_summary
                   or args.vips or args.scoefs or args.heatmap)

    deploy_dir = fromchtc_dir/'deploy'
    model_dir = fromchtc_dir/'model'
    analysis_dir = fromchtc_dir/'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if run_all or args.pred_vs_true:
        for job_dir, fig, ax in plot_pred_vs_true(deploy_dir):
            fig.savefig(analysis_dir/f'{job_dir.name}.png')
            plt.close(fig)

    if run_all or args.metrics_summary:
        metrics_summary(deploy_dir)
        plot_metrics_bars(deploy_dir)

    if run_all or args.vips:
        for job_dir, fig, ax in plot_vips(model_dir):
            trait = _trait_from_job_name(job_dir.name)
            fig.savefig(analysis_dir/f'vips__{trait}.png')
            plt.close(fig)

    if run_all or args.scoefs:
        for job_dir, fig, ax in plot_scoefs(model_dir):
            trait = _trait_from_job_name(job_dir.name)
            fig.savefig(analysis_dir/f'scoefs__{trait}.png')
            plt.close(fig)

    if run_all or args.heatmap:
        metrics = [m for flag, m in [(args.rnrmse, 'rnrmse_mean'),
                                     (args.r2, 'r2_mean')] if flag]
        if not metrics:
            metrics = ['rnrmse_mean']

        for metric in metrics:
            metric_label = metric.removesuffix('_mean')
            for job_dir, fig, ax in plot_heatmaps(deploy_dir, model_dir, metric):
                trait = _trait_from_job_name(job_dir.name)
                fig.savefig(analysis_dir/f'heatmap__{metric_label}__{trait}.png')
                plt.close(fig)


if __name__ == '__main__':
    main()
