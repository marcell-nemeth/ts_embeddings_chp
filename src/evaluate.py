import argparse
from pathlib import Path
from typing import Callable, List

import esig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib.axes import Axes
from pyts.approximation import (DiscreteFourierTransform,
                                PiecewiseAggregateApproximation)
from tssb.utils import load_time_series_segmentation_datasets

from src.models import AutoEncoder, LSTMAutoEncoder, TiDE, DeepTCN, train
from src.common import (calculate_inter, calculate_intra, evaluate,
                        extract_segments, split_into_windows)


SEED = None

ROOT_PATH = Path(__file__).parent.parent

# Where to save plots
PLOTS_DIR = ROOT_PATH / 'results/outputs/plots'

# Where to save tabular data
TABULAR_DIR = ROOT_PATH / 'results/outputs/tabular'


DATASET_PARAMS = {
    'ECG200': {'wsize': 40, 'epochs': 600, 'batch_size': 16}, # 2 classes
    'SonyAIBORobotSurface1': {'wsize': 60, 'epochs': 600, 'batch_size': 32},
    'SonyAIBORobotSurface2': {'wsize': 60, 'epochs': 300, 'batch_size': 32},
    'ItalyPowerDemand': {'wsize': 40, 'epochs': 600, 'batch_size': 32},
    'Trace': {'wsize': 80, 'epochs': 400, 'batch_size': 32}, # 3 classes
    'EOGHorizontalSignal': {'wsize': 80, 'epochs': 800, 'batch_size': 128},
    'Haptics': {'wsize': 70, 'epochs': 800, 'batch_size': 32},
    'CBF': {'wsize': 40, 'epochs': 400, 'batch_size': 32},
    'EOGVerticalSignal': {'wsize': 80, 'epochs': 800, 'batch_size': 128}, # 4 classes
    'InsectWingbeatSound': {'wsize': 30, 'epochs': 150, 'batch_size': 16},
    'Car': {'wsize': 30, 'epochs': 1000, 'batch_size': 32},
    'SwedishLeaf': {'wsize': 100, 'epochs': 200, 'batch_size': 32},
    'UWaveGestureLibraryX': {'wsize': 40, 'epochs': 600, 'batch_size': 32}, # 5 classes
    'UWaveGestureLibraryY': {'wsize': 40, 'epochs': 600, 'batch_size': 32},
    'UWaveGestureLibraryZ': {'wsize': 40, 'epochs': 600, 'batch_size': 32},
    'SyntheticControl': {'wsize': 25, 'epochs': 800, 'batch_size': 32},
}


# The feature extraction methods that we are comparing
FE_METHODS = [
    # ('fourier', lambda *args, **kwargs: fourier_transform(*args, **kwargs)),
    # ('paa', lambda *args, **kwargs: paa_transform(*args, **kwargs)),
    # ('signature', lambda *args, **kwargs: signature_method(*args, **kwargs)),
    ('autoencoder-wo', lambda segments, **kwargs: autoencoder(
        segments, **kwargs,
        autoencoder=AutoEncoder, latent_loss=0.0
    )),
    ('autoencoder', lambda segments, **kwargs: autoencoder(
        segments, **kwargs,
        autoencoder=AutoEncoder, latent_loss=1.0
    )),
    ('lstm-ae-linear', lambda segments, **kwargs: autoencoder(
        segments, **kwargs,
        autoencoder=LSTMAutoEncoder, latent_loss=1.0
    )),
    ('deep-tcn-wo', lambda segments, **kwargs: autoencoder(
        segments, **kwargs,
        autoencoder=DeepTCN, latent_loss=0.0
    )),
    ('deep-tcn', lambda segments, **kwargs: autoencoder(
        segments, **kwargs,
        autoencoder=DeepTCN, latent_loss=1.0
    )),
    ('tide-wo', lambda segments, **kwargs: autoencoder(
        segments, **kwargs,
        autoencoder=TiDE, latent_loss=0.0
    )),
    ('tide', lambda segments, **kwargs: autoencoder(
        segments, **kwargs,
        autoencoder=TiDE, latent_loss=1.0
    )),
]


TABLE_COLUMNS = [(col, subcol)
                 for col in ['intra_sim', 'intra_dist',
                             'inter_sim', 'inter_dist']
                 for subcol in ['mean', 'std']]


def fourier_transform(segments: List[np.ndarray], **kwargs):
    segment_windows = [split_into_windows(segment, kwargs['wsize'], stride=1)
                       for segment in segments]

    dft = DiscreteFourierTransform(n_coefs=16, drop_sum=True)
    segment_features = [dft.fit_transform(windows.squeeze())
                        for windows in segment_windows]
    return segment_features


def paa_transform(segments: List[np.ndarray], **kwargs):
    segment_windows = [split_into_windows(segment, kwargs['wsize'], stride=1)
                       for segment in segments]

    paa = PiecewiseAggregateApproximation(output_size=16, window_size=None)
    segment_features = [paa.fit_transform(windows.squeeze())
                        for windows in segment_windows]
    return segment_features


def signature_method(segments: List[np.ndarray], **kwargs):
    # Augment time series with time stamps
    prev = 0
    for i, segment in enumerate(segments):
        time_stamps = np.arange(start=prev, stop=prev +
                                len(segment)).reshape((-1, 1))
        segments[i] = np.concatenate([segment, time_stamps], axis=-1)
        prev += len(segment)

    segment_windows = [split_into_windows(segment, kwargs['wsize'], stride=1)
                       for segment in segments]
    segment_features = [np.array([esig.stream2sig(window, depth=2) for window in windows])
                        for windows in segment_windows]
    return segment_features


def autoencoder(segments: List[np.ndarray], **kwargs):
    TRAIN_RATIO = 0.8

    # Chop up the data into train and test
    train_segments = [segment[:int(len(segment) * TRAIN_RATIO)]
                      for segment in segments]
    test_segments = [segment[int(len(segment) * TRAIN_RATIO):]
                     for segment in segments]

    # Create windows from test and training data
    train_windows = [torch.tensor(split_into_windows(segment, kwargs['wsize']),
                                  dtype=torch.float).squeeze()
                     for segment in train_segments]
    test_windows = [torch.tensor(split_into_windows(segment, kwargs['wsize']),
                                 dtype=torch.float).squeeze()
                    for segment in test_segments]

    # Create model and train it
    if SEED is not None:
        torch.manual_seed(seed=SEED)

    wsize = kwargs['wsize']
    architecture = kwargs['autoencoder']
    model = architecture(
        in_features=wsize, latent_features=8, hidden_size=min(30, max(20, wsize // 2))
    )
    train(model, train_windows, **kwargs)

    # Return features extracted from test windows
    model.eval()
    with torch.no_grad():
        segment_features = [model.encode(windows).numpy()
                            for windows in test_windows]
    return segment_features


def save_segment_features(fpath, segment_features: List[np.ndarray]):
    with open(fpath, 'w') as f:
        for i, features in enumerate(segment_features):
            print(f'--- Segment {i}', file=f)
            df = pd.DataFrame(features)
            df.to_csv(f, sep=';', index=False, header=False)
            print('', file=f)


def plot_sement_matrix(axis: Axes, matrix: np.ndarray, annotation: Callable[[int, int], str]):
    assert matrix.ndim == 2
    height, width = matrix.shape

    axis.imshow(matrix)
    axis.grid(False)
    axis.xaxis.set_ticks(range(width))
    axis.xaxis.set_ticklabels([f'segment {i}' for i in range(width)])
    axis.xaxis.set_ticks_position('top')
    axis.yaxis.set_ticks(range(height))
    axis.yaxis.set_ticklabels([f'segment {i}' for i in range(width)], rotation='vertical', va='center')
    for i in range(height):
        for j in range(width):
            axis.text(i, j, annotation(i, j),
                      weight='bold',
                      fontsize='small',
                      horizontalalignment='center',
                      verticalalignment='center')


def parse_args():
    parser = argparse.ArgumentParser()
    method_choices = [name for name, _ in FE_METHODS]
    parser.add_argument(
        "--methods", nargs="+", type=str, default="all", choices=method_choices,
        help=f"The methods to use for feature extraction. Available options: {', '.join(method_choices)}",
        metavar="METHOD"
    )
    dataset_choices = list(DATASET_PARAMS.keys())
    parser.add_argument(
        "--datasets", nargs="+", type=str, default="all", choices=dataset_choices, 
        help=f"The datasets to perform feature extraction on. Available options: {', '.join(dataset_choices)}",
        metavar="DATASET"
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        help=f"The seed to use for model training.",
    )
    return parser.parse_args()


def main(datasets: List[str], methods: List):
    datasets = [(
        row['dataset'],                           # Name of the dataset
        np.reshape(row['time_series'], (-1, 1)),  # The time series data
        row['change_points'])                     # The index of the change points
        for _, row in load_time_series_segmentation_datasets(datasets).iterrows()]

    # Extract features for each method and dataset
    for fe_name, method in methods:

        # Create directories to store files in
        tabular_dir = TABULAR_DIR / str(fe_name)
        tabular_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = PLOTS_DIR / str(fe_name)
        plots_dir.mkdir(parents=True, exist_ok=True)

        summary = []
        for ds_name, ts, cps in datasets:
            segments = extract_segments(ts, cps)
            num_segments = len(segments)

            params = DATASET_PARAMS[ds_name]
            segment_features = method(segments, **params)

            # Save the extracted features into csv file
            save_segment_features(tabular_dir / f'{ds_name}.csv',
                                  segment_features)

            mean_sim, mean_dist = evaluate(segment_features, agg=np.mean)
            std_sim, std_dist = evaluate(segment_features, agg=np.std)

            # Create figure from evaluation results
            fig = plt.figure(figsize=(2*num_segments + 2, num_segments + 2))
            plt.set_cmap(cm.coolwarm)
            fig.suptitle(f'{fe_name} on {ds_name}', weight='bold', fontsize=20)
            ax_sim, ax_dist = fig.subplots(nrows=1, ncols=2)
            ax_sim.set_title('Similarity')
            plot_sement_matrix(
                ax_sim, mean_sim,
                lambda i, j: '\n'.join([
                    f'avg: {mean_sim[i,j]:.4f}',
                    f'std: {std_sim[i,j]:.4f}',
                    f'{len(segment_features[i])} - {len(segment_features[j])}',
                ]))
            ax_dist.set_title('Distance')
            plot_sement_matrix(
                ax_dist, mean_dist,
                lambda i, j: '\n'.join([
                    f'avg: {mean_dist[i,j]:.4f}',
                    f'std: {std_dist[i,j]:.4f}',
                    f'{len(segment_features[i])} - {len(segment_features[j])}',
                ]))
            fig.tight_layout()
            fig.savefig(plots_dir / f'{ds_name}.png')
            plt.close(fig)

            # Add results to summarise
            summary.append([
                fn(matrix)
                for fn in [calculate_intra, calculate_inter]
                for matrix in [mean_sim, std_sim, mean_dist, std_dist]])

        # Save summary into .csv and .tex file
        df = pd.DataFrame(
            data=summary,
            index=[ds_name for ds_name, *_ in datasets],
            columns=pd.MultiIndex.from_tuples(TABLE_COLUMNS))
        df.to_csv(tabular_dir / 'summary.csv', sep=';', index_label='dataset')
        # df.style.to_latex(tabular_dir / 'summary.tex', position_float='centering', hrules=True, multicol_align='c')

        print(f'--- {fe_name} ---')
        print(df)
        print()


if __name__ == '__main__':
    args = parse_args()
    main(
        datasets=(
            args.datasets if args.datasets != "all" else list(DATASET_PARAMS.keys())
        ),
        methods=(
            FE_METHODS
            if args.methods == "all"
            else [method for method in FE_METHODS if method[0] in args.methods]
        ),
    )
