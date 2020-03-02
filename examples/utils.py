import numpy as np
import pandas as pd

from pathlib import Path
from typing import Sequence

from sklearn.model_selection import GroupShuffleSplit
from pyspikelib import TrainNormalizeTransform
from examples.dataset_adapters import fcx1_dataset


def fcx1_data_sklearn_format(datapath: Path, samples: int = 20) -> Sequence:
    window, step = 20, 20
    wake_spikes = fcx1_dataset(datapath / 'wake.parq')
    sleep_spikes = fcx1_dataset(datapath / 'sleep.parq')

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
    X = np.hstack([wake_spikes.series.values, sleep_spikes.series.values])
    y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(sleep_spikes.shape[0])])
    groups = np.hstack([wake_spikes.groups.values, sleep_spikes.groups.values])

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})
    normalizer = TrainNormalizeTransform(window=window, step=step, n_samples=samples)
    X_train, y_train = normalizer.transform(X_train, y_train, delimiter=',')
    X_test, y_test = normalizer.transform(X_test, y_test, delimiter=',')
    return X_train, X_test, y_train, y_test
