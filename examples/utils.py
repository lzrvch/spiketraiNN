from pathlib import Path
from typing import Sequence

import numpy as np
import pyspikelib.utils as spkutil


def prepare_fcx1_data_sklearn_format(datapath: Path, samples: int = 20) -> Sequence:
    wake_data = spkutil.load_parquet(datapath / 'wake.parq')
    sleep_data = spkutil.load_parquet(datapath / 'sleep.parq')

    data = {}
    data['wake_train'], data['wake_test'] = spkutil.split_by_spikes(
        wake_data, ratio=0.5
    )
    data['sleep_train'], data['sleep_test'] = spkutil.split_by_spikes(
        sleep_data, ratio=0.5
    )

    window_size = 100
    step_size = 100
    total_samples = 5000

    crop_data = {}
    for key in data:
        crop_data[key] = spkutil.crop_isi_samples(
            data[key],
            window_size=window_size,
            step_size=step_size,
            total_samples=total_samples,
        )

    print(
        'Dataset size: {}'.format(
            [(key, crop_data[key]['series'].shape) for key in crop_data]
        )
    )

    indices = np.random.choice(total_samples, samples)

    X_train = np.concatenate(
        [
            crop_data['wake_train']['series'][indices, :],
            crop_data['sleep_train']['series'][indices, :],
        ]
    )
    y_train = np.array([0] * indices.shape[0] + [1] * indices.shape[0])

    X_test = np.concatenate(
        [
            crop_data['wake_test']['series'][indices, :],
            crop_data['sleep_test']['series'][indices, :],
        ]
    )
    y_test = np.array([0] * indices.shape[0] + [1] * indices.shape[0])

    return X_train, X_test, y_train, y_test
