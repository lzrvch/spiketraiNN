import numpy as np
from typing import Tuple
import spiketraiNN.distance_binding as dst


class SpikeDistance:
    """Calculate spike train distance according to the metric name
        stored in the 'metric' attribute
    """

    def __init__(self, metric: str) -> None:
        self.metric = metric

    def _initialize_metric_parameters(
        self, spike_train1: np.ndarray, spike_train2: np.ndarray, q: float
    ) -> Tuple[float]:
        """Set external parameters used for metric calculation"""
        PARAMETER_FREE_METRICS = [
            'schreiber',
            'isi',
            'spike',
            'van_rossum',
            'max_metric',
            'modulus_metric',
        ]
        safety_eps = 1e-1
        a_parameter, b_parameter = 0, 0
        if self.metric == 'victor_purpura':
            a_parameter = q
        elif self.metric in PARAMETER_FREE_METRICS:
            joint_train = np.concatenate([spike_train1, spike_train2], axis=0)
            a_parameter = np.min(joint_train) - safety_eps
            b_parameter = np.max(joint_train) + safety_eps

        return a_parameter, b_parameter

    def _compute(
        self, spike_train1: np.ndarray, spike_train2: np.ndarray, q: str = None
    ) -> float:
        distance = np.zeros(1, dtype='float64')
        a_parameter, b_parameter = self._initialize_metric_parameters(
            spike_train1, spike_train2, q
        )
        dst.initialize_arrays(spike_train1, spike_train2, distance)
        dst.set_distance_parameters(
            a=a_parameter,
            b=b_parameter,
            n1=spike_train1.shape[0],
            n2=spike_train2.shape[0],
        )
        dst.compute(self.metric)
        return distance[0]

    def __call__(
        self, spike_train1: np.ndarray, spike_train2: np.ndarray, q: float = None
    ) -> float:
        return self._compute(spike_train1, spike_train2, q=q)


def spike_train_distance(
    spike_train1: np.ndarray,
    spike_train2: np.ndarray,
    metric: str = 'isi',
    q: float = None,
) -> float:
    distance = SpikeDistance(metric=metric)
    return distance(spike_train1, spike_train2, q=q)
