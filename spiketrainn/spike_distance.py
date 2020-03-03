import numpy as np
from typing import Tuple
import spiketrainn.distance_binding as dst


class SpikeDistance:
    """Calculate spike train distance according to the metric name
        stored in the 'metric' attribute
    """

    def __init__(self, metric: str, safety_eps: float = 1e-1) -> None:
        self.metric = metric
        self.safety_eps = safety_eps

    def _initialize_metric_parameters(
        self, first_train: np.ndarray, second_train: np.ndarray, q: float
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
        a_parameter, b_parameter = 0., 0.
        if self.metric == 'victor_purpura':
            a_parameter = q
        elif self.metric in PARAMETER_FREE_METRICS:
            joint_train = np.concatenate([first_train, second_train], axis=0)
            a_parameter = np.min(joint_train) - self.safety_eps
            b_parameter = np.max(joint_train) + self.safety_eps

        return a_parameter, b_parameter

    def _compute(
        self, first_train: np.ndarray, second_train: np.ndarray, q: str = None
    ) -> float:
        distance = np.zeros(1, dtype='float64')
        a_parameter, b_parameter = self._initialize_metric_parameters(
            first_train, second_train, q
        )
        dst.initialize_arrays(first_train, second_train, distance)
        dst.set_distance_parameters(
            a=a_parameter,
            b=b_parameter,
            n1=first_train.shape[0],
            n2=second_train.shape[0],
        )
        dst.compute(self.metric)
        return distance[0]

    def __call__(
        self, first_train: np.ndarray, second_train: np.ndarray, q: float = None
    ) -> float:
        return self._compute(first_train, second_train, q=q)


def spike_train_distance(
    first_train: np.ndarray,
    second_train: np.ndarray,
    metric: str = 'isi',
    q: float = None,
) -> float:
    distance = SpikeDistance(metric=metric)
    return distance(first_train, second_train, q=q)
