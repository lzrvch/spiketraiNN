import numpy as np
import spiketraiNN.distance_binding as dst


def spike_train_distance(spike_train1, spike_train2, metric='isi', params=None):
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

    if metric == 'victor_purpura':
        a_parameter = params['q']
    elif metric in PARAMETER_FREE_METRICS:
        joint_train = np.concatenate([spike_train1, spike_train2], axis=0)
        a_parameter = np.min(joint_train) - safety_eps
        b_parameter = np.max(joint_train) + safety_eps

    distance = np.zeros(1, dtype='float64')
    dst.initialize_arrays(spike_train1, spike_train2, distance)
    dst.set_distance_parameters(
        a=a_parameter, b=b_parameter, n1=spike_train1.shape[0], n2=spike_train2.shape[0]
    )
    dst.compute(metric)
    return distance[0]
