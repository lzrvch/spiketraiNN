import numpy as np
import distance_binding as dst


def spike_train_distance(spike_train1, spike_train2,
                         method='victor_purpura',
                         params=None):
    no_param_methods = ['schreiber',
                        'isi',
                        'spike',
                        'van_rossum',
                        'max_metric',
                        'modulus_metric']

    if method == 'victor_purpura':
        a = params['q']
        b = 0.
    elif method in no_param_methods:
        joint_train = np.concatenate([spike_train1, spike_train2], axis=0)
        a = np.min(joint_train) - 0.1
        b = np.max(joint_train) + 0.1
    else:
        a = 0.
        b = 0.

    distance = np.zeros(1, dtype='float64')
    dst.initialize_arrays(spike_train1, spike_train2, distance)
    dst.set_distance_parameters(a=a,
                                b=b,
                                n1=spike_train1.shape[0],
                                n2=spike_train2.shape[0])
    dst.compute(method)
    return distance[0]
