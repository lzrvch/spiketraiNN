cimport numpy as np
import numpy as np
import cython

cdef extern from "distance.h":
    void setDistanceParameters(double parameter_a, double parameter_b, int nSpikes1, int nSpikes2)
    void initializeArrays(double* SpikeTrain1, double* SpikeTrain2, double* distance)
    void computeVPDistance()
    void computeSrbrDistance()
    void computeISIDistance()
    void computeSPIKEDistance()
    void computeVRDistance()
    void computeMaxMetricDistance()
    void computeDTWDistance()
    void computeModulusMetricDistance()

def set_distance_parameters(a, b, n1, n2):
    setDistanceParameters(a, b, n1, n2)

def initialize_arrays(np.ndarray[np.float64_t, ndim=1] spike_train_1,
                      np.ndarray[np.float64_t, ndim=1] spike_train_2,
                      np.ndarray[np.float64_t, ndim=1] distance):

    initializeArrays(<double*> spike_train_1.data, <double*> spike_train_2.data, <double*> distance.data)

def compute(method):
    if method == 'victor_purpura':
        computeVPDistance()
    elif method == 'schreiber':
        computeSrbrDistance()
    elif method == 'isi':
        computeISIDistance()
    elif method == 'spike':
        computeSPIKEDistance()
    elif method == 'van_rossum':
        computeVRDistance()
    elif method == 'max_metric':
        computeMaxMetricDistance()
    elif method == 'dtw':
        computeDTWDistance()
    elif method == 'modulus_metric':
        computeModulusMetricDistance()
