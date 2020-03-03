# spiketraiNN: a lightweight Python library for spike train distance calculation.

By [Ivan Lazarevich](https://lazarevi.ch)

# Introduction

This is a Python wrapper for a variety of spike train distance functors ([C++ implementation](https://github.com/rist-ro/spike-train-metrics) as well as generic DTW distance for time series [C++ implementation](https://github.com/lemire/lbimproved/blob/master/dtw.h)).

Available distance metrics include:

  - Victor-Purpura distance
  - Schreiber distance
  - ISI distance
  - SPIKE distance
  - Van Rossum distance
  - MaxMetric distance
  - ModulusMetric distance
  - DTW distance


# Installation

```
pip install -r requirements.txt
cd ./spiketrainn
make
```

# Usage

```
import numpy as np
import spiketrainn as spknn

n_points = 100
first_train = np.cumsum(np.random.uniform(10, 40, n_points))
second_train = np.cumsum(np.random.uniform(10, 40, n_points))

distance = spknn.distance(
    first_train, second_train, metric='isi'
)
```
