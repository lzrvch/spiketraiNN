import pylab as p
import numpy as np
import pandas as pd
import spiketraiNN as spknn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pathlib import Path
import pyspikelib.utils as spkutil
import pyspikelib.mpladeq as mpladeq

mpladeq.beautify_mpl()


datapath = Path("../pyspikelib/data/")
wake_data = spkutil.load_parquet(datapath / "wake.parq")
sleep_data = spkutil.load_parquet(datapath / "sleep.parq")

data = {}

data["wake_train"], data["wake_test"] = spkutil.split_by_spikes(wake_data, ratio=0.5)
data["sleep_train"], data["sleep_test"] = spkutil.split_by_spikes(sleep_data, ratio=0.5)

p.plot(data["wake_train"]["series"][5][:1000])
mpladeq.prettify()
p.xlabel("ISI # in the spike train")
p.ylabel("ISI value, ms")

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


[(key, crop_data[key]["series"].shape) for key in crop_data]

samples = 1000
indices = np.random.choice(total_samples, samples)

X_train = np.concatenate(
    [
        crop_data["wake_train"]["series"][indices, :],
        crop_data["sleep_train"]["series"][indices, :],
    ]
)
y_train = np.array([0] * indices.shape[0] + [1] * indices.shape[0])

X_test = np.concatenate(
    [
        crop_data["wake_test"]["series"][indices, :],
        crop_data["sleep_test"]["series"][indices, :],
    ]
)
y_test = np.array([0] * indices.shape[0] + [1] * indices.shape[0])

X_train = np.cumsum(X_train, axis=1)
X_test = np.cumsum(X_test, axis=1)

train_metric = lambda spike_train1, spike_train2: spknn.distance(
    spike_train1, spike_train2, metric="isi"
)

knn_classifier = KNeighborsClassifier(
    n_neighbors=5, weights="uniform", metric=train_metric, n_jobs=-1
)

knn_classifier.fit(X_train, y_train)
knn_classifier.predict(X_test)

accuracy_score(y_test, knn_classifier.predict(X_test))
