# Simple example for fcx1 WAKE/SLEEP data and kNN classifier
# choose SPIKE_TRAIN_METRIC from: 'isi', 'victor_purpura', 'schreiber',
# 'spike', 'van_rossum', 'max_metric', 'modulus_metric'

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from utils import fcx1_data_sklearn_format

import spiketraiNN as spknn

np.random.seed(0)

SPIKE_TRAIN_METRIC = 'schreiber'

data_split = fcx1_data_sklearn_format(Path('../pyspikelib/data/'), samples=1000)
X_train, X_test, y_train, y_test = data_split

for features in (X_train, X_test):
    features = np.cumsum(features, axis=1)

train_metric = lambda spike_train1, spike_train2: spknn.distance(
    spike_train1, spike_train2, metric=SPIKE_TRAIN_METRIC
)

print('Fitting classifier...')

knn_classifier = KNeighborsClassifier(
    n_neighbors=2, weights='uniform', metric=train_metric, n_jobs=-1
)

knn_classifier.fit(X_train, y_train)

print('Getting test predictions...')

test_predictions = knn_classifier.predict(X_test)
print(
    'Accuracy on a balanced validation set: {}'.format(
        accuracy_score(y_test, test_predictions)
    )
)
