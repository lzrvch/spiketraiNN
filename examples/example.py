from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from utils import prepare_fcx1_data_sklearn_format

import spiketraiNN as spknn

data_split = prepare_fcx1_data_sklearn_format(Path('../pyspikelib/data/'), samples=10)
X_train, X_test, y_train, y_test = data_split

for features in (X_train, X_test):
    features = np.cumsum(features, axis=1)

train_metric = lambda spike_train1, spike_train2: spknn.distance(
    spike_train1, spike_train2, metric='isi'
)

knn_classifier = KNeighborsClassifier(
    n_neighbors=2, weights='uniform', metric=train_metric, n_jobs=-1
)

knn_classifier.fit(X_train, y_train)

test_predictions = knn_classifier.predict(X_test)
print(
    'Accuracy on a balanced validation set: {}'.format(
        accuracy_score(y_test, test_predictions)
    )
)
