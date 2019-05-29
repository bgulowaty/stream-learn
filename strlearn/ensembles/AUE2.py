from attr import attrs, attrib
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy

from strlearn.ensembles.voting import WeightedMajorityPredictionCombiner

EPS = np.finfo(float).eps


@attrs
class AUE2:
    _base_estimator = attrib()
    _ensemble_size = attrib(default=15)
    _ensemble = attrib(default=[], init=False)
    _classes = attrib(default=[], init=False)
    _ensemble_weights = attrib(default=[], init=False)

    @staticmethod
    def mse_random_classifier(y):
        batch_size = len(y)
        classes, counts = np.unique(y, return_counts=True)
        return np.sum([counts[idx] / batch_size * (1 - counts[idx] / batch_size) ** 2 for idx, _ in enumerate(classes)])

    def partial_fit(self, x, y, classes):
        self._classes = classes

        new_clf = deepcopy(self._base_estimator)
        new_clf.fit(x, y)
        random_clf_mse = self.mse_random_classifier(y)
        clf_weight = 1 / (random_clf_mse + EPS)

        mses = []
        for idx, clf in enumerate(self._ensemble):
            y_pred = clf.predict(x)
            clf_mse = mean_squared_error(y, y_pred)
            mses.append(clf_mse)
            self._ensemble_weights[idx] = 1 / (random_clf_mse + clf_mse + EPS)

        if len(self._ensemble) < self._ensemble_size:
            self._ensemble.append(new_clf)
            self._ensemble_weights.append(clf_weight)
        else:
            worst_classifier_idx = np.argmax(mses, axis=0)
            self._ensemble[worst_classifier_idx] = new_clf
            self._ensemble_weights[worst_classifier_idx] = random_clf_mse

        for clf_idx in range(len(self._ensemble) - 1):
            clf = self._ensemble[clf_idx]
            try:
                clf.partial_fit(x, y, classes=self._classes)
            except (NotImplementedError, AttributeError) as e:
                clf.fit(x, y)

    def predict(self, x):
        current_weights = [self._ensemble_weights[k] for k in range(len(self._ensemble))]
        ensemble_predictions_combiner = WeightedMajorityPredictionCombiner(
            self._ensemble, current_weights, self._classes)

        return ensemble_predictions_combiner.predict(x)
