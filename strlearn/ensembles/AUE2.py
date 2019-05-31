import numpy as np
from attr import attrs, attrib
from copy import deepcopy

from strlearn.ensembles.voting import WeightedMajorityPredictionCombiner

EPS = np.finfo(float).eps


@attrs
class AUE2:
    _base_estimator = attrib()
    _ensemble_size = attrib(default=20)

    _ensemble = attrib(factory=list, init=False)
    _classes = attrib(factory=list, init=False)
    _ensemble_weights = attrib(factory=list, init=False)

    @staticmethod
    def mse_random_classifier(y):
        batch_size = len(y)
        classes, counts = np.unique(y, return_counts=True)
        return np.sum([counts[idx] / batch_size * (1 - counts[idx] / batch_size) ** 2 for idx, _ in enumerate(classes)])

    @staticmethod
    def mse(classes, y_true, y_supports):
        batch_size = len(y_true)

        y_true_index_in_clf = [np.where(classes == y)[0][0] for y in y_true]

        return 1.0 / batch_size * np.sum(
            [(1 - y_supports[idx][y_true_index_in_clf[idx]]) ** 2 for idx in range(batch_size)])

    def partial_fit(self, x, y, classes):
        self._classes = classes

        new_clf = deepcopy(self._base_estimator)
        new_clf.fit(x, y)

        random_clf_mse = AUE2.mse_random_classifier(y)

        clf_weight = 1.0 / (random_clf_mse + EPS)

        mses = []
        for idx, clf in enumerate(self._ensemble):
            y_supports = clf.predict_proba(x)
            clf_mse = AUE2.mse(clf.classes_, y, y_supports)
            mses.append(clf_mse)
            self._ensemble_weights[idx] = 1.0 / (random_clf_mse + clf_mse + EPS)

        if len(self._ensemble) < self._ensemble_size:
            self._ensemble.append(new_clf)
            self._ensemble_weights.append(clf_weight)
        else:
            worst_classifier_idx = np.argmax(mses, axis=0)
            self._ensemble[worst_classifier_idx] = new_clf
            self._ensemble_weights[worst_classifier_idx] = clf_weight

        [clf.partial_fit(x, y, classes=self._classes) for clf in self._ensemble if clf != new_clf]

    def predict(self, x):
        current_weights = [self._ensemble_weights[k] for k in range(len(self._ensemble))]
        ensemble_predictions_combiner = WeightedMajorityPredictionCombiner(
            self._ensemble, current_weights, self._classes)

        return ensemble_predictions_combiner.predict(x)
