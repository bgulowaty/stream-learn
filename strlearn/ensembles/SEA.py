from attr import attrs, attrib
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from copy import deepcopy

from strlearn.ensembles.voting import MajorityPredictionCombiner

EPS = np.finfo(float).eps


@attrs
class SEA:
    _base_estimator = attrib()
    _ensemble_size = attrib(default=15)
    _ensemble = attrib(factory=list, init=False)
    _ensemble_quality_measurer = attrib(default=balanced_accuracy_score)
    _classes = attrib(default=[], init=False)

    def partial_fit(self, x, y, classes):
        self._classes = classes

        new_clf = deepcopy(self._base_estimator)
        new_clf.fit(x, y)

        if len(self._ensemble) < self._ensemble_size:
            self._ensemble.append(new_clf)
        else:
            new_clf_quality = self._ensemble_quality_measurer(y, new_clf.predict(x))

            ensemble_qualities = [self._ensemble_quality_measurer(y, clf.predict(x)) for clf in self._ensemble]

            worst_model_idx = np.argsort(ensemble_qualities)[0]

            if new_clf_quality >= ensemble_qualities[worst_model_idx]:
                self._ensemble[worst_model_idx] = new_clf

    def predict(self, x):
        ensemble_predictions_combiner = MajorityPredictionCombiner(self._ensemble, self._classes)

        return ensemble_predictions_combiner.predict(x)
