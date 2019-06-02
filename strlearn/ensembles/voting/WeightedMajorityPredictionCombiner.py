import numpy as np
from attr import attrs, attrib
from strlearn.ensembles.voting import BaseEnsemblePredictionCombiner


@attrs
class WeightedMajorityPredictionCombiner(BaseEnsemblePredictionCombiner):
    _ensemble = attrib()
    _weights = attrib()
    _classes = attrib()

    def predict(self, x):
        all_members_can_return_supports = all([hasattr(clf, 'predict_proba') for clf in self._ensemble])

        if all_members_can_return_supports:
            supports_by_clf = [clf.predict_proba(x) * weight for (weight, clf) in zip(self._weights, self._ensemble)]
            supports_sum_by_sample = sum(supports_by_clf)
            predictions = [self._classes[idx] for idx in np.argmax(supports_sum_by_sample, axis=1)]

        else:
            predictions_by_clf = [clf.predict(x) for clf in self._ensemble]
            supports_by_clf = [
                np.vstack(
                    [(predictions == clazz).T * weight for clazz in self._classes]
                ) for (weight, predictions) in zip(self._weights, predictions_by_clf)
            ]

            supports_sum_by_sample = sum(supports_by_clf)

            predictions = [self._classes[idx] for idx in np.argmax(supports_sum_by_sample, axis=0)]

        return np.array(predictions)
