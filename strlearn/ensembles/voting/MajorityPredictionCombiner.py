import numpy as np
from attr import attrs, attrib
from functools import reduce

from strlearn.ensembles.voting import BaseEnsemblePredictionCombiner


@attrs
class MajorityPredictionCombiner(BaseEnsemblePredictionCombiner):

    _ensemble = attrib()
    _classes = attrib()

    def predict(self, x):
        all_members_can_return_supports = all([hasattr(clf, 'predict_proba') for clf in self._ensemble])

        if all_members_can_return_supports:
            supports_by_clf = [clf.predict_proba(x) for clf in self._ensemble]
            supports_sum_by_sample = list(reduce(lambda left, right: left + right, supports_by_clf))
            predictions = [self._classes[idx] for idx in np.argmax(supports_sum_by_sample, axis=1)]

            return np.array(predictions)
        else:
            predictions = [{class_name: 0 for class_name in self._classes} for _ in range(len(x))]

            for k, clf in enumerate(self._ensemble):
                y_pred = clf.predict(x)
                for sample_idx, sample_pred in enumerate(y_pred):
                    predictions[sample_idx][sample_pred] += 1

            actual_predictions = []
            for sample_predictions in predictions:
                actual_predictions.append(max(sample_predictions, key=sample_predictions.get))

            return np.array(actual_predictions)
