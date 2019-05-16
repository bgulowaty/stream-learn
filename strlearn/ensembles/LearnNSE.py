from attr import attrs, attrib
import numpy as np
from copy import deepcopy

from strlearn.ensembles.voting import WeightedMajorityPredictionCombiner

EPS = np.finfo(float).eps


@attrs
class LearnNSE:
    _base_estimator = attrib()
    _a = attrib(default=0.5)
    _b = attrib(default=10)

    def __attrs_post_init__(self):
        self._skts = []
        self._bkts = []
        self._wkts = []
        self._skts_est = []
        self._classes = []
        self._ensemble = []

    def partial_fit(self, x, y, classes):
        self._classes = classes
        batch_size = len(x)

        is_first_iteration = len(self._ensemble) == 0
        if is_first_iteration:
            D = np.ones(batch_size) / batch_size
        else:
            y_pred = self.predict(x)

            wrong_predictions_indicies = [idx for idx, pred in enumerate(y_pred) if pred != y[idx]]
            good_predictions_indicies = [idx for idx, pred in enumerate(y_pred) if pred == y[idx]]
            wrong_predictions_count = len(wrong_predictions_indicies)

            ensemble_error = wrong_predictions_count / batch_size

            weights = np.array([ensemble_error / batch_size if prediction_idx in good_predictions_indicies
                                else 1.0 / batch_size
                                for prediction_idx, _ in enumerate(y_pred)])

            D = weights / (weights.sum() + EPS)

        classifier = deepcopy(self._base_estimator)
        classifier.fit(x, y, sample_weight=D)
        self._ensemble.append(classifier)

        self._skts.append([])
        self._bkts.append([])
        self._wkts.append([])
        self._skts_est.append([])

        t = len(self._ensemble) - 1
        for k, classifier in enumerate(self._ensemble):
            y_pred = classifier.predict(x)
            wrong_predictions_indicies = [idx for idx, pred in enumerate(y_pred) if pred != y[idx]]

            ekt = (D * [1 if prediction_idx in wrong_predictions_indicies else 0
                        for prediction_idx, _ in enumerate(y_pred)]).sum()

            if k == t and ekt > 0.5:
                new_clf = deepcopy(self._base_estimator)
                new_clf.fit(x, y, sample_weight=D)
                self._ensemble[k] = new_clf
            elif k < t and ekt > 0.5:
                ekt = 0.5

            bkt = ekt / (1.0 - ekt)
            self._bkts[k].append(bkt)

            skt = 1.0 / (1.0 + np.exp(-self._a * (t - k - self._b)))
            skt = skt / (np.flip(self._skts[k])[:t - k].sum() + EPS)
            self._skts[k].append(skt)

            bkt_est = np.sum(np.flip(self._skts[k])[:t - k] * np.flip(self._bkts[k])[:t - k])

            wkt = np.log(1.0 / (bkt_est + EPS))
            self._wkts[k].append(wkt)

    def predict(self, x):
        current_weights = [self._wkts[k][-1] for k in range(len(self._ensemble))]
        ensemble_predictions_combiner = WeightedMajorityPredictionCombiner(
            self._ensemble, current_weights, self._classes)

        return ensemble_predictions_combiner.predict(x)
