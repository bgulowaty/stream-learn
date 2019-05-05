from attr import attrs, attrib
import numpy as np
from copy import deepcopy

EPS = np.finfo(float).eps


@attrs
class LearnNSE:
    _base_estimator = attrib()
    _a = attrib(default=0.5)
    _b = attrib(default=15)

    _skts = []
    _bkts = []
    _wkts = []
    _skts_est = []
    _classes = []
    _ensemble = []

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
            good_predictions_count = len(y_pred) - wrong_predictions_count

            ensemble_error = wrong_predictions_count / batch_size

            weights = np.array([ensemble_error / batch_size if prediction_idx in good_predictions_indicies
                                else 1.0 / batch_size
                                for prediction_idx, _ in enumerate(y_pred)])

            D = weights / weights.sum()

        classifier = deepcopy(self._base_estimator)
        classifier.fit(x, y, sample_weight=D)
        self._ensemble.append(classifier)

        self._skts.append([])
        self._bkts.append([])
        self._wkts.append([])

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

            skt = 1.0 / (1.0 + np.exp(-self._a * (t - k - self._b)))
            skt = skt / (np.sum(self._skts[k]) + skt)
            self._skts[k].append(skt)

            bkt_est = np.sum(self._skts[k]) * np.sum(self._bkts[k])
            self._bkts[k].append(bkt_est)

            wkt = np.log(1.0 / (bkt_est + EPS))
            self._wkts[k].append(wkt)

    def predict(self, x):
        ensemble_predictions = []
        for k, estimator in enumerate(self._ensemble):
            y_pred = estimator.predict_proba(x)
            ensemble_predictions.append(self._wkts[k][-1] * y_pred)
        ensemble_predictions = np.array(ensemble_predictions).sum(axis=0)
        predicted_classes_indicies = np.argmax(ensemble_predictions, axis=1)
        return np.array([self._classes[idx] for idx in predicted_classes_indicies])
