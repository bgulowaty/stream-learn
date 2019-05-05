import numpy as np
from attr import attrs, attrib
from copy import deepcopy


@attrs
class SERA:
    _base_estimator = attrib()
    _post_balance_ratio = attrib(default=0.5)

    _ensemble_size = attrib(default=15)

    _minority_instances_x = []
    _minority_instances_y = []
    _current_iteration = 0

    _ensemble = []
    _minority_samples_aggregate = {}

    def _mahalanobis(self, X, variable):
        variable = np.array(variable)
        cov_x = np.cov(X, rowvar=False)
        mu = np.mean(X, axis=0)
        return np.sqrt((variable - mu).T @ np.linalg.pinv(cov_x) @ (variable - mu))

    def _split_minority_majority(self, x, y):
        classes, counts = np.unique(y, return_counts=True)
        sorted_indicies = np.argsort(counts)
        sorted_classes = classes[sorted_indicies]

        minority_class = sorted_classes[0]
        other_classes = sorted_classes[1:]

        minority_class_indicies = np.isin(y, minority_class)
        other_classes_indicies = np.isin(y, other_classes)

        return x[minority_class_indicies], y[minority_class_indicies], x[other_classes_indicies], y[
            other_classes_indicies]

    def partial_fit(self, x, y, classes):
        self._current_iteration += 1
        self._classes = classes

        X_min, Y_min, X_maj, Y_maj = self._split_minority_majority(x, y)
        minority_class = Y_min[0]

        # CZY NA PEWNO
        imbalance_ratio = len(Y_min) / len(y)
        if self._post_balance_ratio > (self._current_iteration - 1) * imbalance_ratio:
            X_train = np.concatenate((x, self._minority_instances_x), axis=0) if len(
                self._minority_instances_x) > 0 else x
            Y_train = np.concatenate((y, self._minority_instances_y), axis=0) if len(
                self._minority_instances_y) > 0 else y
        else:
            d = []
            for (min_x_sample, min_y_sample) in zip(self._minority_instances_x, self._minority_instances_y):
                d.append(self._mahalanobis(X_min, min_x_sample))

            d = np.mean(d)
            d_sorted_indicies = np.argsort(d)
            samples_to_pick = (self._post_balance_ratio - imbalance_ratio) * self._current_iteration
            M_x = self._minority_instances_x[d_sorted_indicies[:int(samples_to_pick)]]
            M_y = self._minority_instances_y[d_sorted_indicies[:int(samples_to_pick)]]

            X_train = np.concatenate((x, M_x), axis=0)
            Y_train = np.concatenate((y, M_y), axis=0)

        clf = deepcopy(self._base_estimator)
        clf.fit(X_train, Y_train)

        self._ensemble.append(clf)
        if len(self._ensemble) > self._ensemble_size:
            self._ensemble = np.delete(self._ensemble, 0, axis=0).tolist()

        self._minority_instances_x = np.append(self._minority_instances_x, X_min, axis=0) if len(
            self._minority_instances_x) > 0 else X_min
        self._minority_instances_y = np.append(self._minority_instances_y, Y_min, axis=0) if len(
            self._minority_instances_y) > 0 else Y_min

    def predict(self, x):
        ensemble_predictions = []
        for k, estimator in enumerate(self._ensemble):
            y_pred = estimator.predict_proba(x)
            ensemble_predictions.append(y_pred)
        ensemble_predictions = np.array(ensemble_predictions).sum(axis=0)
        predicted_classes_indicies = np.argmax(ensemble_predictions, axis=1)
        return np.array([self._classes[idx] for idx in predicted_classes_indicies])
