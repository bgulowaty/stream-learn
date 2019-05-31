import numpy as np
from attr import attrs, attrib
from copy import deepcopy
from strlearn.ensembles.voting import WeightedMajorityPredictionCombiner
from strlearn.utils import split_minority_majority
from collections import defaultdict


@attrs
class MuSERA:
    _base_estimator = attrib()
    _post_balance_ratio = attrib(default=0.5)

    _ensemble_size = attrib(default=20)

    _minority_instances = attrib(factory=lambda: defaultdict(
        lambda: {
            'x': [],
            'y': []
        }
    ), init=False)
    _current_iteration = attrib(default=0, init=False)

    _ensemble = attrib(factory=list, init=False)
    _ensemble_weights = attrib(factory=list, init=False)
    _enough_samples_in_aggregate = attrib(default=False)

    def _mahalanobis(self, X, variable):
        variable = np.array(variable)
        cov_x = np.cov(X, rowvar=False)
        mu = np.mean(X, axis=0)
        return np.sqrt((variable - mu).T @ np.linalg.pinv(cov_x) @ (variable - mu))

    def partial_fit(self, x, y, classes):
        self._current_iteration += 1
        self._classes = classes
        batch_size = len(x)

        X_min, Y_min, X_maj, Y_maj = split_minority_majority(x, y)
        minority_class = Y_min[0]
        aggregated_minority_instances = self._minority_instances[minority_class]

        # CZY NA PEWNO
        current_imbalance_ratio = len(X_min)/len(X_maj)

        if not self._enough_samples_in_aggregate:
            X_train = np.concatenate((x, aggregated_minority_instances['x'])) if len(aggregated_minority_instances['x']) != 0 else x
            Y_train = np.concatenate((y, aggregated_minority_instances['y'])) if len(aggregated_minority_instances['y']) != 0 else y
        else:
            d = []
            for min_x in aggregated_minority_instances['x']:
                d.append(self._mahalanobis(X_min, min_x))

            d_sorted_indices = np.argsort(d)
            samples_to_pick = (self._post_balance_ratio - current_imbalance_ratio) * batch_size

            M_x = np.array(aggregated_minority_instances['x'])[d_sorted_indices[:int(samples_to_pick)]]
            M_y = np.array(aggregated_minority_instances['y'])[d_sorted_indices[:int(samples_to_pick)]]

            X_train = np.concatenate((x, M_x), axis=0)
            Y_train = np.concatenate((y, M_y), axis=0)

        minority_sampled_added = len(X_train) - len(x)

        new_ratio = (minority_sampled_added + len(X_min)) / len(X_maj)
        if new_ratio > self._post_balance_ratio:
            self._enough_samples_in_aggregate = True

        new_clf = deepcopy(self._base_estimator)
        new_clf.fit(X_train, Y_train)

        self._ensemble.append(new_clf)
        self._ensemble_weights.append(0)

        # Compute weights
        for idx, clf in enumerate(self._ensemble):
            supports = clf.predict_proba(x)

            y_true_index_in_clf_classes = [np.where(clf.classes_ == y_true)[0][0] for y_true in y]

            e = 1.0 / batch_size * \
                 np.sum([
                     (1.0 - supports[sample_idx][y_true_index_in_clf_classes[sample_idx]]) ** 2
                     for sample_idx in range(batch_size)
                 ])
            w = np.log(1.0/e)
            self._ensemble_weights[idx] = w


        # Prune worst classifier
        if len(self._ensemble) == self._ensemble_size:
            worst_classifier_index = np.argmin(self._ensemble_weights[:-1])
            del self._ensemble[worst_classifier_index]
            del self._ensemble_weights[worst_classifier_index]

        # Aggregate new samples
        for (x, y) in zip(X_min, Y_min):
            self._minority_instances[minority_class]['x'].append(x)
            self._minority_instances[minority_class]['y'].append(y)


    def predict(self, x):
        ensemble_predictions_combiner = WeightedMajorityPredictionCombiner(
            self._ensemble, self._ensemble_weights, self._classes)

        return ensemble_predictions_combiner.predict(x)
