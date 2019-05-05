from attr import attrs, attrib
from imblearn.under_sampling import RandomUnderSampler
from copy import deepcopy
import numpy as np


@attrs
class SE:
    _base_estimator = attrib()
    _sampling_strategy = attrib(default='distance')  # or 'uniform'
    _sampling_rate = attrib(default=0.3)
    _minority_sample_batches_kept = attrib(default=20)
    _ensemble_size = attrib(default=30)
    _models_created_each_iteration = attrib(default=3)

    def __attrs_post_init__(self):
        self._undersampler = RandomUnderSampler(sampling_strategy=1.0 / self._models_created_each_iteration)

    _ensemble = []
    _minority_samples_aggregate = {}

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
        if len(np.unique(y)) < 2:
            return

        self._classes = classes

        X_min, Y_min, X_maj, Y_maj = self._split_minority_majority(x, y)
        minority_class = Y_min[0]
        if minority_class in self._minority_samples_aggregate:
            self._minority_samples_aggregate[minority_class].append((X_min, Y_min))
        else:
            self._minority_samples_aggregate[minority_class] = [(X_min, Y_min)]

        if len(self._minority_samples_aggregate[minority_class]) > self._minority_sample_batches_kept:
            self._minority_samples_aggregate[minority_class] = np.delete(
                self._minority_samples_aggregate[minority_class], 0, axis=0).tolist()

        X_train = []
        Y_train = []
        # Add minority class aggregate to training chunk
        m = len(self._minority_samples_aggregate[minority_class])
        #         print(self._minority_samples_aggregate[minority_class])
        for idx, (chunk_x, chunk_y) in enumerate(self._minority_samples_aggregate[minority_class]):
            if self._sampling_strategy is 'distance':
                r = 1 - (m - idx + 1) * self._sampling_rate
                print(f"len = {m}, idx = {idx}, r = {r}")
            elif self._sampling_strategy is 'uniform':
                r = self._sampling_rate

            samples_count_to_pick = int(np.ceil(len(chunk_x) * 2 * r))
            randomly_choosen_indicies = np.random.choice(range(len(chunk_x)), samples_count_to_pick, replace=False)
            X_train = np.concatenate((X_train, chunk_x[randomly_choosen_indicies])) if len(X_train) > 0 else chunk_x[
                randomly_choosen_indicies]
            Y_train = np.concatenate((Y_train, chunk_y[randomly_choosen_indicies])) if len(Y_train) > 0 else chunk_y[
                randomly_choosen_indicies]

        # flip X_Train and take only most recent elements to make len X_maj = param * len X_min
        X_train = X_train[::-1][:int(len(X_maj) / self._models_created_each_iteration)]
        Y_train = Y_train[::-1][:int(len(Y_maj) / self._models_created_each_iteration)]

        # Undersample majority class to match ratio defined by _models_created_each_iteration
        X_train_undersampled, Y_train_undersampled = self._undersampler.fit_resample(
            np.concatenate((X_maj, X_train)), np.concatenate((Y_maj, Y_train))
        )

        X_train_min, Y_train_min, X_maj_undersampled, Y_maj_undersampled = self._split_minority_majority(
            X_train_undersampled, Y_train_undersampled)

        train_major_x_split = np.array_split(X_maj_undersampled, self._models_created_each_iteration)
        train_major_y_split = np.array_split(Y_maj_undersampled, self._models_created_each_iteration)

        for (X_major_train, Y_major_train) in zip(train_major_x_split, train_major_y_split):
            clf = deepcopy(self._base_estimator)
            clf.fit(np.concatenate([X_major_train, X_train_min]), np.concatenate([Y_major_train, Y_train_min]))
            self._ensemble = np.append(self._ensemble, clf)
            if len(self._ensemble) > self._ensemble_size:
                self._ensemble = np.delete(self._ensemble, 0, axis=0)

    def predict(self, x):
        ensemble_predictions = []
        for k, estimator in enumerate(self._ensemble):
            y_pred = estimator.predict_proba(x)
            ensemble_predictions.append(y_pred)
        ensemble_predictions = np.array(ensemble_predictions).sum(axis=0)
        predicted_classes_indicies = np.argmax(ensemble_predictions, axis=1)
        return np.array([self._classes[idx] for idx in predicted_classes_indicies])


