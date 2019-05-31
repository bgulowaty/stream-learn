import numpy as np
from attr import attrs, attrib
from collections import defaultdict
from copy import deepcopy

from strlearn.ensembles.voting import MajorityPredictionCombiner
from strlearn.utils import split_minority_majority


@attrs
class SE:
    _base_estimator = attrib()
    _minority_sampling_strategy = attrib(default='fixed')  # or 'fading'
    _sampling_rate = attrib(default=0.3)
    _minority_sample_batches_kept = attrib(default=10)
    _ensemble_size = attrib(default=20)

    _ensemble = attrib(factory=list, init=False)
    _minority_samples_aggregate = attrib(factory=lambda: defaultdict(list), init=False)

    def partial_fit(self, x, y, classes):
        self._classes = classes

        X_min, Y_min, X_maj, Y_maj = split_minority_majority(x, y)

        minority_class = Y_min[0]

        self._minority_samples_aggregate[minority_class].append((X_min, Y_min))

        if len(self._minority_samples_aggregate[minority_class]) > self._minority_sample_batches_kept:
            self._minority_samples_aggregate[minority_class] = self._minority_samples_aggregate[minority_class][1:]

        X_min_accumulated = []
        Y_min_accumulated = []

        # Add minority class aggregate to training chunk
        m = len(self._minority_samples_aggregate[minority_class])

        for idx, (chunk_x, chunk_y) in enumerate(self._minority_samples_aggregate[minority_class]):
            if self._minority_sampling_strategy is 'fading':
                r = 1 - (m - idx + 1) * self._sampling_rate
            else:  # fixed
                r = self._sampling_rate

            samples_to_pick = int(np.ceil(len(chunk_x) * r))
            # print(f"idx = {idx}, size = {m}, r = {r}, picking {samples_to_pick}")

            if samples_to_pick > 0:
                all_chunk_indices = range(len(chunk_x))

                randomly_chosen_indices = np.random.choice(all_chunk_indices, samples_to_pick, replace=False)

                X_min_accumulated = np.concatenate((X_min_accumulated, chunk_x[randomly_chosen_indices])) if len(X_min_accumulated) > 0 else chunk_x[
                    randomly_chosen_indices]
                Y_min_accumulated = np.concatenate((Y_min_accumulated, chunk_y[randomly_chosen_indices])) if len(Y_min_accumulated) > 0 else chunk_y[
                    randomly_chosen_indices]

        # Reverse lists to make newest samples come first
        X_min_accumulated = np.flip(X_min_accumulated)
        Y_min_accumulated = np.flip(Y_min_accumulated)

        majority_chunk_size = len(X_maj)
        minority_acc_size = len(X_min_accumulated)
        min_maj_ratio = minority_acc_size/majority_chunk_size


        if min_maj_ratio > 1:
            self._create_clf_and_add_to_ensemble(
                np.concatenate((X_maj, X_min[:majority_chunk_size])),
                np.concatenate((Y_maj, Y_min[:majority_chunk_size]))
            )
        elif min_maj_ratio > 0.5:
            X_maj, Y_maj = self._randomly_undersample(X_maj, Y_maj, minority_acc_size)
            self._create_clf_and_add_to_ensemble(
                np.concatenate((X_maj, X_min)),
                np.concatenate((Y_maj, Y_min))
            )
        else:
            max_models_created = int(majority_chunk_size/minority_acc_size)
            difference = majority_chunk_size - minority_acc_size * max_models_created
            if difference > 0:
                X_maj, Y_maj = self._randomly_undersample(X_maj, Y_maj, majority_chunk_size - difference)

            X_maj_splits = np.array_split(X_maj, max_models_created)
            Y_maj_splits = np.array_split(Y_maj, max_models_created)

            for (X_maj_chunk, Y_maj_chunk) in zip(X_maj_splits, Y_maj_splits):
                self._create_clf_and_add_to_ensemble(
                    np.concatenate((X_maj_chunk, X_min_accumulated)),
                    np.concatenate((Y_maj_chunk, Y_min_accumulated))
                )

    def _randomly_undersample(self, x, y, desired_no):
        indices = list(range(len(x)))
        undersampled_indices = np.random.choice(indices, desired_no, replace=False)
        return x[undersampled_indices], y[undersampled_indices]

    def _create_clf_and_add_to_ensemble(self, x, y):
        clf = deepcopy(self._base_estimator)
        clf.fit(x, y)
        if len(self._ensemble) == self._ensemble_size:
            self._ensemble = self._ensemble[1:]

        self._ensemble.append(clf)

    def predict(self, x):
        voter = MajorityPredictionCombiner(self._ensemble, self._classes)

        return voter.predict(x)
