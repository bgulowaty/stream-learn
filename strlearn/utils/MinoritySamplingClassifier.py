from attr import attrs, attrib
from strlearn.preprocessing.oversampling import smote
from strlearn.utils import split_minority_majority


def default_smote_wrapper(x_min, x_maj, k, sampling_percentage):
    x_min_smoted = smote(x_min, k, sampling_percentage=sampling_percentage, dont_restrict=True)
    return x_min_smoted


@attrs
class MinoritySmotingClassifier:
    _wrapped_classifier = attrib()
    _smote = attrib(default=default_smote_wrapper)
    _k = attrib(default=3)
    _sampling_percentage = attrib(default='auto')

    def partial_fit(self, x, y, classes):
        x_min, y_min, x_maj, y_maj = split_minority_majority(x, y)
        if self._sampling_percentage == 'auto':
            sampling_percentage = (len(x_maj) - len(x_min)) / len(x_min) * 100
            x_min_smoted = self._smote(x_min, x_maj, self._k, sampling_percentage)
        else:
            x_min_smoted = self._smote(x_min, x_maj, self._k, self._sampling_percentage)

        y_min_smoted = [y_min[0] for _ in x_min_smoted]

        if len(y_min_smoted) == 0:
            x_train = x
            y_train = y
        else:
            x_train = np.concatenate((x, x_min_smoted), axis=0)
            y_train = np.concatenate((y, y_min_smoted), axis=0)

        self._wrapped_classifier.partial_fit(x_train, y_train, classes)

    def predict(self, x):
        return self._wrapped_classifier.predict(x)

    def get_wrapped_clf(self):
        return self._wrapped_classifier