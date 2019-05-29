import numpy as np
from attr import attrs, attrib


from strlearn.ensembles.voting import BaseEnsemblePredictionCombiner


@attrs
class MajorityPredictionCombiner(BaseEnsemblePredictionCombiner):

    _ensemble = attrib()
    _classes = attrib()

    def predict(self, x):
        predictions = [{class_name: 0 for class_name in self._classes} for _ in range(len(x))]

        for k, clf in enumerate(self._ensemble):
            y_pred = clf.predict(x)
            for sample_idx, sample_pred in enumerate(y_pred):
                predictions[sample_idx][sample_pred] += 1

        actual_predictions = []
        for sample_predictions in predictions:
            actual_predictions.append(max(sample_predictions, key=sample_predictions.get))

        return np.array(actual_predictions)
