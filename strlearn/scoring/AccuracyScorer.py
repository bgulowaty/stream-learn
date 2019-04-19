from pytypes import typechecked, override
from sklearn.metrics import accuracy_score
from typing import Iterable

from strlearn.scoring.BaseScorer import BaseScorer


class AccuracyScorer(BaseScorer):

    @override
    def get_name(self) -> str:
        return "accuracy"

    @override
    def score(self, y_true: Iterable, y_pred: Iterable, classes: Iterable):
        return accuracy_score(y_true, y_pred)
