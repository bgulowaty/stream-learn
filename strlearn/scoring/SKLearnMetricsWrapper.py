from attr import attrs, attrib
from pytypes import override
from typing import Iterable, Callable

from strlearn.scoring.BaseScorer import BaseScorer


@attrs
class SKLearnMetricsWrapper(BaseScorer):
    _metric: Callable = attrib()
    _name: str = attrib()

    @override
    def get_name(self) -> str:
        return self._name

    @override
    def score(self, y_true: Iterable, y_pred: Iterable, classes: Iterable):
        return self._metric(y_true, y_pred)
