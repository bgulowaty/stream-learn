from attr import attrs, attrib
from pytypes import typechecked, override
from typing import Iterable, Any

from strlearn.scoring.BaseScorer import BaseScorer


@typechecked
@attrs
class SKLearnMetricsWrapper(BaseScorer):
    _metric: Any = attrib()
    _name: str = attrib()

    @override
    def get_name(self) -> str:
        return self._name

    @override
    def score(self, y_true: Iterable, y_pred: Iterable):
        return self._metric(y_true, y_pred)
