from attr import attrs, attrib
from pytypes import typechecked
from typing import Dict


@typechecked
@attrs(frozen=True)
class LearningPartialResults:
    sample_no: int = attrib()
    scores: Dict = attrib()
