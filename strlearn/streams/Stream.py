from itertools import islice

from attr import attrs, attrib
from numpy import array
from typing import Iterable, Tuple, TypeVar

from strlearn.streams.base import BaseStream
from strlearn.streams.base.BaseStream import X as X_type, Y as Y_type


@attrs
class Stream(BaseStream):
    X: Iterable[X_type] = attrib()
    Y: Iterable[Y_type] = attrib()
    _classes: Iterable[Y_type] = attrib()
    _variable_output_on_dry: bool = attrib(default=True)

    def __attrs_post_init__(self):
        self._X_iterator = iter(self.X)
        self._Y_iterator = iter(self.Y)

    _is_dry = False
    _next_X = None
    _next_Y = None

    def is_dry(self) -> bool:
        return self._is_dry

    def get_next_samples(self, chunk_size: int = 1) -> Tuple[Iterable[X_type], Iterable[Y_type]]:
        next_x_batch = list(islice(self._X_iterator, chunk_size))
        next_y_batch = list(islice(self._Y_iterator, chunk_size))

        if len(next_x_batch) != len(next_y_batch):
            raise AttributeError

        if len(next_x_batch) < chunk_size:
            self._is_dry = True

        return array(next_x_batch), array(next_y_batch)

    def get_classes(self) -> Iterable[Y_type]:
        return self._classes
