from itertools import islice, chain

from attr import attrs, attrib
from numpy import array, unique
from typing import Tuple, Collection

from strlearn.streams.base import BaseStream
from strlearn.streams.base.BaseStream import X as X_TYPE, Y as Y_TYPE


@attrs
class Stream(BaseStream):
    X: Collection[X_TYPE] = attrib()
    Y: Collection[Y_TYPE] = attrib()
    _classes: Collection[Y_TYPE] = attrib(default=None)
    _variable_output_on_dry: bool = attrib(default=True)

    def __attrs_post_init__(self):
        self._X_iterator = iter(self.X)
        self._Y_iterator = iter(self.Y)
        self._is_dry = False
        self._next_X = None
        self._next_Y = None

        if self._classes is None:
            self._classes = unique(self.Y)

        if len(self.X) != len(self.Y):
            raise AttributeError


    def is_dry(self) -> bool:
        return self._is_dry

    def get_next_samples(self, chunk_size: int = 1) -> Tuple:

        next_x_batch = list(islice(self._X_iterator, chunk_size))
        next_y_batch = list(islice(self._Y_iterator, chunk_size))

        if len(next_x_batch) < chunk_size or not self.there_are_next_samples():
            self._is_dry = True

        return array(next_x_batch), array(next_y_batch)

    def get_classes(self) -> Collection:
        return self._classes

    def there_are_next_samples(self):
        try:
            next_x_sample = next(self._X_iterator)
            self._X_iterator = chain([next_x_sample], self._X_iterator)
            return True
        except StopIteration:
            return False

