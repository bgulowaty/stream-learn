from attr import attrib, attrs
from pytypes import typechecked
from skmultiflow.data.base_stream import Stream as SkMultiflowStream
from typing import Iterable, Tuple

from strlearn.streams import BaseStream
from strlearn.streams.base.BaseStream import X, Y


# @typechecked
@attrs
class ScikitMultiflowStreamAdapter(BaseStream):
    stream: SkMultiflowStream = attrib()

    _initialized = False

    def is_dry(self) -> bool:
        return not self.stream.has_more_samples()

    def get_next_samples(self, chunk_size: int = 1) -> Tuple[Iterable[X], Iterable[Y]]:
        if not self._initialized:
            self.stream.prepare_for_use()
            self._initialized = True

        return self.stream.next_sample(chunk_size)

    def get_classes(self) -> Iterable[Y]:
        return self.stream.target_values
