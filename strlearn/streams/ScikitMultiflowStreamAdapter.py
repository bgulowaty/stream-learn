from attr import attrib, attrs
from skmultiflow.data.base_stream import Stream as SkMultiflowStream
from typing import Iterable, Tuple, Collection

from strlearn.streams import BaseStream


@attrs
class ScikitMultiflowStreamAdapter(BaseStream):
    stream: SkMultiflowStream = attrib()

    _initialized = False

    def is_dry(self) -> bool:
        return not self.stream.has_more_samples()

    def get_next_samples(self, chunk_size: int = 1) -> Tuple:
        if not self._initialized:
            self.stream.prepare_for_use()
            self._initialized = True

        return self.stream.next_sample(chunk_size)

    def get_classes(self) -> Collection:
        return self.stream.target_values
