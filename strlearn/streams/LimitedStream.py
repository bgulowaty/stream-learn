from attr import attrs, attrib
from pytypes import typechecked
from typing import Tuple, Any

from strlearn.streams import BaseStream


@typechecked
@attrs
class LimitedStream(BaseStream):
    base_stream: BaseStream = attrib()
    limit: int = attrib()

    _processed_samples = 0

    def is_dry(self) -> bool:
        return self._processed_samples == self.limit

    def get_next_samples(self, chunk_size: int = 1) -> Tuple[Any, Any]:
        next_chunk_would_reach_limit = (self._processed_samples + chunk_size) > self.limit
        if next_chunk_would_reach_limit:
            samples_left_to_reach_limit = self.limit - self._processed_samples
            x, y = self.base_stream.get_next_samples(samples_left_to_reach_limit)
            self._processed_samples = self.limit
            return x, y

        x, y = self.base_stream.get_next_samples(chunk_size)
        self._processed_samples += chunk_size
        return x, y

    def get_classes(self):
        return self.base_stream.get_classes()
