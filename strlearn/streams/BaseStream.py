from abc import ABCMeta, abstractmethod
from typing import Tuple, Any


class BaseStream(metaclass=ABCMeta):

    @abstractmethod
    def is_dry(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_next_samples(self, chunk_size: int = 1) -> Tuple[Any, Any]:
        raise NotImplementedError()

    @abstractmethod
    def get_classes(self):
        raise NotImplementedError()
