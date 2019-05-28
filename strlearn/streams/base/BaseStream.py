from abc import ABCMeta, abstractmethod
from typing import Tuple, TypeVar, Collection

X = TypeVar('X')
Y = TypeVar('Y')


class BaseStream(metaclass=ABCMeta):

    @abstractmethod
    def is_dry(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_next_samples(self, chunk_size: int = 1) -> Tuple[Collection[X], Collection[Y]]:
        raise NotImplementedError()

    @abstractmethod
    def get_classes(self) -> Collection[Y]:
        raise NotImplementedError()
