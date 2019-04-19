from abc import ABCMeta, abstractmethod
from typing import Tuple, TypeVar, Iterable, Generic


X = TypeVar('X')
Y = TypeVar('Y')


class BaseStream(Generic[X, Y], metaclass=ABCMeta):

    @abstractmethod
    def is_dry(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_next_samples(self, chunk_size: int = 1) -> Tuple[Iterable[X], Iterable[Y]]:
        raise NotImplementedError()

    @abstractmethod
    def get_classes(self) -> Iterable[Y]:
        raise NotImplementedError()
