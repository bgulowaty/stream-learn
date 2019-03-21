from abc import ABCMeta, abstractmethod
from typing import Iterable


class BaseScorer(metaclass=ABCMeta):

    @abstractmethod
    def score(self, y_true: Iterable, y_pred: Iterable):
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        return 'base_scorer'
