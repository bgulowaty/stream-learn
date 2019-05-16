from abc import ABCMeta, abstractmethod


class BaseEnsemblePredictionCombiner(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError
