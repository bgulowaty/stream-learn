from abc import ABCMeta, abstractmethod

from strlearn.learners.listeners import LearningPartialResults


class BaseLearningResultListener(metaclass=ABCMeta):

    @abstractmethod
    def listen(self, result: LearningPartialResults):
        raise NotImplementedError
