from attr import attrs, attrib
from typing import Iterable

from strlearn.learners.listeners import BaseLearningResultListener, LearningPartialResults


@attrs
class BroadcastingListener(BaseLearningResultListener):
    _listeners: Iterable[BaseLearningResultListener] = attrib()

    def listen(self, result: LearningPartialResults):
        for listener in self._listeners:
            listener.listen(result)


def create_broadcasting_listener(*listeners):
    return BroadcastingListener(*listeners)
