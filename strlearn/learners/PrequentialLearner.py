from attr import attrs, attrib
from collections import Iterable
from typing import Optional

from strlearn.learners.BaseLearner import BaseLearner
from strlearn.learners.listeners import BaseLearningResultListener, LearningPartialResults
from strlearn.streams import BaseStream


# todo(bgulowaty): Wait for reasonable runtime type checker
# every single one fails on factory in attrs
@attrs
class PrequentialLearner(BaseLearner):
    _estimator = attrib()
    _stream: BaseStream = attrib()
    _testing_stream: Optional[BaseStream] = attrib(default=None)
    _listener: Optional[BaseLearningResultListener] = attrib(default=None)
    _chunk_size: int = attrib(default=1)
    _testing_stream_chunk_size: Optional[int] = attrib(default=1)
    _scorers: Iterable = attrib(factory=list)

    def __attrs_post_init__(self):
        self._iteration = 0

    def run(self):
        while self._stream.is_dry() is False:
            x, y = self._stream.get_next_samples(self._chunk_size)

            if self._iteration > 0:
                if self._testing_stream is not None and not self._testing_stream.is_dry():
                    x_test, y_test = self._testing_stream.get_next_samples(self._testing_stream_chunk_size)
                    self._send_scores_to_listener(self.test(x_test, y_test, self._testing_stream.get_classes()))
                elif self._testing_stream is None:
                    self._send_scores_to_listener(self.test(x, y, self._stream.get_classes()))

            self.train(x, y)
            self._iteration += 1

    def test(self, x, y, classes):
        y_pred = self._estimator.predict(x)
        scores = {scorer.get_name(): scorer.score(y, y_pred, classes) for scorer in self._scorers}

        return scores

    def train(self, x, y):
        try:
            self._estimator.partial_fit(x, y, classes=self._stream.get_classes())
        except (NotImplementedError, AttributeError) as e:
            self._estimator.fit(x, y)

    def _send_scores_to_listener(self, scores):
        self._listener.listen(LearningPartialResults(self._iteration, scores))
