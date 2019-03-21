from attr import attrs, attrib
from typing import Iterable, Optional

from strlearn.learners.BaseLearner import BaseLearner
from strlearn.learners.listeners import BaseLearningResultListener, LearningPartialResults
from strlearn.streams import BaseStream


# todo(bgulowaty): Wait for reasonable runtime type checker
# every single one fails on factory in attrs
# @typechecked
@attrs
class PrequentialLearner(BaseLearner):
    _estimator = attrib()
    _stream: BaseStream = attrib()
    _listener: Optional[BaseLearningResultListener] = attrib(default=None)
    _chunk_size: int = attrib(default=1)
    _scorers: Iterable = attrib(factory=list)

    _instances_processed = 0
    _scores = {}

    # @override
    def run(self, processed_samples_limit=None):
        while self._stream.is_dry() is False:
            x, y = self._stream.get_next_samples(self._chunk_size)
            self._instances_processed += self._chunk_size

            estimator_trained_at_least_once = self._instances_processed > self._chunk_size
            if estimator_trained_at_least_once:
                scores = self.test(x, y)
                self._send_scores_to_listener(scores)

            self.train(x, y)

    def test(self, x, y):
        y_pred = self._estimator.predict(x)

        scores = {scorer.get_name(): scorer.score(y, y_pred) for scorer in self._scorers}

        return scores

    def train(self, x, y):
        try:
            self._estimator.partial_fit(x, y, classes=self._stream.get_classes())
        except (TypeError, NotImplementedError, AttributeError) as e:
            self._estimator.fit(x, y)

    def _send_scores_to_listener(self, scores):
        self._listener.listen(LearningPartialResults(self._instances_processed, scores))
