from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin, DensityMixin, \
    OutlierMixin, MetaEstimatorMixin
from typing import Union

Estimator = Union[
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    ClusterMixin,
    TransformerMixin,
    DensityMixin,
    OutlierMixin,
    MetaEstimatorMixin
]


class BaseLearner(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        raise NotImplementedError
