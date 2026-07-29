from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCost(ABC):
    @abstractmethod
    def fit(self, signal):
        pass

    @abstractmethod
    def error(self, start, end):
        pass

    def sum_of_costs(self, bkps):
        return sum(
            self.error(start, end)
            for start, end in zip([0] + list(bkps[:-1]), bkps)
        )

    @property
    @abstractmethod
    def model(self):
        pass


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, signal):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_predict(self, signal, *args, **kwargs):
        pass

