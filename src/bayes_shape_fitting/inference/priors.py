from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import numpy as np

@dataclass(frozen=True)
class Prior(ABC):
    """Base class for priors"""

    @abstractmethod
    def log_prob(self, value: np.ndarray) -> float:
        """Compute log p(value)"""
        pass

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from prior"""
        pass


@dataclass(frozen=True)
class GaussianPrior(Prior):
    mean: np.ndarray
    cov: np.ndarray

    def log_prob(self, value: np.ndarray) -> float:
        from scipy.stats import multivariate_normal
        return multivariate_normal.logpdf(value, self.mean, self.cov)

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.multivariate_normal(self.mean, self.cov, size)


@dataclass(frozen=True)
class UniformPrior(Prior):
    lower: np.ndarray
    upper: np.ndarray

    def log_prob(self, value: np.ndarray) -> float:
        if np.all((value >= self.lower) & (value <= self.upper)):
            return -np.sum(np.log(self.upper - self.lower))
        return -np.inf

    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.uniform(self.lower, self.upper, (size, len(self.lower)))

