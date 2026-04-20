from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class BaseCylinderFitter(ABC):
    @abstractmethod
    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param points: Points to fit
        :param args:
        :param kwargs:
        :return: parameters of the fitted cylinder
        """
        pass
