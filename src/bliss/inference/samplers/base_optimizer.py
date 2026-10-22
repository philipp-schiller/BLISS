from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import numpy as np

class Optimizer(ABC):
    """
    Base class for Bayesian optimizers.

    Subclasses must implement `optimize`.
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol_lp: float = 1e-3,
        tol_params: float = 1e-3,
        verbose: bool = False,
    ):
        self.max_iter = max_iter
        self.tol_lp = tol_lp
        self.tol_params = tol_params
        self.verbose = verbose

    @abstractmethod
    def optimize(
        self,
        posterior,
        shape_init,
        source_init=None,
        noise_init=None,
        shape_bounds=None,
        source_bounds=None,
        noise_bounds=None,
    ):
        """
        Run optimization and return an optimization result.
        """
        pass

@dataclass
class OptimizationResult:
    shape_params: np.ndarray
    source_params: Optional[np.ndarray]
    noise_params: Optional[np.ndarray]
    log_posterior_final: float

    n_iterations: int = 0
    n_function_evals: int = 0
    runtime: float = 0.0
    converged: bool = False
    message: str = ""

    log_posterior_history: Optional[list] = None
    shape_history: Optional[list] = None
    source_history: Optional[list] = None
    noise_history: Optional[list] = None