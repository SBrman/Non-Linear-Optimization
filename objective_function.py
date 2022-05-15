#! python3

__author__ = "Simanta Barman"
__email__ = "barma017@umn.edu"


import numpy as np
from abc import ABC, abstractmethod
from typing import Union

# Custom Type
Vector = Union[np.array, list]


class BaseObjectiveFunction(ABC):

    def __init__(self, A, b):
        self.A = A 
        self.b = b
    
    def __call__(self, x):
        return self.objective_value(x)
    
    @abstractmethod
    def objective_value(self, x: Vector) -> float:
        """Returns the objective value of the function."""
        pass

    @abstractmethod
    def gradient(self, x: Vector) -> np.array:
        """Returns the gradient at x"""
        pass

    @abstractmethod
    def hessian(self, x: Union[Vector, None] = None) -> np.array:
        """Returns the hessian (A in this case)"""
        pass

    @property
    def eigenvalues(self) -> np.array:
        """Returns all the eigenvalues of the matrix A"""
        return np.linalg.eigvals(self.A.T.dot(self.A))

    @property
    def max_eigenvalue(self) -> float:
        """Returns the maximum eigenvalue of the matrix A."""
        return np.max(self.eigenvalues)

    @property
    def is_psd(self) -> bool:
        """Returns whether the function is positive semi-definite."""
        return np.all(self.eigenvalues >= 0)

    @property
    def is_convex(self) -> bool:
        """Returns the whether the function is convex."""
        return self.is_psd
