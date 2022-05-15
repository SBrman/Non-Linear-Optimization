__author__ = "Simanta Barman"
__email__ = "barma017@umn.edu"

import numpy as np
from typing import Union
from objective_function import BaseObjectiveFunction

Vector = Union[list, np.array]

class LinearRegressionModel(BaseObjectiveFunction):
    
    def __repr__(self):
    	return """Linear Regression Model: 
        ||Ax - b||_2 ^ 2"""
    
    def objective_value(self, x: Vector):
        return np.linalg.norm((self.A.dot(x) - self.b), 2) ** 2
    
    def gradient(self, x: Vector):
        return 2 * self.A.T.dot(self.A.dot(x) - self.b)
    
    def hessian(self, x: Vector):
        return 2 * self.A.T.dot(self.A)

    def partial_derivative(self, x: float, i: int):
        return 2 * self.A[i].T.dot(self.A[i] * x[i] - self.b[i])

    
class LogisticRegressionModel(BaseObjectiveFunction):
    
    def __init__(self, A, b):
        super().__init__(A, b)
        self.M, _ = A.shape
        self._step_size = None
    
    def __repr__(self):
    	return """Logistic Regression Model: 
        (1/M) * sum(ln(1 + exp(-b_i * x^T.a_i) for i in range(1, M)))"""
    
    def exp_term(self, x: Vector, i: int):
        """Returns exp(- b[i] * A[i].dot(x))"""
        return np.exp(self.b[i] * x.T.dot(self.A[i]))
    
    def objective_value(self, x: Vector):
        return (1 / self.M) * sum(np.log(1 + np.exp(-self.b[i] * x.T.dot(self.A[i]))) for i in range(self.M))
    
    def gradient(self, x: Vector):
        return - (1 / self.M) * sum((self.b[i] * self.A[i]) / (1 + self.exp_term(x, i)) 
                                    for i in range(self.M))
    
    def hessian(self, x: Vector):
        return (1 / self.M) * sum((self.b[i]**2 * self.A[i].dot(self.A[i].T) * self.exp_term(x, i)) 
                                  / (1 + self.exp_term(x, i)) ** 2 for i in range(self.M))
        
    def partial_derivative(self, x: Vector, i: int):
        """For now computes the gradient and then returns the ith element. Bad way to do this."""
        # TODO: Need to make this efficient by calculating the partial derivative instead of 
        # computing gradient every time its called.
        return self.gradient(x)[i]
    
    @property
    def _lr_step_size(self):
        """
        Returns the fixed step size for logistic regression.
        
        For logistic regression step size does not depend on x or label if label is -1 or 1
        """
        if not self._step_size:
            # Calculate and store step size if not already calculated
            self._step_size = 1 / ((1 / self.M) * sum(np.linalg.norm(self.A[i], 2) **2 for i in range(self.M)))

        return self._step_size


class Perceptron(BaseObjectiveFunction):
    def __init__(self, A, b):
        super().__init__(A, b)
        self.M, _ = A.shape
    
    def __repr__(self):
    	return """Perceptron Learning Model: max(-b_i.x^T.a_i, 0)"""
    
    def objective_value(self, x: Vector):
        # return sum(max(-self.b[i] * x.T.dot(self.A[i]), 0) for i in range(self.M))
        terms = - self.b * self.A.dot(x)
        max_terms = terms[terms >= 0]
        return np.sum(max_terms)
    
    def partial_derivative(self, x: Vector, i: int):
        return self.b[i] * self.A[i]
    
    def gradient(self, x: Vector) -> np.array: 
        raise AssertionError("Should not be called.")
    
    def hessian(self, x: Vector):
        raise AssertionError("Should not be called.")
    

class SVM(BaseObjectiveFunction):
    def __init__(self, A, b, gamma):
        super().__init__(A, b)
        self.gamma = gamma
        self.M, self.N = A.shape
        
    def __repr__(self):
    	return """Support Vector Machine: 
            1/N * sum(max(-b_i.x^T.a_i, 0) for i in N) + gamma (x^T.x)"""
    
    def objective_value(self, x: Vector):
        # return sum(max(-self.b[i] * x.T.dot(self.A[i]), 0) for i in range(self.M))
        terms = - self.b * self.A.dot(x)
        max_terms = terms[terms >= 0]
        return 1/self.M * np.sum(max_terms) + self.gamma * (x.T.dot(x))
    
    def miss_classified(self, x: Vector, i: int) -> bool:
        return bool(self.b[i] * self.A[i].T.dot(x) < 1)
    
    def subgradient_hinge_loss(self, x: Vector, i: int):
        return - self.b[i] * self.A[i] if self.miss_classified(x, i) else 0

    def subgradient(self, x: Vector, i: int):
        pd = 2 * self.gamma * x
        return pd - ((1 / self.M) * self.b[i] * self.A[i]) if self.miss_classified(x, i) else pd
    
    def partial_derivative(self, x: Vector, i: int):
        return self.subgradient(x, i)[i]
    
    def gradient(self, x: Vector) -> np.array: 
        raise AssertionError("Should not be called.")
    
    def hessian(self, x: Vector):
        raise AssertionError("Should not be called.")




class SVM_dual(BaseObjectiveFunction):
    def __init__(self, A, b):
        super().__init__(A, b)
        self.M, self.N = A.shape
        
    def __repr__(self):
        return """Support Vector Machine: 
            sum(lambda[i] for i in range(self.M)) 
            - 0.5 * sum(lambda[i] * lambda[j] * b[i] * b[j] * a[i]^T * a[j])"""
   
    def objective_value(self, lambda_: Vector) -> float:
        """Avoid calling this, takes super long time."""

        first_term = np.sum(lambda_)

        second_term = 0
        for i, (a_i, b_i) in enumerate(zip(self.A, self.b)):
            # outer_first_term = lambda_[i] * b_i * a_i
            inner_first_term = 0
            for j, (a_j, b_j) in enumerate(zip(self.A, self.b)):
                # inner_first_term += lambda_[j] * b_j * a_j
                second_term += lambda_[i] * lambda_[j] * b_i * b_j * a_i.T.dot(a_j)
            # second_term += outer_first_term.dot(inner_first_term.T)

        # original problem is a maximization problem, taking the negative of the objective function.
        return - (first_term - 0.5 * second_term)
    
    def partial_derivative(self, lambda_: Vector, i: int) -> float:
        x_t = sum(lambda_[j] * b_j * a_j for j, (a_j, b_j) in enumerate(zip(self.A, self.b)))

        # original problem is a maximization problem, taking the negative of the objective function.
        return - (1 - self.b[i] * self.A[i].T.dot(x_t))
    
    def gradient(self, lambda_: Vector) -> np.array:
        return np.array([self.partial_derivative(lambda_, i) for i in range(self.N)])
    
    def hessian(self, lambda_: Vector):
        raise AssertionError("Should not be called.")
    