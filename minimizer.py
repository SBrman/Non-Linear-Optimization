#! python3

__author__ = "Simanta Barman"
__email__ = "barma017@umn.edu"


import time
import numpy as np
from typing import Union, Callable
from enum import Enum, auto
from models import LogisticRegressionModel
from numpy.lib.arraysetops import isin

# Custom types
Vector = Union[list, np.array]


class StepSizeRule(Enum):
    CONSTANT = auto()
    DIMINISHING = auto()
    ARMIJO = auto()


class Minimizer:

    def __init__(self, f):
        assert f.is_convex, f'The function {f} is not a convex function.'
        self.f = f
        self.gradf_norm_x0 = None

    def relative_grad_norm(self, x_k: Vector) -> float:
        """Returns the relative gradient norm at x_k."""

        # Compute grad f(x_k)
        gradf_xk = self.f.gradient(x_k)
        gradf_norm_xk = np.linalg.norm(gradf_xk, 2)
        
        # ||grad f(x_k)||_2 / ||grad f(x_0)||_2
        relative_grad_norm = gradf_norm_xk / self.gradf_norm_x0
        
        return relative_grad_norm
        
    def get_norm_grad_f_initial_x(self, x_0: Vector) -> float:
        """Returns the absolute value of the L2 norm of gradient f at initial solution. """

        # Compute grad f at initial x
        grad_f_x0 = self.f.gradient(x_0)

        # Return ||grad f(x_0)||_2
        return np.linalg.norm(grad_f_x0, 2)
    
    def _step_size(self, x: Vector, d: Vector, rule: StepSizeRule, epsilon: float,
                   iteration: int, constant_step_size: float = None, 
                   diminishing_function: Callable[[int], float] = None) -> float:
        """Returns the step size using the input step size rule. """

        if rule == StepSizeRule.CONSTANT:
            if constant_step_size:
                return constant_step_size
            else:
                return self.constant_stepsize(epsilon)

        elif rule == StepSizeRule.DIMINISHING:
            assert diminishing_function is not None, 'Diminishing function is not specified.'
            return self.diminishing_stepsize(iteration, diminishing_function)

        elif rule == StepSizeRule.ARMIJO:
            return self.armijo_stepsize(x, d)

        else:
            raise NotImplementedError('Step size rule unknown.')
    
    def constant_stepsize(self, epsilon: float = 1e-5):
        """Returns alpha in (epsilon, (2 - epsilon) / L)."""
        if isinstance(self.f, LogisticRegressionModel):
            return self.f._lr_step_size

        L = self.f.max_eigenvalue
        return 1 / L
    
    def diminishing_stepsize(self, iteration: int, function: Callable[[int], float]) -> float:
        """
        Returns the diminishing step size based on input function.
        Args:
            function: diminishing funtion. i.e. alpha = 1/r or 1/r^2 or 1/r^0.5
        """
        return function(iteration)

    def armijo_stepsize(self, x: Vector, d: Vector, sigma: float = 0.1, beta: float = 0.5,
                        s: float = 1) -> float:
        """ Returns the stepsize according to Armijo rule. """
        
        assert 0 < sigma < 0.5, "sigma not in (0, 0.5)"
        assert 0 < beta < 1, "beta not in (0, 1)"
         
        fx = self.f(x)
        grad_fx = - d   # negative of gradient is the direction

        k = 0
        while True:
            alpha = beta**k * s
            if self.f(x + alpha * d) - fx <= alpha * sigma * (grad_fx.T.dot(d)):
                return alpha
            k += 1

    def steepest_descent(self, initial_soln: Vector, step_size_rule: StepSizeRule,
                         constant_step_size: float = None, 
                         diminishing_function: Callable[[int], float] = None, 
                         epsilon: float = 1e-4, max_iteration: int = 10000):
        """Returns the solution from Steepest descent method with Armijo step size rule. """
        iteration_times = []
        objective_values = []
        rel_grad_obj_vals = []
                
        # Computing the absolute value of the norm of gradient of f at the initial solution.
        self.gradf_norm_x0 = self.get_norm_grad_f_initial_x(x_0=initial_soln)

        x_new = initial_soln

        for k in range(max_iteration):  # To ensure termination
            # Starting time counting
            start_time = time.perf_counter()
            
            # Step 1: Set initial solution.
            x = x_new

            # Step 2: Compute descent direction based on x.
            d = - self.f.gradient(x)

            # Step 3: Find step-size alpha
            alpha = self._step_size(x=x, d=d, rule=step_size_rule, epsilon=epsilon,
                                    iteration=k, constant_step_size=constant_step_size,
                                    diminishing_function=diminishing_function)

            # Step 4: Update solution
            x_new = x + alpha * d

            # end time counting for kth iteration
            iteration_times.append(time.perf_counter() - start_time)    
            # Keeping the objective values for each iteration.
            objective_values.append(self.f(x_new))
            
            # Stopping criteria
            rel_grad_obj_vals.append(self.relative_grad_norm(x_new))
            if rel_grad_obj_vals[-1] < epsilon:
                break

        return x_new, iteration_times, objective_values, rel_grad_obj_vals

    def coordinate_descent(self, initial_soln: Vector, step_size_rule: StepSizeRule, 
                           constant_step_size: float = None, 
                           diminishing_function: Callable[[int], float] = None, 
                           epsilon: float = 1e-4, max_iteration: int = 10000):
        """Returns the solution using the coordinate descent method."""
        
        iteration_times = []
        objective_values = []
        rel_grad_obj_vals = []
        
        # Set initial solution.
        x = initial_soln
        n = len(x)

        # Computing the absolute value of the norm of gradient of f at the initial solution.
        self.gradf_norm_x0 = self.get_norm_grad_f_initial_x(x_0=initial_soln)

        # Compute descent direction based on initial x.
        d = - self.f.gradient(initial_soln)
        
        for r in range(max_iteration):
            # Starting time counting
            start_time = time.perf_counter()

            # Picking an i
            i = r % n
            
            # Update the descent direction to find the step size
            d[i] = self.f.partial_derivative(x, i)
            
            # Find step-size alpha
            alpha = self._step_size(x=x, d=d, rule=step_size_rule, epsilon=epsilon,
                                    iteration=r, constant_step_size=constant_step_size,
                                    diminishing_function=diminishing_function)
            
            # Update the solution only for ith entry
            x[i] = x[i] - alpha * d[i]
            
            # end time counting for kth iteration
            iteration_times.append(time.perf_counter() - start_time)
            # Keeping the objective values for each iteration.
            objective_values.append(self.f(x))
            
            # Stopping criteria
            rel_grad_obj_vals.append(self.relative_grad_norm(x))
            if rel_grad_obj_vals[-1] < epsilon:
                break

        return x, iteration_times, objective_values, rel_grad_obj_vals

    def perceptron_learning(self, initial_soln: Vector, epsilon: float = 1e-4):
        """Returns the solution using the perceptron learning algorithm."""
        
        iteration_times = []
        objective_values = []
        # Set initial solution.
        
        x_hat = initial_soln
        x = x_hat.copy()
        
        for i in range(self.f.M):
            # Starting time counting
            start_time = time.perf_counter()

            x = x_hat + self.f.b[i] * self.f.A[i]
            
            if self.f(x) < self.f(x_hat):
                x_hat = x
                
            # end time counting for ith iteration
            iteration_times.append(time.perf_counter() - start_time)
            # Keeping the objective values for each iteration.
            objective_values.append(self.f(x_hat))
            
        return x_hat, iteration_times, objective_values


    def svm_stochastic_subgradient_method(self, initial_soln: Vector, T: int = 10000, eta: float = 1):
        
        iteration_times = []
        objective_values = []

        # Set initial solution.
        x = initial_soln

        for r in range(T):
            # Starting time counting
            start_time = time.perf_counter()

            # Choose random training datapoint uniformly
            i = np.random.randint(0, self.f.N)
            
            # Stepsize (According to Leon Bottou's suggestion (Lecture 10, slide 0-32))
            alpha = eta / (r + 1)
            
            # Update
            x = x - alpha * self.f.subgradient(x, i)
            
            # The update step does the same thing as following commented code.
            # if self.f.miss_classified(x, i):
            #     x = x - alpha * (2 * self.f.gamma * x - (1 / self.f.N) * self.f.b[i] * self.f.A[i])
            # else:
            #     x = (1 - alpha * 2 * self.f.gamma) * x
                
            # end time counting for ith iteration
            iteration_times.append(time.perf_counter() - start_time)
            # Keeping the objective values for each iteration.
            objective_values.append(self.f(x))
            
        return x, iteration_times, objective_values
    
    def svm_coordinate_descent(self, initial_soln: Vector, T: int = 10000, eta: float = 1):
        
        iteration_times = []
        objective_values = []

        # Set initial solution.
        x = initial_soln
        n = len(x)

        for r in range(T):
            # Starting time counting
            start_time = time.perf_counter()

            # Choose datapoint
            i = r % n
            
            # Stepsize (According to Leon Bottou's suggestion (Lecture 10, slide 0-32))
            alpha = eta / (r + 1)
            
            # Update
            x[i] = x[i] - alpha * self.f.subgradient(x, i)[i]
                
            # end time counting for ith iteration
            iteration_times.append(time.perf_counter() - start_time)
            # Keeping the objective values for each iteration.
            objective_values.append(self.f(x))
            
        return x, iteration_times, objective_values
        