import numpy as np
import numpy.typing as npt
from typing import Callable
from abc import ABC, abstractmethod

ReactionFunc = Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class ODESolver(ABC):
    def __init__(self, f: ReactionFunc):
        self.f = f

    def set_initial_condition(self, u0: npt.NDArray[np.float64], t0: float = 0.0):
        self.u0 = np.asarray(u0, dtype=np.float64)
        self.shape = self.u0.shape
        
        # Runtime Validation
        test_output = self.f(t0, self.u0)
        if not isinstance(test_output, np.ndarray):
            raise TypeError("Reaction function 'f' must return a numpy array.")
        if test_output.shape != self.shape:
            raise ValueError("Shape mismatch between initial condition and function output.")

    def solve(self, time_range: tuple, N: int, save_interval: int = 1):
        t0, T = time_range
        self.dt = (T - t0) / N
        
        num_saves = N // save_interval + 1
        
        self.t_history = np.zeros(num_saves)
        self.u_history = np.zeros((num_saves, *self.shape))
        
        self.t_history[0] = t0
        self.u_history[0] = self.u0
        
        current_u = self.u0.copy()
        current_t = t0
        save_idx = 1

        for n in range(N):
            # This calls the child's implementation of advance()
            current_u = self.advance(current_t, current_u)
            current_t += self.dt
            
            if (n + 1) % save_interval == 0:
                self.t_history[save_idx] = current_t
                self.u_history[save_idx] = current_u
                save_idx += 1

        return self.t_history[:save_idx], self.u_history[:save_idx]

    @abstractmethod
    def advance(self, t: float, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculates the state at the next time step.
        Must be implemented by subclasses.
        """
        pass