import numpy as np
import numpy.typing as npt
from .ode_solver import ODESolver

class ExplicitEuler(ODESolver):
    
    def advance(self, t: float, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return u + self.dt * self.f(t, u)