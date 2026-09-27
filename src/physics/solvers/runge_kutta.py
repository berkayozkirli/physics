import numpy as np
import numpy.typing as npt
from .ode_solver import ODESolver

class RungeKutta4(ODESolver):
    def advance(self, t: float, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        dt = self.dt
        f = self.f
        k1 = f(t, u)
        k2 = f(t + dt / 2.0, u + k1 * (dt / 2.0))
        k3 = f(t + dt / 2.0, u + k2 * (dt / 2.0))
        k4 = f(t + dt, u + k3 * dt)
        
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)