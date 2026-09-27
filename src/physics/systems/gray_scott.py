from collections.abc import Callable

import numpy as np
import numpy.typing as npt


class GrayScottReaction:
    def __init__(self, F: float = 0.035, k: float = 0.060):
        self.F = F
        self.k = k

    def __call__(self, t: float, state: npt.NDArray[np.float64]):
        u, v = state[..., 0], state[..., 1]
        uv2 = u * v * v
        du = -uv2 + self.F * (1.0 - u)
        dv = uv2 - (self.F + self.k) * v
        return np.stack((du, dv), axis=-1)


class GrayScottDiffusion:
    def __init__(self, discretization: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Du=0.16, Dv=0.08, dx=1.0):
        self.Du = Du
        self.Dv = Dv
        self.dx2 = dx**2
        self.laplacian2d = discretization

    def __call__(self, t: float, state: npt.NDArray[np.float64]):
        u, v = state[..., 0], state[..., 1]
        du = (self.Du / self.dx2) * self.laplacian2d(u)
        dv = (self.Dv / self.dx2) * self.laplacian2d(v)
        return np.stack((du, dv), axis=-1)

class GrayScott:
    def __init__(self, F: float, k: float, discretization: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Du: float, Dv: float, dx: float = 1.0):
        self.reaction = GrayScottReaction(F, k)
        self.diffusion = GrayScottDiffusion(discretization, Du, Dv, dx)

    def __call__(self, t: float, state: npt.NDArray[np.float64]):
        return self.reaction(t, state) + self.diffusion(t, state)
