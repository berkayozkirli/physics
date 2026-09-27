from collections.abc import Callable

import numpy as np
import numpy.typing as npt


class SchnakenbergReaction:
    """
    Reaction part of the Schnakenberg model
    """
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __call__(self, t: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u, v = state[..., 0], state[..., 1]
        u2v = (u**2) * v
        du = self.a - u + u2v
        dv = self.b - u2v
        return np.stack((du, dv), axis=-1)

class SchnakenbergDiffusion:
    """
    This class focuses only on the Diffusion
    """ 
    def __init__(self, discretization: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Du: float, Dv: float, dx: float = 1.0):
        self.Du = Du
        self.Dv = Dv
        self.dx2 = dx**2
        self.laplacian2d = discretization

    def __call__(self, t: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: 
        u, v = state[...,0], state[...,1]
        diff_u = (self.Du / self.dx2) * self.laplacian2d(u)
        diff_v = (self.Dv / self.dx2) * self.laplacian2d(v)
        return np.stack((diff_u, diff_v), axis=-1)

class Schnakenberg:
    """
    Reaction + Diffusion System
   
    ∂u/∂t = Du ∇²u + a - u + u²v
    ∂v/∂t = Dv ∇²v + b - u²v    
    """
    def __init__(self, a: float, b: float, discretization: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Du: float, Dv: float, dx: float = 1.0):
        self.reaction = SchnakenbergReaction(a, b) 
        self.diffusion = SchnakenbergDiffusion(discretization, Du, Dv, dx)

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        d_reaction = self.reaction(t, state)
        d_diffusion = self.diffusion(t, state)
        return d_reaction + d_diffusion



