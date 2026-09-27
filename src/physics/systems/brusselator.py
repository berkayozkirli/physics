from collections.abc import Callable

import numpy as np
import numpy.typing as npt


class BrusselatorReaction:
    """
    This class focuses only on the Reaction
    """ 
    def __init__(self, A: float, B: float):
        self.A = A
        self.B = B

    def __call__(self, t: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: 
        u, v = state[..., 0], state[..., 1]
        u2v = (u**2) * v
        du = self.A - (self.B + 1.0) * u + u2v
        dv = self.B * u - u2v
        return np.stack((du, dv), axis=-1)

class BrusselatorDiffusion:
    """
    This class focuses only on the Reaction
    """ 
    def __init__(self, discretization: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Du: float, Dv: float, dx: float = 1.0, mode:str ="periodic"):
        self.Du = Du
        self.Dv = Dv
        self.dx2 = dx**2
        self.laplacian2d = discretization


    def __call__(self, t: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: 
        u, v = state[...,0], state[...,1]
        diff_u = (self.Du / self.dx2) * self.laplacian2d(u)
        diff_v = (self.Dv / self.dx2) * self.laplacian2d(v)
        return np.stack((diff_u, diff_v), axis=-1)

class Brusselator:
    """
    Reaction + Diffusion System    
    """
    def __init__(self, A: float, B: float, discretization: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Du: float, Dv: float, dx: float = 1.0, mode="periodic"):
        self.reaction = BrusselatorReaction(A, B) 
        self.diffusion = BrusselatorDiffusion(discretization, Du, Dv, dx, mode=mode)

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        d_reaction = self.reaction(t, state)
        d_diffusion = self.diffusion(t, state)
        return d_reaction + d_diffusion