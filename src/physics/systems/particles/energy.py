import numpy as np
import numpy.typing as npt


class Energy:
  def __init__(self, g, q, m, N):
    self.g = g
    self.q = q
    self.m = m
    self.N = N

  def __call__(self, states: npt.NDArray[np.float64]) -> np.float64:
    """
    Compute the total energy of the system as
    V + U = sum_i(v_i**2/2 + gh_i + sum_j(k/r_ij)
    """
    u = states[:,0:2]
    v = states[:,2:4]
    heights = states[:,1]
    kinetic = np.sum(1/2 * self.m * np.sum(v**2, axis=-1))
    gravitational = np.sum(-self.g * self.m * heights)
    delta_u = u[:, np.newaxis, :] - u[np.newaxis, :, :]
    dist = np.linalg.norm(delta_u, axis=-1)
    np.fill_diagonal(dist, np.inf)
    electric_matrix = 0.5 * (self.q**2) / dist
    electric = np.sum(electric_matrix)
    return kinetic + gravitational + electric
