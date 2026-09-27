import numpy as np
import numpy.typing as npt


class Motion:
    def __init__(self, g: float, q: float, m: float, N: int):
        self.g = g
        self.q = q
        self.m = m
        self.N = N

    def __call__(self, t: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        state is an (N, 4) array.
        Columns 0, 1 are x, y positions.
        Columns 2, 3 are vx, vy velocities.
        """
        # Extract all positions (R) and velocities (V) at once
        R = state[:, 0:2] 
        V = state[:, 2:4] 
        # Base gravity for all particles
        A = np.zeros_like(R)
        A[:, 1] = self.g
        # Subtracting them broadcasts into an (N, N, 2) matrix of all pairwise distances
        delta_R = R[:, np.newaxis, :] - R[np.newaxis, :, :]
        # Calculate scalar distances (N, N)
        dist = np.linalg.norm(delta_R, axis=-1)
        np.fill_diagonal(dist, np.inf)
        mag = (self.q**2 / self.m) / (dist**3)
        # Calculate scalar distances (N, N, 1)
        mag = mag[..., np.newaxis]
        # Multiply magnitude by direction, and sum along the j-axis to get total force on i
        A += np.sum(mag * delta_R, axis=1)
        # Derivative of position is velocity; derivative of velocity is acceleration.
        d_state = np.empty_like(state)
        d_state[:, 0:2] = V
        d_state[:, 2:4] = A

        return d_state