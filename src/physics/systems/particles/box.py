import numpy as np
import numpy.typing as npt


class Box:
	def __init__(self, box_size: float):
		self.box_size = box_size

	def collide(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
		"""
		Instantly checks and reflects all particles against the box walls using 
		boolean masking, preventing them from escaping the simulation bounds.
		"""
		# X-axis bounds (Left and Right walls)
		hit_left = state[:, 0] <= 0
		hit_right = state[:, 0] >= self.box_size
		state[hit_left | hit_right, 2] *= -1.0
		state[hit_left, 0] = 0.1
		state[hit_right, 0] = self.box_size - 0.1
		# Y-axis bounds (Floor and Ceiling)
		hit_floor = state[:, 1] <= 0
		hit_ceiling = state[:, 1] >= self.box_size
		state[hit_floor | hit_ceiling, 3] *= -1.0
		state[hit_floor, 1] = 0.1                       
		state[hit_ceiling, 1] = self.box_size - 0.1
		return state