import numpy as np


def laplacian2d_periodic(grid):
    """
    Computes the 2D discrete Laplacian with periodic boundary conditions.
    """
    top = np.roll(grid, 1, axis=0)
    bottom = np.roll(grid, -1, axis=0)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)
    return top + bottom + left + right - 4.0 * grid

def laplacian2d_reflective(grid):
    """
    Computes the 2D discrete Laplacian with reflective (zero-flux) boundary conditions.
    """
    padded = np.pad(grid, pad_width=1, mode='edge')    
    top = padded[0:-2, 1:-1]
    bottom = padded[2:, 1:-1]
    left = padded[1:-1, 0:-2]
    right = padded[1:-1, 2:]
    return top + bottom + left + right - 4.0 * grid
