import os
import numpy as np
import matplotlib.pyplot as plt

def export_grid_frames(grid_history: np.ndarray, frames_path="grids", cmap='viridis'):
    """
    Exports a time-series history of 2D grids as a sequence of PNG frames.
    
    Parameters:
    - grid_history: A 3D NumPy array of shape (num_frames, height, width).
    - frames_path: Path for the frames.
    - cmap: Matplotlib colormap to use (e.g., 'viridis', 'plasma', 'gray').
    """
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    num_frames = grid_history.shape[0]
    vmin, vmax = np.min(grid_history), np.max(grid_history)
    for i in range(num_frames):
        frame_data = grid_history[i]
        filename = os.path.join(frames_path, f"frame_{i:04d}.png")
        plt.imsave(filename, frame_data, cmap=cmap, vmin=vmin, vmax=vmax)