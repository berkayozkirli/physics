import os
import numpy as np
import matplotlib.pyplot as plt

def export_particle_frames(state_history: np.ndarray, box_size: float, frames_path="frames"):
    """
    Renders an entire particle simulation history directly to PNG frames.
    
    Parameters:
    - state_history: A 3D array of shape (num_frames, N, 4)
    - box_size: The physical dimension of the box for axis limits
    - frames_path: Path for the frames.    
    """
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    num_frames, N, _ = state_history.shape
    # Create a consistent color mapping for N particles
    colors = plt.cm.hsv(np.linspace(0, 1, N))
    for i in range(num_frames):
        x_coords = state_history[i, :, 0]
        y_coords = state_history[i, :, 1]        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.scatter(x_coords, y_coords, s=40, c=colors, marker='o')        
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)        
        ax.set_aspect('equal')
        ax.axis('off')        
        filename = os.path.join(frames_path, f"frame_{i:04d}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)        
        plt.close(fig)