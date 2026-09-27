import os
import subprocess
import numpy as np
from .frame_exporters import export_particle_frames, export_grid_frames
from .utils import sync_real_time

class Video:
    """
    Acts as an Observer for the simulation, calculating the required sampling rate
    to achieve the desired playback speed, and handling automated MP4 compilation.
    """
    def __init__(self, name: str, t0: float, t_max: float, dt: float, playback_speed: float = 1.0, fps: int = 60):
        self.name = name
        self.t0 = t0
        self.t_max = t_max
        self.T = t_max - t0
        self.dt = dt
        self.playback_speed = playback_speed
        self.fps = fps        
        self.frames_path = os.path.join("animations", "frames", self.name)        
        sync_data = sync_real_time(self.T, self.dt, self.playback_speed, self.fps)
        self.save_interval = sync_data["save_interval"]
        self.num_saves = sync_data["num_saves"]

    def create_from_particles(self, states: np.ndarray, box_size: float):
        print(f"\n[Video: {self.name}] Exporting particle frames...")
        export_particle_frames(state_history=states, box_size=box_size, frames_path=self.frames_path)
        self._compile_ffmpeg()

    def create_from_grid(self, states: np.ndarray, cmap: str = 'viridis'):
        print(f"\n[Video: {self.name}] Exporting grid frames...")
        export_grid_frames(grid_history=states, frames_path=self.frames_path, cmap=cmap)
        self._compile_ffmpeg()

    def _compile_ffmpeg(self):
        print(f"\n[Video: {self.name}] Compiling MP4 with FFmpeg...")        
        output_file = os.path.join("animations", "videos", f"{self.name}.mp4")        
        
        cmd = [
            "ffmpeg",
            "-y", 
            "-framerate", str(self.fps),
            "-i", f"{self.frames_path}/frame_%04d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_file
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Video: {self.name}] Success! Video saved as '{output_file}'")
        except subprocess.CalledProcessError:
            print(f"[Video: {self.name}] Error: FFmpeg compilation failed.")