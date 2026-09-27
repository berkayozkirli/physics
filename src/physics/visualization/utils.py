import math

def sync_real_time(t_max: float, dt: float, playback_speed: float = 1.0, fps: int = 60) -> dict:
    """
    Calculates the exact save_interval needed to sync simulation time 
    with real-world video playback using a speed multiplier.
    
    Parameters:
    - t_max: Total simulation time.
    - dt: Integration time step.
    - playback_speed: 1.0 is real-time, 2.0 is double speed, 0.5 is half speed.
    - fps: Desired frame rate of the video.
    """
    total_steps = int(t_max / dt)
    target_video_duration = t_max / playback_speed
    target_frames = int(target_video_duration * fps)
    
    # Safety check: Cannot save more frames than total steps
    if target_frames > total_steps:
        print(f"Warning: Requested {target_frames} frames, but only have {total_steps} steps.")
        print(f"Defaulting to saving every single step (save_interval = 1).")
        save_interval = 1
    else:
        save_interval = max(1, total_steps // target_frames)
        
    actual_frames = total_steps // save_interval
    actual_duration = actual_frames / fps
    
    print("\n--- 🎬 Video Synchronization Report ---")
    print(f"Simulation Setup : t=[0, {t_max}] with dt={dt} ({total_steps} total steps)")
    print(f"Target Speed     : {playback_speed}x Real-time ({target_video_duration:.2f}s @ {fps} FPS)")
    print(f"Calculated Sync  : Set save_interval = {save_interval}")
    print(f"Actual Output    : {actual_duration:.2f} seconds ({actual_frames} total frames)")
    print("---------------------------------------\n")
    
    return {
        "save_interval": save_interval,
        "num_saves": actual_frames    
        }