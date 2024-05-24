import json
import numpy as np
import imageio
import os

def export_to_video(video_frames, output_video_path, fps = 12):
    # Ensure all frames are NumPy arrays and determine video dimensions from the first frame
    assert all(isinstance(frame, np.ndarray) for frame in video_frames), "All video frames must be NumPy arrays."
    # Ensure output_video_path is ending with .mp4
    if not output_video_path.endswith('.mp4'):
        output_video_path += '.mp4'
    # Create a video file at the specified path and write frames to it
    with imageio.get_writer(output_video_path, fps=fps, format='mp4') as writer:
        for frame in video_frames:
            writer.append_data(
                (frame * 255).astype(np.uint8)
            )

def save_generation(video_frames, configs, base_path, file_name=None):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    p_config = configs["pipe_configs"]
    frames, steps, fps = p_config["num_frames"], p_config["steps"], p_config["fps"]
    if not file_name:
        index = [int(each.split('_')[0]) for each in os.listdir(base_path)]
        max_idex = max(index) if index else 0
        idx_str = str(max_idex + 1).zfill(6)


        key_info = '_'.join([str(frames), str(steps), str(fps)])
        file_name = f'{idx_str}_{key_info}'

    with open(f'{base_path}/{file_name}.json', 'w') as f:
        json.dump(configs, f, indent=4)

    export_to_video(video_frames, f'{base_path}/{file_name}.mp4', fps=p_config["export_fps"])

    return file_name

