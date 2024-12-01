import os
import math
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips

def merge_videos_to_grid(video_files, grid_size, output_filename):
    # Load the video clips
    clips = [VideoFileClip(vf) for vf in video_files]
    
    # Remove audio to prevent issues during merging
    clips = [clip.without_audio() for clip in clips]
    
    # Ensure all clips have the same duration
    min_duration = min(clip.duration for clip in clips)
    clips = [clip.subclip(0, min_duration) for clip in clips]
    
    # Resize clips to the smallest width and height among them
    widths, heights = zip(*[(clip.w, clip.h) for clip in clips])
    min_width = min(widths)
    min_height = min(heights)
    clips = [clip.resize((min_width, min_height)) for clip in clips]
    
    # Pad the last batch if necessary
    num_clips_needed = grid_size * grid_size
    if len(clips) < num_clips_needed:
        # Create a black clip to pad
        black_clip = clips[0].fx(lambda gf, t: 0, apply_to='mask')
        black_clip = black_clip.set_duration(min_duration).resize((min_width, min_height))
        clips.extend([black_clip] * (num_clips_needed - len(clips)))
    
    # Arrange clips into a grid
    grid_clips = []
    for i in range(0, len(clips), grid_size):
        row_clips = clips[i:i+grid_size]
        grid_clips.append(row_clips)
    grid_video = clips_array(grid_clips)
    
    # Write the result to a file
    grid_video.write_videofile(output_filename, codec='libx264')
    
    # Close the clips
    for clip in clips:
        clip.close()
    grid_video.close()
    print(f"Created grid video: {output_filename}")

def process_all_videos(input_dirs, grid_size=2, merged_output_dir='merged_videos', final_output='final_video.mp4'):
    # Collect all video files
    video_files = []
    for input_dir in input_dirs:
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith('.mp4'):
                video_files.append(os.path.join(input_dir, filename))
    
    # Sort video files to ensure consistent ordering
    video_files.sort()
    
    # Create directory for merged videos
    if not os.path.exists(merged_output_dir):
        os.makedirs(merged_output_dir)
    
    # Group videos into batches
    num_videos = len(video_files)
    batch_size = grid_size * grid_size
    num_batches = math.ceil(num_videos / batch_size)
    merged_videos = []
    for i in range(num_batches):
        batch_videos = video_files[i*batch_size:(i+1)*batch_size]
        output_filename = os.path.join(merged_output_dir, f'merged_{i+1}.mp4')
        merge_videos_to_grid(batch_videos, grid_size, output_filename)
        merged_videos.append(output_filename)
    
    # # Concatenate the merged videos into a final video
    # merged_clips = [VideoFileClip(mv) for mv in merged_videos]
    # final_clip = concatenate_videoclips(merged_clips)
    # final_clip.write_videofile(final_output, codec='libx264')
    # print(f"Final video created: {final_output}")

    # Concatenate the merged videos into a final video
    merged_clips = [VideoFileClip(mv) for mv in merged_videos]
    final_clip = concatenate_videoclips(merged_clips)

    # Halve the FPS for the final output
    original_fps = merged_clips[0].fps  # Assuming all merged videos have the same FPS
    final_fps = original_fps // 2

    # Write the final video file with halved FPS
    final_clip.write_videofile(final_output, codec='libx264', fps=final_fps)
    print(f"Final video created: {final_output}")
    
    # Close all clips
    for clip in merged_clips:
        clip.close()
    final_clip.close()

if __name__ == '__main__':
    # Directories containing your videos
    input_dirs = ['./k=0.001, r=0.005', './k=0.05, r=0.01']
    
    # Choose grid size (2 for 2x2 grid, 3 for 3x3 grid)
    grid_size = 3  # or set to 3 for 3x3 grid
    
    process_all_videos(
        input_dirs=input_dirs,
        grid_size=grid_size,
        merged_output_dir='merged_videos',
        final_output='final_video.mp4'
    )