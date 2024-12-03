import cv2
import os
import random

# Specify the path to the video file
video_path = r'v1\k=0.005, r=0.005, U_rr=0.05, rho=7800.0, is_disk=True, omega1_0=5.0, omega2_0=0.0.mp4'

# Specify the output image file name
output_image_path = 'simulation_frame.png'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Set the frame number to the 10th second
frame_number = fps * 10

# Set the video to the frame number
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the frame
ret, frame = cap.read()

if ret:
    # Save the frame as an image file
    cv2.imwrite(output_image_path, frame)
    print(f"Frame {frame_number} extracted and saved as {output_image_path}")
else:
    print("Failed to extract frame")

# Release the VideoCapture object
cap.release()