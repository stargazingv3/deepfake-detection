import os
import cv2
import random

# Define the paths
video_dir = 'datasets/Celeb-DF-v2/'
output_dir = 'datasets/Celeb-DF-v2-images/'

# Function to extract a random frame from a video
def extract_random_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame = random.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()

# Iterate over the directories and process each video
for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith('.mp4'):  # Assuming the videos are in .mp4 format
            video_path = os.path.join(root, file)
            # Create the corresponding output directory structure
            relative_path = os.path.relpath(root, video_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.jpg")
            extract_random_frame(video_path, output_path)

print("Random frames have been extracted and saved to the output directory while maintaining the directory structure.")
