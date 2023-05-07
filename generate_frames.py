import cv2
import os
from tqdm import tqdm

# Open the video file

def preprocess_video(path):

    cap = cv2.VideoCapture(path)

    # Create a directory to store the extracted frames
    name = path.split('/')[-1]
    name = name.split('.')[0].split('_')[-1]
    mode = path.split('/')[-4]
    
    output_dir = f'extracted_frames/{mode}/{name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the time interval to extract frames (in milliseconds)
    interval = 100

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip
    skip_frames = int(fps * (interval / 1000))

    # Initialize variables
    frame_count = 0
    success = True

    # Loop through the video frames
    while success:
        # Read a frame from the video
        success, frame = cap.read()
        # Check if the frame was read successfully
        if success:
            # Check if the current frame is a multiple of the skip frames
            if frame_count % skip_frames == 0:
                # Save the extracted frame as an image file
                output_path = os.path.join(output_dir, f"frame{frame_count}.jpg")
                cv2.imwrite(output_path, frame)
                

            # Increment the frame count
            frame_count += 1

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


import glob
path = 'data/'
videos = [f for f in glob.glob(path + "**/*.mp4", recursive=True)]
for video in tqdm(videos):
    preprocess_video(video)