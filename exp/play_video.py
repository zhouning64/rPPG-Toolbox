import cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os


#file_path = "dataset/MMPD_mini_stationary/subject1/Copy of p1_0.mat"
file_path = "dataset/MMPD_mini_stationary/subject1/Copy of p1_0.mat"
if os.path.exists(file_path):
    mat_file = loadmat(file_path)
else:
    print("File not found.")


# Load the .mat file
mat_file = loadmat("dataset/MMPD_mini_stationary/subject1/Copy\\ of\\ p1_0.mat")

# Extract data
video = mat_file['video']
ppg_waveform = mat_file['GT_ppg']
light = mat_file['light'].item()
motion = mat_file['motion'].item()

# Display metadata
print("Light condition:", light)
print("Motion type:", motion)

# Process and visualize PPG signal
plt.plot(ppg_waveform)
plt.title("PPG Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Extract video data
video = mat_file['video']  # Shape [t, w, h, c]

# Convert video frames to uint8 (if needed)
video = (video * 255).astype(np.uint8) if video.max() <= 1 else video

print(video.shape)
# Loop through each frame and display it
for i in range(video.shape[0]):  # Iterate over time (frames)
    frame = video[i]  # Get the i-th frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    cv2.imshow('Video Playback', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # Wait for 30ms between frames, press 'q' to quit
        break

cv2.destroyAllWindows()