import cv2
import time
import torch


def record_video(duration=10,
                 output_file="record.mp4",
                 camera_index=0,
                 width=640,
                 height=480,
                 fps=20.0):
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Enforce the desired width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Now read back the actual frame size to confirm
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested size: {width}x{height}, actual size: {actual_width}x{actual_height}")

    # Define the codec and create VideoWriter (MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (actual_width, actual_height))

    print(f"Recording video for {duration} seconds...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        out.write(frame)  # Save frame

        # (Optional) Display frame
        cv2.imshow('Recording...', frame)

        # Stop after duration or if 'q' pressed
        if (time.time() - start_time) > duration:
            print("Time is up.")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_file}")


def load_mpg_into_tensor(filepath):
    """
    Reads a .mpg (MPEG) video file using OpenCV,
    converts each frame to a torch tensor,
    and returns a single tensor of shape (num_frames, channels, height, width).
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {filepath}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            # No more frames or error reading
            break

        # frame is in BGR format by default (OpenCV)
        # Convert to RGB to maintain consistent color channel ordering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert NumPy array (H, W, C) to a PyTorch tensor with shape (C, H, W)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # (3, H, W)

        frames.append(frame_tensor)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from file: {filepath}")

    # Stack all frame tensors along a new dimension at the front
    # Result shape: (batch_size, num_frames, 3, height, width)
    video_tensor = torch.stack(frames, dim=0)
    video_tensor = video_tensor.unsqueeze(0)
    video_tensor = video_tensor.float()
    video_tensor = video_tensor[:, :600, :, :, :]
    return video_tensor



if __name__ == "__main__":
    record_video(
        duration=35,
        output_file="../run_video/record.mp4",
        camera_index=0,
        width=60,
        height=80,
        fps=30.0
    )
    load_mpg_into_tensor("../run_video/record.mp4")