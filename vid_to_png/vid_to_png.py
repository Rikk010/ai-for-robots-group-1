import cv2
import os
import csv

# Configuration
video_path = "vid_to_png/slamtest4/slamtest4.avi"  # Your .avi file
output_dir = "vid_to_png/slamtest4/data"  # Directory to save frames
csv_filename = "data.csv"
fps = 30  # Target frame rate
ns_per_frame = int(1e9 / fps)  # Nanoseconds per frame

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {video_path}")

# Prepare CSV file
csv_path = os.path.join(os.path.dirname(output_dir), csv_filename)
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["nanosecond", "filename"])

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Calculate timestamp in nanoseconds
    timestamp_ns = frame_idx * ns_per_frame
    filename = f"{timestamp_ns}.png"
    filepath = os.path.join(output_dir, filename)

    # Save frame as PNG
    cv2.imwrite(filepath, frame)

    # Write to CSV
    csv_writer.writerow([timestamp_ns, filename])

    frame_idx += 1

# Clean up
cap.release()
csv_file.close()

print(f"Extracted {frame_idx} frames to '{output_dir}' and wrote timestamps to '{csv_path}'.")