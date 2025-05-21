import cv2
import argparse

from ultralytics import YOLO
from utils import tracking, depth
from transformers import pipeline

# !This script is created for testing purposes (on the PC) and is not part of the final solution!
def main():
    parser = argparse.ArgumentParser(description="Track a specific object and estimate depth in its bounding box")
    parser.add_argument("-c","--camera",            type = int,     default = 0,        help = "Camera index")
    parser.add_argument("-t_c","--target_class",    type = int,     default = 0,        help = "Class to track (0 = Person & 1 = Helmet)")
    parser.add_argument("-t_id","--target_id",      type = int,     default = 1,        help = "Track ID to focus on")
    parser.add_argument("-r_f","--resize_frame",    type = bool,    default = True,     help = "Resize the frame to 320x240 for simulation purposes")
    args = parser.parse_args()

    # Init
    detect_model    = YOLO('./models/helmet-medium.pt')
    depth_pipe      = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')

    # Initialize the camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera '{args.camera}'!")

    try:
        while True:
            ret, frame = cap.read()
            # Check if frame is read correctly
            if not ret:
                print("Error: Could not read frame.")
                break

            # Resize the frame for simulation purposes
            if (args.resize_frame):
                frame = cv2.resize(frame, (320, 240))

            # 1) Run tracker on full frame
            person = tracking.track_specific(detect_model, frame, args.target_class, args.target_id)

            # If not found, show the frame and continue
            if person is None:
                cv2.imshow("Depth Tracking", frame)
                continue
        
            # 2) Crop the frame to the bounding box of the tracked object
            _, _, x1, y1, x2, y2 = person
            depth_person = depth.get_depth(depth_pipe, frame, (x1, y1, x2, y2), normalize=True)

            # 3) Draw the bounding box and depth on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Depth: {depth_person:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            
            # 5) Show the frame
            cv2.imshow("Depth Tracking", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()