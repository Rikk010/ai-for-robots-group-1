import cv2
import argparse

from ultralytics import YOLO
from utils import tracking

# !This script is created for testing purposes (on the PC) and is not part of the final solution!
def main():
    parser = argparse.ArgumentParser(description="Track a specific object")
    parser.add_argument("-c","--camera",            type = int,     default = 0,        help = "Camera index")
    parser.add_argument("-t_c","--target_class",    type = int,     default = 0,        help = "Class to track (0 = Person & 1 = Helmet)")
    parser.add_argument("-t_id","--target_id",      type = int,     default = 1,        help = "Track ID to focus on")
    parser.add_argument("-r_f","--resize_frame",    type = bool,    default = True,     help = "Resize the frame to 320x240 for simulation purposes")
    args = parser.parse_args()

    # Init
    detect_model = YOLO("./models/helmet-medium.pt")
    track_x1, track_x2, track_y1, track_y2 = 0.0, 0.0, 0.0, 0.0

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
            tracks = tracking.track(detect_model, frame)
    
            # 2) Check for the target class and ID
            class_names = ["Person", "Helmet"]
            for track_id, cls_id, x1, y1, x2, y2, conf in tracks:
                if track_id == args.target_id and cls_id == args.target_class:
                    track_x1, track_x2, track_y1, track_y2 = x1, x2, y1, y2

                    # Draw a rectangle around the tracked object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"TARGET | Class: {class_names[cls_id]} | Id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                else:
                    # Draw a rectangle around the detected object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"OTHER | Class: {class_names[cls_id]} | Id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

            # 3) Calculate the relative position of the tracked object
            relative_x = (track_x1 + track_x2) / 2 / frame.shape[1]
            relative_y = (track_y1 + track_y2) / 2 / frame.shape[0]

            # 4) Print the relative position state
            if relative_x < 0.5:
                print("Object is on the left side of the frame -> Rotating to the left...")
            elif relative_x > 0.5:
                print("Object is on the right side of the frame -> Rotating to the right...")

            # 5) Show the frame
            cv2.imshow("Frame Tracking", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()