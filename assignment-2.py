import argparse
import cv2
import time
import numpy as np

from transformers import pipeline
from PIL import Image
from utils import tracking, depth
from ultralytics import YOLO

def main():
    
    parser = argparse.ArgumentParser(
        description="Track a specific object and estimate depth in its bounding box"
    )
    parser.add_argument("-c","--camera", type=int, default=0, help="Camera index")

    args = parser.parse_args()

    # Init pipelines
    depth_pipe  = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')
    detect_model = YOLO('./models/helmet-medium.pt')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera")

    class_names = ["Person", "Helmet"]
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) Run tracker on full frame
            person = tracking.track_specific(detect_model, frame, 0, 1)

            if person is None:
                cv2.imshow("Depth Tracking", frame)
                continue
        
            track_id, cls_id, x1, y1, x2, y2 = person
            
            depth_person = depth.get_depth(depth_pipe, frame, (x1, y1, x2, y2), normalize=True)
            # Draw a rectangle around the tracked object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw depth
            cv2.putText(frame, f"Depth: {depth_person:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Depth Tracking", frame)
           

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()