import argparse
import cv2
import time
import numpy as np

from transformers import pipeline
from PIL import Image
from utils import tracking
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Track a specific object and estimate depth in its bounding box"
    )
    parser.add_argument("-c","--camera",       type=int,   default=0,                                           help="Camera index")
    parser.add_argument("-m","--depth_model",  type=str,   default="depth-anything/Depth-Anything-V2-Small-hf", help="Depth‐estimation model")
    parser.add_argument("-d","--detect_model", type=str,   default="./models/helmet-medium.pt",                 help="YOLO detection model")
    parser.add_argument("--target_class",      type=int,   default=0,                                           help="Class to track (0=Person,1=Helmet)")
    parser.add_argument("--target_id",         type=int,   default=1,                                           help="Track ID to focus on")
    args = parser.parse_args()

    # Init pipelines
    depth_pipe  = pipeline(task="depth-estimation", model=args.depth_model)
    detect_model = YOLO(args.detect_model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    class_names = ["Person", "Helmet"]
    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) Run tracker on full frame
            # Returns (track_id, cls_id, x1, y1, x2, y2, conf)
            tracks = tracking.track(detect_model, frame)

            # 2) Find the desired track and draw its bbox + label
            roi = None
            for track_id, cls_id, x1, y1, x2, y2, conf in tracks:
                if track_id == args.target_id and cls_id == args.target_class:
                    roi = (x1, y1, x2, y2)
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw name, id and accuracy
                    label = f"Class: {class_names[cls_id]} | Id: {track_id} | Acc: {conf:.2f}"
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
                    break

            # 3) If found, crop & do depth only on that patch
            if roi:
                x1, y1, x2, y2 = roi
                patch = frame[y1:y2, x1:x2]
                rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                depth = depth_pipe(Image.fromarray(rgb))["depth"]
                depth_np = np.array(depth, dtype=np.float32)
                depth_8u = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_bgr = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)
                frame[y1:y2, x1:x2] = depth_bgr

                # Compute a single depth value for the ROI
                median_depth = float(np.median(depth_np))
                print(f"Median ROI depth value = {median_depth:.3f}")

                # Display depth value just below the tracked box
                cv2.putText(
                    frame, f"Depth: {median_depth:.2f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

            # 4) Calculate and display FPS
            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Exit on 'q' key press
            cv2.imshow("Depth Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()