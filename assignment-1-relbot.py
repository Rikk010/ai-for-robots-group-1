import cv2

from utils import tracking
from ultralytics import YOLO

detect_model = YOLO("./models/helmet-medium.pt")

def assignment_1(frame, target_class=0, target_id=1):
    
    tracks = tracking.track(detect_model, frame)

    class_names = ["Person", "Helmet"]
    for track_id, cls_id, x1, y1, x2, y2, conf in tracks:
        if track_id == target_id and cls_id == target_class:
            # Draw a rectangle around the tracked object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"TARGET | Class: {class_names[cls_id]} | Id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
        else:
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"OTHER | Class: {class_names[cls_id]} | Id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

    relative_x = (x1 + x2) / 2 / frame.shape[1]
    relative_y = (y1 + y2) / 2 / frame.shape[0]

    if relative_x < 0.5:
        print("Object is on the left side of the frame, rotating....")
        # TODO: Rotate robot to the left
    elif relative_x > 0.5:
        print("Object is on the right side of the frame, rotating....")
        # TODO: Rotate robot to the right
    
    cv2.imshow("Frame Tracking", frame)

    horizontal_position = (x1 + x2) / 2
    return horizontal_position