from utils import tracking
from utils import detection

from ultralytics import YOLO
import cv2

CAMERA_ID= 1
TARGET_CLASS = 0 # 0: Person 1:Helmet
TARGET_ID = 1 # 1 is the first ID given

model = YOLO("./models/helmet-medium.pt")


cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break


    tracks = tracking.track(model, frame)

    class_names = ["Person", "Helmet"]
    for track_id, cls_id, x1, y1, x2, y2 in tracks:
        if track_id == TARGET_ID and cls_id == TARGET_CLASS:
            # Draw a rectangle around the tracked object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracked: {class_names[cls_id]} id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Other {class_names[cls_id]} id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    relative_x = (x1 + x2) / 2 / frame.shape[1]
    relative_y = (y1 + y2) / 2 / frame.shape[0]

    if relative_x < 0.5:
        print("Object is on the left side of the frame, rotating....")
        # TODO: rotate robot
    elif relative_x > 0.5:
        print("Object is on the right side of the frame, rotating....")

    cv2.imshow("Webcam Feed", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




