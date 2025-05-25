import cv2
import numpy as np

from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

detect_model = YOLO('./models/helmet-medium.pt')
depth_pipe   = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')

def track_specific(model, frame, target_cls_id, target_track_id):
    """
    Track a specific object by class and track ID.

    Args:
        model:           YOLO model instance with tracking enabled.
        frame:           Input image frame (BGR numpy array).
        target_cls_id:   Integer class ID to match (0=Person, 1=Helmet).
        target_track_id: Integer track ID to match (1=first detected).

    Returns:
        A tuple (track_id, cls_id, x1, y1, x2, y2) of the bounding box
        and identifiers for the first matching object, or None if not found.
    """
    res = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes_data = res[0].boxes

    # If no boxes detected, return None
    if boxes_data is None or boxes_data.id is None or boxes_data.cls is None:
        return None

    boxes   = boxes_data.xyxy.cpu().numpy()
    ids     = boxes_data.id.cpu().numpy()
    classes = boxes_data.cls.cpu().numpy()

    for box, track_id, cls_id in zip(boxes, ids, classes):
        if cls_id != target_cls_id or track_id != target_track_id:
            x1, y1, x2, y2 = map(int, box)
            return track_id, cls_id, x1, y1, x2, y2

    # If no matching object found, return None
    return None

def get_depth(pipe, frame, box, normalize=False):
    """
    Get depth of the object in the box.
    Higher values mean closer objects; lower values mean farther objects.

    Args:
        pipe:      Depth-estimation pipeline.
        frame:     Image frame (BGR numpy array).
        box:       Bounding box tuple (x1, y1, x2, y2).
        normalize: If True, normalise the depth map to [0, 1] before averaging.

    Returns:
        Mean depth value inside the box.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth = pipe(Image.fromarray(rgb))["depth"]

    # Get mean depth in box
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    depth = np.array(depth)
    if normalize:
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    # Crop depth to box
    depth = depth[y1:y2, x1:x2]
    depth = np.mean(depth)

    return depth

def get_target_position(frame, target_class = 0, target_id = 1, depth_factor = 20000):
    """
    Track a specific object and estimate its depth and screen position.

    Args:
        frame:        The input frame from the camera (BGR numpy array).
        target_class: The class to track (0=Person, 1=Helmet).
        target_id:    The ID of the object to track (1=first detected).
        depth_factor: Scale factor to convert normalised depth into relative units.

    Returns:
        (depth, x_center, y_center) or None if the target was not found.
    """
    # 1) Run tracker on full frame
    person = track_specific(detect_model, frame, target_class, target_id)

    # If not found, show the frame and continue
    if person is None:
        return None

    # 2) Crop the frame to the bounding box of the tracked object
    _, _, x1, y1, x2, y2 = person
    depth_person = get_depth(depth_pipe, frame, (x1, y1, x2, y2), normalize=True)
    depth_person = depth_person * depth_factor
    
    horizontal_position = (x1 + x2) / 2
    vertical_position = (y1 + y2) / 2

    return depth_person, horizontal_position, vertical_position

if __name__ == "__main__":
    # Run setup
    run = True

    # Run assignment 2 in loop()
    cap = cv2.VideoCapture(1)
    while run == True: 
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))
        
        DEPTH_FACTOR       = 20000
        OBSTACLE_THRESHOLD = 5000

        # Get the depth of the left and right side of the frame & Scale them
        left_side_depth  = get_depth(depth_pipe, frame, (0, 0, int(frame.shape[1]/4), frame.shape[0]), normalize=True)
        right_side_depth = get_depth(depth_pipe, frame, (int(frame.shape[1]/4*3), 0, frame.shape[1], frame.shape[0]), normalize=True)
        
        left_side_depth  = left_side_depth * DEPTH_FACTOR
        right_side_depth = right_side_depth * DEPTH_FACTOR

        # Check if there are obstacles on the left or right side
        is_left_side_obstacle  = False
        is_right_side_obstacle = False

        if left_side_depth > OBSTACLE_THRESHOLD:
            is_left_side_obstacle = True
            cv2.circle(frame, (int(frame.shape[1]/4*3), int(frame.shape[0]/2)), 5, (0, 0, 255), -1)
        if right_side_depth > OBSTACLE_THRESHOLD:
            is_right_side_obstacle = True
            cv2.circle(frame, (int(frame.shape[1]/4), int(frame.shape[0]/2)), 5, (0, 0, 255), -1)

        print(f"SIDE DEPTHS | Left: {left_side_depth} & Right: {right_side_depth}")

        # Put this in relbot code
        target = get_target_position(frame, target_class = 0, target_id = 1, depth_factor = DEPTH_FACTOR)
        if target is None:
            # No target found, don't change anything
            cv2.imshow("Depth Tracking", frame)
            continue

        # If target is found, assign the values
        person_z, person_x, person_y = target
        target_x, target_y, target_z = person_x, person_y, person_z

        # If there is an obstacle on the left side, set the target pos to the opposite side
        if is_left_side_obstacle:
            target_x = 320
        if is_right_side_obstacle:
            target_x = 0

        # Draw the debug circle on the tracked person
        cv2.circle(frame, (int(person_x), int(person_y)), 5, (0, 255, 0), -1)
        print(f"Person depth: {person_z:.2f}, Person horizontal position: {person_x:.2f}")

        # End of relbot

        # Show frame
        cv2.imshow("Depth Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
