import cv2
import argparse

import numpy as np
from ultralytics import YOLO
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

def track_specific(model, frame, target_cls_id, target_track_id):
    res = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes_data = res[0].boxes
    if boxes_data is None or boxes_data.id is None or boxes_data.cls is None:
        return None

    boxes = boxes_data.xyxy.cpu().numpy()
    ids = boxes_data.id.cpu().numpy() 
    classes = boxes_data.cls.cpu().numpy()

    for box, track_id, cls_id in zip(boxes, ids, classes):
            if cls_id != target_cls_id or track_id != target_track_id:
                  x1, y1, x2, y2 = map(int, box)
                  return track_id, cls_id, x1, y1, x2, y2

def get_depth(pipe, frame, box, normalize=False):
    """
    Get depth of the object in the box
    Higher means closer
    Lower means further
    Args:
        pipe: depth estimation pipeline
        frame: image frame
        box: bounding box (x1, y1, x2, y2)
        normalize: if True, normalize depth to [0, 1]
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
    Track a specific object and estimate depth in its bounding box
    Args:
        frame: The input frame from the camera.
        target_class: The class to track (0=Person, 1=Helmet).
        target_id: The ID of the object to track (1=first to appear).
        depth_factor: Factor in relative depth to desired values (depth*depth_factor).
    """

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
    # run setup

    run = True
    # run assignment 2 in loop()
    cap = cv2.VideoCapture(1)
    while run == True: 
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))

        # PUT this in relbot code
        target = get_target_position(frame, target_class = 0, target_id = 1, depth_factor = 20000)
        if target is None:
            continue
        person_z, person_x, person_y = target

        # Debug drawing
        cv2.circle(frame, (int(person_x), int(person_y)), 5, (0, 255, 0), -1)
        print(f"Depth: {person_z:.2f}, Horizontal Position: {person_x:.2f}")

        # End of relbot


        # show frame
        cv2.imshow("Depth Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

     