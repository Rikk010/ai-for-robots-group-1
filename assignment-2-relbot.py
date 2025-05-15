import argparse
import cv2
import time
import numpy as np

from transformers import pipeline
from PIL import Image
from utils import tracking, depth
from ultralytics import YOLO

depth_pipe  = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')
detect_model = YOLO('./models/helmet-medium.pt')
def assignment_2(frame, target_class=0,target_id=1, depth_factor=20000):
    """
    Track a specific object and estimate depth in its bounding box
    Args:
        frame: The input frame from the camera.
        target_class: The class to track (0=Person, 1=Helmet).
        target_id: The ID of the object to track (1=first to appear).
        depth_factor: Factor in relative depth to desired values (depth*depth_factor).
    """

    class_names = ["Person", "Helmet"]

    person = tracking.track_specific(detect_model, frame, target_class, target_id)

    if person is None:
        cv2.imshow("Depth Tracking", frame)
        return None

    track_id, cls_id, x1, y1, x2, y2 = person
    
    depth_person = depth.get_depth(depth_pipe, frame, (x1, y1, x2, y2), normalize=True)
    depth_person = depth_person * depth_factor
    
    # Draw a rectangle around the tracked object
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # draw depth
    cv2.putText(frame, f"Depth: {depth_person:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Depth Tracking", frame)
    