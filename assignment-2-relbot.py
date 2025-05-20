import cv2

from transformers import pipeline
from utils import tracking, depth
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2

detect_model    = YOLO('./models/helmet-medium.pt')
depth_pipe      = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')

def assignment_2(frame, target_class = 0, target_id = 1, depth_factor = 20000):
    """
    Track a specific object and estimate depth in its bounding box
    Args:
        frame: The input frame from the camera.
        target_class: The class to track (0=Person, 1=Helmet).
        target_id: The ID of the object to track (1=first to appear).
        depth_factor: Factor in relative depth to desired values (depth*depth_factor).
    """

    # 1) Run tracker on full frame
    person = tracking.track_specific(detect_model, frame, target_class, target_id)

    # If not found, show the frame and continue
    if person is None:
        cv2.imshow("Depth Tracking", frame)
        return None

    # 2) Crop the frame to the bounding box of the tracked object
    _, _, x1, y1, x2, y2 = person
    depth_person = depth.get_depth(depth_pipe, frame, (x1, y1, x2, y2), normalize=True)
    depth_person = depth_person * depth_factor
    
    # 3) Draw the bounding box and depth on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Depth: {depth_person:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Depth Tracking", frame)

    horizontal_position = (x1 + x2) / 2
    return depth_person, horizontal_position


if __name__ == "__main__":
    # run setup

    run = True
    # run assignment 2 in loop()
    while run == True: 
        frame = 0
        target_class = 0
        target_id = 1
        depth_factor = 20000

        assignment_2(frame, target_class = target_class, target_id = target_id, depth_factor = depth_factor)