#!/usr/bin/env python3
from tkinter import Image
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2

from transformers import pipeline
from ultralytics import YOLO

gi.require_version('Gst', '1.0')
from gi.repository import Gst

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
    vertical_position   = (y1 + y2) / 2

    return depth_person, horizontal_position, vertical_position

class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        self.position_pub = self.create_publisher(Point, '/object_position', 10)

        self.declare_parameter('gst_pipeline', (
            'udpsrc port=5000 caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96" ! '
            'rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw,format=RGB ! appsink name=sink'
        ))
        pipeline_str = self.get_parameter('gst_pipeline').value

        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink     = self.pipeline.get_by_name('sink')
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

    def on_timer(self):
        sample = self.sink.emit('pull-sample')
        if not sample:
            return

        buf         = sample.get_buffer()
        caps        = sample.get_caps()
        width       = caps.get_structure(0).get_value('width')
        height      = caps.get_structure(0).get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return

        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        buf.unmap(mapinfo)

        # --- Video recording setup ---
        if not hasattr(self, 'video_writer'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
        # Save a copy of the original frame for recording (no annotations)
        frame_for_record = frame.copy()
        # ----------------------------

        # --- Draw boxes for all detected persons ---
        res = detect_model.track(frame, persist=True, tracker="bytetrack.yaml")
        boxes_data = res[0].boxes
        if boxes_data is not None and boxes_data.id is not None and boxes_data.cls is not None:
            boxes   = boxes_data.xyxy.cpu().numpy()
            ids     = boxes_data.id.cpu().numpy()
            classes = boxes_data.cls.cpu().numpy()
            for box, track_id, cls_id in zip(boxes, ids, classes):
                if cls_id == 0:  # Only draw for persons
                    x1, y1, x2, y2 = map(int, box)
                    if track_id == 1:
                        color = (0, 255, 0)  # Green for tracked person (ID 1)
                        thickness = 3
                    else:
                        color = (0, 0, 255)  # Red for others
                        thickness = 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        # --- End box drawing ---

        # -------------------------
        # Start Assignment 2B
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

        target = get_target_position(frame, target_class = 0, target_id = 1, depth_factor = DEPTH_FACTOR)
        if target is None:
            msg   = Point()
            msg.x = 160.0
            msg.y = 0.0
            msg.z = 10001.0
            self.position_pub.publish(msg)
            print("No target found")
            return
        
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
        print(f"Person depth: {person_z:.2f} | Person horizontal position: {person_x:.2f}")

        msg = Point()
        msg.x = target_x
        msg.y = target_y
        msg.z = target_z
        self.position_pub.publish(msg)

        # Write the unannotated frame to the video file
        self.video_writer.write(frame_for_record)
        # End Assignment 2B
        # -------------------------
   
    def show_debug_window(self, frame, title="Preview"):
        resized_frame = cv2.resize(frame, (320, 240))
        cv2.imshow(title, resized_frame)
        cv2.moveWindow(title, 100, 100)
        cv2.waitKey(1)

    def destroy_node(self):
        # --- Release video writer if present ---
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        # --------------------------------------
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()

def main(args=None):
    print('Starting...')
    rclpy.init(args=args)
    node = VideoInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # --- Release video writer if present on interrupt ---
        if hasattr(node, 'video_writer'):
            node.video_writer.release()
        # --------------------------------------
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
