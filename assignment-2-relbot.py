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

        # -------------------------
        # Start Assignment 2(A)
        target = get_target_position(frame, target_class = 0, target_id = 1, depth_factor = 20000)
        if target is None:
            msg   = Point()
            msg.x = 160.0
            msg.y = 0.0
            msg.z = 10001.0
            self.position_pub.publish(msg)
            print("No target found")
            return

        person_z, person_x, person_y = target
        print(f"Person depth: {person_z:.2f} | Person horizontal position: {person_x:.2f}")

        msg = Point()
        msg.x = person_x
        msg.y = person_y
        msg.z = person_z
        self.position_pub.publish(msg)

        # End Assignment 2(A)
        # -------------------------

    def show_debug_window(self, frame, title="Preview"):
        resized_frame = cv2.resize(frame, (320, 240))
        cv2.imshow(title, resized_frame)
        cv2.moveWindow(title, 100, 100)
        cv2.waitKey(1)

    def destroy_node(self):
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()

def main(args=None):
    print('Starting...')
    rclpy.init(args=args)
    node = VideoInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
