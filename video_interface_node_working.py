#!/usr/bin/env python3

import gi
import rclpy
import numpy as np
import cv2

from PIL import Image
from gi.repository import Gst
from rclpy.node import Node
from geometry_msgs.msg import Point
from transformers import pipeline
from ultralytics import YOLO

# ------------------------------------------------------
# Constants
# ------------------------------------------------------
GST_PIPELINE_DEFAULT = (
    'udpsrc port=5000 caps="application/x-rtp,media=video,'
    'encoding-name=H264,payload=96" ! '
    'rtph264depay ! avdec_h264 ! tee name=t '
    't. ! queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink '
    't. ! queue ! x264enc ! mp4mux ! filesink location=output.mp4'
)
ASSIGNMENT_TO_RUN   = 1
TARGET_CLASS        = 0       # 0 = Person | 1 = Helmet
TARGET_ID           = 1
DEPTH_FACTOR        = 20000

# ------------------------------------------------------
# Initialise GStreamer
# ------------------------------------------------------
gi.require_version('Gst', '1.0')
Gst.init(None)

# ------------------------------------------------------
# Load models
# ------------------------------------------------------
DETECT_MODEL    = YOLO('./models/helmet-medium.pt')
DEPTH_PIPELINE  = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def get_depth(pipe, frame, box, normalise=False):
    """
    Calculate mean depth inside the given bounding box.
    Higher values indicate nearer objects.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map = pipe(Image.fromarray(rgb))["depth"]
    depth_arr = np.array(depth_map)

    if normalise:
        depth_arr = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min())

    x1, y1, x2, y2 = map(int, box)
    crop = depth_arr[y1:y2, x1:x2]
    return float(np.mean(crop))


def track(model, frame):
    """
    Run object tracking and return list of entries:
    (track_id, class_id, x1, y1, x2, y2, confidence).
    """
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes

    xyxy = boxes.xyxy.cpu().numpy()
    ids  = boxes.id.cpu().numpy()   if hasattr(boxes, "id")   and boxes.id   is not None else [None] * len(xyxy)
    cls  = boxes.cls.cpu().numpy()  if hasattr(boxes, "cls")  and boxes.cls  is not None else [None] * len(xyxy)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else [None] * len(xyxy)

    entries = []
    for (x1, y1, x2, y2), track_id, cls_id, c in zip(xyxy, ids, cls, conf):
        if track_id is None or cls_id is None:
            continue
        entries.append((
            int(track_id),
            int(cls_id),
            int(x1), int(y1), int(x2), int(y2),
            float(c) if c is not None else None
        ))
    return entries

def track_specific(model, frame, target_class, target_track_id):
    """
    Find and return the first other object that is NOT the target.
    """
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    ids  = boxes.id.cpu().numpy()  if hasattr(boxes, "id")  and boxes.id  is not None else [None] * len(xyxy)
    cls  = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") and boxes.cls is not None else [None] * len(xyxy)

    for (x1, y1, x2, y2), track_id, cls_id in zip(xyxy, ids, cls):
        if cls_id != target_class or track_id != target_track_id:
            return int(track_id), int(cls_id), int(x1), int(y1), int(x2), int(y2)

    return None

# ------------------------------------------------------
# Main node
# ------------------------------------------------------
class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        self.position_pub = self.create_publisher(Point, '/object_position', 10)

        self.declare_parameter('gst_pipeline', GST_PIPELINE_DEFAULT)
        pipeline_str = self.get_parameter('gst_pipeline').value

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink     = self.pipeline.get_by_name('sink')
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialised, streaming at 30 Hz')

    def on_timer(self):
        sample = self.sink.emit('pull-sample')
        if not sample:
            return

        buf    = sample.get_buffer()
        caps   = sample.get_caps()
        width  = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return

        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        buf.unmap(mapinfo)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV/YOLO
        horizontal_position = 160.0                         # Fallback value

        if ASSIGNMENT_TO_RUN == 1:
            result = self.assignment_1(frame_bgr)   # Returns the horizontal position
            if result is not None:
                horizontal_position = frame_bgr
        elif ASSIGNMENT_TO_RUN == 2:
            result = self.assignment_2(frame_bgr)   # Returns the depth and horizontal position
            if result is not None:
                depth_person, horizontal_position = result
        elif ASSIGNMENT_TO_RUN == 3:
            self.get_logger().info("Assignment 3 is not implemented yet.")
            return
        else:
            self.get_logger().error(f"Invalid assignment number: {ASSIGNMENT_TO_RUN}. Please set it to 1, 2 or 3.")
            return

        msg = Point(x=horizontal_position, y=0.0, z=10001.0)
        self.position_pub.publish(msg)

    def assignment_1(self, frame, target_class = TARGET_CLASS, target_id = TARGET_ID):
        tracks = track(DETECT_MODEL, frame)

        class_names = ["Person", "Helmet"]
        tracked = False
        track_x1, track_x2 = 0, 0
        for track_id, cls_id, x1, y1, x2, y2, conf in tracks:
            if track_id == target_id and cls_id == target_class:
                tracked = True
                track_x1, track_x2 = x1, x2

                # Draw a rectangle around the target object 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"TARGET | Class: {class_names[cls_id]} | Id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
            else:
                # Draw a rectangle around the other object(s)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"OTHER | Class: {class_names[cls_id]} | Id: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

        self.show_debug_window(frame, title = "Frame Tracking")

        if not tracked:
            return None

        horizontal_position = (track_x1 + track_x2) / 2
        return horizontal_position

    def assignment_2(self, frame, target_class = 0, target_id = 1, depth_factor = 20000):
            person = track_specific(DETECT_MODEL, frame, target_class, target_id)

            if person is None:
                self.show_debug_window(frame, title="Depth Tracking")
                return None

            _, _, x1, y1, x2, y2 = person
            depth_person = get_depth(DEPTH_PIPELINE, frame, (x1, y1, x2, y2), normalize=True)
            depth_person = depth_person * depth_factor

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Depth: {depth_person:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.show_debug_window(frame, title = "Depth Tracking")

            horizontal_position = (x1 + x2) / 2
            return depth_person, horizontal_position

    def show_debug_window(self, frame, title):
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
