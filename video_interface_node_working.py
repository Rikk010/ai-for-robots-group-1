#!/usr/bin/env python3
import rclpy
import gi
import numpy as np
import cv2

from rclpy.node import Node
from geometry_msgs.msg import Point
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Load models globally
detect_model = YOLO('./models/helmet-medium.pt')
depth_pipe = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')

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

def track(model, frame):
    res     = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes   = res[0].boxes.xyxy.cpu().numpy()
    ids     = res[0].boxes.id.cpu().numpy() if hasattr(res[0].boxes, "id") and res[0].boxes.id is not None else [None] * len(boxes)
    classes = res[0].boxes.cls.cpu().numpy() if hasattr(res[0].boxes, "cls") and res[0].boxes.cls is not None else [None] * len(boxes)
    confs   = res[0].boxes.conf.cpu().numpy() if hasattr(res[0].boxes, "conf") and res[0].boxes.conf is not None else [None] * len(boxes)

    tracking_entries = []
    for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
            if track_id is None or cls_id is None:
                  continue
            x1, y1, x2, y2 = map(int, box)
            tracking_entries.append((
                int(track_id),
                int(cls_id),
                x1, y1, x2, y2,
                float(conf) if conf is not None else None
            ))
    return tracking_entries

def track_specific(model, frame, target_cls_id, target_track_id):
    res     = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes   = res[0].boxes.xyxy.cpu().numpy()
    ids     = res[0].boxes.id.cpu().numpy() if hasattr(res[0].boxes, "id") and res[0].boxes.id is not None else [None] * len(boxes)
    classes = res[0].boxes.cls.cpu().numpy() if hasattr(res[0].boxes, "cls") and res[0].boxes.cls is not None else [None] * len(boxes)

    for box, track_id, cls_id in zip(boxes, ids, classes):
            if cls_id != target_cls_id or track_id != target_track_id:
                  x1, y1, x2, y2 = map(int, box)
                  return track_id, cls_id, x1, y1, x2, y2


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
        self.sink = self.pipeline.get_by_name('sink')
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')

    def on_timer(self):
        sample = self.sink.emit('pull-sample')
        if not sample:
            return

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return

        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        buf.unmap(mapinfo)

        # Convert RGB to BGR for OpenCV/YOLO
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        result = self.assignment_1(frame_bgr)

        if result:
            depth_person, horizontal_position = result
            msg = Point()
            msg.x = horizontal_position
            msg.y = 0.0
            msg.z = 10001.0
            self.position_pub.publish(msg)
        else:
            msg = Point()
            msg.x = 160.0
            msg.y = 0.0
            msg.z = 10001.0
            self.position_pub.publish(msg)

    def assignment_1(self, frame, target_class=0, target_id=1):
        tracks = track(detect_model, frame)

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

        self.show_debug_window(frame, title="Frame Tracking")

        if not tracked:
            return None

        horizontal_position = (track_x1 + track_x2) / 2
        return 0.0, horizontal_position  # depth value is mocked as 0.0 for compatibility

    def assignment_2(self, frame, target_class=0, target_id=1, depth_factor=20000):
        person = track_specific(detect_model, frame, target_class, target_id)

        if person is None:
            self.show_debug_window(frame, title="Depth Tracking")
            return None

        _, _, x1, y1, x2, y2 = person
        depth_person = get_depth(depth_pipe, frame, (x1, y1, x2, y2), normalize=True)
        depth_person = depth_person * depth_factor

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Depth: {depth_person:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.show_debug_window(frame, title="Depth Tracking")

        horizontal_position = (x1 + x2) / 2
        return depth_person, horizontal_position

    def show_debug_window(self, frame, title="Preview"):
        # Resize frame for smaller preview (e.g., 320x240)
        resized_frame = cv2.resize(frame, (320, 240))
        cv2.imshow(title, resized_frame)
        cv2.moveWindow(title, 100, 100)  # Position the window on screen
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
