#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2

from transformers import pipeline
from ultralytics import YOLO
from utils import tracking, depth  # Ensure these are correctly imported

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Load models globally
detect_model = YOLO('./models/helmet-medium.pt')
depth_pipe = pipeline(task="depth-estimation", model='depth-anything/Depth-Anything-V2-Small-hf')

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

        result = self.assignment_2(frame_bgr)

        if result:
            depth_person, horizontal_position = result
            msg = Point()
            msg.x = horizontal_position
            msg.y = 0.0
            msg.z = depth_person
            self.position_pub.publish(msg)

    def assignment_2(self, frame, target_class=0, target_id=1, depth_factor=20000):
        person = tracking.track_specific(detect_model, frame, target_class, target_id)

        if person is None:
            cv2.imshow("Depth Tracking", frame)
            cv2.waitKey(1)
            return None

        _, _, x1, y1, x2, y2 = person
        depth_person = depth.get_depth(depth_pipe, frame, (x1, y1, x2, y2), normalize=True)
        depth_person = depth_person * depth_factor

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Depth: {depth_person:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Depth Tracking", frame)
        cv2.waitKey(1)

        horizontal_position = (x1 + x2) / 2
        return depth_person, horizontal_position

    def destroy_node(self):
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()

def main(args=None):
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
