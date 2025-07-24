#!/usr/bin/env python3
import logging
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import numpy as np

# suppress ultralytics and root logs
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.FATAL)

class SignDetectionNode(Node):
    def __init__(self):
        super().__init__('sign_detection_node')
        # suppress ROS logs below FATAL
        self.get_logger().set_level(LoggingSeverity.FATAL)

        # Subscribe to the robot’s camera topic
        self.subscription = self.create_subscription(
            CompressedImage,
            '/video_source/compressed',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.bridge = CvBridge()
        self.yolo_model = YOLO('/home/goma/ros2_ws/src/yolov8_ros2/yolov8_ros2/yolov8_best-v2.pt')
        self.confidence_threshold = 0.6

        # Single window, sized 640×480
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Detection", 640, 480)

    def image_callback(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError:
            return

        h, w = frame.shape[:2]
        results = self.yolo_model(frame, conf=self.confidence_threshold)

        # Draw boxes manually with corrected left/right
        annotated = frame.copy()
        for r in results:
            for box in (r.boxes or []):
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                orig_label = self.yolo_model.names[cls].lower()

                # override left/right based on box center x
                if orig_label in ('left', 'right'):
                    cx = (x1 + x2) / 2
                    label = 'RIGHT' if cx > w / 2 else 'LEFT'
                else:
                    label = orig_label.upper()

                # choose color per class
                color = {
                    'STOP': (0, 0, 255),
                    'STRAIGHT': (255, 0, 0),
                    'LEFT': (0, 255, 0),
                    'RIGHT': (0, 255, 255)
                }.get(label, (255, 255, 255))

                # draw
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )

        # resize and display
        annotated = cv2.resize(annotated, (640, 480))
        cv2.imshow("YOLO Detection", annotated)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
