#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2

from yolo_msg.msg import InferenceResult
from yolo_msg.msg import Yolov8Inference

class YoloInference(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # ✅ Load your custom YOLOv8 model
        self.model = YOLO('/home/goma/ros2_ws/src/yolov8_ros2/yolov8_ros2/yolov8_best.pt')

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # Use webcam

        # ROS 2 publishers
        self.yolo_pub = self.create_publisher(Yolov8Inference, '/Yolov8_inference', 10)
        self.yolo_img_pub = self.create_publisher(Image, '/inference_result', 10)

        # Run at 20Hz
        self.timer = self.create_timer(1/20, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("❌ Failed to read from webcam.")
            return

        results = self.model(frame)
        yolo_msg = Yolov8Inference()
        yolo_msg.header.frame_id = 'inference'
        yolo_msg.header.stamp = self.get_clock().now().to_msg()

        for r in results:
            for box in r.boxes:
                inf = InferenceResult()
                b = box.xyxy[0].cpu().numpy()
                c = int(box.cls[0])
                inf.class_name = self.model.names[c]
                inf.left = int(b[0])
                inf.top = int(b[1])
                inf.right = int(b[2])
                inf.bottom = int(b[3])
                yolo_msg.yolov8_inference.append(inf)

        # Draw results
        result_frame = results[0].plot()
        frame_uint8 = result_frame.astype(np.uint8)

        # Publish ROS messages
        self.yolo_img_pub.publish(self.bridge.cv2_to_imgmsg(frame_uint8, encoding='bgr8'))
        self.yolo_pub.publish(yolo_msg)

        # ✅ Show window with detections
        cv2.imshow("YOLOv8 Inference", frame_uint8)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()  # ✅ Close OpenCV window
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    y_i = YoloInference()
    rclpy.spin(y_i)
    y_i.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
