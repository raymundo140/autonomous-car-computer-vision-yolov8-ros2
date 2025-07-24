#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import math
import torch
from simple_pid import PID
from ultralytics.engine.results import Boxes

class FollowLineNode(Node):
    def __init__(self):
        super().__init__('follow_line_node')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            '/video_source/compressed',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.bridge = CvBridge()
        self.yolo_model = YOLO('/home/goma/ros2_ws/src/yolov8_ros2/yolov8_ros2/yolov8_best-v2.pt')

        max_yaw = math.radians(60)
        self.max_thr = 0.15
        self.yaw_pid = PID(Kp=0.6, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

        self.paused = False
        self.detected_stop = False
        self.confidence_threshold = 0.6

        self.get_logger().info('🟢 PID + YOLOv8 + Traffic light circle-based detection active.')

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("⚠️ Could not decode image from /video_source/compressed")
            return

        self.process_frame(frame)

    def process_frame(self, frame):
        results = self.yolo_model(frame)
        for r in results:
            conf_thresh = self.confidence_threshold
            if r.boxes is not None:
                filtered_data = [b.data[0] for b in r.boxes if float(b.conf[0]) >= conf_thresh]
                for b in r.boxes:
                    conf = float(b.conf[0])
                    if conf < conf_thresh:
                        continue
                    class_id = int(b.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    self.get_logger().info(f"🧠 Detected: {class_name} (conf: {conf:.2f})")
                    if class_name.lower() == 'stop':
                        self.detected_stop = True
                if filtered_data:
                    r.boxes = Boxes(torch.stack(filtered_data), r.orig_shape)
                else:
                    r.boxes = None

        self.detect_traffic_light_circles(frame)
        throttle, yaw, key = self.follow_line(frame)

        if key == ord('q'):
            self.paused = True
            self.get_logger().info("⏸️ Robot paused.")
        elif key == ord('p'):
            self.paused = False
            self.get_logger().info("▶️ Robot resumed.")

        twist = Twist()
        if not self.paused:
            if self.detected_stop:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                twist.linear.x = float(throttle)
                twist.angular.z = float(yaw)

        self.publisher.publish(twist)

        result_frame = results[0].plot()
        cv2.imshow("YOLOv8 Detection", cv2.resize(result_frame, (640, 480)))
        if hasattr(self, 'line_display_frame'):
            cv2.imshow("Line Following", cv2.resize(self.line_display_frame, (640, 480)))

        cv2.waitKey(1)

    def detect_traffic_light_circles(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        def detect_circles(mask, color_name, bgr_color):
            blurred = cv2.GaussianBlur(mask, (9, 9), 2)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                       param1=50, param2=30, minRadius=5, maxRadius=50)
            if circles is not None:
                self.get_logger().info(f"🔵 Detected {color_name} circle ({len(circles[0])} total)")
                for circle in np.uint16(np.around(circles[0])):
                    x, y, r = circle
                    cv2.circle(frame, (x, y), r, bgr_color, 2)
                    cv2.circle(frame, (x, y), 2, bgr_color, 3)

        # Color masks
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        )
        yellow_mask = cv2.inRange(hsv, np.array([18, 100, 100]), np.array([30, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([90, 255, 255]))

        # Detect circles per color
        detect_circles(red_mask, 'Red', (0, 0, 255))
        detect_circles(yellow_mask, 'Yellow', (0, 255, 255))
        detect_circles(green_mask, 'Green', (0, 255, 0))

        combined = np.hstack([
            cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow("Color Masks (R | Y | G)", cv2.resize(combined, (960, 320)))


    def follow_line(self, frame):
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        dark_thres = 100

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, dark_thres, 255, cv2.THRESH_BINARY_INV)
        mask[:int(height * 0.7), :] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 1000]

        throttle, yaw = 0.0, 0.0
        if contours:
            def contour_key(c):
                _, _, angle, cx, _ = self.get_contour_line(c)
                max_angle = 80
                angle = max(min(angle, max_angle), -max_angle)
                ref_x = (width / 2) + (angle / max_angle) * (width / 2)
                return abs(cx - ref_x)

            best_contour = sorted(contours, key=contour_key)[0]
            x, y, w, h = cv2.boundingRect(best_contour)
            cx = x + w // 2
            normalized_x = (cx - (width / 2)) / (width / 2)

            yaw = self.yaw_pid(normalized_x)
            alignment = 1 - abs(normalized_x)
            align_thres = 0.3
            throttle = self.max_thr * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0

            pt1, pt2, angle, line_cx, line_cy = self.get_contour_line(best_contour)
            cv2.drawContours(display_frame, [best_contour], -1, (0, 255, 0), 2)
            cv2.line(display_frame, pt1, pt2, (255, 0, 0), 2)
            cv2.circle(display_frame, (int(line_cx), int(line_cy)), 5, (0, 0, 255), -1)
            cv2.line(display_frame, (int(width // 2), height), (int(line_cx), int(line_cy)), (0, 255, 255), 2)

        self.line_display_frame = display_frame.copy()
        key = cv2.waitKey(1)
        return throttle, yaw, key

    def get_contour_line(self, c, fix_vert=True):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        scale = 100
        pt1 = (int(cx - vx * scale), int(cy - vy * scale))
        pt2 = (int(cx + vx * scale), int(cy + vy * scale))
        angle = math.degrees(math.atan2(vy, vx))
        if fix_vert:
            angle -= 90 * np.sign(angle)
        return pt1, pt2, angle, cx, cy

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FollowLineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
