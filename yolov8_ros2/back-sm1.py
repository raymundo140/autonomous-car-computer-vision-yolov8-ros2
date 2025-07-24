#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from ultralytics import YOLO
import logging
import cv2
import numpy as np
import math
from enum import Enum
from simple_pid import PID
import time
import copy

logging.getLogger("ultralytics").setLevel(logging.ERROR)

class RobotState(Enum):
    FOLLOW_LINE = 1
    WAIT_GREEN = 2
    SLOW_DOWN = 3
    EXECUTE_SIGN = 4
    STOPPED = 5

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
        self.yolo_model.overrides['verbose'] = False

        max_yaw = math.radians(60)
        self.max_thr = 0.15
        self.yaw_pid = PID(Kp=0.6, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

        self.state = RobotState.FOLLOW_LINE
        self.confidence_threshold = 0.6
        self.last_sign_detected = None
        self.last_state_change_time = time.time()

        self.started = False
        self.paused = False

        self.status_window_name = "Robot Status"
        cv2.namedWindow(self.status_window_name)
        cv2.setMouseCallback(self.status_window_name, self.handle_button_click)

    def handle_button_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 20 <= x <= 120 and 260 <= y <= 290:
                self.started = True
                self.paused = False
            elif 140 <= x <= 240 and 260 <= y <= 290:
                self.paused = True

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("⚠️ Could not decode image")
            return
        self.process_frame(frame)

    def process_frame(self, frame):
        detected_sign = None
        results = self.yolo_model(frame)

        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    conf = float(b.conf[0])
                    if conf < self.confidence_threshold:
                        continue
                    class_id = int(b.cls[0])
                    class_name = self.yolo_model.names[class_id].lower()
                    if class_name in ['stop', 'right', 'around']:
                        detected_sign = class_name

        light_color, hsv_val = self.detect_traffic_light_color(frame)
        now = time.time()

        if self.started and not self.paused:
            if self.state == RobotState.FOLLOW_LINE:
                if light_color == 'red':
                    self.state = RobotState.WAIT_GREEN
                    self.last_state_change_time = now
                elif light_color == 'yellow':
                    self.state = RobotState.SLOW_DOWN
                    self.last_state_change_time = now
                elif detected_sign:
                    self.state = RobotState.EXECUTE_SIGN
                    self.last_sign_detected = detected_sign
                    self.last_state_change_time = now

            elif self.state == RobotState.WAIT_GREEN:
                if light_color == 'green':
                    self.state = RobotState.FOLLOW_LINE

            elif self.state == RobotState.SLOW_DOWN:
                if light_color == 'red':
                    self.state = RobotState.WAIT_GREEN

            elif self.state == RobotState.EXECUTE_SIGN:
                if now - self.last_state_change_time > 30:
                    self.state = RobotState.STOPPED if self.last_sign_detected == 'stop' else RobotState.FOLLOW_LINE
                    self.last_state_change_time = now

        twist = Twist()
        if self.started and not self.paused:
            if self.state == RobotState.FOLLOW_LINE:
                throttle, yaw, _ = self.follow_line(frame)
                twist.linear.x = float(throttle)
                twist.angular.z = float(yaw)
            elif self.state == RobotState.WAIT_GREEN:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            elif self.state == RobotState.SLOW_DOWN:
                throttle, yaw, _ = self.follow_line(frame)
                twist.linear.x = float(throttle * 0.3)
                twist.angular.z = float(yaw)
            elif self.state == RobotState.EXECUTE_SIGN:
                if self.last_sign_detected == 'around':
                    twist.linear.x = 0.15
                    twist.angular.z = 0.095
                elif self.last_sign_detected == 'right':
                    twist.linear.x = 0.05
                    twist.angular.z = -1.0
            elif self.state == RobotState.STOPPED:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                if now - self.last_state_change_time > 3:
                    self.state = RobotState.FOLLOW_LINE
                    self.last_state_change_time = now

        self.publisher.publish(twist)

        result_frame = results[0].plot()
        cv2.imshow("YOLOv8 Detection", cv2.resize(result_frame, (640, 480)))
        if hasattr(self, 'line_display_frame'):
            cv2.imshow("Line Following", cv2.resize(self.line_display_frame, (640, 480)))
        self.show_robot_status(frame, light_color, detected_sign, hsv_val)
        cv2.waitKey(1)

    def detect_traffic_light_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.circle_frame = copy.deepcopy(frame)

        def detect_circle(mask, color_name, bgr_color):
            blurred = cv2.GaussianBlur(mask, (9, 9), 2)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                       param1=50, param2=30, minRadius=5, maxRadius=50)
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                x, y, r = circles[0]
                cv2.circle(self.circle_frame, (x, y), r, bgr_color, 2)
                cv2.circle(self.circle_frame, (x, y), 2, bgr_color, 3)
                return True, (x, y)
            return False, None

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([160, 70, 50]), np.array([179, 255, 255]))
        )
        yellow_mask = cv2.inRange(hsv, np.array([17, 70, 60]), np.array([38, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

        for mask, color, bgr in [(red_mask, "Red", (0, 0, 255)),
                                 (yellow_mask, "Yellow", (0, 255, 255)),
                                 (green_mask, "Green", (0, 255, 0))]:
            found, coord = detect_circle(mask, color, bgr)
            if found:
                return color.lower(), hsv[coord[1], coord[0]]

        return None, None

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
            best_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
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
        return throttle, yaw, None

    def get_contour_line(self, c, fix_vert=True):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        scale = 100
        pt1 = (int(cx - vx * scale), int(cy - vy * scale))
        pt2 = (int(cx + vx * scale), int(cy + vy * scale))
        angle = math.degrees(math.atan2(vy, vx))
        if fix_vert:
            angle -= 90 * np.sign(angle)
        return pt1, pt2, angle, cx, cy

    def show_robot_status(self, frame, light_color, detected_sign, hsv_val=None):
        status_frame = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.putText(status_frame, f"State: {self.state.name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        color_text = light_color.upper() if light_color else "None"
        cv2.putText(status_frame, f"Traffic Light: {color_text}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        sign_text = detected_sign.upper() if detected_sign else "None"
        cv2.putText(status_frame, f"Sign Detected: {sign_text}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if hsv_val is not None:
            h, s, v = map(int, hsv_val)
            cv2.putText(status_frame, f"HSV: ({h},{s},{v})", (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        # Draw Start button
        cv2.rectangle(status_frame, (20, 260), (120, 290), (0, 255, 0), -1)
        cv2.putText(status_frame, "Start", (40, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # Draw Pause button
        cv2.rectangle(status_frame, (140, 260), (240, 290), (0, 255, 255), -1)
        cv2.putText(status_frame, "Pause", (160, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow(self.status_window_name, status_frame)
        if hasattr(self, 'circle_frame'):
            cv2.imshow("Traffic Light Circles", cv2.resize(self.circle_frame, (640, 480)))

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FollowLineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()