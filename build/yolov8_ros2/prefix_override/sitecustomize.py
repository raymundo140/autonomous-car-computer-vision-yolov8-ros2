import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/goma/ros2_ws/src/yolov8_ros2/install/yolov8_ros2'
