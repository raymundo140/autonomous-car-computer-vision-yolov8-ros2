# Autonomous Car with Computer Vision using YOLOv8 and ROS 2

This project implements a 1:10-scale autonomous vehicle capable of navigating an indoor environment using computer vision and deep learning. The system detects traffic lights and traffic signs in real time using a YOLOv8 model, while following road lines using a PID-controlled vision pipeline.

> Developed as part of the Intelligent Robotics Implementation course at Tecnológico de Monterrey, in collaboration with Manchester Robotics.

## 🧠 Features

- Real-time traffic sign recognition with YOLOv8
- Traffic light detection using color segmentation
- Line following with computer vision and PID control
- Hierarchical finite state machines for decision logic
- Jetson Nano-based edge computing with ROS 2 and micro-ROS
- Ethical analysis on the societal impact of autonomy

## 🚗 Hardware

- Jetson Nano 4GB
- HackerBoard v2 (dual H-bridge + ESC)
- Raspberry Pi Camera v2
- DC motors with encoders
- 20,000mAh QC 3.0 Powerbank
- 3D-printed chassis, mounts, and accessories

## 🧰 Software Stack

- Ubuntu 22.04 + ROS 2 Humble + micro-ROS
- Python, OpenCV, NumPy
- YOLOv8 (Ultralytics)
- RViz2 (optional for visualization)
- Roboflow (for annotation)

## 📊 Results

- ✅ 92.3% mAP@0.5 in traffic sign detection
- ✅ RMS lateral error < 3 cm
- ✅ 95.1% F1-score in traffic light classification


