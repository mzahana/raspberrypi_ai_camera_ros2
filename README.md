# Raspberry Pi AI Camera ROS2 Package

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Launching the Node](#launching-the-node)
  - [Directly Running the Node](#directly-running-the-node)
- [Published Topics](#published-topics)
- [Visualizing Detections](#visualizing-detections)
- [Saving Detection Images](#saving-detection-images)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Debugging Tips](#debugging-tips)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Raspberry Pi AI Camera ROS2 Package** provides a robust ROS2 node for real-time object detection using the IMX500 AI camera ([Raspberry Pi AI camera](https://www.raspberrypi.com/products/ai-camera/)). This package leverages the Picamera2 library to interface with the camera hardware and utilizes pre-trained quantized object detection models to identify and localize objects within the camera's field of view. Detection results are published as ROS2 messages, enabling seamless integration with other ROS2-based systems and tools.

## Features

- **Real-Time Object Detection:** Utilize the IMX500 AI camera for efficient and accurate object detection.
- **ROS2 Integration:** Publish raw images, detection images, and detection data using standard ROS2 message types.
- **Configurable Parameters:** Adjust detection thresholds, inference rates, and other parameters to suit your application's needs.
- **Image Saving:** Optionally save images with bounding boxes and labels for offline analysis.
- **Comprehensive Logging:** Detailed logs for easy debugging and monitoring.
- **Modular Design:** Easily extendable to incorporate additional features or integrate with other ROS2 nodes.

## Prerequisites

Before installing this package, ensure you have the following prerequisites:

- **Hardware:**
  - Raspberry Pi (preferably Pi 4 or later) with sufficient processing power.
  - Raspberry Pi AI camera.

- **Software:**
  - **Operating System:** Pi OS Bookworm (tested with `Kernel: Linux 6.6.62+rpt-rpi-2712`).
  - **ROS2:** Installed and properly configured. This package is compatible with ROS2 Humble  (not tested with any other  distributions). The ROS2 installation was done from source.
  - **Python 3.10+**
  - **Picamera2:** For interfacing with the IMX500 camera.
  - **OpenCV:** For image processing and visualization (should be included in the ROS2 installation).
  - **cv_bridge:** For converting between ROS2 images and OpenCV images (should be included in the ROS2 installation).

## Installation

### 1. Clone the Repository

First, clone the repository into your ROS2 workspace's `src` directory:

```bash
cd ~/ros2_ws/src
git clone https://github.com/yourusername/raspberrypi_ai_camera_ros2.git
```

### 2. Install Dependencies

Ensure all necessary dependencies are installed. You can install them using `apt` and `pip`:

```bash
# Update package lists
sudo apt update 

sudo apt install imx500-all imx500-tools

# Install prerequisites of picamera2
sudo apt install python3-opencv python3-munkres

# Install some dependencies of picamera2
git clone -b next https://github.com/raspberrypi/picamera2
cd picamera2
pip install -e .  --break-system-packages

# reboot
sudo reboot
```


### 3. Build the Package

Navigate back to your workspace and build the package using `colcon`:

```bash
cd ~/ros2_ws
colcon build --packages-select raspberrypi_ai_camera_ros2
```

After a successful build, source the workspace:

```bash
source install/setup.bash
```

## Configuration

The node offers various parameters to customize its behavior. These can be set via a launch file or directly using ROS2 parameter settings.

### Parameters

| Parameter               | Type    | Default          | Description                                                                 |
|-------------------------|---------|------------------|-----------------------------------------------------------------------------|
| `network_package`       | string  | _required_       | Path to the object detection network package (`.rpk` file).                 |
| `labels_file`           | string  | _required_       | Path to the labels file containing class names.                             |
| `frame_id`              | string  | `camera_frame`   | Frame ID to use in message headers.                                         |
| `detection_threshold`   | double  | `0.55`           | Confidence score threshold for detections.                                  |
| `iou_threshold`         | double  | `0.65`           | Intersection over Union (IOU) threshold for non-maximum suppression.        |
| `max_detections`        | integer | `10`             | Maximum number of detections per frame.                                     |
| `ignore_dash_labels`    | bool    | `false`          | Whether to ignore labels that start with a dash (`-`).                      |
| `preserve_aspect_ratio` | bool    | `false`          | Whether to preserve the aspect ratio of the input image.                    |
| `inference_rate`        | integer | `30`             | Inference rate (frames per second).                                         |
| `output_directory`      | string  | `detect`         | Directory to save detection images if `save_images` is enabled.             |
| `save_images`           | bool    | `false`          | Whether to save images with 
| `normalized_coordinates`           | bool    | `false`          | Whether the bounding boxes coordinates received from the network are normalized or not .                                     |

## Usage


### Launching the Node

A launch file is provided to simplify running the node with the necessary parameters.

####  Run the Launch File

Use the following command to launch the node:

```bash
ros2 launch raspberrypi_ai_camera_ros2 object_detection_launch.py \
    network_package:="/home/pi5/src/ai_camera_firmware/dds/dds.veddesta.parts0-6.aug.v8n.320.det/network.rpk" \
    labels_file:="/home/pi5/src/ai_camera_firmware/dds/labels.txt" \
    frame_id:="camera_frame" \
    detection_threshold:=0.60 \
    iou_threshold:=0.70 \
    max_detections:=15 \
    ignore_dash_labels:=false \
    preserve_aspect_ratio:=true \
    inference_rate:=25 \
    output_directory:="/home/pi5/detections" \
    save_images:=true
```

> **Note:** Replace the paths and parameter values with those appropriate for your setup.

### Directly Running the Node

Alternatively, you can run the node directly without a launch file by setting parameters using command-line arguments:

```bash
ros2 run raspberrypi_ai_camera_ros2 object_detection_node.py \
    --ros-args \
    -p network_package:="/path/to/network.rpk" \
    -p labels_file:="/path/to/labels.txt" \
    -p frame_id:="camera_frame" \
    -p detection_threshold:=0.60 \
    -p iou_threshold:=0.70 \
    -p max_detections:=15 \
    -p ignore_dash_labels:=false \
    -p preserve_aspect_ratio:=true \
    -p inference_rate:=25 \
    -p output_directory:="/path/to/output" \
    -p save_images:=true
```

## Published Topics

The node publishes the following ROS2 topics:

| Topic Name                | Message Type                   | Description                                                              |
|---------------------------|--------------------------------|--------------------------------------------------------------------------|
| `/camera/raw_image`       | `sensor_msgs/Image`            | Raw image captured from the camera without detections.                   |
| `/camera/detection_image` | `sensor_msgs/Image`            | Image with bounding boxes and labels drawn on detected objects.          |
| `/camera/detections`      | `vision_msgs/Detection2DArray` | Array of detection results, including bounding boxes and class probabilities.|

## Visualizing Detections

To visualize the raw and detection images, you can use `rqt_image_view`:

1. **Run `rqt_image_view`:**

   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```

2. **Select Topics:**

   - **Raw Image:** Select `/camera/raw_image` to view the raw camera feed.
   - **Detection Image:** Select `/camera/detection_image` to view images with detected objects highlighted.

## Saving Detection Images

If the `save_images` parameter is enabled (`true`), the node will save images with detections to the specified `output_directory`. Ensure that the directory exists or that the node has permissions to create it.

**Example:**

With `output_directory` set to `/home/pi5/detections`, images will be saved as `detection_1.jpg`, `detection_2.jpg`, etc.

## Troubleshooting

### Common Issues

1. **Parameter Errors:**

   - **Symptom:** Errors related to missing or incorrect parameters.
   - **Solution:** Ensure all required parameters (`network_package` and `labels_file`) are correctly set and that file paths are valid.


2. **No Images Published:**

   - **Symptom:** No messages are being published on `/camera/raw_image` or `/camera/detection_image`.
   - **Solution:** Verify that the camera is properly connected and that the `Picamera2` library is correctly interfacing with the IMX500 camera.

3. **Permissions Issues:**

   - **Symptom:** Errors related to accessing the camera or writing to the output directory.
   - **Solution:** Ensure that the user running the ROS2 node has the necessary permissions to access the camera device and write to the specified directories.

### Debugging Tips

- **Enable Detailed Logging:**

  Set the ROS2 log level to `DEBUG` to get more detailed output:

  ```bash
  export RCLCPP_LOG_LEVEL=DEBUG
  ```

- **Check Node Status:**

  Verify that the node is running and healthy:

  ```bash
  ros2 node list
  ```

- **Inspect Published Messages:**

  Use `ros2 topic echo` to inspect messages being published:

  ```bash
  ros2 topic echo /camera/detections
  ```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

4. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

Please ensure that your contributions adhere to the project's coding standards and include appropriate tests and documentation.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Disclaimer:** This project is provided "as is" without any warranty. Use it at your own risk.
