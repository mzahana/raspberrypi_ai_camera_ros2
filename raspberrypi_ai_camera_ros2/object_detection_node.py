#!/usr/bin/env python3
"""
IMX500 Object Detection ROS2 Node

This node performs real-time object detection using the IMX500 AI camera connected to a Raspberry Pi.
It uploads a quantized object detection model to the camera, processes the detections, and publishes
the results using ROS2 topics. It publishes raw images, images with detections, and detection results.

Usage:
    ros2 run raspberrypi_ai_camera_ros2 object_detection_node.py --ros-args \
        -p network_package:=/path/to/network.rpk \
        -p labels_file:=/path/to/labels.txt \
        -p frame_id:=camera_frame \
        [other parameters]

Parameters:
    network_package (string, required): Path to the network file (.rpk file)
    labels_file (string, required): Path to the labels file
    frame_id (string, default: 'camera_frame'): Frame ID to use in message headers
    detection_threshold (float, default: 0.55): Detection confidence threshold
    iou_threshold (float, default: 0.65): IOU threshold for detections
    max_detections (int, default: 10): Maximum number of detections per frame
    ignore_dash_labels (bool, default: False): Ignore labels starting with a dash
    preserve_aspect_ratio (bool, default: False): Preserve aspect ratio of the input image
    inference_rate (int, default: 30): Inference rate (frames per second)
    output_directory (string, default: 'detect'): Directory to save detection images
    save_images (bool, default: False): Save images with detections
    normalized_coordinates (bool, default: True): Whether the network outputs normalized coordinates (between 0 and 1)
"""

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import os
import sys

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance

from cv_bridge import CvBridge


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('imx500_object_detection_node')

        # Declare and get parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('network_package', ''),
                ('labels_file', ''),
                ('frame_id', 'camera_frame'),
                ('detection_threshold', 0.55),
                ('iou_threshold', 0.65),
                ('max_detections', 10),
                ('ignore_dash_labels', False),
                ('preserve_aspect_ratio', False),
                ('inference_rate', 30),
                ('output_directory', 'detect'),
                ('save_images', False),
                ('normalized_coordinates', True),
                ('bbx_order', 'yx'),
            ]
        )

        # Retrieve parameters with error checking
        try:
            network_package = self.get_parameter('network_package').get_parameter_value().string_value
            if not network_package:
                raise ValueError("Parameter 'network_package' is required but not set or is empty.")

            labels_file = self.get_parameter('labels_file').get_parameter_value().string_value
            if not labels_file:
                raise ValueError("Parameter 'labels_file' is required but not set or is empty.")

            frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
            if not frame_id:
                self.get_logger().warn("Parameter 'frame_id' is empty. Using default 'camera_frame'.")
                frame_id = 'camera_frame'

            detection_threshold = self.get_parameter('detection_threshold').get_parameter_value().double_value
            iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
            max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
            ignore_dash_labels = self.get_parameter('ignore_dash_labels').get_parameter_value().bool_value
            preserve_aspect_ratio = self.get_parameter('preserve_aspect_ratio').get_parameter_value().bool_value
            inference_rate = self.get_parameter('inference_rate').get_parameter_value().integer_value
            output_directory = self.get_parameter('output_directory').get_parameter_value().string_value
            save_images = self.get_parameter('save_images').get_parameter_value().bool_value
            normalized_coordinates = self.get_parameter('normalized_coordinates').get_parameter_value().bool_value  # New parameter
        except ValueError as e:
            self.get_logger().error(f"Parameter error: {e}")
            sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"Unexpected error retrieving parameters: {e}")
            sys.exit(1)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize variables
        self.last_detections = []
        self.detection_counter = 0  # Counter for detections
        self.frame_counter = 0  # Counter for frames
        self.output_directory = output_directory
        self.save_images = save_images  # Control image saving
        self.frame_id = frame_id  # Store frame_id for use in message headers
        self.normalized_coordinates = normalized_coordinates  # Store the new parameter
        self.bbx_order = self.get_parameter('bbx_order').get_parameter_value().string_value

        # Validate network package file
        if not os.path.isfile(network_package):
            self.get_logger().error(f"Network package file not found: {network_package}")
            sys.exit(1)
            
        self.get_logger().info(f"Network package file found: {network_package}")

        # Validate labels file
        if not os.path.isfile(labels_file):
            self.get_logger().error(f"Labels file not found: {labels_file}")
            sys.exit(1)
            
        self.get_logger().info(f"Labels file found: {labels_file}")

        # Load labels
        try:
            self.labels = self.get_labels(labels_file, ignore_dash_labels=ignore_dash_labels)
        except Exception as e:
            self.get_logger().error(f"Failed to load labels: {e}")
            sys.exit(1)
            
        self.get_logger().info(f"Number of loaded labels: {len(self.labels)}")

        # Initialize IMX500 and Picamera2
        try:
            self.imx500 = IMX500(network_package)
            self.picam2 = Picamera2(self.imx500.camera_num)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize IMX500 or Picamera2: {e}")
            sys.exit(1)

        # Set up network intrinsics
        try:
            self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
            self.intrinsics.task = "object detection"
            self.intrinsics.threshold = detection_threshold
            self.intrinsics.iou = iou_threshold
            self.intrinsics.max_detections = max_detections
            self.intrinsics.ignore_dash_labels = ignore_dash_labels
            self.intrinsics.postprocess = ""
            self.intrinsics.preserve_aspect_ratio = preserve_aspect_ratio
            self.intrinsics.inference_rate = inference_rate
            self.intrinsics.labels = self.labels
            self.intrinsics.update_with_defaults()
        except Exception as e:
            self.get_logger().error(f"Failed to set up network intrinsics: {e}")
            sys.exit(1)

        # Camera configuration
        try:
            self.config = self.picam2.create_preview_configuration(
                controls={"FrameRate": self.intrinsics.inference_rate}, buffer_count=12)
        except Exception as e:
            self.get_logger().error(f"Failed to create camera preview configuration: {e}")
            sys.exit(1)

        # Initialize Picamera2
        try:
            self.imx500.show_network_fw_progress_bar()
            self.picam2.start(self.config, show_preview=False)  # show_preview=False, since we use OpenCV for display
        except Exception as e:
            self.get_logger().error(f"Failed to start Picamera2: {e}")
            sys.exit(1)

        if self.intrinsics.preserve_aspect_ratio:
            try:
                self.imx500.set_auto_aspect_ratio()
            except Exception as e:
                self.get_logger().error(f"Failed to set auto aspect ratio: {e}")
                sys.exit(1)

        # Initialize Publishers
        try:
            self.raw_image_publisher = self.create_publisher(Image, 'camera/raw_image', 10)
            self.detection_image_publisher = self.create_publisher(Image, 'camera/detection_image', 10)
            self.detections_publisher = self.create_publisher(Detection2DArray, 'camera/detections', 10)
        except Exception as e:
            self.get_logger().error(f"Failed to create publishers: {e}")
            sys.exit(1)

        # Timer to process frames at the specified inference rate
        try:
            timer_period = 1.0 / inference_rate  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
        except Exception as e:
            self.get_logger().error(f"Failed to create timer: {e}")
            sys.exit(1)

        self.get_logger().info("raspberrypi_ai_camera_ros2 Node has been started.")

    @staticmethod
    def get_labels(labels_file, ignore_dash_labels=True):
        """Reads labels from the labels file."""
        try:
            with open(labels_file, 'r') as f:
                labels = f.read().splitlines()
                if not labels:
                    raise ValueError("Labels file is empty.")
                if (ignore_dash_labels):
                    labels = [label for label in labels if label and label != "-"]
            return labels
        except Exception as e:
            raise RuntimeError(f"Error reading labels file: {e}")

    class Detection:
        def __init__(self, coords, category, conf, imx500, picam2, logger, normalized_coordinates, bbx_order='xy'):
            """
            Creates a Detection object with correctly computed bounding box.

            Args:
                coords (list or array): [x_min, y_min, x_max, y_max]
                category (int): Class ID
                conf (float): Confidence score
                imx500 (IMX500): IMX500 instance
                picam2 (Picamera2): Picamera2 instance
                logger (rclpy.logging.Logger): ROS2 logger
                normalized_coordinates (bool): Whether the coordinates are normalized
            """
            try:
                # Validate coords format
                if not isinstance(coords, (list, tuple, np.ndarray)):
                    raise TypeError(f"Detection coordinates must be a list, tuple, or ndarray, got {type(coords)}")

                if len(coords) != 4:
                    raise ValueError(f"Detection coordinates expected to have 4 elements, got {len(coords)}")

                if (bbx_order == 'yx'):
                    y_min, x_min, y_max, x_max = coords
                else:
                    x_min, y_min, x_max, y_max = coords
                

                # Input image size of the model (libcamera._libcamera.Size object)
                input_size = imx500.get_input_size()
                logger.debug(f"get_input_size() returned: {input_size} (type: {type(input_size)})")

                if hasattr(input_size, 'width') and hasattr(input_size, 'height'):
                    input_w = input_size.width
                    input_h = input_size.height
                    logger.debug(f"Input Size: width={input_w}, height={input_h}")
                elif hasattr(input_size, 'get_width') and hasattr(input_size, 'get_height'):
                    input_w = input_size.get_width()
                    input_h = input_size.get_height()
                    logger.debug(f"Input Size via methods: width={input_w}, height={input_h}")
                elif isinstance(input_size, tuple) and len(input_size) == 2:
                    input_w, input_h = input_size
                    logger.debug(f"Input Size unpacked from tuple: width={input_w}, height={input_h}")
                else:
                    raise AttributeError("get_input_size() did not return an object with 'width' and 'height' attributes.")

                # Output image size (ISP) (libcamera._libcamera.Size object)
                isp_size = imx500.get_isp_output_size(picam2)
                logger.debug(f"get_isp_output_size() returned: {isp_size} (type: {type(isp_size)})")

                if hasattr(isp_size, 'width') and hasattr(isp_size, 'height'):
                    isp_w = isp_size.width
                    isp_h = isp_size.height
                    logger.debug(f"ISP Output Size: width={isp_w}, height={isp_h}")
                elif hasattr(isp_size, 'get_width') and hasattr(isp_size, 'get_height'):
                    isp_w = isp_size.get_width()
                    isp_h = isp_size.get_height()
                    logger.debug(f"ISP Output Size via methods: width={isp_w}, height={isp_h}")
                elif isinstance(isp_size, tuple) and len(isp_size) == 2:
                    isp_w, isp_h = isp_size
                    logger.debug(f"ISP Output Size unpacked from tuple: width={isp_w}, height={isp_h}")
                else:
                    raise AttributeError("get_isp_output_size() did not return an object with 'width' and 'height' attributes.")

                # Confirm that the model's input size is not zero to prevent division by zero
                if input_w == 0 or input_h == 0:
                    raise ZeroDivisionError("Input width or height is zero, cannot compute scaling factors.")

                # Scaling factors
                scale_x = isp_w / input_w
                scale_y = isp_h / input_h
                logger.debug(f"Scaling Factors: scale_x={scale_x}, scale_y={scale_y}")

                # If coordinates are normalized, scale them to input image dimensions
                if normalized_coordinates:
                    x_min *= input_w
                    y_min *= input_h
                    x_max *= input_w
                    y_max *= input_h
                    logger.debug(f"Normalized Coordinates scaled to input size: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                else:
                    logger.debug(f"Non-normalized coordinates received: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                # Scale coordinates to ISP (output image) size
                x_min_scaled = x_min * scale_x
                y_min_scaled = y_min * scale_y
                x_max_scaled = x_max * scale_x
                y_max_scaled = y_max * scale_y
                logger.debug(f"Scaled Coordinates: x_min_scaled={x_min_scaled}, y_min_scaled={y_min_scaled}, x_max_scaled={x_max_scaled}, y_max_scaled={y_max_scaled}")

                # Compute bounding box
                x1 = int(x_min_scaled)
                y1 = int(y_min_scaled)
                width = int(x_max_scaled - x_min_scaled)
                height = int(y_max_scaled - y_min_scaled)
                logger.debug(f"Bounding Box: x1={x1}, y1={y1}, width={width}, height={height}")

                # Ensure values are within image boundaries
                x1 = max(0, min(x1, isp_w - 1))
                y1 = max(0, min(y1, isp_h - 1))
                width = max(0, min(width, isp_w - x1))
                height = max(0, min(height, isp_h - y1))
                logger.debug(f"Clamped Bounding Box: x1={x1}, y1={y1}, width={width}, height={height}")

                self.category = category
                self.conf = conf
                self.box = (x1, y1, width, height)
            except Exception as e:
                logger.error(f"Error processing detection coordinates: {e}")
                raise

    def parse_detections(self, metadata, imx500, picam2, labels, detection_threshold, logger):
        """Parses the output tensor into detected objects."""
        try:
            np_outputs = imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is None:
                logger.debug("No outputs received from the network.")
                return []

            # Confirm the structure of np_outputs
            if not isinstance(np_outputs, (list, tuple)) or len(np_outputs) < 3:
                raise ValueError(f"Expected np_outputs to be a list/tuple with at least 3 elements, got {type(np_outputs)} with length {len(np_outputs) if isinstance(np_outputs, (list, tuple)) else 'N/A'}")

            # Extract outputs from the tensor
            # Adjust indices based on your model's output format
            boxes = np_outputs[0][0]
            scores = np_outputs[1][0]
            classes = np_outputs[2][0]

            # Validate shapes
            if not (len(boxes) == len(scores) == len(classes)):
                raise ValueError(f"Mismatch in lengths: boxes({len(boxes)}), scores({len(scores)}), classes({len(classes)})")

            detections = []
            for idx, (box, score, category) in enumerate(zip(boxes, scores, classes)):
                if score > detection_threshold:
                    try:
                        detection = self.Detection(
                            coords=box,
                            category=int(category),
                            conf=float(score),
                            imx500=imx500,
                            picam2=picam2,
                            logger=logger,
                            normalized_coordinates=self.normalized_coordinates,
                            bbx_order=self.bbx_order
                        )
                        detections.append(detection)
                        logger.debug(f"Detection {idx}: Category={category}, Confidence={score}, Box={detection.box}")
                    except Exception as e:
                        logger.error(f"Failed to create Detection object: {e}")
                        continue  # Skip this detection

            return detections
        except Exception as e:
            logger.error(f"Error parsing detections: {e}")
            return []

    def draw_detections(self, image, detections, labels, logger):
        """Draws the detections on the given image."""
        try:
            for detection in detections:
                x, y, w, h = detection.box
                logger.debug(f"Drawing detection: x={x}, y={y}, w={w}, h={h}")
                class_id = detection.category

                # Validate bounding box dimensions
                if w <= 0 or h <= 0:
                    logger.warning(f"Invalid bounding box dimensions: width={w}, height={h}. Skipping this detection.")
                    continue

                # Ensure coordinates are within image boundaries
                img_height, img_width, _ = image.shape
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(1, min(w, img_width - x))
                h = max(1, min(h, img_height - y))

                # Get label
                if class_id >= len(labels):
                    logger.warning(f"Class ID {class_id} out of range for labels.")
                    label = f"Unknown ({detection.conf:.2f})"
                else:
                    label = f"{labels[class_id]} ({detection.conf:.2f})"

                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

                # Compute text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = x + 5
                text_y = y - 10 if y - 10 > 10 else y + h + 20

                # Draw background rectangle for text
                cv2.rectangle(image,
                              (text_x - 2, text_y - text_height - 2),
                              (text_x + text_width + 2, text_y + baseline + 2),
                              (255, 255, 255),
                              cv2.FILLED)

                # Draw text on the image
                cv2.putText(image, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Increment detection counter
                self.detection_counter += 1

            logger.debug(f"Total detections so far: {self.detection_counter}")
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")

    def create_detection_msg(self, detections, header, labels, logger):
        """Creates a Detection2DArray message from detections."""
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            try:
                detection_msg = Detection2D()
                detection_msg.header = header

                # Object Hypothesis
                hypothesis = ObjectHypothesisWithPose()
                class_id = detection.category
                if class_id >= len(labels):
                    logger.warning(f"Class ID {class_id} out of range for labels.")
                    hypothesis.hypothesis.class_id = "Unknown"
                else:
                    hypothesis.hypothesis.class_id = labels[class_id]
                hypothesis.hypothesis.score = detection.conf

                # Pose is unknown; using default PoseWithCovariance
                hypothesis.pose = PoseWithCovariance()

                detection_msg.results.append(hypothesis)

                # Bounding Box
                bbox = detection_msg.bbox
                # Correctly assign x and y through position
                bbox.center.position.x = float(detection.box[0] + detection.box[2] / 2.0)
                bbox.center.position.y = float(detection.box[1] + detection.box[3] / 2.0)
                bbox.center.theta = 0.0  # Assuming no rotation
                bbox.size_x = float(detection.box[2])
                bbox.size_y = float(detection.box[3])

                # Assign an ID (optional)
                detection_msg.id = f"detection_{self.detection_counter}"

                detection_array.detections.append(detection_msg)
            except Exception as e:
                logger.error(f"Failed to create Detection2D message: {e}")
                continue  # Skip this detection

        return detection_array

    def timer_callback(self):
        """Callback function to process and publish frames at each timer tick."""
        try:
            # Check if ROS is still running
            if not rclpy.ok():
                self.get_logger().warn("ROS2 is shutting down. Skipping frame processing.")
                return

            # Retrieve next request
            request = self.picam2.capture_request()
            if request is None:
                self.get_logger().debug("No request available.")
                return

            metadata = request.get_metadata()
            detections = self.parse_detections(
                metadata=metadata,
                imx500=self.imx500,
                picam2=self.picam2,
                labels=self.labels,
                detection_threshold=self.intrinsics.threshold,
                logger=self.get_logger()
            )

            # Check if image data exists
            with MappedArray(request, "main") as m:
                raw_image = m.array.copy()  # Raw image without detections
                if raw_image is None or raw_image.size == 0:
                    self.get_logger().warn("No image data received from the camera.")
                    request.release()
                    return  # Skip publishing if no image

                # Convert RGB to BGR for OpenCV
                raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

            # Increment frame counter only when a new image is captured
            self.frame_counter += 1

            # Publish raw image
            try:
                raw_image_msg = self.bridge.cv2_to_imgmsg(raw_image_bgr, encoding="bgr8")
                current_time = self.get_clock().now()
                raw_image_msg.header.stamp = current_time.to_msg()
                raw_image_msg.header.frame_id = self.frame_id  # Use the frame_id parameter
                self.raw_image_publisher.publish(raw_image_msg)
                self.get_logger().debug(f"Published raw image #{self.frame_counter}")
            except Exception as e:
                self.get_logger().error(f"Failed to publish raw image: {e}")

            # Create a copy for detection image
            detection_image = raw_image_bgr.copy()

            if detections:
                self.get_logger().debug(f"{len(detections)} objects detected.")
                self.draw_detections(detection_image, detections, self.labels, self.get_logger())

                # Save image if save_images is True
                if self.save_images:
                    try:
                        if not os.path.exists(self.output_directory):
                            os.makedirs(self.output_directory)

                        # Use frame counter in filename
                        filename = os.path.join(self.output_directory, f'detection_{self.frame_counter}.jpg')
                        cv2.imwrite(filename, detection_image)
                        self.get_logger().debug(f"Image with detections saved to: {filename}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to save detection image: {e}")

                # Publish detection image
                try:
                    detection_image_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding="bgr8")
                    detection_image_msg.header.stamp = raw_image_msg.header.stamp
                    detection_image_msg.header.frame_id = self.frame_id  # Use the frame_id parameter
                    self.detection_image_publisher.publish(detection_image_msg)
                    self.get_logger().debug(f"Published detection image #{self.frame_counter}")
                except Exception as e:
                    self.get_logger().error(f"Failed to publish detection image: {e}")

                # Create and publish Detection2DArray message
                try:
                    detection_array_msg = self.create_detection_msg(
                        detections=detections,
                        header=raw_image_msg.header,
                        labels=self.labels,
                        logger=self.get_logger()
                    )
                    detection_array_msg.header.frame_id = self.frame_id  # Ensure frame_id consistency
                    self.detections_publisher.publish(detection_array_msg)
                    self.get_logger().debug(f"Published Detection2DArray message #{self.frame_counter}")
                except Exception as e:
                    self.get_logger().error(f"Failed to publish detections: {e}")
            else:
                self.get_logger().debug("No objects detected. Skipping detection image and detections publishing.")

            # Release the request
            try:
                request.release()
            except Exception as e:
                self.get_logger().error(f"Failed to release camera request: {e}")

        except Exception as e:
            self.get_logger().error(f"Unexpected error in timer callback: {e}")

    def destroy_node(self):
        """Handles node shutdown."""
        self.get_logger().info("Shutting down raspberrypi_ai_camera_ros2 Node.")
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().error(f"Error destroying OpenCV windows: {e}")
        try:
            self.picam2.stop()
        except Exception as e:
            self.get_logger().error(f"Error stopping Picamera2: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    except Exception as e:
        node.get_logger().error(f"Unexpected exception: {e}")
    finally:
        # Ensure shutdown is called only once
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
