#!/usr/bin/env python3
"""
Launch file for raspberrypi_ai_camera_ros2 Object Detection Node

This launch file initializes the object detection node with specified parameters
and remaps its topics for flexibility and integration within different systems.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the launch directory
    package_name = 'raspberrypi_ai_camera_ros2'
    launch_dir = os.path.join(get_package_share_directory(package_name), 'launch')

    # Declare the launch arguments (parameters)
    network_package_arg = DeclareLaunchArgument(
        'network_package',
        default_value=os.path.join(
            get_package_share_directory(package_name),
            'networks',
            'imx500_network_yolov8n_640x640_pp.rpk'
        ),
        description='Path to the network package (.rpk) file'
    )

    labels_file_arg = DeclareLaunchArgument(
        'labels_file',
        default_value=os.path.join(
            get_package_share_directory(package_name),
            'labels',
            'coco_yolo.txt'
        ),
        description='Path to the labels file'
    )

    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='camera_frame',
        description='Frame ID to use in message headers'
    )

    detection_threshold_arg = DeclareLaunchArgument(
        'detection_threshold',
        default_value='0.55',
        description='Detection confidence threshold'
    )

    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.65',
        description='IOU threshold for detections'
    )

    max_detections_arg = DeclareLaunchArgument(
        'max_detections',
        default_value='10',
        description='Maximum number of detections per frame'
    )

    ignore_dash_labels_arg = DeclareLaunchArgument(
        'ignore_dash_labels',
        default_value='true',
        description='Whether to ignore labels starting with a dash'
    )

    preserve_aspect_ratio_arg = DeclareLaunchArgument(
        'preserve_aspect_ratio',
        default_value='false',
        description='Whether to preserve the aspect ratio of the input image'
    )

    inference_rate_arg = DeclareLaunchArgument(
        'inference_rate',
        default_value='30',
        description='Inference rate in frames per second'
    )

    output_directory_arg = DeclareLaunchArgument(
        'output_directory',
        default_value='detections',
        description='Directory to save detection images'
    )

    save_images_arg = DeclareLaunchArgument(
        'save_images',
        default_value='false',
        description='Whether to save images with detections'
    )

    normalized_coordinates_arg = DeclareLaunchArgument(
        'normalized_coordinates',
        default_value='true',
        description='Whether the network outputs normalized coordinates (values between 0 and 1)'
    )
    
    bbx_order_arg = DeclareLaunchArgument(
        'bbx_order',
        default_value='yx',
        description='Bounding box coordinates order y0,x0,y1,x1 OR x0,y0,xmax,ymax'
    )

    # Remappings for topics
    remappings = [
        ('/camera/raw_image', '/image_raw'),
        ('/camera/detection_image', '/detections_image'),
        ('/camera/detections', '/detections')
    ]

    # Define the node
    object_detection_node = Node(
        package=package_name,
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
        parameters=[
            {
                'network_package': LaunchConfiguration('network_package'),
                'labels_file': LaunchConfiguration('labels_file'),
                'frame_id': LaunchConfiguration('frame_id'),
                'detection_threshold': LaunchConfiguration('detection_threshold'),
                'iou_threshold': LaunchConfiguration('iou_threshold'),
                'max_detections': LaunchConfiguration('max_detections'),
                'ignore_dash_labels': LaunchConfiguration('ignore_dash_labels'),
                'preserve_aspect_ratio': LaunchConfiguration('preserve_aspect_ratio'),
                'inference_rate': LaunchConfiguration('inference_rate'),
                'output_directory': LaunchConfiguration('output_directory'),
                'save_images': LaunchConfiguration('save_images'),
                'normalized_coordinates': LaunchConfiguration('normalized_coordinates'),
                'bbx_order': LaunchConfiguration('bbx_order'),
            }
        ],
        arguments=['--ros-args', '--log-level', 'INFO'],
        remappings=remappings
    )

    # Log message to indicate launch start
    log_launch_start = LogInfo(
        msg='Starting raspberrypi_ai_camera_ros2 Object Detection Node with topic remappings.'
    )

    return LaunchDescription([
        network_package_arg,
        labels_file_arg,
        frame_id_arg,
        detection_threshold_arg,
        iou_threshold_arg,
        max_detections_arg,
        ignore_dash_labels_arg,
        preserve_aspect_ratio_arg,
        inference_rate_arg,
        output_directory_arg,
        save_images_arg,
        normalized_coordinates_arg,
        bbx_order_arg,
        log_launch_start,
        object_detection_node
    ])
