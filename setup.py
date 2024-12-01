from setuptools import setup
import os
from glob import glob

package_name = 'raspberrypi_ai_camera_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # Include package.xml
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include other necessary directories like networks and labels if needed
        (os.path.join('share', package_name, 'networks'), glob('networks/*')),
        (os.path.join('share', package_name, 'labels'), glob('labels/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohamed Abdelkader',
    maintainer_email='mohamedashraf123@gmail.com',
    description='Raspberry Pi AI Camera ROS2 Node for IMX500 Object Detection',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = raspberrypi_ai_camera_ros2.object_detection_node:main',
        ],
    },
)
