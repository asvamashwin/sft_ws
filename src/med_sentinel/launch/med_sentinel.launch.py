"""Launch file for Med-Sentinel 360 scene builder and robot controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory('med_sentinel')
    config_file = os.path.join(pkg_share, 'config', 'scene_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Isaac Sim in headless mode',
        ),
        DeclareLaunchArgument(
            'config',
            default_value=config_file,
            description='Path to scene_params.yaml config file',
        ),

        LogInfo(msg='Launching Med-Sentinel 360 scene...'),

        Node(
            package='med_sentinel',
            executable='scene_builder',
            name='med_sentinel_scene',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config'),
                'headless': LaunchConfiguration('headless'),
            }],
        ),
    ])
