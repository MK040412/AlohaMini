import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('Aloha')

    urdf_path = os.path.join(pkg_dir, 'urdf', 'Aloha.urdf')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'ik_demo.rviz')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    demo_mode_arg = DeclareLaunchArgument(
        'demo_mode', default_value='circle',
        description='IK demo mode: circle, figure8, or interactive'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
        }],
    )

    ik_node = Node(
        package='Aloha',
        executable='aloha_ik_node.py',
        name='aloha_ik_node',
        output='screen',
        parameters=[{
            'demo_mode': LaunchConfiguration('demo_mode'),
            'urdf_path': urdf_path,
            'rate': 30.0,
        }],
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    return LaunchDescription([
        demo_mode_arg,
        robot_state_publisher,
        ik_node,
        rviz2,
    ])
