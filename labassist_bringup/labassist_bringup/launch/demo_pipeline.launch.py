from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='labassist_nodes', executable='sim_actions', name='sim_actions'),
        Node(package='labassist_nodes', executable='fsm_validator', name='fsm_validator'),
        Node(package='labassist_nodes', executable='notifier_console', name='notifier'),
    ])
