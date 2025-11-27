from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, ThisLaunchFileDir
from launch_ros.actions import Node


def generate_launch_description():
    video = LaunchConfiguration("video")
    fps = LaunchConfiguration("fps")
    ckpt = LaunchConfiguration("ckpt")
    class_index = LaunchConfiguration("class_index")
    device = LaunchConfiguration("device")
    repo_root = LaunchConfiguration("repo_root")
    collect_metrics = LaunchConfiguration("collect_metrics")

    return LaunchDescription(
        [
            DeclareLaunchArgument("video", default_value=""),
            DeclareLaunchArgument("fps", default_value="0.0"),
            DeclareLaunchArgument("ckpt", default_value=""),
            DeclareLaunchArgument("class_index", default_value=""),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument(
                "repo_root",
                default_value=PathJoinSubstitution(
                    [ThisLaunchFileDir(), "..", "..", ".."]
                ),
                description="Root of the labassist workspace containing hpc_scripts.",
            ),
            DeclareLaunchArgument(
                "collect_metrics",
                default_value="false",
                description="Enable latency metrics collection for baseline evaluation.",
            ),

            Node(
                package="labassist_nodes",
                executable="camera_player",
                name="camera",
                parameters=[{"video": video, "fps": fps, "loop": True}],
            ),
            Node(
                package="labassist_nodes",
                executable="feature_resnet18",
                name="feature_resnet18",
                parameters=[{"imgsz": 224, "device": device, "collect_metrics": collect_metrics}],
            ),
            Node(
                package="labassist_nodes",
                executable="mstcn_infer",
                name="mstcn_infer",
                parameters=[
                    {
                        "ckpt": ckpt,
                        "class_index": class_index,
                        "hidden": 256,
                        "stages": 6,
                        "window": 512,
                        "device": device,
                        "repo_root": repo_root,
                        "collect_metrics": collect_metrics,
                    }
                ],
            ),
            Node(
                package="labassist_nodes",
                executable="fsm_validator",
                name="fsm_validator",
            ),
            Node(
                package="labassist_nodes",
                executable="notifier_console",
                name="notifier",
            ),
            # Metrics collector node - only launched when collect_metrics is true
            Node(
                package="labassist_nodes",
                executable="metrics_collector",
                name="metrics_collector",
                condition=IfCondition(collect_metrics),
                parameters=[
                    {
                        "output_dir": "~/.labassist/metrics",
                        "log_interval": 100,
                        "save_on_shutdown": True,
                    }
                ],
            ),
        ]
    )

