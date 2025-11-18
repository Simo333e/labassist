from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    video = LaunchConfiguration("video")
    fps = LaunchConfiguration("fps")
    ckpt = LaunchConfiguration("ckpt")
    class_index = LaunchConfiguration("class_index")
    device = LaunchConfiguration("device")

    return LaunchDescription(
        [
            DeclareLaunchArgument("video", default_value=""),
            DeclareLaunchArgument("fps", default_value="0.0"),
            DeclareLaunchArgument("ckpt", default_value=""),
            DeclareLaunchArgument("class_index", default_value=""),
            DeclareLaunchArgument("device", default_value="cpu"),

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
                parameters=[{"imgsz": 224, "device": device}],
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
        ]
    )

