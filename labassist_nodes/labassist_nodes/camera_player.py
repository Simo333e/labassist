#!/usr/bin/env python3

import pathlib

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraPlayer(Node):
    """Publish frames from a video file or camera device as Image messages."""

    def __init__(self):
        super().__init__("camera_player")

        self.declare_parameter("video", "")
        self.declare_parameter("fps", 0.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("resize_w", 0)
        self.declare_parameter("resize_h", 0)

        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)

        video_path = self.get_parameter("video").get_parameter_value().string_value
        self.cap = self._open_source(video_path)
        if self.cap is None:
            raise RuntimeError("Failed to open video source. Check 'video' parameter.")

        native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = self.get_parameter("fps").get_parameter_value().double_value or native_fps
        period = 1.0 / max(1e-6, fps)

        self.loop = self.get_parameter("loop").get_parameter_value().bool_value
        self.resize_w = int(self.get_parameter("resize_w").get_parameter_value().integer_value)
        self.resize_h = int(self.get_parameter("resize_h").get_parameter_value().integer_value)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"Streaming {video_path or 'camera device'} at {fps:.2f} Hz (loop={self.loop})"
        )

    def _open_source(self, source: str):
        if not source:
            cap = cv2.VideoCapture(0)
            return cap if cap.isOpened() else None
        path = pathlib.Path(source)
        cap = cv2.VideoCapture(str(path) if path.exists() else source)
        return cap if cap.isOpened() else None

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok or frame is None:
                self.get_logger().info("End of video stream; stopping timer.")
                self.timer.cancel()
                return

        if self.resize_w > 0 and self.resize_h > 0:
            frame = cv2.resize(
                frame,
                (self.resize_w, self.resize_h),
                interpolation=cv2.INTER_LINEAR,
            )

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        self.publisher.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(CameraPlayer())
    rclpy.shutdown()


if __name__ == "__main__":
    main()

