#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from torchvision.models import ResNet18_Weights, resnet18
import torchvision.transforms as T


class FeatureResNet18(Node):
    """Convert incoming images into 512-dim ResNet18 features."""

    def __init__(self):
        super().__init__("feature_resnet18")
        self.declare_parameter("imgsz", 224)
        self.declare_parameter("device", "cpu")

        imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        device = self.get_parameter("device").get_parameter_value().string_value
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.model = self._load_model()
        self.tfm = self._build_transform(imgsz)

        self.publisher = self.create_publisher(Float32MultiArray, "/features", 10)
        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self._on_image, 10
        )
        self.get_logger().info(f"Feature node running on device={self.device}")

    def _load_model(self) -> torch.nn.Module:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)

        self._feat_store: dict[str, torch.Tensor] = {}

        def hook(_mod, _inp, out):
            self._feat_store["z"] = out.flatten(1)

        model.avgpool.register_forward_hook(hook)
        model.eval()
        model.to(self.device)
        return model

    def _build_transform(self, imgsz: int):
        weights = ResNet18_Weights.IMAGENET1K_V1
        normalize = T.Normalize(mean=weights.meta["mean"], std=weights.meta["std"])
        return T.Compose(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Resize((imgsz, imgsz), antialias=True),
                normalize,
            ]
        )

    def _on_image(self, msg: Image):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
        except ValueError:
            self.get_logger().warning("Invalid image buffer; skipping frame")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.tfm(rgb).unsqueeze(0).to(self.device)

        self._feat_store.clear()
        with torch.no_grad():
            _ = self.model(tensor)

        feat = self._feat_store.get("z")
        if feat is None:
            self.get_logger().warning("Feature hook missing output.")
            return

        feat_np = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
        msg_out = Float32MultiArray(
            layout=MultiArrayLayout(
                dim=[
                    MultiArrayDimension(
                        label="feat", size=feat_np.size, stride=feat_np.size
                    )
                ],
                data_offset=0,
            ),
            data=feat_np.tolist(),
        )
        self.publisher.publish(msg_out)


def main():
    rclpy.init()
    rclpy.spin(FeatureResNet18())
    rclpy.shutdown()


if __name__ == "__main__":
    main()

