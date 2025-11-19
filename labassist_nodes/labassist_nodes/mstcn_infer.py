#!/usr/bin/env python3

import json
import pathlib
import sys
from collections import deque
import importlib.util  

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header

from labassist_interfaces.msg import ActionEvent


class MSTCNInfer(Node):
    """Online MS-TCN inference node consuming feature vectors."""

    def __init__(self):
        super().__init__("mstcn_infer")
        self.declare_parameter("ckpt", "")
        self.declare_parameter("class_index", "")
        self.declare_parameter("hidden", 256)
        self.declare_parameter("stages", 6)
        self.declare_parameter("window", 512)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("repo_root", "")
        
        self.declare_parameter("num_layers_pg", 10)
        self.declare_parameter("num_layers_r", 10)
        self.declare_parameter("num_r", 5)
        self.declare_parameter("num_f_maps", 64)

        self.ckpt_path = pathlib.Path(
            self.get_parameter("ckpt").get_parameter_value().string_value
        )
        self.class_index_path = pathlib.Path(
            self.get_parameter("class_index").get_parameter_value().string_value
        )
        self.hidden = int(self.get_parameter("hidden").get_parameter_value().integer_value)
        self.stages = int(self.get_parameter("stages").get_parameter_value().integer_value)
        self.window = int(self.get_parameter("window").get_parameter_value().integer_value)
        self.device = self._resolve_device(
            self.get_parameter("device").get_parameter_value().string_value
        )
        repo_root_param = (
            self.get_parameter("repo_root").get_parameter_value().string_value
        )
        self.repo_root = self._resolve_repo_root(repo_root_param)

        self.num_layers_pg = int(
            self.get_parameter("num_layers_pg").get_parameter_value().integer_value
        )
        self.num_layers_r = int(
            self.get_parameter("num_layers_r").get_parameter_value().integer_value
        )
        self.num_r = int(
            self.get_parameter("num_r").get_parameter_value().integer_value
        )
        self.num_f_maps = int(
            self.get_parameter("num_f_maps").get_parameter_value().integer_value
        )

        self._mstcn_ctor = self._import_mstcn(self.repo_root)

        self.names = self._load_class_index(self.class_index_path)
        self.model: torch.nn.Module | None = None
        self.buf: deque[np.ndarray] = deque(maxlen=self.window)

        self.subscription = self.create_subscription(
            Float32MultiArray, "/features", self._on_feature, 50
        )
        self.publisher = self.create_publisher(ActionEvent, "/actions", 10)
        self.get_logger().info(
            f"MS-TCN infer ready (classes={len(self.names)}, device={self.device})"
        )

    def _resolve_device(self, target: str) -> torch.device:
        if target.startswith("cuda") and torch.cuda.is_available():
            return torch.device(target)
        return torch.device("cpu")

    @staticmethod
    def _load_class_index(path: pathlib.Path) -> list[str]:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            inv = {v: k for k, v in data.items()}
            return [inv[i] if i in inv else f"class_{i}" for i in range(len(inv))]
        return list(data)

    def _lazy_init(self, dim: int):
        if self.model is not None:
            return
        ncls = len(self.names)
        # Instantiate MS_TCN2 using training hyperparameters
        self.model = (
            self._mstcn_ctor(
                self.num_layers_pg,
                self.num_layers_r,
                self.num_r,
                self.num_f_maps,
                dim,
                ncls,
            )
            .to(self.device)
            .eval()
        )
        state = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.get_logger().info("Loaded MS-TCN checkpoint.")

    @torch.no_grad()
    def _on_feature(self, msg: Float32MultiArray):
        vec = np.array(msg.data, dtype=np.float32)
        if vec.ndim != 1:
            self.get_logger().warning("Expected 1-D feature vector; skipping.")
            return
        self.buf.append(vec)
        self._lazy_init(vec.shape[0])

        if self.model is None:
            return

        xb = torch.from_numpy(np.stack(self.buf, axis=0)).unsqueeze(0).to(self.device)
        logits = self.model(xb)[-1][0, -1]
        probs = torch.softmax(logits, dim=-1)
        top_idx = int(torch.argmax(probs).item())

        event = ActionEvent(
            header=Header(stamp=self.get_clock().now().to_msg()),
            action_name=self.names[top_idx],
            actor="system",
            objects=[],
            confidence=float(probs[top_idx].item()),
        )
        self.publisher.publish(event)

    @staticmethod
    def _resolve_repo_root(explicit_path: str) -> pathlib.Path:
        if explicit_path:
            candidate = pathlib.Path(explicit_path).expanduser().resolve()
            return candidate
        resolved = pathlib.Path(__file__).resolve()
        try:
            return resolved.parents[6]
        except IndexError:  
            return resolved.parent

    @staticmethod
    def _import_mstcn(repo_root: pathlib.Path):
        """
        Load MS_TCN2 class from hpc_scripts/MS-TCN/causal_model.py by file path.
        """
        causal_path = repo_root / "hpc_scripts" / "MS-TCN" / "causal_model.py"
        if not causal_path.exists():
            raise FileNotFoundError(
                f"Could not find causal_model.py at {causal_path}. "
                "Ensure repo_root is set correctly."
            )
        spec = importlib.util.spec_from_file_location(
            "labassist_ms_tcn_causal", str(causal_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {causal_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "MS_TCN2"):
            raise AttributeError(
                f"Module {causal_path} does not define MS_TCN2; "
                "check the causal_model.py contents."
            )
        return module.MS_TCN2


def main():
    rclpy.init()
    rclpy.spin(MSTCNInfer())
    rclpy.shutdown()


if __name__ == "__main__":
    main()