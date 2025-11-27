import json
import pathlib
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class MetricsCollector(Node):
    """Collects feature, MS-TCN, and end-to-end latency metrics."""

    def __init__(self):
        super().__init__("metrics_collector")
        
        self.declare_parameter("output_dir", "~/.labassist/metrics")
        self.declare_parameter("log_interval", 100)
        
        output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_interval = int(self.get_parameter("log_interval").get_parameter_value().integer_value)
        
        self.pending: dict[int, dict] = {}
        
        # Completed latencies (ms)
        self.feature_latencies: deque[float] = deque(maxlen=10000)
        self.mstcn_latencies: deque[float] = deque(maxlen=10000)
        self.e2e_latencies: deque[float] = deque(maxlen=10000)
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # Subscriptions
        self.create_subscription(Float32MultiArray, "/metrics/feature_timing", self._on_feature, 50)
        self.create_subscription(Float32MultiArray, "/metrics/mstcn_timing", self._on_mstcn, 50)
        
        self.get_logger().info(f"Metrics collector started → {self.output_dir}")

    def _on_feature(self, msg: Float32MultiArray):
        """Feature timing: [frame_seq, t_frame_pub, t_start, t_end]"""
        if len(msg.data) < 4:
            return
        seq = int(msg.data[0])
        self.pending[seq] = {
            "t_frame_pub": msg.data[1],
            "feat_start": msg.data[2],
            "feat_end": msg.data[3],
        }
        feat_ms = (msg.data[3] - msg.data[2]) / 1e6
        self.feature_latencies.append(feat_ms)
        self.frame_count += 1

    def _on_mstcn(self, msg: Float32MultiArray):
        """MS-TCN timing: [frame_seq, t_start, t_end, pred_idx, conf]"""
        if len(msg.data) < 3:
            return
        seq = int(msg.data[0])
        mstcn_start, mstcn_end = msg.data[1], msg.data[2]
        
        mstcn_ms = (mstcn_end - mstcn_start) / 1e6
        self.mstcn_latencies.append(mstcn_ms)
        
        if seq in self.pending:
            feat_data = self.pending.pop(seq)
            e2e_ms = (mstcn_end - feat_data["feat_start"]) / 1e6
            self.e2e_latencies.append(e2e_ms)
        
        if len(self.mstcn_latencies) % self.log_interval == 0:
            self._log_summary()

    def _log_summary(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        def stats(arr):
            if not arr:
                return "N/A"
            a = np.array(arr)
            return f"{np.mean(a):.1f}±{np.std(a):.1f}ms (p95={np.percentile(a, 95):.1f})"
        
        self.get_logger().info(
            f"[{self.frame_count} frames, {fps:.1f} FPS] "
            f"Feature: {stats(self.feature_latencies)} | "
            f"MS-TCN: {stats(self.mstcn_latencies)} | "
            f"E2E: {stats(self.e2e_latencies)}"
        )

    def _save_metrics(self):
        """Save summary to JSON."""
        def stats_dict(arr):
            if not arr:
                return {}
            a = np.array(arr)
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
            }
        
        elapsed = time.time() - self.start_time
        summary = {
            "total_frames": self.frame_count,
            "elapsed_s": elapsed,
            "fps": self.frame_count / elapsed if elapsed > 0 else 0,
            "feature_latency_ms": stats_dict(self.feature_latencies),
            "mstcn_latency_ms": stats_dict(self.mstcn_latencies),
            "e2e_latency_ms": stats_dict(self.e2e_latencies),
        }
        
        out_path = self.output_dir / f"latency_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        self.get_logger().info(f"Saved metrics → {out_path}")

    def destroy_node(self):
        if self.frame_count > 0:
            self._log_summary()
            self._save_metrics()
        super().destroy_node()


def main():
    rclpy.init()
    node = MetricsCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
