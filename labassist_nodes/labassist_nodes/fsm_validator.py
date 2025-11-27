#!/usr/bin/env python3
import rclpy, time
from rclpy.node import Node
from std_msgs.msg import Header
from labassist_interfaces.msg import ActionEvent, MixParams, Alert

class FSMValidator(Node):
    def __init__(self):
        super().__init__('fsm_validator')
        self.sub_actions = self.create_subscription(ActionEvent, '/actions', self.on_action, 10)
        self.sub_params  = self.create_subscription(MixParams,  '/mix_params', self.on_params, 10)
        self.pub_alert   = self.create_publisher(Alert, '/alert', 10)

        # Simplified mode: no hard-coded FSM, just log observed actions
        self.params = {}

    def on_params(self, msg: MixParams):
        self.params = dict(zip(msg.keys, msg.values))
        self.get_logger().info(f'Loaded params: {self.params}')

    def send_alert(self, level, code, message, observed=''):
        a = Alert(
            header=Header(),
            level=level, code=code, message=message,
            expected_next='',            # no sequence comparison
            observed_action=observed,
        )
        self.pub_alert.publish(a)
        self.get_logger().info(f'[{level}] {code}: {message}')

    def on_action(self, msg: ActionEvent):
        # Just log what we observed; no sequence/timeout enforcement
        msg_txt = (
            f"Observed action: {msg.action_name} "
            f"(conf={msg.confidence:.3f}, actor={msg.actor})"
        )
        self.send_alert('info', 'OBSERVED', msg_txt, msg.action_name)

def main():
    rclpy.init()
    rclpy.spin(FSMValidator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
