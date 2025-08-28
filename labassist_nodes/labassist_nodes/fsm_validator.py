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

        # expected sequence (can be paramized later)
        self.expected = ['pipette_A_to_B', 'vortex', 'incubate_start', 'incubate_end']
        self.idx = 0
        self.params = {}
        self.step_started_at = self.get_clock().now()
        self.timeout_s = {
            'pipette_A_to_B': 3.0,
            'vortex':         5.0,
            'incubate_start': 10.0,
            'incubate_end':   60.0
        }
        self.create_timer(0.2, self.check_timeout)

    def on_params(self, msg: MixParams):
        self.params = dict(zip(msg.keys, msg.values))
        self.get_logger().info(f'Loaded params: {self.params}')

    def send_alert(self, level, code, message, observed=''):
        a = Alert(
            header=Header(),
            level=level, code=code, message=message,
            expected_next=self.expected[self.idx] if self.idx < len(self.expected) else '',
            observed_action=observed
        )
        self.pub_alert.publish(a)
        self.get_logger().info(f'[{level}] {code}: {message}')

    def on_action(self, msg: ActionEvent):
        if self.idx >= len(self.expected):
            self.send_alert('soft', 'OK', 'Run already complete', msg.action_name)
            return

        exp = self.expected[self.idx]
        # exact match OK
        if msg.action_name == exp:
            self.step_started_at = self.get_clock().now()
            self.idx += 1
            self.send_alert('soft', 'OK', f'Accepted step: {msg.action_name}', msg.action_name)
            return

        # if action belongs to a future step, flag out-of-order
        if msg.action_name in self.expected[self.idx+1:]:
            self.send_alert('hard', 'OUT_OF_ORDER',
                            f'Observed {msg.action_name} but expected {exp}', msg.action_name)
            return

        # unknown / extra action
        self.send_alert('soft', 'UNKNOWN', f'Unexpected action: {msg.action_name}', msg.action_name)

    def check_timeout(self):
        if self.idx >= len(self.expected):
            return
        exp = self.expected[self.idx]
        elapsed = (self.get_clock().now() - self.step_started_at).nanoseconds / 1e9
        if elapsed > self.timeout_s.get(exp, 30.0):
            self.send_alert('hard', 'TIMEOUT', f'Timeout waiting for step: {exp}')
            # keep waiting; you could also auto-advance or stop here

def main():
    rclpy.init()
    rclpy.spin(FSMValidator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
