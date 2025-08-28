#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from labassist_interfaces.msg import Alert

class NotifierConsole(Node):
    def __init__(self):
        super().__init__('notifier_console')
        self.sub = self.create_subscription(Alert, '/alert', self.on_alert, 10)

    def on_alert(self, a: Alert):
        tag = '🔔' if a.level=='soft' else '🚨'
        self.get_logger().info(f"{tag} {a.level}/{a.code} | expected: {a.expected_next} | observed: {a.observed_action} | {a.message}")

def main():
    rclpy.init()
    rclpy.spin(NotifierConsole())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
