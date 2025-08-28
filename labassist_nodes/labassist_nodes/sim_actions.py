#!/usr/bin/env python3
import rclpy, time
from rclpy.node import Node
from std_msgs.msg import Header
from labassist_interfaces.msg import ActionEvent, MixParams

class SimActions(Node):
    def __init__(self):
        super().__init__('sim_actions')
        self.pub_act = self.create_publisher(ActionEvent, '/actions', 10)
        self.pub_params = self.create_publisher(MixParams, '/mix_params', 10)
        self.timer = self.create_timer(1.0, self.tick)
        self.t0 = self.get_clock().now()
        self.step = 0

        # publish params once
        mp = MixParams(header=Header(), keys=['volume_uL','source','dest'], values=['400','A','B'])
        self.pub_params.publish(mp)
        self.get_logger().info('Published /mix_params')

        # scripted scenario (you can reorder or skip to test FSM)
        self.script = [
            ('pipette_A_to_B', ['pipette','hand'], ['A','B']),
            ('vortex',         ['hand'],           []),
            ('incubate_start', ['timer'],          []),
            ('incubate_end',   ['timer'],          []),
        ]

    def tick(self):
        if self.step >= len(self.script):
            return
        name, actors, objects = self.script[self.step]
        msg = ActionEvent(
            header=Header(),
            action_name=name,
            actor=actors[0],
            objects=objects,
            confidence=0.95
        )
        self.pub_act.publish(msg)
        self.get_logger().info(f'Published action: {name}')
        self.step += 1

def main():
    rclpy.init()
    rclpy.spin(SimActions())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
