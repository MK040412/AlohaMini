#!/usr/bin/python3
"""
ROS2 node for AlohaMini IK demo.

Modes:
  circle    - Both EEs trace circular trajectories (default)
  figure8   - Both EEs trace figure-8 (Lissajous) trajectories
  interactive - RViz2 interactive markers for dragging EE targets

Publishes:
  /joint_states       (sensor_msgs/JointState)   @ 30 Hz
  /ik_markers         (visualization_msgs/MarkerArray) - targets + trail
"""

import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers import InteractiveMarkerServer
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3

from aloha_kinematics import AlohaMiniFK, AlohaMiniIK


class AlohaIKNode(Node):
    """ROS2 node that computes IK and publishes joint states + markers."""

    # All joints that robot_state_publisher needs
    ALL_JOINT_NAMES = [
        'wheel1', 'wheel2', 'wheel3',
        'vertical_move',
        'left_joint1', 'left_joint2', 'left_joint3',
        'left_joint4', 'left_joint5', 'left_joint6',
        'right_joint1', 'right_joint2', 'right_joint3',
        'right_joint4', 'right_joint5', 'right_joint6',
    ]

    def __init__(self):
        super().__init__('aloha_ik_node')

        self.declare_parameter('demo_mode', 'circle')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('rate', 30.0)

        self.mode = self.get_parameter('demo_mode').value
        urdf_path = self.get_parameter('urdf_path').value
        rate = self.get_parameter('rate').value

        if not urdf_path:
            urdf_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'urdf', 'Aloha.urdf'
            )
            if not os.path.exists(urdf_path):
                try:
                    from ament_index_python.packages import get_package_share_directory
                    urdf_path = os.path.join(
                        get_package_share_directory('Aloha'), 'urdf', 'Aloha.urdf'
                    )
                except Exception:
                    pass

        urdf_path = os.path.abspath(urdf_path)
        self.get_logger().info(f'Loading URDF: {urdf_path}')
        self.get_logger().info(f'Demo mode: {self.mode}')

        # FK/IK setup
        self.fk = AlohaMiniFK(urdf_path)
        self.ik = AlohaMiniIK(
            self.fk,
            damping=0.01,
            max_iter=30,       # fewer iterations per tick for real-time
            tol=5e-4,
            alpha=0.3,
            null_space_gain=0.5,   # strong regularization toward rest pose
            max_step_rad=0.03,
        )

        # Start from the rest pose
        self.q = dict(AlohaMiniFK.REST_POSE)

        # Compute rest-pose EE positions as trajectory centers
        T_left = self.fk.fk_left(self.q)
        T_right = self.fk.fk_right(self.q)
        self.left_center = T_left[:3, 3].copy()
        self.right_center = T_right[:3, 3].copy()
        self.get_logger().info(f'Left  EE center: {self.left_center}')
        self.get_logger().info(f'Right EE center: {self.right_center}')

        # Interactive targets
        self.left_target = self.left_center.copy()
        self.right_target = self.right_center.copy()

        # Trail history
        self.left_trail: list = []
        self.right_trail: list = []
        self.max_trail = 200

        # Wheel animation state
        self.wheel_angle = 0.0

        # Publishers
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/ik_markers', 10)

        # Interactive marker server
        self.im_server = None
        if self.mode == 'interactive':
            self._setup_interactive_markers()

        # Warm up: pre-solve IK at t=0 so first frame looks correct
        target_l, target_r = self._generate_targets(0.0)
        for _ in range(5):
            self.q, _, _, _ = self.ik.solve_both(
                target_l, target_r, self.q, q_ref=AlohaMiniFK.REST_POSE)
        self.get_logger().info('IK warm-up complete.')

        # Timer
        self.t0 = self.get_clock().now()
        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('AlohaIKNode started.')

    def _setup_interactive_markers(self):
        self.im_server = InteractiveMarkerServer(self, 'ik_targets')

        for side, center, color in [
            ('left', self.left_center, (0.2, 0.8, 0.2, 0.8)),
            ('right', self.right_center, (0.8, 0.2, 0.2, 0.8)),
        ]:
            im = InteractiveMarker()
            im.header.frame_id = 'base_link'
            im.name = f'{side}_target'
            im.description = f'{side.capitalize()} EE Target'
            im.scale = 0.08
            im.pose.position.x = float(center[0])
            im.pose.position.y = float(center[1])
            im.pose.position.z = float(center[2])

            sphere = Marker()
            sphere.type = Marker.SPHERE
            sphere.scale = Vector3(x=0.03, y=0.03, z=0.03)
            sphere.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])

            ctrl_vis = InteractiveMarkerControl()
            ctrl_vis.always_visible = True
            ctrl_vis.markers.append(sphere)
            im.controls.append(ctrl_vis)

            for axis_name, quat in [
                ('move_x', (1.0, 0.0, 0.0, 1.0)),
                ('move_y', (0.0, 1.0, 0.0, 1.0)),
                ('move_z', (0.0, 0.0, 1.0, 1.0)),
            ]:
                ctrl = InteractiveMarkerControl()
                ctrl.name = axis_name
                ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
                norm = math.sqrt(sum(c * c for c in quat))
                ctrl.orientation.w = quat[3] / norm
                ctrl.orientation.x = quat[0] / norm
                ctrl.orientation.y = quat[1] / norm
                ctrl.orientation.z = quat[2] / norm
                im.controls.append(ctrl)

            self.im_server.insert(im, feedback_callback=self._im_feedback)

        self.im_server.applyChanges()

    def _im_feedback(self, feedback: InteractiveMarkerFeedback):
        pos = feedback.pose.position
        target = np.array([pos.x, pos.y, pos.z])
        if feedback.marker_name == 'left_target':
            self.left_target = target
        elif feedback.marker_name == 'right_target':
            self.right_target = target

    def _generate_targets(self, t: float) -> tuple:
        radius = 0.025  # 2.5 cm radius — small enough to stay in workspace
        freq = 0.25     # Hz — slow enough for smooth tracking

        if self.mode == 'circle':
            angle = 2.0 * math.pi * freq * t
            left_target = self.left_center + np.array([
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.0,
            ])
            right_target = self.right_center + np.array([
                radius * math.cos(-angle),
                radius * math.sin(-angle),
                0.0,
            ])
        elif self.mode == 'figure8':
            angle = 2.0 * math.pi * freq * t
            left_target = self.left_center + np.array([
                radius * math.sin(angle),
                radius * math.sin(2.0 * angle) * 0.5,
                radius * math.cos(angle) * 0.3,
            ])
            right_target = self.right_center + np.array([
                radius * math.sin(-angle),
                radius * math.sin(-2.0 * angle) * 0.5,
                radius * math.cos(-angle) * 0.3,
            ])
        else:
            left_target = self.left_target
            right_target = self.right_target

        return left_target, right_target

    def _tick(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        left_target, right_target = self._generate_targets(t)

        # Solve IK with null-space regularization toward rest pose
        self.q, l_err, r_err, ok = self.ik.solve_both(
            left_target, right_target, self.q,
            q_ref=AlohaMiniFK.REST_POSE,
        )

        # Animate wheels (continuous spin for visual effect)
        self.wheel_angle += 2.0 * self.dt  # 2.0 rad/s
        if self.wheel_angle > 2.0 * math.pi:
            self.wheel_angle -= 2.0 * math.pi

        # Build joint_states message
        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = list(self.ALL_JOINT_NAMES)
        js.position = []
        for name in self.ALL_JOINT_NAMES:
            if name.startswith('wheel'):
                js.position.append(self.wheel_angle)
            else:
                js.position.append(self.q.get(name, 0.0))
        self.js_pub.publish(js)

        # Update trail
        T_left = self.fk.fk_left(self.q)
        T_right = self.fk.fk_right(self.q)
        self.left_trail.append(T_left[:3, 3].copy())
        self.right_trail.append(T_right[:3, 3].copy())
        if len(self.left_trail) > self.max_trail:
            self.left_trail.pop(0)
        if len(self.right_trail) > self.max_trail:
            self.right_trail.pop(0)

        self._publish_markers(left_target, right_target, now)

    def _publish_markers(self, left_target, right_target, stamp):
        ma = MarkerArray()
        stamp_msg = stamp.to_msg()

        for idx, (target, color) in enumerate([
            (left_target, (0.2, 0.9, 0.2)),
            (right_target, (0.9, 0.2, 0.2)),
        ]):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = stamp_msg
            m.ns = 'targets'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(target[0])
            m.pose.position.y = float(target[1])
            m.pose.position.z = float(target[2])
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=0.02, y=0.02, z=0.02)
            m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.9)
            m.lifetime.sec = 0
            ma.markers.append(m)

        for idx, (trail, color) in enumerate([
            (self.left_trail, (0.2, 0.7, 0.2)),
            (self.right_trail, (0.7, 0.2, 0.2)),
        ]):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = stamp_msg
            m.ns = 'trails'
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.005
            m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.6)
            m.lifetime.sec = 0
            for pt in trail:
                p = Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
                m.points.append(p)
            ma.markers.append(m)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = AlohaIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
