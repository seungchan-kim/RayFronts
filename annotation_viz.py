import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray
import json
import os
import re
import math
import colorsys
import numpy as np


class AnnotationViz(Node):
    def __init__(self):
        super().__init__('annotation_viz')

        script_name = os.environ.get('ISAAC_SIM_SCRIPT_NAME', '')
        env_name = re.sub(r'_Launch\.py$', '', script_name)

        spawn_x = float(os.environ.get('DRONE_X',  '0.0'))
        spawn_y = float(os.environ.get('DRONE_Y',  '0.0'))
        spawn_z = float(os.environ.get('DRONE_Z',  '0.07'))
        qz      = float(os.environ.get('DRONE_QZ', '0.0'))
        qw      = float(os.environ.get('DRONE_QW', '1.0'))
        yaw = 2.0 * math.atan2(qz, qw)

        raw_path = f'rayfronts/annotations/{env_name}.json'
        self.annotations = self._load_and_transform(raw_path, spawn_x, spawn_y, spawn_z, yaw)

        half = -yaw / 2.0
        self._orient = (0.0, 0.0, math.sin(half), math.cos(half))

        self.pub = self.create_publisher(MarkerArray, '/annotation_bboxes_all', 10)
        self.create_timer(1.0, self._publish)

        self._last_pos = None
        self._total_distance = 0.0
        self.traj_length_pub = self.create_publisher(String, 'trajectory_length', 10)
        self.create_subscription(
            Odometry,
            '/robot_1/odometry_conversion/odometry',
            self._odom_callback,
            10,
        )

        self.get_logger().info(
            f'env={env_name}, spawn=({spawn_x:.2f},{spawn_y:.2f},{spawn_z:.2f}), '
            f'yaw={math.degrees(yaw):.1f}deg, annotations={len(self.annotations)}'
        )

    def _load_and_transform(self, path, tx, ty, tz, yaw):
        if not os.path.exists(path):
            self.get_logger().error(f'Annotation file not found: {path}')
            return []

        with open(path, 'r') as f:
            data = json.load(f)

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        out = []
        for item in data:
            cx, cy, cz = item['bbox_world']['center_xyz_m']
            sx, sy, sz = item['bbox_world']['size_xyz_m']
            cx -= tx
            cy -= ty
            cz -= tz
            cx_ =  cx * cos_y + cy * sin_y
            cy_ = -cx * sin_y + cy * cos_y
            out.append({'class': item['class'], 'center': [cx_, cy_, cz], 'size': [sx, sy, sz]})
        return out

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        pos = np.array([p.x, p.y, p.z])
        if self._last_pos is not None:
            self._total_distance += float(np.linalg.norm(pos - self._last_pos))
        self._last_pos = pos
        out = String()
        out.data = str(round(self._total_distance, 2))
        self.traj_length_pub.publish(out)

    def _class_color(self, name):
        hue = (hash(name) & 0xFFFF) / 0xFFFF
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        return ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.4)

    def _publish(self):
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()
        qx, qy, qz, qw = self._orient
        mid = 0

        corners_template = [
            (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
            (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
        ]
        edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]

        for ann in self.annotations:
            cx, cy, cz = ann['center']
            sx, sy, sz = ann['size']
            color = self._class_color(ann['class'])
            color.a = 1.0

            box = Marker()
            box.header.frame_id = 'map'
            box.header.stamp = now
            box.ns = ann['class']
            box.id = mid; mid += 1
            box.type = Marker.LINE_LIST
            box.action = Marker.ADD
            box.pose.position.x = float(cx)
            box.pose.position.y = float(cy)
            box.pose.position.z = float(cz)
            box.pose.orientation.x = qx
            box.pose.orientation.y = qy
            box.pose.orientation.z = qz
            box.pose.orientation.w = qw
            box.scale.x = 0.05
            box.color = color
            hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
            corners = [(ex*hx, ey*hy, ez*hz) for ex, ey, ez in corners_template]
            for a, b in edges:
                for px, py, pz in (corners[a], corners[b]):
                    p = Point(); p.x, p.y, p.z = float(px), float(py), float(pz)
                    box.points.append(p)
            box.lifetime.sec = 2
            msg.markers.append(box)

            fill = Marker()
            fill.header.frame_id = 'map'
            fill.header.stamp = now
            fill.ns = ann['class'] + '_fill'
            fill.id = mid; mid += 1
            fill.type = Marker.CUBE
            fill.action = Marker.ADD
            fill.pose = box.pose
            fill.scale.x = float(sx)
            fill.scale.y = float(sy)
            fill.scale.z = float(sz)
            fill.color = self._class_color(ann['class'])
            fill.color.a = 0.15
            fill.lifetime.sec = 2
            msg.markers.append(fill)

            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = now
            label.ns = ann['class'] + '_label'
            label.id = mid; mid += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(cx)
            label.pose.position.y = float(cy)
            label.pose.position.z = float(cz) + float(sz) / 2.0 + 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.8
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            label.text = ann['class']
            label.lifetime.sec = 2
            msg.markers.append(label)

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AnnotationViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
