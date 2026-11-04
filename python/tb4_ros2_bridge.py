#!/usr/bin/env python3
"""The TurtleBot 4 simulator as real ROS 2 robots: a bridge between the Node API (server.mjs) and ROS 2 Humble.

    node server.mjs --site ../robotSimulator           # or with --key for https://robotics.dinno.be
    python3 tb4_ros2_bridge.py --world botopia --docked

Every simulated TurtleBot 4 appears as a real one, with the topics, actions and services of the real robot,
so the same code (Nav2, slam_toolbox, robot_localization, RViz, your own nodes) runs against the simulator
and against the lab's robots:

  out   /scan  /odom  /tf (odom → base_link)  /tf_static  /imu  /hazard_detection  /dock_status
        /battery_state  /oakd/rgb/preview/image_raw  /oakd/rgb/preview/camera_info  /clock
        /sim/ground_truth (nav_msgs/Odometry in the frame 'track': what the robot does not know)
  in    /cmd_vel
  actions   /undock /dock /drive_distance /rotate_angle /navigate_to_position (irobot_create_msgs)
  services  /e_stop /reset_pose

Robot k is on ROS_DOMAIN_ID (base + k), as in the lab, each in its own rclpy context; --namespace puts
them all on one domain under /robot0, /robot1, ... instead. Time is the simulation's: /clock, so use
use_sim_time:=true in your nodes. --rate 1 runs in real time, 0 as fast as the simulator can.

Odometry errors (default none: the simulator's odometry, which only drifts when a wheel slips):
--odom-scale (distance), --odom-drift (heading per metre), --odom-noise, --gyro-bias. They make the
simulation's odometry as poor as the real robot's: what an EKF must correct, and what sim-to-real tunes.

Frames: odom → base_link from the wheel odometry (it drifts when a wheel slips, as on the real robot);
base_link → rplidar_link (4 cm behind the middle, angle 0 straight ahead) and → oakd_rgb_camera_frame /
oakd_rgb_camera_optical_frame. Note: on the real TurtleBot 4 the URDF turns rplidar_link by 90 degrees, so
code that uses TF works on both, code that reads ranges[0] as "straight ahead" does not.
"""
import argparse
import io
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from botopia import BotopiaAPI          # noqa: E402

import numpy as np                       # noqa: E402
import rclpy                             # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse    # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup                # noqa: E402
from rclpy.executors import MultiThreadedExecutor                       # noqa: E402
from rclpy.node import Node                                             # noqa: E402
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, qos_profile_sensor_data   # noqa: E402
from builtin_interfaces.msg import Time                                 # noqa: E402
from rosgraph_msgs.msg import Clock                                     # noqa: E402
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped     # noqa: E402
from nav_msgs.msg import Odometry                                       # noqa: E402
from sensor_msgs.msg import LaserScan, Imu, Image, CameraInfo, BatteryState   # noqa: E402
from tf2_msgs.msg import TFMessage                                      # noqa: E402
from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector, DockStatus   # noqa: E402
from irobot_create_msgs.action import Dock, Undock, DriveDistance, RotateAngle, NavigateToPosition   # noqa: E402
from irobot_create_msgs.srv import EStop, ResetPose                     # noqa: E402

try:
    from PIL import Image as PILImage
except ImportError:                      # no camera without Pillow
    PILImage = None

# The robot's fixed frames, as the simulator draws them (indexTurtleBot4.php: LIDAR, CAM_AT).
LIDAR_X = -0.04
CAM = {'standard': (0.087, 0.331), 'lite': (0.122, 0.135)}
CAM_TILT = math.radians(5)
IMG = 250                                # the OAK-D preview: 250 x 250 bgr8
VFOV = math.radians(55)


def quat_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def quat_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def stamp_of(t):
    s = int(t)
    return Time(sec=s, nanosec=int((t - s) * 1e9))


class RobotNode(Node):
    """One simulated TurtleBot 4: its publishers, subscriber, actions and services."""

    def __init__(self, bridge, k, context, ns):
        super().__init__('tb4_sim_bridge', namespace=ns, context=context)
        self.bridge, self.k = bridge, k
        self.cb = ReentrantCallbackGroup()
        sd = qos_profile_sensor_data
        self.p_scan = self.create_publisher(LaserScan, 'scan', sd)
        self.p_odom = self.create_publisher(Odometry, 'odom', 10)
        self.p_tf = self.create_publisher(TFMessage, 'tf' if ns else '/tf', 50)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.p_tf_static = self.create_publisher(TFMessage, 'tf_static' if ns else '/tf_static', latched)
        self.p_imu = self.create_publisher(Imu, 'imu', sd)
        self.p_haz = self.create_publisher(HazardDetectionVector, 'hazard_detection', sd)
        self.p_dock = self.create_publisher(DockStatus, 'dock_status', sd)
        self.p_batt = self.create_publisher(BatteryState, 'battery_state', sd)
        self.p_img = self.create_publisher(Image, 'oakd/rgb/preview/image_raw', sd)
        self.p_info = self.create_publisher(CameraInfo, 'oakd/rgb/preview/camera_info', sd)
        self.p_truth = self.create_publisher(Odometry, 'sim/ground_truth', 10)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10, callback_group=self.cb)
        for name, typ in (('undock', Undock), ('dock', Dock), ('drive_distance', DriveDistance),
                          ('rotate_angle', RotateAngle), ('navigate_to_position', NavigateToPosition)):
            ActionServer(self, typ, name, execute_callback=self.make_execute(name, typ), callback_group=self.cb,
                         goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.create_service(EStop, 'e_stop', self.on_estop, callback_group=self.cb)
        self.create_service(ResetPose, 'reset_pose', self.on_reset_pose, callback_group=self.cb)
        self.cmd = None                   # (Twist dict, wall time) of the last /cmd_vel
        self.static_sent = False
        # The odometry's errors (--odom-scale, --odom-drift, --odom-noise, --gyro-bias): all 0 passes the
        # simulator's own odometry through; otherwise the bridge integrates its own from the true motion.
        self.err = bridge.err
        self.est = self.prev = None        # the odometry with errors, and the true pose of the step before
        self.imu_yaw = 0.0

    # ---------- in ----------
    def on_cmd_vel(self, msg):
        self.cmd = ({'linear': {'x': msg.linear.x, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': msg.angular.z}}, time.time())

    def make_execute(self, name, typ):
        def execute(handle):
            g = handle.request
            msg = {}
            if name == 'drive_distance':
                msg = {'distance': g.distance, 'max_translation_speed': g.max_translation_speed}
            elif name == 'rotate_angle':
                msg = {'angle': g.angle, 'max_rotation_speed': g.max_rotation_speed}
            elif name == 'navigate_to_position':
                p = g.goal_pose.pose
                msg = {'achieve_goal_heading': g.achieve_goal_heading, 'max_translation_speed': g.max_translation_speed,
                       'max_rotation_speed': g.max_rotation_speed,
                       'goal_pose': {'pose': {'position': {'x': p.position.x, 'y': p.position.y, 'z': 0.0},
                                              'orientation': {'x': p.orientation.x, 'y': p.orientation.y, 'z': p.orientation.z, 'w': p.orientation.w}}}}
            answer = self.bridge.request(self.k, {'kind': 'action', 'name': '/' + name, 'msg': msg}, wait_result=True,
                                         cancelled=lambda: handle.is_cancel_requested)
            res = typ.Result()
            if name in ('dock', 'undock'):
                res.is_docked = bool((answer or {}).get('data', {}).get('is_docked', name == 'dock'))
            if handle.is_cancel_requested:
                handle.canceled()
            elif answer and answer.get('status') == 'SUCCEEDED':
                handle.succeed()
            else:
                handle.abort()
            return res
        return execute

    def on_estop(self, req, res):
        a = self.bridge.request(self.k, {'kind': 'service', 'name': '/e_stop', 'msg': {'e_stop_on': req.e_stop_on}})
        res.success = bool(a) and a.get('code') == 0
        res.message = (a or {}).get('text', '') or (a or {}).get('msg', '')
        return res

    def on_reset_pose(self, req, res):
        self.bridge.request(self.k, {'kind': 'service', 'name': '/reset_pose', 'msg': {}})
        self.est = None                    # the odometry with errors starts again at 0 too
        return res

    def odom_with_errors(self, pose, sim_odom):
        """The odometry as wheels with errors would give it, from the true motion of this step."""
        tx, ty, tyaw = float(pose[0]), -float(pose[1]), float(pose[2])
        p, o = sim_odom['pose']['pose']['position'], sim_odom['pose']['pose']['orientation']
        sim = (float(p['x']), float(p['y']), 2 * math.atan2(float(o['z']), float(o['w'])))
        if self.est is None or self.prev is None or (abs(sim[0]) + abs(sim[1]) < 1e-4 and abs(self.est[0]) + abs(self.est[1]) > 0.01):
            self.est, self.imu_yaw = list(sim), sim[2]          # a start, a reset or a dock: start again where the simulator is
        else:
            px, py, pyaw = self.prev
            dx, dy = tx - px, ty - py
            fwd = math.cos(pyaw) * dx + math.sin(pyaw) * dy
            side = -math.sin(pyaw) * dx + math.cos(pyaw) * dy
            dyaw = math.atan2(math.sin(tyaw - pyaw), math.cos(tyaw - pyaw))
            e, rnd = self.err, np.random.normal
            n = e['noise'] * math.sqrt(abs(fwd) + 1e-9)
            fwd2 = fwd * (1 + e['scale']) + rnd(0, n)
            dyaw2 = dyaw + e['drift'] * abs(fwd) + rnd(0, n)
            c, sn = math.cos(self.est[2]), math.sin(self.est[2])
            self.est[0] += c * fwd2 - sn * side
            self.est[1] += sn * fwd2 + c * side
            self.est[2] = math.atan2(math.sin(self.est[2] + dyaw2), math.cos(self.est[2] + dyaw2))
            self.imu_yaw += dyaw + e['gyro_bias'] * self.bridge.a.dt
        self.prev = (tx, ty, tyaw)
        return self.est

    # ---------- out ----------
    def publish_all(self, t, topics, pose, imu, image, model, with_scan=True):
        st = stamp_of(t)
        self.p_clock.publish(Clock(clock=st))
        if not self.static_sent:
            self.send_static(st, model)
        odom = topics.get('/odom')
        if odom:
            m = Odometry()
            m.header.stamp, m.header.frame_id, m.child_frame_id = st, 'odom', 'base_link'
            p, o = odom['pose']['pose']['position'], odom['pose']['pose']['orientation']
            if self.err['on'] and pose is not None:
                ex, ey, eyaw = self.odom_with_errors(pose, odom)
                p, o = {'x': ex, 'y': ey}, {'x': 0.0, 'y': 0.0, 'z': math.sin(eyaw / 2), 'w': math.cos(eyaw / 2)}
            m.pose.pose.position.x, m.pose.pose.position.y = float(p['x']), float(p['y'])
            m.pose.pose.orientation.x, m.pose.pose.orientation.y = float(o['x']), float(o['y'])
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = float(o['z']), float(o['w'])
            tw = odom['twist']['twist']
            m.twist.twist.linear.x, m.twist.twist.angular.z = float(tw['linear']['x']), float(tw['angular']['z'])
            m.pose.covariance[0] = m.pose.covariance[7] = 0.01
            m.pose.covariance[35] = 0.02
            m.twist.covariance[0] = m.twist.covariance[35] = 0.01
            self.p_odom.publish(m)
            tf = TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = st, 'odom', 'base_link'
            tf.transform.translation.x, tf.transform.translation.y = m.pose.pose.position.x, m.pose.pose.position.y
            tf.transform.rotation = m.pose.pose.orientation
            self.p_tf.publish(TFMessage(transforms=[tf]))
            yaw_odom = 2 * math.atan2(float(o['z']), float(o['w']))
        else:
            yaw_odom = 0.0
        scan = topics.get('/scan') if with_scan else None
        if scan:
            m = LaserScan()
            m.header.stamp, m.header.frame_id = st, 'rplidar_link'
            m.angle_min, m.angle_max, m.angle_increment = float(scan['angle_min']), float(scan['angle_max']), float(scan['angle_increment'])
            m.scan_time, m.time_increment = float(scan['scan_time']), 0.0
            m.range_min, m.range_max = float(scan['range_min']), float(scan['range_max'])
            m.ranges = [float('inf') if r is None else float(r) for r in scan['ranges']]
            self.p_scan.publish(m)
        if imu is not None:
            m = Imu()
            m.header.stamp, m.header.frame_id = st, 'imu_link'
            yaw_imu = self.imu_yaw if self.err['on'] else yaw_odom
            m.orientation.z, m.orientation.w = math.sin(yaw_imu / 2), math.cos(yaw_imu / 2)
            g, a = list(imu.get('gyro', [0, 0, 0])), imu.get('accel', [0, 0, 9.81])
            g[2] = float(g[2]) + self.err['gyro_bias']
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(v) for v in g)
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (float(v) for v in a)
            m.orientation_covariance[0] = m.orientation_covariance[4] = m.orientation_covariance[8] = 0.01
            m.angular_velocity_covariance[8] = 0.001
            m.linear_acceleration_covariance[0] = m.linear_acceleration_covariance[4] = 0.01
            self.p_imu.publish(m)
        haz = topics.get('/hazard_detection')
        if haz is not None:
            m = HazardDetectionVector()
            m.header.stamp = st
            for d in haz.get('detections', []):
                h = HazardDetection()
                h.header.stamp, h.header.frame_id, h.type = st, d.get('header', {}).get('frame_id', 'base_link'), int(d.get('type', 1))
                m.detections.append(h)
            self.p_haz.publish(m)
        dock = topics.get('/dock_status')
        if dock is not None:
            m = DockStatus()
            m.header.stamp, m.dock_visible, m.is_docked = st, bool(dock.get('dock_visible')), bool(dock.get('is_docked'))
            self.p_dock.publish(m)
        batt = topics.get('/battery_state')
        if batt is not None:
            m = BatteryState()
            m.header.stamp = st
            m.voltage, m.percentage = float(batt.get('voltage', 16)), float(batt.get('percentage', 1))
            m.current, m.power_supply_status = float(batt.get('current', 0)), int(batt.get('power_supply_status', 2))
            self.p_batt.publish(m)
        if pose is not None:                # the truth: x to the right of the track, y up (away from the start straight)
            x, z, yaw = pose
            m = Odometry()
            m.header.stamp, m.header.frame_id, m.child_frame_id = st, 'track', 'base_link'
            m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(-z)
            qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = qz, qw
            self.p_truth.publish(m)
        if image is not None:
            image.header.stamp = st
            self.p_img.publish(image)
            info = CameraInfo()
            info.header.stamp, info.header.frame_id = st, 'oakd_rgb_camera_optical_frame'
            info.width = info.height = IMG
            f = (IMG / 2) / math.tan(VFOV / 2)
            info.k = [f, 0.0, IMG / 2, 0.0, f, IMG / 2, 0.0, 0.0, 1.0]
            info.p = [f, 0.0, IMG / 2, 0.0, 0.0, f, IMG / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.distortion_model, info.d = 'plumb_bob', [0.0] * 5
            self.p_info.publish(info)

    def send_static(self, st, model):
        cx, cz = CAM.get(model, CAM['standard'])
        frames = [('base_link', 'base_footprint', (0.0, 0.0, 0.0), (0, 0, 0)),     # slam_toolbox and Nav2 look for either
                  ('base_link', 'rplidar_link', (LIDAR_X, 0.0, 0.19), (0, 0, 0)),
                  ('base_link', 'imu_link', (0.0, 0.0, 0.05), (0, 0, 0)),
                  ('base_link', 'oakd_rgb_camera_frame', (cx, 0.0, cz), (0, CAM_TILT, 0)),
                  ('oakd_rgb_camera_frame', 'oakd_rgb_camera_optical_frame', (0, 0, 0), (-math.pi / 2, 0, -math.pi / 2))]
        out = []
        for parent, child, (x, y, z), (r, p, yw) in frames:
            tf = TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = st, parent, child
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = float(x), float(y), float(z)
            q = quat_rpy(r, p, yw)
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = q
            out.append(tf)
        self.p_tf_static.publish(TFMessage(transforms=out))
        self.static_sent = True


class Bridge:
    def __init__(self, a):
        self.a = a
        self.api = BotopiaAPI(a.api)
        opts = {'world': a.world}
        if a.docked:
            opts['docked'] = 1
        if a.robots > 1:
            opts['robots'] = a.robots
        if a.variant != 'standard':
            opts['variant'] = a.variant
        self.err = {'scale': a.odom_scale, 'drift': a.odom_drift, 'noise': a.odom_noise, 'gyro_bias': a.gyro_bias}
        self.err['on'] = any(self.err.values())
        env = self.api.create('turtlebot4', options=opts)
        self.id = env['id']
        self.lock = threading.Lock()
        self.queue = []                    # ros requests for the next step
        self.answers = []                  # ros answer events, newest last
        self.t = 0.0
        self.nodes, self.contexts = [], []
        base = int(os.environ.get('ROS_DOMAIN_ID', '0'))
        for k in range(a.robots):
            if a.namespace:
                ctx = self.contexts[0] if self.contexts else rclpy.Context()
                if not self.contexts:
                    rclpy.init(context=ctx, domain_id=base)
                    self.contexts.append(ctx)
                ns = f'/robot{k}'
            else:
                ctx = rclpy.Context()
                rclpy.init(context=ctx, domain_id=base + k)
                self.contexts.append(ctx)
                ns = ''
            self.nodes.append(RobotNode(self, k, ctx, ns))
            where = f'namespace {ns} on ROS_DOMAIN_ID {base}' if ns else f'ROS_DOMAIN_ID {base + k}'
            self.nodes[-1].get_logger().info(f'simulated TurtleBot 4 #{k} ({a.world}{", docked" if a.docked else ""}): {where}')

    def request(self, k, req, wait_result=False, cancelled=lambda: False, timeout=120.0):
        """Send a ROS request to robot k with the next step; return its answer (the goal's end with wait_result)."""
        req = dict(req, domain=k)
        with self.lock:
            mark = len(self.answers)
            self.queue.append(req)
        end = time.time() + timeout
        sent_cancel = False
        while time.time() < end:
            time.sleep(0.02)
            with self.lock:
                for e in self.answers[mark:]:
                    if e.get('name') == req['name'] and e.get('domain', k) == k and (not wait_result or e.get('status') or e.get('code', 0) != 0):
                        return e
            if wait_result and cancelled() and not sent_cancel:
                with self.lock:
                    self.queue.append({'kind': 'topic', 'name': '/cmd_vel', 'msg': {'linear': {'x': 0.0}, 'angular': {'z': 0.0}}, 'domain': k})
                sent_cancel = True
        return None

    def loop(self):
        a = self.a
        cam_every = max(1, round(1 / (a.dt * a.camera_hz))) if a.camera_hz > 0 and PILImage else 0
        scan_every = max(1, round(1 / (a.dt * 5.5)))      # the RPLIDAR A1 turns about 5.5 times a second
        n, t0 = 0, time.time()
        while all(rclpy.ok(context=c) for c in self.contexts):
            with self.lock:
                reqs, self.queue = self.queue, []
            now = time.time()
            for node in self.nodes:            # /cmd_vel is sent again every step while it is fresh (the base stops 0.5 s after the last)
                if node.cmd and now - node.cmd[1] < 0.5:
                    reqs.append({'kind': 'topic', 'name': '/cmd_vel', 'msg': node.cmd[0], 'domain': node.k})
            image = {'view': 'robot', 'width': 336, 'quality': 0.9} if cam_every and n % cam_every == 0 else None
            r = self.api.step(self.id, {'ros': reqs} if reqs else {}, dt=a.dt, image=image)
            obs = r['obs']
            self.t = float(obs.get('t', self.t + a.dt))
            ev = [e for e in r.get('events', []) if e.get('type') == 'ros' or 'kind' in e]
            if ev:
                with self.lock:
                    self.answers.extend(ev)
                    del self.answers[:-500]
            extra = obs.get('extra', {})
            img = self.decode(r.get('image')) if image else None
            robots = extra.get('robots') or [dict(domain=0, pose=None, **{'topics': extra.get('topics', {})})]
            for node in self.nodes:
                info = next((rb for rb in robots if rb.get('domain') == node.k), None)
                if not info:
                    continue
                node.publish_all(self.t, info.get('topics', {}), info.get('pose'), obs.get('imu') if node.k == 0 else None,
                                 img if node.k == extra.get('selected', 0) else None, extra.get('model', 'standard'), n % scan_every == 0)
            n += 1
            if a.rate > 0:                     # keep the simulation in step with the wall clock
                ahead = (n * a.dt) / a.rate - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)

    @staticmethod
    def decode(jpeg):
        if not jpeg or PILImage is None:
            return None
        im = np.asarray(PILImage.open(io.BytesIO(jpeg)).convert('RGB'))
        h, w = im.shape[:2]
        x0, y0 = (w - IMG) // 2, max(0, (h - IMG) // 2)
        crop = np.ascontiguousarray(im[y0:y0 + IMG, x0:x0 + IMG, ::-1])          # RGB → BGR
        m = Image()
        m.header.frame_id = 'oakd_rgb_camera_optical_frame'
        m.height, m.width, m.encoding, m.is_bigendian, m.step = crop.shape[0], crop.shape[1], 'bgr8', 0, crop.shape[1] * 3
        m.data = crop.tobytes()
        return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--api', default='http://localhost:8780', help='the Node API (server.mjs)')
    ap.add_argument('--world', default='botopia', help='botopia (the race track), course, house, warehouse, maze, ...')
    ap.add_argument('--docked', action='store_true', help='start on the dock, as the lab robots do')
    ap.add_argument('--robots', type=int, default=1, help='1 to 3 TurtleBot 4s in one world')
    ap.add_argument('--namespace', action='store_true', help='all robots on one domain, under /robot0, /robot1, ...')
    ap.add_argument('--variant', default='standard', choices=['standard', 'lite'])
    ap.add_argument('--dt', type=float, default=0.05, help='simulated seconds per step (0.05: 20 Hz)')
    ap.add_argument('--rate', type=float, default=1.0, help='speed against the wall clock (1: real time, 0: as fast as possible)')
    ap.add_argument('--odom-scale', type=float, default=0.0, help='odometry: distance error, e.g. 0.02 = 2%% too far')
    ap.add_argument('--odom-drift', type=float, default=0.0, help='odometry: heading error in rad per metre driven, e.g. 0.01')
    ap.add_argument('--odom-noise', type=float, default=0.0, help='odometry: random error, m and rad per square root of a metre')
    ap.add_argument('--gyro-bias', type=float, default=0.0, help='IMU: a constant error of the turning speed in rad/s, e.g. 0.002')
    ap.add_argument('--camera-hz', type=float, default=10.0, help='camera pictures per simulated second (0: no camera)')
    a = ap.parse_args()
    b = Bridge(a)
    for ctx in b.contexts:                 # an executor per context (per ROS_DOMAIN_ID)
        ex = MultiThreadedExecutor(num_threads=6, context=ctx)
        for node in b.nodes:
            if node.context is ctx:
                ex.add_node(node)
        threading.Thread(target=ex.spin, daemon=True).start()
    try:
        b.loop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            b.api.close(b.id)
        except Exception:
            pass
        for c in b.contexts:
            if rclpy.ok(context=c):
                rclpy.shutdown(context=c)


if __name__ == '__main__':
    main()
