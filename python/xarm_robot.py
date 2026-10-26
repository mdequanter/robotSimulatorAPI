"""One Python interface for the UFACTORY xArm 6 and xArm 7: the Botopia simulator or the real arm.

    from xarm_robot import connect, deproject
    arm = connect("sim", model=7)         # the xArm 7 simulator (Node API: node server.mjs)
    arm = connect("10.2.172.20")          # the real arm (pip install xarm-python-sdk); 6 or 7 axes: it asks the arm
    arm = connect("fake", model=7)        # a pretend arm inside Python: try a script without robot or simulator

    arm.move_to(0.35, 0.0, 0.25)          # gripper point (m, base frame), gripper pointing down
    arm.set_gripper(80)                   # opening in mm
    arm.move_to(0.35, 0.0, 0.027)
    arm.set_gripper(0)                    # closes; stops on a block
    print(arm.joints(), arm.tcp())
    jpeg = arm.camera()                   # wrist camera: the simulator's image or the real RealSense D435
    jpeg = arm.camera("external")         # the fixed camera that sees the whole arm
    mm = arm.depth()                      # its depth (the wrist RealSense D435): numpy uint16, millimetres, 0 = none
    x, y, z = deproject(arm.intrinsics(), 320, 240, mm[240, 320])   # a pixel → a point in the camera frame (m)
    arm.close()

The wrist camera is a RealSense D435 (colour and depth); the external camera is a normal USB webcam:
    python -c "import xarm_robot; xarm_robot.list_cameras()"          # the RealSense serial numbers
    arm = connect("10.2.172.20", wrist_serial="…", external_camera=1)  # external: the webcam's OpenCV index

The same script then runs on both. Units: metres, degrees, mm (gripper), seconds.
Base frame: origin where the arm stands on the table, x forward, y left, z up. The "gripper point" is
the middle between the finger pads, 155 mm from the flange (xArm Gripper).

In the simulator time only runs inside a call: moves with wait=True return when they are done, and
arm.sleep(t) lets t seconds pass (on the real arm it simply waits).
"""
import math
import threading
import time

import numpy as np

# ---------- Kinematics (UFACTORY's standard Denavit-Hartenberg parameters, as in indexXArm6.php) ----------
T2 = math.atan2(0.2845, 0.0535)
MODELS = {
    # rows: a, alpha, d, theta offset (m, rad)
    6: {"dh": [(0.0, -math.pi / 2, 0.267, 0.0), (math.hypot(0.2845, 0.0535), 0.0, 0.0, -T2),
               (0.0775, -math.pi / 2, 0.0, T2), (0.0, math.pi / 2, 0.3425, 0.0),
               (0.076, -math.pi / 2, 0.0, 0.0), (0.0, 0.0, 0.097, 0.0)],
        "limits": [[-360, 360], [-118, 120], [-225, 11], [-360, 360], [-97, 180], [-360, 360]],
        "ready": [0, -31, -68.6, 0, 99.6, 0]},
    7: {"dh": [(0.0, -math.pi / 2, 0.267, 0.0), (0.0, math.pi / 2, 0.0, 0.0), (0.0525, math.pi / 2, 0.293, 0.0),
               (0.0775, math.pi / 2, 0.0, 0.0), (0.0, math.pi / 2, 0.3425, 0.0), (0.076, -math.pi / 2, 0.0, 0.0),
               (0.0, 0.0, 0.097, 0.0)],
        "limits": [[-360, 360], [-118, 120], [-360, 360], [-11, 225], [-360, 360], [-97, 180], [-360, 360]],
        "ready": [0, -30.6, 0, 67.7, 0, 98.3, 0]},
}
TOOL = 0.155          # gripper point from the flange (m)
TIP = 0.175           # fingertips from the flange
GRIP_MAX = 85.0       # mm
MIN_Z = 0.005         # fingertips never lower than this above the table (m)

# ---------- Depth (RealSense D435) ----------
# Depth pictures are numpy arrays of millimetres (uint16, as the D435's z16 stream), aligned to the colour
# picture; 0 = no depth there. The simulator's cameras: 640 × 480, 55° vertical field of view.
SIM_FOV_V = 55.0


def depth_from_png(png):
    """The simulator's depth picture (PNG bytes, mm = red · 256 + green) as a uint16 array of mm."""
    from io import BytesIO
    from PIL import Image
    a = np.asarray(Image.open(BytesIO(png)).convert("RGB")).astype(np.uint16)
    return a[..., 0] * 256 + a[..., 1]


def depth_colormap(mm, near=None, far=None):
    """A depth array as a coloured RGB picture (uint8): near blue … far red, black where there is no depth.
    near / far in mm (default: the 2nd and 98th percentile of the valid depths)."""
    valid = mm > 0
    if not valid.any():
        return np.zeros(mm.shape + (3,), np.uint8)
    lo = near if near is not None else np.percentile(mm[valid], 2)
    hi = far if far is not None else np.percentile(mm[valid], 98)
    t = np.clip((mm.astype(np.float32) - lo) / max(1.0, hi - lo), 0, 1)
    rgb = np.stack([np.clip(1.5 - abs(4 * t - 3), 0, 1), np.clip(1.5 - abs(4 * t - 2), 0, 1), np.clip(1.5 - abs(4 * t - 1), 0, 1)], -1)
    rgb[~valid] = 0
    return (rgb * 255).astype(np.uint8)


def sim_intrinsics(width=640, height=480, fov_v=SIM_FOV_V):
    """Pinhole intrinsics (pixels) of a simulator camera: fx, fy, cx, cy."""
    f = (height / 2) / math.tan(math.radians(fov_v) / 2)
    return {"width": width, "height": height, "fx": f, "fy": f, "cx": width / 2, "cy": height / 2}


def deproject(intr, u, v, mm):
    """Pixel (u, v) at depth mm → point in the camera frame (m): x right, y down, z forward (RealSense)."""
    z = mm / 1000.0
    return ((u - intr["cx"]) * z / intr["fx"], (v - intr["cy"]) * z / intr["fy"], z)


def fk(q, model=6, tool=TOOL):
    """Pose (4×4) of the tool point for joint angles q (degrees), in the base frame."""
    T = np.eye(4)
    for (a, al, d, off), th in zip(MODELS[model]["dh"], np.radians(q)):
        th += off
        c, s, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
        T = T @ np.array([[c, -s * ca, s * sa, a * c], [s, c * ca, -c * sa, a * s], [0, sa, ca, d], [0, 0, 0, 1]])
    return T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, tool], [0, 0, 0, 1]])


def _down(yaw):
    """Target axes (columns) of a gripper pointing straight down, turned yaw (rad) round the vertical."""
    return np.array([[math.cos(yaw), math.sin(yaw), 0], [math.sin(yaw), -math.cos(yaw), 0], [0, 0, -1]]).T


def _rot_err(R, Rt):
    return 0.5 * sum(np.cross(R[:, k], Rt[:, k]) for k in range(3))


def ik(p, yaw=0.0, start=None, model=6):
    """See _ik; if it doesn't find a way from `start`, it tries once more from the look-down pose."""
    q = _ik(p, yaw, start, model)
    if q is None and start is not None:
        q = _ik(p, yaw, None, model)
        if q is not None:
            q = _unwound(q, start)
    return q


def _ik(p, yaw=0.0, start=None, model=6):
    """Joint angles (degrees) that put the gripper point at p (m), pointing down, turned yaw (rad).
    Damped least squares from `start` (default: the look-down pose). None when out of reach.
    The xArm 7 has one joint more than it needs: of all solutions, the upper arm roll (joint 3) is
    pulled gently towards 0° (in the "null space": moves that don't move the gripper), so the elbow
    doesn't wander from move to move and the arm moves much like an xArm 6."""
    M = MODELS[model]
    lim = np.array(M["limits"], dtype=float)
    q = np.array(start if start is not None else M["ready"], dtype=float)
    q_start = q.copy()
    # Start with the base already turned towards the target: the straight way there may pass the
    # arm's own column, where the solver would get stuck.
    T0 = fk(q, model)
    turn = math.atan2(p[1], p[0]) - math.atan2(T0[1, 3], T0[0, 3])
    q[0] = float(np.clip(q[0] + math.degrees(math.atan2(math.sin(turn), math.cos(turn))), lim[0, 0], lim[0, 1]))
    n, Rt = len(q), _down(yaw)
    found = None
    for _ in range(300):
        T = fk(q, model)
        e = np.concatenate([np.asarray(p) - T[:3, 3], _rot_err(T[:3, :3], Rt)])
        J = np.zeros((6, n))
        for j in range(n):
            dq = np.zeros(n)
            dq[j] = math.degrees(1e-4)
            T2_ = fk(q + dq, model)
            J[:3, j] = (T2_[:3, 3] - T[:3, 3]) / 1e-4
            J[3:, j] = -_rot_err(T2_[:3, :3], T[:3, :3]) / 1e-4
        Jp = J.T @ np.linalg.inv(J @ J.T + 0.02 ** 2 * np.eye(6))            # damped pseudo-inverse
        step, null = Jp @ e, np.zeros(n)
        if n > 6:                                                            # redundant: use the null space
            P = np.eye(n) - J.T @ np.linalg.solve(J @ J.T + 1e-9 * np.eye(6), J)   # moves that keep the gripper still
            pull = np.zeros(n)
            pull[2] = -0.1 * math.radians(q[2])                             # joint 3 towards 0°
            null = P @ pull
            step += null
        if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
            found = _unwound(q, q_start)                                     # there (the elbow may still settle)
            if np.abs(null).max() < 2e-3:
                return found
        big = np.abs(step).max()
        if big > 0.2:
            step *= 0.2 / big
        q = np.clip(q + np.degrees(step), lim[:, 0], lim[:, 1])
    return found


def _unwound(q, start):
    """The base (first) and flange (last) joint can turn ±360°, but a turn of 360° gives the same pose:
    of those, take the one nearest to where the joint is, within ±270° (the cables don't wind up)."""
    q = [float(v) for v in q]
    for i in (0, len(q) - 1):
        options = [q[i] + k * 360.0 for k in (-2, -1, 0, 1, 2) if abs(q[i] + k * 360.0) <= 270.0]
        q[i] = min(options, key=lambda v: abs(v - start[i]))
    return q


class UnsafeMove(ValueError):
    pass


def check_safe(q, model=6, opening_mm=GRIP_MAX, min_z=MIN_Z):
    """Refuses joint targets outside the limits or with the fingertips below min_z (the table)."""
    lim = np.array(MODELS[model]["limits"], dtype=float)
    q = np.asarray(q, dtype=float)
    if q.shape != (len(lim),) or not np.all(np.isfinite(q)):
        raise UnsafeMove(f"need {len(lim)} joint angles for the xArm {model}, got {q.size}")
    if np.any(q < lim[:, 0]) or np.any(q > lim[:, 1]):
        raise UnsafeMove(f"joint angle out of range: {np.round(q, 1).tolist()}")
    T = fk(q, model, TIP)
    low = min(T[2, 3] + s * T[2, 1] * ((opening_mm or 0) / 2000 + 0.012) for s in (-1, 1))
    if low < min_z:
        raise UnsafeMove(f"the fingertips would be {low * 100:.1f} cm above the table (minimum {min_z * 100:.1f} cm)")


# ---------- The common interface ----------
class XArm:
    """What every backend offers. Subclasses implement joints, gripper, move_joints, set_gripper."""

    model = 6
    min_z = MIN_Z

    def tcp(self):
        """Gripper point (x, y, z in m) and its turn round the vertical (degrees)."""
        T = fk(self.joints(), self.model)
        return (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), math.degrees(math.atan2(T[1, 0], T[0, 0])))

    def move_to(self, x, y, z, yaw_deg=0.0, speed=30, wait=True):
        """Move the gripper point to (x, y, z), gripper pointing down, turned yaw_deg."""
        q = ik([x, y, z], math.radians(yaw_deg), self.joints(), self.model)
        if q is None:
            raise UnsafeMove(f"({x:.3f}, {y:.3f}, {z:.3f}) is out of reach")
        self.move_joints(q, speed=speed, wait=wait)

    def check(self, q, opening_mm=None):
        check_safe(q, self.model, self.gripper() if opening_mm is None else opening_mm, self.min_z)

    def open_gripper(self, wait=True):
        self.set_gripper(GRIP_MAX, wait=wait)

    def close_gripper(self, wait=True):
        self.set_gripper(0, wait=wait)

    def camera(self, which="wrist"):
        return None

    def depth(self, which="wrist"):
        """Depth picture of the wrist RealSense D435: uint16 array of mm (0 = none), or None (no depth camera)."""
        return None

    def intrinsics(self, which="wrist"):
        """Pinhole intrinsics of that camera's (colour-aligned) pictures: fx, fy, cx, cy in pixels."""
        return sim_intrinsics()

    def sleep(self, seconds):
        time.sleep(seconds)

    def stop(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SimXArm(XArm):
    """The xArm simulator (indexXArm6.php / indexXArm7.php) through the Botopia Node API."""

    def __init__(self, url="http://localhost:8780", model=7, seed=None):
        from botopia import BotopiaAPI
        self.model = model
        self.api = BotopiaAPI(url)
        created = self.api.create(f"xarm{model}", seed=seed)
        self.id, self._obs = created["id"], created["obs"]
        self._targets = {}

    def _step(self, action=None, dt=0.1):
        action = dict(action or {})
        action.setdefault("motors", {}).update({k: v for k, v in self._targets.items() if k not in action["motors"]})
        self._obs = self.api.step(self.id, action, dt=dt, frame=1 / 30)["obs"]
        return self._obs

    def _motor(self, mid):
        return next(m["deg"] for m in self._obs["motors"] if m["id"] == mid)

    def joints(self):
        return [self._motor(f"joint{i + 1}") for i in range(self.model)]

    def gripper(self):
        return self._motor("gripper")

    def move_joints(self, q, speed=30, wait=True):
        self.check(q)
        self._targets.update({f"joint{i + 1}": float(v) for i, v in enumerate(q)})
        self._step({"jointSpeed": speed})
        t0 = self._obs["t"]
        while wait and max(abs(a - b) for a, b in zip(self.joints(), q)) > 0.1 and self._obs["t"] - t0 < 60:
            self._step()

    def set_gripper(self, mm, wait=True):
        mm = float(min(GRIP_MAX, max(0.0, mm)))
        self._targets["gripper"] = mm
        self._step()
        last, still = None, 0
        while wait and abs(self.gripper() - mm) > 0.5 and still < 3:      # stops early on a block
            self._step()
            still = still + 1 if last is not None and abs(self.gripper() - last) < 0.05 else 0
            last = self.gripper()

    def sleep(self, seconds):
        for _ in range(max(1, round(seconds / 0.1))):
            self._step()

    def camera(self, which="wrist"):
        """JPEG of the wrist camera or of the external camera (the whole arm)."""
        return self.api.frame(self.id, 0.9, "robot" if which == "wrist" else "external", 640)

    def depth(self, which="wrist"):
        """Depth of the wrist D435 (640 × 480 mm, aligned to camera()); 0 = no depth. The external camera
        is a normal camera: None."""
        if which != "wrist":
            return None
        return depth_from_png(self.api.frame(self.id, 1, "depth", 640))

    def below(self):
        """Simulator: what lies under the gripper, along its axis, as the wrist D435 sees it: depth_mm (its
        depth there), gap_mm (under the fingertips), height_mm (above the table), block (id or None) and
        u, v (where it is in the 640 × 480 picture)."""
        return self._step(dt=1 / 30)["extra"].get("below")

    def blocks(self):
        """Simulator only: the coloured blocks (id, color, x, y, z, yawDeg, held)."""
        return self._step(dt=1 / 30)["extra"]["blocks"]

    def stop(self):
        self._targets.update({f"joint{i + 1}": v for i, v in enumerate(self.joints())})
        self._step()

    def close(self):
        self.api.close(self.id)


class FakeXArm(XArm):
    """A pretend arm inside Python (joints move at their speed in a thread): no robot needed."""

    def __init__(self, model=7):
        self.model = model
        self.q = np.array(MODELS[model]["ready"], dtype=float)
        self.target, self.speed = self.q.copy(), 30.0
        self.grip = self.grip_t = 60.0
        self.enabled, self._run = True, True
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            with self._lock:
                d = self.target - self.q
                far = np.abs(d).max()
                if far > 1e-6 and self.enabled:
                    self.q += d * min(1.0, self.speed * 0.02 / far)
                self.grip += max(-1.8, min(1.8, self.grip_t - self.grip))
            time.sleep(0.02)

    def joints(self):
        with self._lock:
            return [float(v) for v in self.q]

    def gripper(self):
        return float(self.grip)

    def move_joints(self, q, speed=30, wait=True):
        self.check(q, self.grip)
        with self._lock:
            self.target, self.speed = np.array(q, dtype=float), float(speed)
        while wait and np.abs(np.array(self.joints()) - self.target).max() > 0.1:
            time.sleep(0.02)

    def follow(self, q, mm, speed):
        self.move_joints(q, speed, wait=False)
        if mm is not None:
            self.set_gripper(mm, wait=False)

    def set_gripper(self, mm, wait=True):
        self.grip_t = float(min(GRIP_MAX, max(0.0, mm)))
        while wait and abs(self.grip - self.grip_t) > 0.5:
            time.sleep(0.02)

    def enable(self):
        self.enabled = True

    def stop(self):
        with self._lock:
            self.target = self.q.copy()

    def status(self):
        return {"robot": f"pretend xArm {self.model} (--fake)", "model": self.model, "enabled": self.enabled, "error": 0, "warn": 0}

    def close(self):
        self._run = False


def list_cameras():
    """Print the RealSense cameras on this PC (name and serial number) and return their serials."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not installed: pip install pyrealsense2")
        return []
    serials = []
    for dev in rs.context().query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"{dev.get_info(rs.camera_info.name)}  serial {serial}  (USB {dev.get_info(rs.camera_info.usb_type_descriptor)})")
        serials.append(serial)
    if not serials:
        print("No RealSense camera found")
    return serials


class FakeCamera:
    """A pretend camera (for --fake): a grey picture with its name and the time."""

    def __init__(self, name):
        self.name = name

    def jpeg(self, quality=80, width=480):
        from io import BytesIO
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (width, width * 3 // 4), (40, 52, 66))
        ImageDraw.Draw(im).text((12, 12), f"pretend {self.name} camera  {time.strftime('%H:%M:%S')}", fill=(255, 255, 255))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def depth(self):
        """A pretend depth picture: the table 500 mm away, a 4 cm block in the middle."""
        mm = np.full((480, 640), 500, np.uint16)
        mm[200:280, 280:360] = 460
        return mm

    def intrinsics(self):
        return sim_intrinsics()

    def close(self):
        pass


class RealSense:
    """A RealSense D435 over USB (pip install pyrealsense2): colour, and depth aligned to the colour
    picture. serial: which camera, when there are several (see list_cameras); None = the first one found."""

    def __init__(self, serial=None, width=640, height=480, fps=30, depth=True):
        import pyrealsense2 as rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipe.start(cfg)
        self.serial, self.has_depth = serial, depth
        self.align = rs.align(rs.stream.color) if depth else None
        # Depth units (m per step; 0.001 on a D435): to millimetres.
        self.mm_per_unit = profile.get_device().first_depth_sensor().get_depth_scale() * 1000 if depth else 1
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intr = {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy}

    def depth(self):
        """Depth aligned to the colour picture: uint16 array of mm (0 = no depth), or None without depth."""
        if not self.has_depth:
            return None
        frames = self.align.process(self.pipe.wait_for_frames(2000))
        d = np.asanyarray(frames.get_depth_frame().get_data())
        return d if self.mm_per_unit == 1 else (d * self.mm_per_unit).astype(np.uint16)

    def intrinsics(self):
        return dict(self._intr)

    def jpeg(self, quality=80, width=None):
        from io import BytesIO
        from PIL import Image
        frame = self.pipe.wait_for_frames(2000).get_color_frame()
        im = Image.fromarray(np.asanyarray(frame.get_data()))
        if width:
            im = im.resize((width, round(im.height * width / im.width)))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def close(self):
        self.pipe.stop()


class UsbCamera:
    """A normal USB camera (webcam) through OpenCV (pip install opencv-python): the external camera.
    index: OpenCV's camera number (0, 1, ...; a laptop's own camera is usually 0)."""

    def __init__(self, index=0, width=640, height=480):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(int(index))
        if not self.cap.isOpened():
            raise RuntimeError(f"no camera at index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def jpeg(self, quality=80, width=None):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("the camera gave no picture")
        if width:
            frame = self.cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])))
        return self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()

    def close(self):
        self.cap.release()


class RealXArm(XArm):
    """The real arm through UFACTORY's SDK (pip install xarm-python-sdk); camera: RealSense over USB.

    The model (6 or 7 axes) is read from the arm. speed_limit caps every move (°/s). The arm's own
    safety settings (collision sensitivity, reduced mode, safety boundary in UFACTORY Studio) still
    apply: set them before you start."""

    def __init__(self, ip, camera=True, speed_limit=60, model=None, wrist_serial=None, external_camera=None):
        from xarm.wrapper import XArmAPI
        self.arm = XArmAPI(ip, is_radian=False)
        self.ip, self.speed_limit = ip, speed_limit
        axes = getattr(self.arm, "axis", None)
        self.model = model or (axes if axes in MODELS else 7)
        if model and axes and axes != model:
            raise RuntimeError(f"the arm at {ip} has {axes} axes, not {model}")
        self.enable()
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(3000)
        self._cams = {}
        if camera:
            # The wrist camera: a RealSense D435 (serial number: which one, when there are several).
            try:
                self._cams["wrist"] = RealSense(wrist_serial)
            except Exception as e:                       # not connected, or no pyrealsense2
                print(f"RealSense wrist camera not available: {e}")
            # The external camera: a normal USB webcam (its OpenCV index), if given.
            if external_camera is not None:
                try:
                    self._cams["external"] = UsbCamera(external_camera)
                except Exception as e:
                    print(f"External camera not available: {e}")
        self._last_grip = None

    def enable(self):
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)              # all joints
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)

    def _mode(self, mode):
        if self.arm.mode != mode:
            self.arm.set_mode(mode)
            self.arm.set_state(0)
            time.sleep(0.2)
        return self.arm.mode == mode

    def joints(self):
        code, angles = self.arm.get_servo_angle()
        if code != 0:
            raise RuntimeError(f"xArm: can't read the joints (code {code})")
        return [float(a) for a in angles[:self.model]]

    def gripper(self):
        code, pos = self.arm.get_gripper_position()
        return None if code != 0 or pos is None else float(pos) / 10.0       # 0 … 850 → mm

    def move_joints(self, q, speed=30, wait=True):
        self.check(q, self.gripper() or 0)
        speed = float(min(speed, self.speed_limit))
        self._mode(0)
        code = self.arm.set_servo_angle(angle=[float(v) for v in q], speed=speed, mvacc=speed * 10, wait=wait)
        if code != 0:
            raise RuntimeError(f"xArm: move refused (code {code}, error {self.arm.error_code})")

    def follow(self, q, mm, speed):
        """Online target (bridge): the newest target replaces the last one (mode 6, joint online
        trajectory planning). If the firmware doesn't have it: point-to-point (mode 0)."""
        self.check(q, mm)
        speed = float(min(speed, self.speed_limit))
        if not self._mode(6):
            self._mode(0)
        self.arm.set_servo_angle(angle=[float(v) for v in q], speed=speed, mvacc=speed * 10, wait=False)
        if mm is not None and (self._last_grip is None or abs(mm - self._last_grip) > 2):
            self.arm.set_gripper_position(max(0, min(850, mm * 10)), wait=False)
            self._last_grip = mm

    def set_gripper(self, mm, wait=True):
        self.arm.set_gripper_position(max(0, min(850, float(mm) * 10)), wait=wait)
        self._last_grip = mm

    def camera(self, which="wrist"):
        """JPEG of the wrist camera or of the external camera, None if it isn't there."""
        cam = self._cams.get(which)
        return cam.jpeg() if cam else None

    def depth(self, which="wrist"):
        """Depth of the wrist RealSense, aligned to camera(): uint16 array of mm (0 = none); None if it isn't
        there (the external camera is a normal webcam)."""
        cam = self._cams.get(which)
        return cam.depth() if cam else None

    def intrinsics(self, which="wrist"):
        cam = self._cams.get(which)
        return cam.intrinsics() if cam else sim_intrinsics()

    def stop(self):
        self.arm.set_state(4)                           # stop the motion; enable() to go on

    def status(self):
        a = self.arm
        return {"robot": f"xArm {self.model} at {self.ip}", "model": self.model,
                "enabled": a.state in (0, 1, 2) and a.error_code == 0,
                "error": a.error_code, "warn": a.warn_code, "state": a.state, "mode": a.mode}

    def close(self):
        for cam in self._cams.values():
            cam.close()
        self.arm.disconnect()


def connect(target="sim", model=7, **kw):
    """'sim' (or an http:// URL of the Node API), 'fake', or the arm's IP address (model: read from the arm)."""
    if target == "sim" or str(target).startswith("http"):
        url = target if str(target).startswith("http") else kw.pop("url", "http://localhost:8780")
        return SimXArm(url, model=model, **kw)
    if target == "fake":
        return FakeXArm(model)
    return RealXArm(target, **kw)
