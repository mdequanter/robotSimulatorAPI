"""Tutorial 20: drive around the house until YOLO finds the object you chose, then drive up to it.

The TurtleBot3 with its OpenMANIPULATOR-X and the YOLO model of tutorial 19 (python/models/house_yolo11n.pt). The robot
only uses what a real one has: the gripper camera (through YOLO), the LiDAR (/scan) and the odometry (/odom), and it
drives with /cmd_vel. Three things it does in turn:

    look      stand still, the arm turns the gripper camera from left to right: is the object in one of the pictures?
    explore   not seen: drive to the nearest place it has not looked around from yet, along a path on its own map
              (a grid of what the LiDAR saw: free floor and walls), through the doors
    approach  seen: turn the base to it, then drive to it, the box kept in the middle of the picture, until the LiDAR
              says it is close

    python python/house_seek.py                   # asks which object
    python python/house_seek.py --target bed      # chair, table, desk, sofa, bed, bookcase, plant, fridge, toilet, person

Start the server first: node server.mjs --key rsk_...   (--show to watch). Needs: pip install ultralytics
"""
import argparse
import base64
import heapq
import io
import math
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from botopia import BotopiaAPI  # noqa: E402
from house_yolo import CLASSES, MODEL, draw  # noqa: E402

WIDTH, HEIGHT = 640, 480
HFOV = 2 * math.atan(math.tan(math.radians(62 / 2)) * WIDTH / HEIGHT)   # the gripper camera: 62° high, so 77° wide
JOINTS = ["joint1", "joint2", "joint3", "joint4"]
LOOK = [-1.0, 0.3, 0.7]          # joints 2-4: the arm upright above the base (its start pose), the camera level
PANS = [p / 10 for p in range(-24, 25, 4)]   # joint1 from -2.4 to +2.4 rad: with the 77° of the camera, all but behind
STEP = 0.25                      # seconds per step; /cmd_vel stops the wheels when it is not sent for half a second
V, W = 0.18, 0.8                 # m/s and rad/s (the Waffle Pi: at most 0.26 m/s and 1.82 rad/s)
HALF = 0.21                      # metres: half the width of the Waffle Pi (0.306 m) and a margin
CLEAR = 0.35                     # metres: something closer than this in the robot's path (LiDAR) and it stops
ARRIVE = 0.55                    # metres: the object this close in front, and the robot has arrived
NEW = 1.5                        # metres: a place at least this far from where the robot looked around is worth a look
HOUSE = [-5.9, -3.9, 5.9, 3.9]   # world x0, z0, x1, z1: the robot searches inside the house, not in the garden


def twist(v, w):
    """/cmd_vel: forward v m/s, turning w rad/s (left = positive)."""
    return {"kind": "topic", "name": "/cmd_vel", "type": "geometry_msgs/msg/Twist",
            "msg": {"linear": {"x": v, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": w}}}


def arm(pan, sec=0.4):
    """/arm_controller/joint_trajectory: joint1 = pan (left = positive), the other joints in the look pose."""
    return {"kind": "topic", "name": "/arm_controller/joint_trajectory", "type": "trajectory_msgs/msg/JointTrajectory",
            "msg": {"joint_names": JOINTS, "points": [{"positions": [pan, *LOOK],
                                                        "time_from_start": {"sec": int(sec), "nanosec": int(sec % 1 * 1e9)}}]}}


def odom(obs):
    """x, y (metres from the start) and yaw (radians) from /odom."""
    p = obs["extra"]["topics"]["/odom"]["pose"]["pose"]
    q = p["orientation"]
    return p["position"]["x"], p["position"]["y"], 2 * math.atan2(q["z"], q["w"])


def scan(obs):
    """/scan: 360 distances, index i = i degrees to the left of the front; no echo (None) counts as 8 m."""
    return [r if r is not None else 8.0 for r in obs["extra"]["topics"]["/scan"]["ranges"]]


def free(ranges, deg, half=20):
    """The shortest LiDAR distance in a cone of ±half degrees round direction deg (0 = in front, + = left)."""
    return min(ranges[(round(deg) + d) % 360] for d in range(-half, half + 1))


def ahead(ranges):
    """How far the robot can drive straight on: the nearest LiDAR point in a lane as wide as the robot, in front."""
    d = 8.0
    for i, r in enumerate(ranges):
        x, y = r * math.cos(math.radians(i)), r * math.sin(math.radians(i))
        if x > 0 and abs(y) < HALF:
            d = min(d, x)
    return d


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Map:
    """What the LiDAR has seen, on a grid of 10 cm cells in odometry coordinates (the start in the middle). Per cell
    it counts the beams that passed through it and the beams that ended in it: a wall where more ended than passed.
    (One beam is not enough: a beam that just grazes a door post would close the door on the map.)"""
    RES, N = 0.1, 300            # 30 x 30 m

    def __init__(self):
        self.passes = np.zeros((self.N, self.N), np.int32)
        self.hits = np.zeros((self.N, self.N), np.int32)
        self.blocked = np.zeros((self.N, self.N), np.int32)  # bumped into (block): for so many more plans

    @property
    def free(self):
        return self.passes > 0

    @property
    def wall(self):
        return (self.hits > self.passes) | (self.blocked > 0)

    def cell(self, x, y):
        return int(round(x / self.RES)) + self.N // 2, int(round(y / self.RES)) + self.N // 2

    def xy(self, c):
        return (c[0] - self.N // 2) * self.RES, (c[1] - self.N // 2) * self.RES

    def add(self, x, y, yaw, ranges):
        for i in range(360):
            r = min(ranges[i], 7.9)
            a = yaw + math.radians(i)
            d = np.arange(0.0, r - self.RES, self.RES / 2)
            cx = np.round((x + d * math.cos(a)) / self.RES).astype(int) + self.N // 2
            cy = np.round((y + d * math.sin(a)) / self.RES).astype(int) + self.N // 2
            cells = np.unique(cx * self.N + cy)
            self.passes.flat[cells] += 1
            if ranges[i] < 7.9:
                self.hits[self.cell(x + r * math.cos(a), y + r * math.sin(a))] += 1

    def block(self, x, y, yaw):
        """Something the LiDAR does not see (a table top, a chair leg low down) just in front: a wall on the map, for the
        next three plans (a bump in a door does not close that door for good)."""
        for d in (0.25, 0.35):
            for s in (-0.15, 0.0, 0.15):
                self.blocked[self.cell(x + d * math.cos(yaw) - s * math.sin(yaw), y + d * math.sin(yaw) + s * math.cos(yaw))] = 3

    @staticmethod
    def grow(wall, r):
        """The walls made r cells thicker on every side."""
        out = wall.copy()
        for dx in range(-int(r), int(r) + 1):
            for dy in range(-int(r), int(r) + 1):
                if dx * dx + dy * dy <= r * r:
                    out |= np.roll(np.roll(wall, dx, 0), dy, 1)
        return out

    def plan(self, start, looked, inside):
        """The shortest path (Dijkstra on the grid) to the nearest free cell at least NEW metres from every place where
        the robot looked around: a list of cells, or None when the whole map has been looked at. The path may cross
        cells the LiDAR has not seen yet (at double cost), never walls.
        inside: the cells of the fence (a boolean grid); the path stays in it."""
        self.blocked[self.blocked > 0] -= 1
        wall = self.wall
        room = ~self.grow(wall, 2.5) & inside                # room for the robot: 0.25 m from every wall
        ok = room & self.free
        unknown = room & ~self.free                          # not seen yet: allowed (the LiDAR checks it on the way)
        near = self.grow(wall, 5)                            # within 0.5 m of a wall: allowed, but it costs more
        s = self.cell(*start)
        ok[s[0] - 3:s[0] + 4, s[1] - 3:s[1] + 4] |= self.free[s[0] - 3:s[0] + 4, s[1] - 3:s[1] + 4]   # leave the wall
        far = ok.copy()                                      # the goal: free floor seen by the LiDAR ...
        ix, iy = np.indices(ok.shape)
        for lx, ly in looked:                                # ... at least NEW m from every place it looked from
            cx, cy = self.cell(lx, ly)
            far &= (ix - cx) ** 2 + (iy - cy) ** 2 >= (NEW / self.RES) ** 2
        dist, prev, todo = {s: 0.0}, {}, [(0.0, s)]
        while todo:
            d, c = heapq.heappop(todo)
            if d > dist[c]:
                continue
            if far[c]:
                path = [c]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return path[::-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (c[0] + dx, c[1] + dy)
                if 0 <= n[0] < self.N and 0 <= n[1] < self.N and (ok[n] or unknown[n]):
                    # Through the middle of a door, and over floor it has seen rather than floor it has not.
                    nd = d + math.hypot(dx, dy) * (3 if near[n] else 1) * (2 if unknown[n] else 1)
                    if nd < dist.get(n, 1e9):
                        dist[n], prev[n] = nd, c
                        heapq.heappush(todo, (nd, n))
        return None


class Seeker:
    def __init__(self, api, eid, model, target, show, start):
        """start: where the robot stands on the house plan (world x, z, yaw), as a robot that localises itself on a
        known map knows it. It only sets the fence: everything else comes from the robot's own sensors."""
        self.api, self.eid, self.model, self.target = api, eid, model, CLASSES.index(target)
        self.show, self.panel, self.pan, self.obs = show, None, 0.0, None
        self.map, self.n = Map(), 0
        self.looked = []                 # odometry x, y of the places where the robot looked around
        self.given_up = []               # goals it could not reach: not worth a look any more
        self.trail = []
        self.start = start
        ix, iy = np.indices((Map.N, Map.N))
        wx, wz = self.to_world((ix - Map.N // 2) * Map.RES, (iy - Map.N // 2) * Map.RES)
        self.inside = (wx > HOUSE[0]) & (wx < HOUSE[2]) & (wz > HOUSE[1]) & (wz < HOUSE[3])

    def to_world(self, x, y):
        """Odometry x (forward at the start), y (left) to world x, z of the house plan."""
        x0, z0, yaw0 = self.start["x"], self.start["z"], self.start["yaw"]
        c, s = np.cos(yaw0), np.sin(yaw0)
        return x0 + x * c - y * s, z0 - x * s - y * c

    def step(self, v=0.0, w=0.0, pan=None, dt=STEP, picture=False):
        ros = [twist(v, w)]
        if pan is not None:
            ros.append(arm(pan, max(0.3, abs(pan - self.pan) / 1.5)))
            self.pan = pan
        action = {"ros": ros}
        if self.show and self.panel:
            action["show"] = self.panel
        r = self.api.step(self.eid, action, dt=dt,
                          image={"view": "gripper", "width": WIDTH, "quality": 0.85} if picture else None)
        self.obs = r["obs"]
        self.n += 1
        if self.n % 2 == 0:                                  # every half second: the scan on the map
            self.map.add(*odom(self.obs), scan(self.obs))
        return r.get("image")

    def detect(self, jpeg, close=False):
        """YOLO on a picture: the best box of the target, as (confidence, angle to the left of the camera's middle, box height
        as a fraction of the picture), or None. The picture with every box goes to the page (action show).
        close: the robot is driving up to it, so a box from the top of the picture to the bottom is fine."""
        from PIL import Image
        res = self.model.predict(Image.open(io.BytesIO(jpeg)), imgsz=WIDTH, conf=0.4, verbose=False)[0]
        boxes = [(int(b.cls), float(b.conf), *[float(v) for v in b.xyxy[0]]) for b in res.boxes]
        # Looking around, a box from the top of the picture to the bottom is not trusted: tutorial 19 left such close-ups out of
        # the dataset, so the model never learned them (a white washing machine right in front can come out as a fridge).
        best = max((b for b in boxes if b[0] == self.target and b[1] >= 0.55
                    and (close or b[5] - b[3] < 0.95 * HEIGHT)), key=lambda b: b[1], default=None)
        label = "YOLO: " + (", ".join(f"{CLASSES[c]} {p:.2f}" for c, p, *_ in boxes) or "nothing")
        self.panel = {"image": base64.b64encode(draw(jpeg, boxes, CLASSES)).decode(), "label": label}
        if not best:
            return None
        _, conf, x0, y0, x1, y1 = best
        u = ((x0 + x1) / 2 - WIDTH / 2) / (WIDTH / 2)          # -1 (left edge) .. +1 (right edge)
        return conf, -math.atan(u * math.tan(HFOV / 2)), (y1 - y0) / HEIGHT

    def look(self):
        """Turn the gripper camera from left to right; the direction of the target (radians, left of the base's front)."""
        x, y, yaw = odom(self.obs)
        self.looked.append((x, y))
        self.map.add(x, y, yaw, scan(self.obs))
        self.step(pan=PANS[0], dt=1.2)
        for pan in PANS:
            seen = self.detect(self.step(pan=pan, dt=0.4, picture=True))
            if seen:
                conf, ang, _ = seen
                # Where is it? The LiDAR distance in that direction gives the point; outside the house: not ours.
                d = max(0.0, min(scan(self.obs)[round(math.degrees(pan + ang)) % 360], 8.0) - 0.4)   # the wall behind it is further
                wx, wz = self.to_world(x + d * math.cos(yaw + pan + ang), y + d * math.sin(yaw + pan + ang))
                if not (HOUSE[0] < wx < HOUSE[2] and HOUSE[1] < wz < HOUSE[3]):
                    print(f"  seen: {CLASSES[self.target]} {conf:.2f}, but outside the house: not that one", flush=True)
                    continue
                print(f"  seen: {CLASSES[self.target]} {conf:.2f}, {math.degrees(pan + ang):+.0f}° from the front", flush=True)
                return pan + ang
        return None

    def turn_to(self, bearing):
        """Turn the base in place by bearing radians (odometry), the arm back to the front meanwhile."""
        goal = wrap(odom(self.obs)[2] + bearing)
        self.step(pan=0.0)
        self.trail = []
        for _ in range(60):
            err = wrap(goal - odom(self.obs)[2])
            if abs(err) < 0.06:
                break
            self.step(w=max(-W, min(W, 2.5 * err)))
            if self.stuck():
                self.free_up()
                break
        self.step()

    def free_up(self):
        """Wedged (against a door leaf, a chair): a little back, a little forward, a turn either way, until the LiDAR
        picture changes again."""
        print("  stuck: wriggling free", flush=True)
        for v, w in ((-0.1, 0.0), (0.1, 0.0), (0.0, -0.6), (0.0, 0.6), (-0.1, -0.6), (-0.1, 0.6)):
            self.trail = []
            for _ in range(5):
                self.step(v=v, w=w)
            if not self.stuck():
                break
        self.trail = []
        self.step()

    def stuck(self):
        """Driving or turning, but the LiDAR picture has not changed in the last second: against something it does not see.
        (Not the odometry: it counts wheel turns, and wheels that slip against a table leg keep turning.)"""
        r = scan(self.obs)
        front = [r[i % 360] for i in range(-60, 61, 4)]
        self.trail = (self.trail + [front])[-5:]
        return len(self.trail) == 5 and sum(abs(a - b) for a, b in zip(front, self.trail[0])) / len(front) < 0.01

    def explore(self):
        """Plan a path on the map to the nearest place not looked at yet, and follow it for up to 3 m. False when
        there is no such place left."""
        x, y, _ = odom(self.obs)
        path = self.map.plan((x, y), self.looked + self.given_up, self.inside)
        if path is None and self.map.blocked.any():           # hemmed in by its own bumps: forget them, plan again
            self.map.blocked[:] = 0
            path = self.map.plan((x, y), self.looked + self.given_up, self.inside)
        if path is None:
            return False
        goal = self.map.xy(path[-1])
        print(f"  explore: to {goal[0]:+.1f}, {goal[1]:+.1f} ({len(path) * Map.RES:.1f} m)", flush=True)
        pts = [self.map.xy(c) for c in path]
        x0, y0 = x, y
        self.trail = []
        for _ in range(200):
            x, y, yaw = odom(self.obs)
            if math.hypot(goal[0] - x, goal[1] - y) < 0.25 or math.hypot(x - x0, y - y0) > 3.0:
                break
            # Follow the path: steer to the point on it 0.5 m beyond the nearest one (pure pursuit).
            k = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - x, pts[i][1] - y))
            tx, ty = pts[min(k + 5, len(pts) - 1)]
            err = wrap(math.atan2(ty - y, tx - x) - yaw)
            v = V if abs(err) < 0.5 else 0.0                 # facing the wrong way: turn on the spot first
            w = max(-W, min(W, 2.0 * err))
            if v and ahead(scan(self.obs)) < CLEAR:
                self.map.block(x, y, yaw)                    # not this way: the next plan goes round it
                for _ in range(4):
                    self.step(v=-0.1)
                break
            if self.stuck():
                if v:
                    self.map.block(x, y, yaw)
                self.free_up()
                break
            self.step(v=v, w=w)
        self.step()
        x, y, _ = odom(self.obs)
        before, now = math.hypot(goal[0] - x0, goal[1] - y0), math.hypot(goal[0] - x, goal[1] - y)
        if now > 0.6 and before - now < 0.5:
            self.given_up.append(goal)                       # hardly got closer: that goal is out of reach
        return True

    def avoid(self):
        """Something in the way of the approach: back off a little and turn to the freer side, then look again."""
        print("  in the way: around it", flush=True)
        self.map.block(*odom(self.obs))
        for _ in range(6):
            self.step(v=-0.1)
        r = scan(self.obs)
        self.turn_to(math.radians(55 if free(r, 60, 30) > free(r, -60, 30) else -55))

    def approach(self):
        """Drive to the target, its box kept in the middle of the picture. "arrived" when it is close in front,
        "blocked" when something else is in the way, "lost" when YOLO no longer sees it."""
        v = w = 0.0
        lost, self.trail = 0, []
        for _ in range(240):                                 # at most a minute of driving
            seen = self.detect(self.step(v=v, w=w, picture=True), close=True)
            r = scan(self.obs)
            front = ahead(r)
            if seen:
                lost = 0
                _, ang, h = seen
                # Arrived: the LiDAR beams in the direction of the box say it is close (and the box is not small: a wall
                # beside a far table is close too), or it fills the picture, or the robot can go no further and the box
                # is already big (a low table: the beams pass between its legs).
                there = free(r, math.degrees(ang), 4)
                if (there < ARRIVE and h > 0.2) or h > 0.85 or (front < CLEAR and h > 0.4):
                    self.step()
                    print(f"  arrived: {CLASSES[self.target]} {there:.2f} m in front", flush=True)
                    return "arrived"
                w = max(-W, min(W, 1.8 * ang))
                v = V if abs(ang) < 0.35 else 0.05
            else:
                lost += 1
                v, w = 0.0, 0.0
                if lost >= 6:
                    break
            if v > 0 and (front < CLEAR or self.stuck()):
                self.step()
                return "blocked"
        self.step()
        print("  lost it", flush=True)
        return "lost"

    def run(self, seconds):
        self.step(dt=0.5)
        t_end = self.obs["t"] + seconds                      # simulated seconds
        misses = 0                                           # approaches from here that did not get there
        while self.obs["t"] < t_end:
            bearing = self.look() if misses < 2 else None
            if bearing is None:
                misses = 0
                if not self.explore():
                    print("  looked everywhere", flush=True)
                    return False
                continue
            self.turn_to(bearing)
            result = self.approach()
            if result == "arrived":
                return True
            if result == "blocked":
                self.avoid()
            misses += 1
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=CLASSES, help="the object to find (default: asks)")
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--model", default=MODEL, help="the .pt file of tutorial 19")
    ap.add_argument("--seed", type=int, default=7, help="where in the house the robot starts")
    ap.add_argument("--seconds", type=float, default=900, help="give up after this long (simulated seconds)")
    a = ap.parse_args()
    target = a.target
    while target not in CLASSES:
        target = input(f"Which object? ({', '.join(CLASSES)}): ").strip().lower()

    from ultralytics import YOLO
    model = YOLO(a.model)
    api = BotopiaAPI(a.url)
    eid = api.create("tb3manipulator")["id"]
    api.reset(eid, world="house", seed=a.seed)
    # A random free spot in the house, facing a random way: the robot does not know where it is.
    r = api.step(eid, {"place": {"random": True, "area": [-5.8, -3.8, 5.8, 3.8]}}, dt=0.5)
    print(f"Looking for: {target}", flush=True)
    s = Seeker(api, eid, model, target, show=True, start=r["placed"])
    t0 = time.time()
    found = s.run(a.seconds)
    x, y, _ = odom(s.obs)
    print(f"{'Found' if found else 'Not found'}: {target} after {s.obs['t']:.0f} s in the simulator ({time.time() - t0:.0f} s here), "
          f"{len(s.looked)} times looked around, now at {x:+.1f}, {y:+.1f} m from the start")
    if found:
        # The simulator's own labels as the referee: is it really that object, in the middle of the picture?
        lab = api.labels(eid, view="gripper", width=WIDTH, classes=CLASSES)
        mid = [b for b in lab["boxes"] if b["cls"] == s.target and b["x"] < WIDTH / 2 < b["x"] + b["w"]]
        print("simulator:", f"yes, a {target} in the middle of the picture" if mid else f"no {target} in the middle of the picture")
    input("Press Enter to close the simulator...")
    api.close(eid)


if __name__ == "__main__":
    main()
