"""Watch what a simulated robot sees, live, and drive it from the window.

    node server.mjs                       (in another terminal; port 8780)
    python python/camera_view.py          (TurtleBot 4 on the start square)
    python python/camera_view.py --robot skid --lane track --auto

The window shows the robot's own camera (left) and the 3D view of the simulator (right). On the
camera picture the light pixels of the road are tinted green, with a cross at their centre: that is
what the --auto driver steers by.

Keys (click the window first):  W / S forward / back,  A / D turn,  Q / E sideways (robots that can),
space stop,  F auto-drive on / off (follow the road),  R reset,  P save a picture,  Esc quit.

Needs: numpy, opencv-python (pip install numpy opencv-python).
"""
import argparse, os, sys, time
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("This script needs OpenCV: pip install opencv-python")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))     # botopia.py lies next to this script
from botopia import BotopiaAPI


def decode(jpg):
    """JPEG bytes → OpenCV image (BGR)."""
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def find_road(img):
    """Light pixels in the lower part of the camera picture: the road. Returns the mask (whole
    picture size) and the horizontal position of its centre (−1 left … +1 right), or None."""
    h, w = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), np.uint8)
    lower = grey[int(h * 0.58):, :] > 115                    # the floor just in front of the robot
    mask[int(h * 0.58):, :] = lower * 255
    if lower.sum() < h * w * 0.004:
        return mask, None
    cols = np.nonzero(lower)[1]
    return mask, (cols.mean() - w / 2) / (w / 2)


def main():
    ap = argparse.ArgumentParser(description="Show the robot's camera live and drive it.")
    ap.add_argument("--url", default="http://localhost:8780", help="Node API (server.mjs)")
    ap.add_argument("--robot", default="turtlebot4", help="robot id, see api.html (e.g. skid, go2, g1, jetbot)")
    ap.add_argument("--lane", default="free", help="free, track, stairs, corridor, … (see api.html)")
    ap.add_argument("--width", type=int, default=640, help="width of the camera picture in pixels")
    ap.add_argument("--auto", action="store_true", help="start with auto-drive (follow the road) on")
    ap.add_argument("--dt", type=float, default=0.05, help="simulated seconds per step")
    a = ap.parse_args()

    api = BotopiaAPI(a.url)
    try:
        env = api.create(a.robot, lane=a.lane)
    except Exception as e:
        sys.exit(f"Could not reach the simulator API at {a.url}: {e}\nStart it with: node server.mjs")
    eid, axes = env["id"], env["info"]["drive"]
    cam = {"view": "robot", "width": a.width, "quality": 0.85}
    obs, jpg = api.reset(eid, lane=a.lane, image=cam)
    print(f"{env['info']['name']} ({a.robot}) · drive axes: {', '.join(axes) or 'none'}")
    print(__doc__.split("Keys")[1].split("Needs")[0].strip())

    auto, drive, shot_n = a.auto, {"fwd": 0.0, "turn": 0.0, "strafe": 0.0}, 0
    keys = {ord("w"): ("fwd", 0.6), ord("s"): ("fwd", -0.5), ord("a"): ("turn", 0.7), ord("d"): ("turn", -0.7),
            ord("q"): ("strafe", 0.6), ord("e"): ("strafe", -0.6)}
    win = f"Botopia · {a.robot}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_last, fps = time.time(), 0.0
    try:
        while True:
            img = decode(jpg)
            mask, centre = find_road(img)

            # Auto-drive: steer towards the centre of the road; no road in view → turn to find it.
            if auto:
                if centre is None:
                    drive = {"fwd": 0.0, "turn": 0.6, "strafe": 0.0}
                else:
                    drive = {"fwd": 0.5, "turn": float(np.clip(-1.5 * centre, -1, 1)), "strafe": 0.0}

            action = {"drive": {k: v for k, v in drive.items() if k in axes}} if axes else {}
            r = api.step(eid, action, dt=a.dt, image=cam)
            jpg, obs = r["image"], r["obs"]
            for ev in r["events"]:
                if ev["type"] not in ("blocked", "collision", "stall"):
                    print(f"{ev['t']:7.2f}s  event {ev['type']}: {ev}")

            # Picture: camera with the road tinted green, its centre marked; the 3D view beside it.
            view = img.copy()
            view[mask > 0] = (0.55 * view[mask > 0] + 0.45 * np.array([60, 220, 60])).astype(np.uint8)
            h, w = view.shape[:2]
            if centre is not None:
                cx = int((centre + 1) * w / 2)
                cv2.drawMarker(view, (cx, int(h * 0.8)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.line(view, (w // 2, int(h * 0.58)), (w // 2, h), (255, 255, 255), 1)
            scene = decode(api.frame(eid, view="scene", quality=0.7))
            scene = cv2.resize(scene, (int(scene.shape[1] * h / scene.shape[0]), h))
            frame = np.hstack([view, scene])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_last)
            t_last = now
            p = obs["pose"]
            lines = [f"{'AUTO' if auto else 'keys'}  fwd {drive['fwd']:+.2f} turn {drive['turn']:+.2f}",
                     f"x {p['x']:+.2f} z {p['z']:+.2f} yaw {np.degrees(p['yaw']):+.0f}  {fps:4.1f} fps",
                     f"along the lane {obs['progress']['distance']:.2f} m" + (f"  stuck: {obs['stuck']}" if obs["stuck"] else "")]
            for i, t in enumerate(lines):
                cv2.putText(frame, t, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, t, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:     # Esc or window closed
                break
            if k in keys:
                auto = False
                ax, v = keys[k]
                drive[ax] = 0.0 if drive[ax] == v else v                            # press again: that axis stops
            elif k == ord(" "):
                auto, drive = False, {"fwd": 0.0, "turn": 0.0, "strafe": 0.0}
            elif k == ord("f"):
                auto = not auto
                if not auto:
                    drive = {"fwd": 0.0, "turn": 0.0, "strafe": 0.0}
            elif k == ord("r"):
                obs, jpg = api.reset(eid, lane=a.lane, image=cam)
            elif k == ord("p"):
                shot_n += 1
                name = f"robot_camera_{shot_n:03d}.jpg"
                cv2.imwrite(name, img)
                print("saved", os.path.abspath(name))
    finally:
        cv2.destroyAllWindows()
        api.close(eid)


if __name__ == "__main__":
    main()
