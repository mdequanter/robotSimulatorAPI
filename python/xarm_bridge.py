"""Bridge between the xArm simulator page in your browser and the real UFACTORY xArm 6 or xArm 7.

A web page can't talk to the arm's control box itself (the SDK protocol is plain TCP on port 502), so
this program does, on a PC in the robot's network:

    browser (indexXArm7.php)  <-- WebSocket ws://localhost:8770 -->  xarm_bridge.py  <-- xArm SDK -->  xArm 6
                                                                         |
                          a RealSense D435 on the wrist + a USB webcam that sees the whole arm (optional)

What the page can do through it:
  * digital twin: the real arm's joints are shown as a blue see-through arm in the simulator;
  * follow (only when you tick it on the page): the real arm follows the simulated one, speed-limited;
  * the cameras: the wrist camera and the external camera (the whole arm) appear on the page.
    With the page's "RealSense depth view" ticked, it sends the wrist D435's depth instead (in colour).

    pip install xarm-python-sdk websockets pyrealsense2 pillow numpy opencv-python
    python xarm_bridge.py --ip 10.2.172.20            # your arm's IP (see UFACTORY Studio); 6 or 7 axes: asked
    python xarm_bridge.py --fake --model 7            # no robot: a pretend xArm 7, to try the page
    python xarm_bridge.py --list-cameras              # the RealSense serial numbers, then:
    python xarm_bridge.py --ip 10.2.172.20 --wrist-camera 123456789012 --external-camera 1
                                                      # --external-camera: the webcam's OpenCV index (0, 1, ...)

Then open the simulator page of the same model (indexXArm7.php for an xArm 7), "Real robot" → Connect.

Safety: following is off until you tick it on the page; it is limited to --max-speed (°/s); the
fingertips are never sent below --min-z above the table; the arm stops when you press STOP or Esc,
when the page closes or loses the connection, and when no target came for 1 s. Set the arm's own
collision sensitivity and safety boundary in UFACTORY Studio, and keep the emergency stop in reach.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xarm_robot  # noqa: E402

WATCHDOG = 1.0          # s without a target while following → stop


def depth_jpeg(camera, quality=70, width=480):
    """The camera's depth picture in colour (near blue … far red, black: no depth), as a JPEG."""
    from io import BytesIO
    from PIL import Image
    mm = camera.depth()
    im = Image.fromarray(xarm_robot.depth_colormap(mm))
    im = im.resize((width, round(im.height * width / im.width)))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class Bridge:
    def __init__(self, robot, cameras, max_speed, cam_fps):
        self.robot, self.cameras = robot, cameras            # {"wrist": RealSense, "external": UsbCamera}
        self.max_speed, self.cam_fps = max_speed, cam_fps
        self.clients, self.owner = set(), None          # owner: the page that makes the arm follow
        self.follow, self.last_target, self.msg = False, 0.0, ""
        self.joints, self.grip = None, None
        self.depth_view = False                         # the page shows the depth pictures (its depth view)

    async def broadcast(self, message):
        data = json.dumps(message)
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except Exception:
                self.clients.discard(ws)

    def stop_follow(self, why):
        if self.follow:
            print(f"Stop following: {why}")
        self.follow, self.owner, self.msg = False, None, why
        try:
            self.robot.stop()
        except Exception as e:
            print(f"stop failed: {e}")

    async def handle(self, ws, m):
        cmd = m.get("cmd")
        if cmd == "depth":                              # the page's "RealSense depth view": send depth, not colour
            self.depth_view = bool(m.get("on"))
        elif cmd == "enable":
            await asyncio.to_thread(self.robot.enable)
            self.msg = "motors enabled"
        elif cmd == "stop":
            self.stop_follow("stopped from the page")
        elif cmd == "follow":
            if m.get("on") and len(m.get("joints") or []) != self.robot.model:
                self.follow, self.msg = False, (f"this is an xArm {self.robot.model}: open the xArm {self.robot.model} "
                                                f"simulator (indexXArm{self.robot.model}.php) to follow it")
                return
            if m.get("on"):
                self.follow, self.owner, self.last_target = True, ws, time.time()
                self.msg = "following the simulator"
                print("Following the simulator")
                if m.get("joints"):
                    await self.target(m["joints"], m.get("gripper"))
            else:
                self.follow, self.owner, self.msg = False, None, "not following"
                print("Stopped following (unticked on the page)")
        elif cmd == "target" and self.follow and ws is self.owner:
            self.last_target = time.time()
            await self.target(m["joints"], m.get("gripper"))

    async def target(self, joints, gripper):
        try:
            await asyncio.to_thread(self.robot.follow, joints, gripper, self.max_speed)
        except xarm_robot.UnsafeMove as e:
            self.stop_follow(f"refused: {e}")
        except Exception as e:
            self.stop_follow(f"robot error: {e}")

    async def client(self, ws):
        self.clients.add(ws)
        print(f"Page connected ({len(self.clients)} open)")
        from websockets.exceptions import ConnectionClosed
        try:
            async for raw in ws:
                try:
                    await self.handle(ws, json.loads(raw))
                except (ValueError, KeyError, TypeError) as e:
                    print(f"bad message: {e}")
        except ConnectionClosed:
            pass                                         # the page went away without saying goodbye
        finally:
            self.clients.discard(ws)
            if ws is self.owner:
                self.stop_follow("the page closed")
            print(f"Page disconnected ({len(self.clients)} open)")

    async def state_loop(self):
        """10 times a second: the real arm's state to every page; the watchdog."""
        n = 0
        while True:
            if self.follow and time.time() - self.last_target > WATCHDOG:
                self.stop_follow("no target from the page for 1 s")
            try:
                self.joints = await asyncio.to_thread(self.robot.joints)
                if n % 2 == 0:                               # the gripper is a slower request
                    self.grip = await asyncio.to_thread(self.robot.gripper)
                st = self.robot.status()
            except Exception as e:
                st = {"robot": "xArm 6", "enabled": False, "error": -1, "warn": 0}
                self.msg = f"can't read the arm: {e}"
            if self.clients:
                await self.broadcast({"type": "state", **st, "joints": self.joints, "gripper": self.grip,
                                      "follow": self.follow, "msg": self.msg, "cameras": list(self.cameras)})
            n += 1
            await asyncio.sleep(0.1)

    async def camera_loop(self, name, camera):
        while True:
            if self.clients:
                try:
                    if self.depth_view and name == "wrist" and getattr(camera, "depth", None):   # only the wrist D435 has depth
                        jpg = await asyncio.to_thread(depth_jpeg, camera, 70, 480)
                    else:
                        jpg = await asyncio.to_thread(camera.jpeg, 70, 480)
                    await self.broadcast({"type": "camera", "name": name, "jpeg": base64.b64encode(jpg).decode()})
                except Exception as e:
                    print(f"{name} camera: {e}")
                    await asyncio.sleep(1)
            await asyncio.sleep(1 / self.cam_fps)


async def main_async(args):
    import websockets
    if args.list_cameras:
        xarm_robot.list_cameras()
        return
    if args.fake:
        robot = xarm_robot.FakeXArm(args.model)
    else:
        print(f"Connecting to the xArm at {args.ip} …")
        robot = xarm_robot.RealXArm(args.ip, camera=False, speed_limit=args.max_speed)
        print(f"Connected: an xArm {robot.model}")
    robot.min_z = args.min_z
    # Cameras: the wrist RealSense D435 (colour + depth) and, if given, a USB webcam for the external view.
    cameras = {}
    if not args.no_camera:
        try:
            cameras["wrist"] = xarm_robot.RealSense(args.wrist_camera)
            print("RealSense wrist camera: on" + (f" (serial {args.wrist_camera})" if args.wrist_camera else ""))
        except Exception as e:
            print(f"RealSense wrist camera not available ({e}); go on without it")
        if args.external_camera is not None:
            try:
                cameras["external"] = xarm_robot.UsbCamera(args.external_camera)
                print(f"External camera: on (webcam {args.external_camera})")
            except Exception as e:
                print(f"External camera not available ({e}); go on without it")
    if args.fake and not args.no_camera and not (args.wrist_camera or args.external_camera is not None):
        cameras = {name: xarm_robot.FakeCamera(name) for name in ("wrist", "external")}   # pretend images
    bridge = Bridge(robot, cameras, args.max_speed, args.camera_fps)
    async with websockets.serve(bridge.client, args.host, args.port, max_size=2 ** 22):
        print(f"Bridge ready on ws://{args.host}:{args.port}  ->  open the xArm {robot.model} simulator, 'Real robot' -> Connect")
        print(f"Following is off until you tick it on the page; at most {args.max_speed:g} deg/s.  Ctrl+C to quit.")
        await asyncio.gather(bridge.state_loop(), *(bridge.camera_loop(n, c) for n, c in cameras.items()))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", help="the arm's IP address")
    ap.add_argument("--fake", action="store_true", help="no robot: a pretend arm (to try the page)")
    ap.add_argument("--model", type=int, default=7, choices=[6, 7], help="--fake: pretend to be an xArm 6 or 7")
    ap.add_argument("--host", default="127.0.0.1", help="where the page connects (keep 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--max-speed", type=float, default=30, help="°/s, at most, while following")
    ap.add_argument("--min-z", type=float, default=0.01, help="fingertips at least this high above the table (m)")
    ap.add_argument("--no-camera", action="store_true", help="no cameras")
    ap.add_argument("--wrist-camera", help="serial number of the RealSense on the wrist (with several RealSenses)")
    ap.add_argument("--external-camera", type=int, help="OpenCV index (0, 1, ...) of the USB webcam that sees the whole arm")
    ap.add_argument("--list-cameras", action="store_true", help="show the RealSense cameras and their serial numbers")
    ap.add_argument("--camera-fps", type=float, default=10)
    args = ap.parse_args()
    if not args.ip and not args.fake and not args.list_cameras:
        ap.error("give the arm's --ip, or --fake to try without a robot")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Bye")


if __name__ == "__main__":
    main()
