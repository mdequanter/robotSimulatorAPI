"""The Unitree Go2 follows the path of the ε track: the YOLO model of tutorial 8 finds the path in the Go2's
camera image, a PPO policy steers the dog so it stays in the middle. First in the simulator, then on the
real Go2 over WebRTC.

    camera image ──► YOLO11n-seg ──► mask of the path ──► PPO policy ──► Move(x, 0, z) on rt/api/sport/request

Put this file in the python/ folder of robotSimulatorAPI, next to botopia.py and track_yolo.py (it uses their
functions), start the simulators with  node server.mjs --key rsk_xxxxxxxx  and run:

    python go2_track_yolo.py check                 # does the YOLO model of tutorial 8 find the path in the Go2's camera?
    python go2_track_yolo.py dataset               # Go2 camera images + labels   (~1 min)
    python go2_track_yolo.py finetune              # retrain YOLO on them, from the tutorial-8 weights (~10 min CPU)
    python go2_track_yolo.py rl                    # train the steering policy on the Go2   (~20 min)
    python go2_track_yolo.py rl --resume           # and on: the dog needs about 500 000 steps (~20 min)
    python go2_track_yolo.py eval                  # a lap with YOLO in the loop (and --masks sim)
    python go2_track_yolo.py watch                 # the same lap in real time, in the visible simulator (server --show)
    python go2_track_yolo.py watch --map --route "50.8417,4.3229 50.8425,4.3240"   # with the GPS map of the page
    python go2_track_yolo.py watch --route saved_paths/gnss-laerbeekbos.json      # a route from a JSON file
    python go2_track_yolo.py real --ip 192.168.1.42   # on the real Go2 (pip install unitree_webrtc_connect)

Tutorial: https://robotics.dinno.be/tutorial-rl-go2-path.html
"""
import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_yolo  # noqa: E402
from botopia import BotopiaAPI  # noqa: E402
from track_yolo import CAM_W, TRACK_LENGTH, Car, YoloMasks, lap, make_dataset, make_vec_env, report, sim_mask  # noqa: E402

ROBOT = "go2"
# The Go2 simulator turns a drive command into velocities as the joystick of the app does, at speed
# level "normal": forward 1.0 m/s and turning 2.2 rad/s at full stick (indexGo2.php, SPEED).
# The car of tutorial 8 drives at 0.6; the dog reacts later (it needs a step or two before its body
# turns) and leaves the path in the sharp bends of the ε track at that speed, even with a controller that
# knows its exact offset. At 0.4 it keeps up. So: forward 0.4 m/s, the policy steers −1 … 1, and on the
# real dog that is Move(x = 0.4 m/s, z = steer · 2.2).
THROTTLE = track_yolo.THROTTLE = 0.4            # track_yolo's Car and RL environment use it
V_NORMAL, W_NORMAL = 1.0, 2.2
TUTORIAL3_YOLO = "runs/segment/track/weights/best.pt"


def iou(a, b):
    a, b = a > 127, b > 127
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def check(api, yolo, robots, n=40, seed=3):
    """How well YOLO finds the path, per robot: IoU of its mask with the simulator's true mask."""
    rng = np.random.default_rng(seed)
    for robot in robots:
        i, scores = api.create(robot, "track")["id"], []
        for _ in range(n):
            _, cam = api.reset(i, lane="track", start=float(rng.uniform(0, 42.56)), offset=float(rng.uniform(-0.3, 0.3)),
                               yaw=float(rng.uniform(-20, 20)), image={"view": "robot", "width": CAM_W, "quality": 0.92})
            truth = sim_mask(api.frame(i, view="mask", width=CAM_W))
            if (truth > 127).mean() < 0.05:              # turned away from the path: nothing to find
                continue
            scores.append(iou(yolo(cam), truth))
        api.close(i)
        print(f"{robot:10s} path found by YOLO: IoU mean {np.mean(scores):.2f}, worst {np.min(scores):.2f}  ({len(scores)} images with the path in view)")


class Steering:
    """The trained policy: the last two masks (as in training) → steering −1 … 1."""

    def __init__(self, path):
        from stable_baselines3 import PPO
        self.policy, self.prev = PPO.load(path, device="cpu"), None

    def __call__(self, mask, obs=None):
        self.prev = mask if self.prev is None else self.prev
        stack = np.stack([self.prev, mask], axis=2)
        self.prev = mask
        return float(self.policy.predict(stack, deterministic=True)[0][0])


class MapCar(Car):
    """The Go2 with the GPS map of the simulator page (common/gpsmap.js): ?map=1 shows the map in the API's
    page too (in the --show window), and the route goes with the first step. Every metre the dog walks on
    the track is a metre along the route; it prints the waypoints as it reaches them."""

    def __init__(self, api, robot, route=None, notes=None):
        super().__init__(api, robot, options={"map": 1})
        self.pending = {"route": route, "start": True} if route else None
        self.notes = notes or []

    def command(self, steer):
        action = super().command(steer)
        if self.pending:
            action["gps"], self.pending = self.pending, None
        return action

    def result(self, r):
        for e in r["events"]:
            if e.get("type") == "waypoint":
                print(f"waypoint {e['index']} reached: {e['lat']:.6f}, {e['lon']:.6f}")
                if e["index"] < len(self.notes) and self.notes[e["index"]]:
                    print(f"    {self.notes[e['index']]}")
            elif e.get("type") == "route_end":
                print(f"end of the route: {e['distance']:.0f} m")
        return super().result(r)


def parse_route(text):
    """ "lat,lon lat,lon ..." (or ; between the points), or a JSON file: [[lat, lon], ...], or
    {"points": [{"lat": …, "lon": … (or "lng"), "instruction": "…"}, ...]} → ([[lat, lon], ...], [instruction, ...])"""
    if not os.path.isfile(text):
        return [[float(v) for v in p.split(",")] for p in text.replace(";", " ").split()], []
    with open(text, encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("points", data.get("route", [])) if isinstance(data, dict) else data
    route, notes = [], []
    for p in points:
        if isinstance(p, dict):
            route.append([float(p["lat"]), float(p.get("lon", p.get("lng")))])
            notes.append(p.get("instruction") or p.get("name") or "")
        else:
            route.append([float(p[0]), float(p[1])])
            notes.append("")
    return route, notes


def route_length(route):
    """Metres along the route (haversine, as the page does)."""
    r, total = 6371008.8, 0.0
    for (la1, lo1), (la2, lo2) in zip(route, route[1:]):
        a = math.sin(math.radians(la2 - la1) / 2) ** 2 +             math.cos(math.radians(la1)) * math.cos(math.radians(la2)) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
        total += 2 * r * math.asin(min(1.0, math.sqrt(a)))
    return total


def run_real(args):
    """On the real Go2: frames of its camera → YOLO → mask → policy → Move, 10 times a second."""
    import asyncio
    import cv2
    from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
    from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod

    yolo, steer = YoloMasks(args.yolo, args.device), Steering(args.policy)
    # The camera gives 16:9 (1280 × 720); the simulator's camera and YOLO's training images are 4:3:
    # cut the middle 4:3 out and shrink it to 320 × 240.
    latest = {"img": None}
    x = args.speed if args.speed is not None else THROTTLE * V_NORMAL
    w = W_NORMAL * x / (THROTTLE * V_NORMAL)          # the same curve at another speed: turn rate scales with speed

    async def frames(track):
        while True:
            f = await track.recv()
            latest["img"] = f.to_ndarray(format="bgr24")

    async def main():
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
        await conn.connect()
        conn.video.switchVideoChannel(True)
        conn.video.add_track_callback(frames)
        sport = RTC_TOPIC["SPORT_MOD"]
        await conn.datachannel.pub_sub.publish_request_new(sport, {"api_id": SPORT_CMD["BalanceStand"]})
        print(f"Connected. Walking at {x:.2f} m/s for at most {args.seconds:.0f} s; Ctrl+C stops it.")
        t_end, n = time.time() + args.seconds, 0
        try:
            while time.time() < t_end:
                t0 = time.time()
                img = latest["img"]
                if img is not None:
                    h, wd = img.shape[:2]
                    cw = min(wd, h * 4 // 3)
                    img = cv2.resize(img[:, (wd - cw) // 2:(wd + cw) // 2], (CAM_W, CAM_W * 3 // 4))
                    jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                    mask = yolo(jpeg)
                    s = steer(mask) if mask.mean() > 2 else 0.0          # no path in sight: go straight (and slow down)
                    vx = x if mask.mean() > 2 else 0.0
                    await conn.datachannel.pub_sub.publish_request_new(
                        sport, {"api_id": SPORT_CMD["Move"], "parameter": {"x": round(vx, 2), "y": 0, "z": round(s * w, 2)}})
                    n += 1
                    if n % 10 == 0:
                        print(f"steer {s:+.2f}  path pixels {mask.mean() / 2.55:.0f} %")
                await asyncio.sleep(max(0.0, 0.1 - (time.time() - t0)))
        finally:
            await conn.datachannel.pub_sub.publish_request_new(sport, {"api_id": SPORT_CMD["StopMove"]})
            print("Stopped.")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Ctrl+C: the Go2 stops by itself within a second (Move is not repeated any more).")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "dataset", "finetune", "rl", "eval", "watch", "real"])
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--data", default="go2_data", help="dataset folder")
    ap.add_argument("--images", type=int, default=800, help="dataset: training images (+ 20 %% for validation)")
    ap.add_argument("--epochs", type=int, default=15, help="finetune: training epochs")
    ap.add_argument("--yolo", default=None, help="YOLO weights (default: the fine-tuned ones if they exist, else tutorial 8's)")
    ap.add_argument("--steps", type=int, default=250_000, help="rl: training steps (the dog needs about 500 000: then rl --resume)")
    ap.add_argument("--envs", type=int, default=8, help="rl: simulators at the same time")
    ap.add_argument("--policy", default="go2_path_ppo.zip")
    ap.add_argument("--resume", action="store_true", help="rl: train the saved policy further instead of starting again")
    ap.add_argument("--masks", choices=["yolo", "sim"], default="yolo", help="eval / watch: where the mask comes from")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ip", help="real: the Go2's IP address")
    ap.add_argument("--speed", type=float, help="real: forward speed in m/s (default 0.4, as in training)")
    ap.add_argument("--seconds", type=float, default=60, help="real: walk at most this long")
    ap.add_argument("--map", action="store_true", help="eval / watch: show the GPS map of the page (click waypoints there, or --route)")
    ap.add_argument("--route", help='eval / watch: GPS route "lat,lon lat,lon ..." or a JSON file with its points: the dog walks rounds until its end (implies --map)')
    ap.add_argument("--laps", type=int, default=1, help="eval / watch: rounds of the track (with --route: as many as the route needs)")
    args = ap.parse_args()
    finetuned = "runs/segment/track_go2/weights/best.pt"
    if args.yolo is None:                         # fine-tuned on the Go2 if you did, else the one of tutorial 8
        args.yolo = finetuned if os.path.exists(finetuned) else TUTORIAL3_YOLO

    if args.command == "real":
        if not args.ip:
            ap.error("real needs --ip, the Go2's address (the Unitree app shows it)")
        run_real(args)
        return

    api = BotopiaAPI(args.url)
    if args.command == "check":
        print(f"YOLO weights: {args.yolo}")
        check(api, YoloMasks(args.yolo, args.device), [ROBOT, "ackermann"])
        return
    if args.command == "dataset":
        make_dataset(api, args.data, args.images, args.images // 5, [ROBOT])
        print(f"Dataset ready: {args.data}/track.yaml")
        return
    if args.command == "finetune":
        from ultralytics import YOLO
        model = YOLO(args.yolo)                       # start from what it learned on the other robots' cameras
        model.train(data=f"{args.data}/track.yaml", imgsz=CAM_W, epochs=args.epochs, batch=16, device=args.device,
                    project=os.path.abspath("runs/segment"), name="track_go2", exist_ok=True, workers=0, fliplr=0.5)
        return
    if args.command == "rl":
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        venv = VecFrameStack(VecMonitor(make_vec_env(args.envs, ROBOT, args.url)), 2, channels_order="last")
        if args.resume:
            model = PPO.load(args.policy, env=venv, device=args.device)
        else:
            model = PPO("CnnPolicy", venv, n_steps=256, batch_size=256, n_epochs=4, learning_rate=lambda f: 2e-4 * f,
                        target_kl=0.03, gamma=0.95, ent_coef=0.003, verbose=1, seed=1, device=args.device)
        t0 = time.time()
        model.learn(total_timesteps=args.steps, reset_num_timesteps=not args.resume)
        model.save(args.policy)
        print(f"Saved {args.policy} after {(time.time() - t0) / 60:.1f} min")
        venv.close()
        return

    yolo = YoloMasks(args.yolo, args.device) if args.masks == "yolo" else None
    laps = args.laps
    if args.map or args.route:
        route, notes = parse_route(args.route) if args.route else (None, None)
        if route:
            laps = max(laps, math.ceil(route_length(route) / TRACK_LENGTH))
            print(f"Route of {len(route)} waypoints, {route_length(route):.0f} m: {laps} rounds of the track")
        car = MapCar(api, ROBOT, route, notes)
    else:
        car = Car(api, ROBOT)
    max_time = laps * TRACK_LENGTH / (THROTTLE * V_NORMAL) * 1.5      # 1.5 × the time it needs at its speed
    r = lap(car, Steering(args.policy), masks=args.masks, yolo=yolo, realtime=args.command == "watch",
            laps=laps, max_time=max_time)
    report(f"Go2, PPO steering, masks from {args.masks.upper()}", r)
    car.close()


if __name__ == "__main__":
    main()
