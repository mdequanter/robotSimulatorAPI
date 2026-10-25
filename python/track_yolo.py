"""Tutorial 3: a YOLO segmentation model finds the path, reinforcement learning keeps the robot in the
middle of it.

    camera image ──► YOLO11n-seg (Ultralytics) ──► mask of the path ──► PPO policy ──► steering

1. dataset   camera images of the ε track plus their labels, made by the simulator itself
2. yolo      train YOLO11n-seg to find the path (one class: "path")
3. rl        train a PPO policy that steers from a path mask, to stay in the middle
4. eval      drive laps with YOLO in the loop and measure how well it stays in the middle

Start the simulator API first: node server.mjs
pip install ultralytics stable-baselines3 gymnasium

    python track_yolo.py dataset
    python track_yolo.py yolo
    python track_yolo.py rl
    python track_yolo.py baseline
    python track_yolo.py eval                 # masks from YOLO (and --masks sim for comparison)
    python track_yolo.py watch --gif lap.gif

Tutorial: https://robotics.dinno.be/tutorial-yolo-track.html
"""
import argparse
import io
import os
import sys
import time

import numpy as np
from gymnasium import spaces
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from botopia import BotopiaAPI  # noqa: E402

TRACK_LENGTH = 42.56     # metres round the ε track
HALF_BAND = 0.35         # the path is 70 cm wide
CAM_W = 320              # camera images for YOLO: 320 × 240
MW, MH = 64, 48          # path mask the policy sees
THROTTLE = 0.6           # fixed speed: this tutorial is about steering


# ---------- 1. Dataset ----------
def mask_to_polygons(mask):
    """Binary mask (H, W) → YOLO segmentation label lines: "0 x1 y1 x2 y2 …" (0 … 1)."""
    import cv2
    h, w = mask.shape
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if cv2.contourArea(c) < 25:                                # tiny specks far away
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
    return lines


def make_dataset(api, folder, n_train, n_val, robots, seed=1):
    """Camera images from random spots on the track (also beside the centre line and at an angle, so
    YOLO sees the path from every view a driving robot gets), labels from the simulator's mask view."""
    rng = np.random.default_rng(seed)
    envs = {r: api.create(r, "track")["id"] for r in robots}
    for split, n in (("train", n_train), ("val", n_val)):
        os.makedirs(f"{folder}/images/{split}", exist_ok=True)
        os.makedirs(f"{folder}/labels/{split}", exist_ok=True)
        for k in range(n):
            robot = robots[k % len(robots)]
            i = envs[robot]
            _, cam = api.reset(i, lane="track", start=float(rng.uniform(0, TRACK_LENGTH)),
                               offset=float(rng.uniform(-0.55, 0.55)), yaw=float(np.clip(rng.normal(0, 18), -50, 50)),
                               image={"view": "robot", "width": CAM_W, "quality": 0.92})
            mask = np.asarray(Image.open(io.BytesIO(api.frame(i, view="mask", width=CAM_W))).convert("L"))
            name = f"{split}_{k:05d}_{robot}"
            open(f"{folder}/images/{split}/{name}.jpg", "wb").write(cam)
            open(f"{folder}/labels/{split}/{name}.txt", "w").write("\n".join(mask_to_polygons(mask)))
        print(f"{split}: {n} images")
    with open(f"{folder}/track.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(folder)}\ntrain: images/train\nval: images/val\nnames:\n  0: path\n")
    for i in envs.values():
        api.close(i)


# ---------- 2. YOLO → a path mask for the policy ----------
class YoloMasks:
    """Runs the trained YOLO model on a camera image and returns the path as a (48, 64) 0/255 mask."""

    def __init__(self, weights, device="cpu"):
        from ultralytics import YOLO
        self.model, self.device = YOLO(weights), device

    def __call__(self, jpeg):
        img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))[:, :, ::-1]     # RGB → BGR (as OpenCV)
        r = self.model.predict(img, imgsz=CAM_W, conf=0.25, device=self.device, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            return np.zeros((MH, MW), np.uint8)
        full = (r.masks.data.cpu().numpy().max(axis=0) > 0.5).astype(np.uint8) * 255   # all instances together
        return np.asarray(Image.fromarray(full).resize((MW, MH), Image.BILINEAR))


def sim_mask(png):
    """The simulator's own mask (the ground truth) at the policy's size."""
    return np.asarray(Image.open(io.BytesIO(png)).convert("L").resize((MW, MH), Image.BILINEAR))


def noisy(mask, rng):
    """Training masks get YOLO-like flaws, so the policy doesn't rely on a perfect mask: a ragged
    edge (shrink / grow by a pixel), some flipped pixels and now and then a missing chunk."""
    m = mask > 127
    if rng.random() < 0.5:
        shift = np.roll(m, rng.choice([-1, 1]), axis=rng.integers(0, 2))
        m = m & shift if rng.random() < 0.5 else m | shift
    m ^= rng.random(m.shape) < 0.02
    if rng.random() < 0.15:
        y, x = rng.integers(0, MH - 8), rng.integers(0, MW - 12)
        m[y:y + 8, x:x + 12] = False
    return m.astype(np.uint8) * 255


# ---------- 3. RL: steer from the mask, stay in the middle ----------
class Car:
    """One simulator: steering → drive command, step result → reward."""

    def __init__(self, api, robot, dt=0.1, reward="squared"):
        self.api, self.dt, self.reward_kind = api, dt, reward
        self.id = api.create(robot, "track")["id"]

    def command(self, steer):
        return {"drive": {"fwd": THROTTLE, "turn": float(np.clip(steer, -1, 1))}}

    def reset(self, start, offset=0.0, yaw=0.0, view="mask"):
        obs, img = self.api.reset(self.id, lane="track", start=start, offset=offset, yaw=yaw,
                                  image={"view": view, "width": MW if view == "mask" else CAM_W})
        self.s, self.driven, self.steps = obs["progress"]["distance"], 0.0, 0
        return img, obs

    def result(self, r):
        """reward: 1 per step in the exact middle, 0 at the edge of the path, −10 and the end of the
        attempt when it leaves the path. "squared" (1 − e²) hardly minds small deviations, "linear"
        (1 − |e|) pulls harder towards the exact middle."""
        obs = r["obs"]
        p = obs["progress"]
        ds = (p["distance"] - self.s + TRACK_LENGTH / 2) % TRACK_LENGTH - TRACK_LENGTH / 2
        self.s, self.driven, self.steps = p["distance"], self.driven + ds, self.steps + 1
        off = p["offset"] is None or abs(p["offset"]) > HALF_BAND
        e = 1.0 if off else p["offset"] / HALF_BAND
        reward = -10.0 if off else 1.0 - (e * e if self.reward_kind == "squared" else abs(e))
        return reward, off, obs

    def close(self):
        self.api.close(self.id)


def make_vec_env(n, robot, url, max_steps=300, reward="squared"):
    """n simulators in one batch call; observation = the simulator's path mask with YOLO-like flaws."""
    from stable_baselines3.common.vec_env import VecEnv

    class MaskVecEnv(VecEnv):
        def __init__(self):
            self.api = BotopiaAPI(url)
            self.cars = [Car(self.api, robot, reward=reward) for _ in range(n)]
            self.rng = np.random.default_rng(1)
            super().__init__(n, spaces.Box(0, 255, shape=(MH, MW, 1), dtype=np.uint8),
                             spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))

        def _obs(self, png):
            return noisy(sim_mask(png), self.rng)[:, :, None]

        def _start(self, c):
            # Start anywhere, also off-centre and at an angle: it must learn to come back to the middle.
            png, _ = c.reset(start=float(self.rng.uniform(0, TRACK_LENGTH)), offset=float(self.rng.uniform(-0.2, 0.2)),
                             yaw=float(self.rng.uniform(-20, 20)))
            return self._obs(png)

        def reset(self):
            return np.stack([self._start(c) for c in self.cars])

        def step_async(self, actions):
            self.actions = actions

        def step_wait(self):
            results = self.api.step_many([{"id": c.id, "action": c.command(a[0]), "dt": c.dt,
                                           "image": {"view": "mask", "width": MW}} for c, a in zip(self.cars, self.actions)])
            obs, rewards, dones, infos = [], [], [], []
            for c, r in zip(self.cars, results):
                reward, off, o = c.result(r)
                ob = self._obs(r["image"])
                truncated = c.steps >= max_steps and not off
                info = {}
                if off or truncated:
                    info = {"terminal_observation": ob, "TimeLimit.truncated": truncated}
                    ob = self._start(c)
                obs.append(ob); rewards.append(reward); dones.append(off or truncated); infos.append(info)
            return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones), infos

        def close(self):
            for c in self.cars:
                c.close()

        def get_attr(self, name, indices=None):
            return [getattr(self.cars[i], name, None) for i in self._get_indices(indices)]

        def set_attr(self, name, value, indices=None):
            for i in self._get_indices(indices):
                setattr(self.cars[i], name, value)

        def env_method(self, name, *args, indices=None, **kw):
            return [getattr(self.cars[i], name)(*args, **kw) for i in self._get_indices(indices)]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return MaskVecEnv()


# ---------- 4. A lap ----------
def lap(car, steer_fn, masks="sim", yolo=None, max_time=150, frames=None, realtime=False):
    """One lap from the start line. masks "sim": the policy gets the simulator's mask; "yolo": the
    camera image goes through YOLO first. Reports how far from the middle it drove."""
    view = "mask" if masks == "sim" else "robot"
    img, obs = car.reset(start=0.0, view=view)
    offs, n_off, was_off, t, ms = [], 0, False, 0.0, []
    while t < max_time:
        t0 = time.time()
        mask = sim_mask(img) if masks == "sim" else yolo(img)
        ms.append(time.time() - t0)
        r = car.api.step(car.id, car.command(steer_fn(mask, obs)), dt=car.dt,
                         image={"view": view, "width": MW if view == "mask" else CAM_W})
        img = r["image"]
        _, off, obs = car.result(r)
        t = obs["t"]
        o = obs["progress"]["offset"]
        offs.append(HALF_BAND if o is None else abs(o))                 # (0.0 is a valid offset)
        n_off += off and not was_off
        was_off = off
        if frames is not None:
            frames.append((car.api.frame(car.id, 0.85), mask))
        if realtime:
            time.sleep(car.dt)
        if car.driven >= TRACK_LENGTH:
            break
    return {"time": t if car.driven >= TRACK_LENGTH else None, "mean": float(np.mean(offs)), "max": float(np.max(offs)),
            "off": n_off, "mask_ms": 1000 * float(np.mean(ms))}


def report(name, r):
    done = f"lap {r['time']:.1f} s" if r["time"] else "no full lap"
    print(f"{name}: {done}, from the middle: mean {r['mean'] * 100:.1f} cm, max {r['max'] * 100:.1f} cm, "
          f"left the path {r['off']} times, mask {r['mask_ms']:.1f} ms per image")


def main():
    # A Windows console (cp1252) can't print ε or …: show "?" instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["dataset", "yolo", "rl", "baseline", "eval", "watch"])
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--robot", default="ackermann")
    ap.add_argument("--data", default="track_data", help="dataset folder")
    ap.add_argument("--images", type=int, default=1500, help="dataset: training images (+ 20 %% for validation)")
    ap.add_argument("--epochs", type=int, default=30, help="yolo: training epochs")
    ap.add_argument("--yolo", default="runs/segment/track/weights/best.pt", help="trained YOLO weights")
    ap.add_argument("--steps", type=int, default=150_000, help="rl: training steps")
    ap.add_argument("--policy", default="track_middle_ppo.zip")
    ap.add_argument("--reward", choices=["squared", "linear"], default="squared", help="rl: how off-centre driving is punished")
    ap.add_argument("--masks", choices=["yolo", "sim"], default="yolo", help="eval / watch: where the mask comes from")
    ap.add_argument("--device", default="cpu", help="cpu, cuda (0) or auto")
    ap.add_argument("--gif", help="watch: save the lap as an animated GIF")
    args = ap.parse_args()
    api = BotopiaAPI(args.url)

    if args.command == "dataset":
        # Mostly the Ackermann's camera, plus two robots with their camera at another height.
        make_dataset(api, args.data, args.images, args.images // 5, [args.robot, args.robot, "skid", "4wd"])
        print(f"Dataset ready: {args.data}/track.yaml")
        return

    if args.command == "yolo":
        from ultralytics import YOLO
        model = YOLO("yolo11n-seg.pt")             # pretrained on COCO: it already knows shapes and edges
        model.train(data=f"{args.data}/track.yaml", imgsz=CAM_W, epochs=args.epochs, batch=16,
                    # an absolute folder: otherwise Ultralytics puts it under its own settings' runs_dir
                    device=args.device if args.device != "auto" else None, project=os.path.abspath("runs/segment"), name="track",
                    exist_ok=True, workers=0, plots=True, fliplr=0.5, mosaic=1.0)
        return

    if args.command == "baseline":
        # Knows its exact distance to the centre line (the policy doesn't): steers against it, with the
        # gains that kept it closest to the middle at this speed (we tried kp 3…15, kd 0.5…3).
        car, prev = Car(api, args.robot), [0.0]

        def pd(mask, obs):
            e = obs["progress"]["offset"] or 0.0
            steer = -10.0 * e - 0.5 * (e - prev[0]) / car.dt
            prev[0] = e
            return steer
        report("Line follower (knows its offset)", lap(car, pd, masks="sim"))
        car.close()
        return

    from stable_baselines3 import PPO
    if args.command == "rl":
        import torch
        from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
        torch.set_num_threads(min(4, os.cpu_count() or 4))     # leave the CPU cores to the simulators
        venv = VecFrameStack(VecMonitor(make_vec_env(8, args.robot, args.url, reward=args.reward)), 2, channels_order="last")
        model = PPO("CnnPolicy", venv, n_steps=256, batch_size=256, n_epochs=4, learning_rate=lambda f: 2e-4 * f,
                    target_kl=0.03, gamma=0.95, ent_coef=0.003, verbose=1, seed=1, device=args.device)
        t0 = time.time()
        model.learn(total_timesteps=args.steps)
        model.save(args.policy)
        print(f"Saved {args.policy} after {(time.time() - t0) / 60:.1f} min")
        venv.close()
        return

    policy = PPO.load(args.policy, device="cpu")
    yolo = YoloMasks(args.yolo, args.device if args.device != "auto" else "cpu") if args.masks == "yolo" else None
    car, prev = Car(api, args.robot), [None]

    def steer(mask, obs):                          # the last 2 masks, as in training
        prev[0] = mask if prev[0] is None else prev[0]
        stack = np.stack([prev[0], mask], axis=2)
        prev[0] = mask
        return float(policy.predict(stack, deterministic=True)[0][0])

    if args.command == "eval":
        report(f"PPO steering, masks from {args.masks.upper()}", lap(car, steer, masks=args.masks, yolo=yolo))
    else:
        frames = [] if args.gif else None
        report(f"PPO steering, masks from {args.masks.upper()}",
               lap(car, steer, masks=args.masks, yolo=yolo, frames=frames, realtime=not args.gif))
        if args.gif:
            pics = []
            for scene, mask in frames[::2]:
                im = Image.open(io.BytesIO(scene)).convert("RGB")
                im = im.resize((480, round(im.height * 480 / im.width)))
                m = Image.fromarray(mask).resize((MW * 2, MH * 2), Image.NEAREST).convert("RGB")
                im.paste(m, (im.width - m.width - 6, 6))                      # the mask the policy steers by
                pics.append(im.convert("P", palette=Image.ADAPTIVE))
            pics[0].save(args.gif, save_all=True, append_images=pics[1:], duration=int(car.dt * 2000), loop=0)
            print(f"Saved {args.gif} ({len(pics)} frames)")
    car.close()


if __name__ == "__main__":
    main()
