"""Reinforcement learning tutorial 7: drive the ε track as fast as possible, seeing only the camera.

The robot (Ackermann car by default) gets nothing but the image of its own front camera: no
position, no map, no speed. It learns to steer and to choose its speed with PPO and a convolutional
neural network (stable-baselines3 CnnPolicy).

Start the simulator API first: node server.mjs
pip install stable-baselines3 gymnasium pillow

    python track_vision_rl.py baseline              # a driver that knows where the centre line is
    python track_vision_rl.py train                 # PPO from camera images, saves track_vision_ppo.zip
    python track_vision_rl.py eval                  # lap time of the trained model
    python track_vision_rl.py watch --gif lap.gif   # one lap: 3D view + what the network sees

Tutorial: https://robotics.dinno.be/tutorial-rl-vision.html
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
HALF_BAND = 0.35         # the painted band is 70 cm wide
W, H = 64, 48            # camera image the network sees (greyscale)
IMAGE = {"view": "robot", "width": W, "quality": 0.9}


def to_gray(jpeg):
    """JPEG bytes from the API → (48, 64, 1) uint8 array."""
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), dtype=np.uint8)[:, :, None]


class Driver:
    """One simulator on the track: turns an action into a drive command and a step result into a
    reward. Shared by the training environments and the test laps."""

    def __init__(self, api, robot, dt=0.1, time_penalty=0.05, min_throttle=0.25, speed=1.0, vmax=None, camh=None, camtilt=None):
        self.api, self.dt, self.time_penalty, self.min_throttle = api, dt, time_penalty, min_throttle
        self.speed = speed                         # scales the throttle the network chooses (eval / watch)
        # vmax: the robot's top speed in m/s (a faster motor / gearing in the simulator; Ackermann: 0.9)
        # camh / camtilt: the front camera's height (m) and downward tilt (°) (Ackermann; default low, 8°)
        options = {k: v for k, v in (("vmax", vmax), ("camh", camh), ("camtilt", camtilt)) if v}
        self.id = api.create(robot, "track", options=options)["id"]

    def command(self, action):
        """action = [throttle, steering], both −1 … 1 (what the network outputs)."""
        throttle = float(np.clip(action[0], -1, 1)) * 0.5 + 0.5          # 0 … 1
        fwd = self.min_throttle + (1 - self.min_throttle) * throttle     # never stands still
        return {"drive": {"fwd": fwd * self.speed, "turn": float(np.clip(action[1], -1, 1))}}

    def reset(self, start):
        obs, jpeg = self.api.reset(self.id, lane="track", start=start, image=IMAGE)
        self.s, self.driven, self.steps = obs["progress"]["distance"], 0.0, 0
        return to_gray(jpeg), obs

    def result(self, r):
        """(image, reward, off_the_band, obs) after one step."""
        obs = r["obs"]
        p = obs["progress"]
        ds = p["distance"] - self.s
        if ds < -TRACK_LENGTH / 2:                                     # crossed the start line
            ds += TRACK_LENGTH
        elif ds > TRACK_LENGTH / 2:
            ds -= TRACK_LENGTH
        self.s, self.driven, self.steps = p["distance"], self.driven + ds, self.steps + 1
        off = p["offset"] is None or abs(p["offset"]) > HALF_BAND
        reward = 10 * ds - self.time_penalty - (10.0 if off else 0.0)
        return to_gray(r["image"]), reward, off, obs

    def close(self):
        self.api.close(self.id)


def make_vec_env(n, robot, url, max_steps=300, **kwargs):
    """n simulators as one stable-baselines3 VecEnv, all stepped in ONE API call (with images).
    Every attempt starts somewhere else on the track and ends when the robot leaves the band or
    after max_steps (30 s)."""
    from stable_baselines3.common.vec_env import VecEnv

    class TrackCameraVecEnv(VecEnv):
        def __init__(self):
            self.api = BotopiaAPI(url)
            self.drivers = [Driver(self.api, robot, **kwargs) for _ in range(n)]
            space = spaces.Box(0, 255, shape=(H, W, 1), dtype=np.uint8)
            super().__init__(n, space, spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32))
            self.rng = np.random.default_rng(1)

        def _start(self, d):
            return d.reset(start=float(self.rng.uniform(0, TRACK_LENGTH)))[0]

        def reset(self):
            return np.stack([self._start(d) for d in self.drivers])

        def step_async(self, actions):
            self.actions = actions

        def step_wait(self):
            results = self.api.step_many([{"id": d.id, "action": d.command(a), "dt": d.dt, "image": IMAGE}
                                          for d, a in zip(self.drivers, self.actions)])
            obs, rewards, dones, infos = [], [], [], []
            for d, r in zip(self.drivers, results):
                img, reward, off, o = d.result(r)
                truncated = d.steps >= max_steps and not off
                info = {"driven": d.driven, "speed": d.driven / (d.steps * d.dt)}
                if off or truncated:
                    info["terminal_observation"] = img
                    info["TimeLimit.truncated"] = truncated
                    img = self._start(d)
                obs.append(img); rewards.append(reward); dones.append(off or truncated); infos.append(info)
            return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones), infos

        def close(self):
            for d in self.drivers:
                d.close()

        def get_attr(self, name, indices=None):
            return [getattr(self.drivers[i], name, None) for i in self._get_indices(indices)]

        def set_attr(self, name, value, indices=None):
            for i in self._get_indices(indices):
                setattr(self.drivers[i], name, value)

        def env_method(self, name, *args, indices=None, **kw):
            return [getattr(self.drivers[i], name)(*args, **kw) for i in self._get_indices(indices)]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return TrackCameraVecEnv()


# ---------- One lap from the start line ----------
def lap(driver, policy, max_time=180, frames=None, realtime=False, playback=1.0):
    """Drives from the start line until one full lap or max_time seconds. policy(image, obs) → action.
    Returns the lap time (None if not finished), how often it left the band and its mean speed."""
    img, obs = driver.reset(start=0.0)
    off_count, was_off, t = 0, False, 0.0
    while t < max_time:
        r = driver.api.step(driver.id, driver.command(policy(img, obs)), dt=driver.dt, image=IMAGE)
        img, _, off, obs = driver.result(r)
        t = obs["t"]
        if off and not was_off:
            off_count += 1
        was_off = off
        if frames is not None:
            frames.append((driver.api.frame(driver.id, 0.85), img))
        if realtime:
            time.sleep(driver.dt / playback)                           # playback 2 = twice as fast
        if driver.driven >= TRACK_LENGTH:
            return {"time": t, "off": off_count, "speed": driver.driven / t}
    return {"time": None, "off": off_count, "speed": driver.driven / max(t, 1e-9), "driven": driver.driven}


def report(name, r):
    if r["time"] is None:
        print(f"{name}: no full lap in time ({r['driven']:.1f} m), left the band {r['off']} times")
    else:
        print(f"{name}: lap in {r['time']:.1f} s ({r['speed'] * 100:.0f} cm/s), left the band {r['off']} times")


def KeepBest(path):
    """Saves the model whenever the mean reward of the last 100 attempts is a new record, so a
    training run that gets worse later still leaves its best version behind."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _KeepBest(BaseCallback):
        best = -np.inf

        def _on_rollout_end(self):
            buf = self.model.ep_info_buffer
            if len(buf) >= 20:
                mean = float(np.mean([e["r"] for e in buf]))
                if mean > self.best:
                    self.best = mean
                    self.model.save(path)
                    print(f"  new best mean reward {mean:.1f}: saved {path}")

        def _on_step(self):
            return True
    return _KeepBest()


class FrameStack:
    """The last n images side by side in the channel axis, as VecFrameStack does during training."""

    def __init__(self, n):
        self.n, self.frames = n, None

    def __call__(self, img):
        if self.frames is None:
            self.frames = [img] * self.n
        self.frames = self.frames[1:] + [img]
        return np.concatenate(self.frames, axis=2)


def main():
    # A Windows console (cp1252) can't print ε or …: show "?" instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["baseline", "train", "eval", "watch"])
    ap.add_argument("--robot", default="ackermann", help="robot with fwd + turn: ackermann, skid, 4wd, tracked, ...")
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--steps", type=int, default=400_000, help="training steps (train)")
    ap.add_argument("--envs", type=int, default=8, help="simulators stepped at the same time (train)")
    ap.add_argument("--device", default="auto", help="cpu, cuda or auto: where the neural network runs (train)")
    ap.add_argument("--stack", type=int, default=3, help="camera images stacked into one observation")
    ap.add_argument("--model", default="track_vision_ppo.zip", help="train saves this and <name>_best.zip")
    ap.add_argument("--gif", help="watch: save the lap as an animated GIF")
    ap.add_argument("--throttle", type=float, default=0.6, help="baseline: fixed throttle 0 … 1")
    ap.add_argument("--speed", type=float, default=1.0, help="eval / watch: multiply the network's throttle (0.5 = half speed)")
    ap.add_argument("--camh", type=float, help="camera height in m (Ackermann default 0.08): higher sees further ahead; train and eval with the same value")
    ap.add_argument("--camtilt", type=float, help="camera tilt down in degrees (default 8)")
    ap.add_argument("--init", help="train: start from this trained model instead of from scratch (fine-tuning)")
    ap.add_argument("--vmax", type=float, help="top speed of the robot in m/s (Ackermann default 0.9): a faster car; train with the same value")
    ap.add_argument("--playback", type=float, default=1.0, help="watch: 0.5 = slow motion, 2 = twice as fast (only how you see it)")
    args = ap.parse_args()
    api = BotopiaAPI(args.url)

    if args.command == "baseline":
        # Knows exactly how far it is from the centre line (the network doesn't): steer against the
        # offset and how fast it changes, at a fixed throttle.
        for throttle in sorted({args.throttle, 1.0}):
            d = Driver(api, args.robot, vmax=args.vmax, camh=args.camh, camtilt=args.camtilt)
            prev = [0.0]

            def pd(img, obs):
                e = obs["progress"]["offset"] or 0.0
                turn = -3.0 * e - 1.5 * (e - prev[0]) / d.dt
                prev[0] = e
                return [throttle * 2 - 1, turn]
            report(f"Line follower (knows its offset), throttle {throttle:.1f}", lap(d, pd))
            d.close()
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor

    if args.command == "train":
        import torch
        # PyTorch would claim every CPU core and starve the Chrome tabs that run the simulators.
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        venv = VecFrameStack(VecMonitor(make_vec_env(args.envs, args.robot, args.url, vmax=args.vmax, camh=args.camh, camtilt=args.camtilt)), args.stack, channels_order="last")
        # Small, careful updates: a learning rate that goes down to 0, and target_kl stops an update
        # early when the policy would change too much (big jumps can wreck what it learned).
        if args.init:
            # Fine-tuning: start from a model that can already drive (e.g. at a lower top speed) and
            # let it adapt, with a smaller learning rate so it doesn't forget what it knows.
            model = PPO.load(args.init, env=venv, device=args.device, learning_rate=lambda remaining: 5e-5 * remaining,
                             target_kl=0.03, verbose=1)
        else:
            model = PPO("CnnPolicy", venv, n_steps=max(32, 2048 // args.envs), batch_size=256, n_epochs=4,
                        learning_rate=lambda remaining: 1e-4 * remaining, target_kl=0.03, gamma=0.98,
                        clip_range=0.2, ent_coef=0.005, verbose=1, seed=1, device=args.device)
        t0 = time.time()
        model.learn(total_timesteps=args.steps, callback=KeepBest(args.model.replace(".zip", "_best.zip")))
        model.save(args.model)
        print(f"Saved {args.model} after {(time.time() - t0) / 60:.1f} min of training")
        venv.close()
        return

    if not os.path.exists(args.model):
        sys.exit(f"No trained model '{args.model}' in {os.getcwd()}.\nTrain one first: python {sys.argv[0]} train")
    model = PPO.load(args.model, device="cpu")
    d = Driver(api, args.robot, speed=args.speed, vmax=args.vmax, camh=args.camh, camtilt=args.camtilt)
    stack = FrameStack(args.stack)
    policy = lambda img, obs: model.predict(stack(img), deterministic=True)[0]   # noqa: E731
    if args.command == "eval":
        for run in range(2):
            stack.frames = None
            report(f"PPO from the camera, lap {run + 1}", lap(d, policy))
    else:
        frames = [] if args.gif else None
        report("PPO from the camera", lap(d, policy, frames=frames, realtime=not args.gif, playback=args.playback))
        if args.gif:
            pics = []
            for scene, img in frames[::2]:                                    # 5 frames per second: smaller file
                im = Image.open(io.BytesIO(scene)).convert("RGB")
                im = im.resize((480, round(im.height * 480 / im.width)))
                eye = Image.fromarray(img[:, :, 0]).resize((W * 2, H * 2), Image.NEAREST).convert("RGB")
                im.paste(eye, (im.width - eye.width - 6, 6))                   # what the network sees
                pics.append(im.convert("P", palette=Image.ADAPTIVE))
            pics[0].save(args.gif, save_all=True, append_images=pics[1:], duration=int(d.dt * 2000), loop=0)
            print(f"Saved {args.gif} ({len(pics)} frames)")
    d.close()


if __name__ == "__main__":
    main()
