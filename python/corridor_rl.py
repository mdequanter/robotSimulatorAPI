"""Reinforcement learning tutorial: drive the 60 cm corridor (sharp right-hand bend) as fast as possible.

Uses the Botopia Node API (start it first: node server.mjs) and PPO from
stable-baselines3 (pip install stable-baselines3 gymnasium).

    python corridor_rl.py baseline              # hand-written driver: the time to beat
    python corridor_rl.py train                 # train PPO (150 000 steps), saves corridor_ppo.zip
    python corridor_rl.py eval                  # lap time of the trained model
    python corridor_rl.py watch --gif lap.gif   # one lap as an animated GIF (needs Pillow)

Tutorial: https://robotics.dinno.be/tutorial-rl-corridor.html
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from botopia import BotopiaAPI  # noqa: E402

# ---------- Corridor geometry (world metres, from common/terrain.js) ----------
# The robot starts at (0.9, 6.3) facing +x. The corridor centre line runs straight to the bend at
# x = 2.42, then turns right (+z). The lane counts as done ("goal") at z = 7.35.
P0, P1, P2 = (0.9, 6.3), (2.42, 6.3), (2.42, 8.0)
SEG1 = P1[0] - P0[0]                  # length of the first straight
GOAL_Z = 7.35
HEAD1, HEAD2 = 0.0, -math.pi / 2      # path heading on both straights (yaw + = left, so right = −90°)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def path_frame(x, z):
    """Where the robot is relative to the centre line: (s, e, heading) =
    distance along the line, sideways offset (+ = left of the line), direction of the line there."""
    s1 = min(max(x - P0[0], 0.0), SEG1)               # nearest point on the first straight
    d1 = math.hypot(x - (P0[0] + s1), z - P0[1])
    s2 = min(max(z - P1[1], 0.0), P2[1] - P1[1])      # ... and on the second
    d2 = math.hypot(x - P1[0], z - (P1[1] + s2))
    if d1 <= d2:
        return s1, P0[1] - z, HEAD1                   # left of a +x line = smaller z
    return SEG1 + s2, x - P1[0], HEAD2                # left of a +z line = larger x


def heading_at(s):
    return HEAD1 if s < SEG1 else HEAD2


# ---------- The environment ----------
class CorridorEnv(gym.Env):
    """Gymnasium environment around one Botopia simulator in the corridor lane.

    Action: [fwd, turn] in −1 … 1 (the robot's own drive commands, like the W/A/S/D keys).
    Observation (10 numbers, roughly −1 … 1):
      0   progress along the centre line / 3 m
      1   sideways offset from the centre line / 0.3 m (the walls are at ±1)
      2,3 sin / cos of the heading error (robot heading − line heading)
      4,5 sin / cos of the heading error to the line 0.5 m further on (sees the bend coming)
      6   distance to the bend / 1.5 m (0 after it)
      7   forward speed / 0.6 m/s
      8   turn rate / 2 rad/s
      9   1 if the robot touched a wall in the last step
    Reward per step: 10 × metres gained along the line − time_penalty (time is precious); +10 at the goal.
    Touching a wall costs 5 and ends the episode (end_on_hit; otherwise it costs 1 per step).
    An episode also ends at the goal or after max_steps (then: too slow).
    """

    metadata = {"render_modes": []}

    def __init__(self, robot="mecanum", url="http://localhost:8780", dt=0.1, max_steps=200, end_on_hit=True,
                 time_penalty=0.1, api=None):
        super().__init__()
        self.api = api or BotopiaAPI(url)
        self.robot, self.dt, self.max_steps, self.end_on_hit = robot, dt, max_steps, end_on_hit
        self.time_penalty = time_penalty
        created = self.api.create(robot, "corridor")
        self.id = created["id"]
        self.axes = list(created["info"]["drive"])
        if self.axes[:2] != ["fwd", "turn"]:
            raise ValueError(f"{robot}: needs a robot that drives with fwd + turn, it has {self.axes}")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(10,), dtype=np.float32)

    def _features(self, obs, hit):
        p, v = obs["pose"], obs["velocity"]
        s, e, h = path_frame(p["x"], p["z"])
        err = wrap(p["yaw"] - h)
        err_ahead = wrap(p["yaw"] - heading_at(s + 0.5))
        to_bend = max(0.0, SEG1 - s)
        f = [s / 3, e / 0.3, math.sin(err), math.cos(err), math.sin(err_ahead), math.cos(err_ahead),
             min(to_bend, 1.5) / 1.5, v["vx"] / 0.6, v["wz"] / 2, 1.0 if hit else 0.0]
        return np.clip(np.array(f, dtype=np.float32), -5, 5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs = self.api.reset(self.id, lane="corridor", seed=seed)
        self.steps, self.s = 0, 0.0
        return self._features(self.obs, False), {}

    def to_action(self, action):
        a = [float(np.clip(v, -1, 1)) for v in action]
        return {"drive": {"fwd": a[0], "turn": a[1]}}

    def step(self, action):
        return self.result(self.api.step(self.id, self.to_action(action), dt=self.dt))

    def result(self, r):
        """Observation, reward and end of the episode from one API step result."""
        self.obs, self.steps = r["obs"], self.steps + 1
        p = self.obs["pose"]
        s, _, _ = path_frame(p["x"], p["z"])
        hit = any(ev["type"] in ("collision", "blocked", "stall") for ev in r["events"])
        goal = p["z"] >= GOAL_Z and p["x"] > SEG1 + P0[0] - 0.4

        reward = 10 * (s - self.s) - self.time_penalty + (10.0 if goal else 0.0)
        if hit:
            reward -= 5.0 if self.end_on_hit else 1.0
        self.s = s
        terminated = goal or (hit and self.end_on_hit)
        truncated = self.steps >= self.max_steps and not terminated
        info = {"time": self.obs["t"], "goal": goal, "hit": hit}
        return self._features(self.obs, hit), reward, terminated, truncated, info

    def frame(self):
        return self.api.frame(self.id)

    def close(self):
        self.api.close(self.id)


# ---------- Many simulators at once, for training ----------
def make_vec_env(n, robot, url, **kwargs):
    """n corridor simulators as one stable-baselines3 VecEnv. Every step is ONE batch call to the API,
    so the server steps all simulators at the same time (each Chrome tab on its own CPU core)."""
    from stable_baselines3.common.vec_env import VecEnv

    class CorridorVecEnv(VecEnv):
        def __init__(self):
            self.api = BotopiaAPI(url)
            self.envs = [CorridorEnv(robot, url, api=self.api, **kwargs) for _ in range(n)]
            super().__init__(n, self.envs[0].observation_space, self.envs[0].action_space)
            self.actions = None

        def reset(self):
            return np.stack([e.reset(seed=i)[0] for i, e in enumerate(self.envs)])

        def step_async(self, actions):
            self.actions = actions

        def step_wait(self):
            results = self.api.step_many([{"id": e.id, "action": e.to_action(a), "dt": e.dt}
                                          for e, a in zip(self.envs, self.actions)])
            obs, rewards, dones, infos = [], [], [], []
            for e, r in zip(self.envs, results):
                o, reward, terminated, truncated, info = e.result(r)
                if terminated or truncated:                     # VecEnv rule: restart it right away
                    info["terminal_observation"] = o
                    info["TimeLimit.truncated"] = truncated and not terminated
                    o = e.reset()[0]
                obs.append(o); rewards.append(reward); dones.append(terminated or truncated); infos.append(info)
            return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones), infos

        def close(self):
            for e in self.envs:
                e.close()

        def get_attr(self, name, indices=None):
            return [getattr(self.envs[i], name) for i in self._get_indices(indices)]

        def set_attr(self, name, value, indices=None):
            for i in self._get_indices(indices):
                setattr(self.envs[i], name, value)

        def env_method(self, name, *args, indices=None, **kw):
            return [getattr(self.envs[i], name)(*args, **kw) for i in self._get_indices(indices)]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return CorridorVecEnv()


# ---------- A hand-written driver: the benchmark ----------
def baseline_action(env):
    """Follow the centre line: aim at a point 0.5 m ahead on it, slow down while turning hard."""
    p = env.obs["pose"]
    s, _, _ = path_frame(p["x"], p["z"])
    t = s + 0.5
    tx, tz = (P0[0] + t, P0[1]) if t < SEG1 else (P1[0], P1[1] + (t - SEG1))
    want = math.atan2(-(tz - p["z"]), tx - p["x"])     # heading to that point (yaw: + = left)
    err = wrap(want - p["yaw"])
    turn = max(-1.0, min(1.0, 2.5 * err))
    fwd = 1.0 if abs(err) < 0.3 else 0.4
    return [fwd, turn]


def run_lap(env, policy, seed=0, frames=None, realtime=False):
    obs, _ = env.reset(seed=seed)
    total, hits = 0.0, 0
    while True:
        obs, reward, done, truncated, info = env.step(policy(obs))
        total += reward
        hits += info["hit"]
        if frames is not None:
            frames.append(env.frame())
        if realtime:
            time.sleep(env.dt)
        if done or truncated:
            return {"goal": info["goal"], "time": info["time"], "hits": hits, "reward": total}


def report(name, lap):
    result = f"{lap['time']:.2f} s" if lap["goal"] else f"did not finish in {lap['time']:.0f} s"
    print(f"{name}: {result}, wall hits {lap['hits']}, total reward {lap['reward']:.1f}")


# ---------- Commands ----------
def main():
    # A Windows console (cp1252) can't print ε or …: show "?" instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["baseline", "train", "eval", "watch"])
    ap.add_argument("--robot", default="skid", help="robot with fwd + turn: skid, tracked, 4wd, whegs, ackermann, ...")
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--steps", type=int, default=150000, help="training steps (train)")
    ap.add_argument("--envs", type=int, default=8, help="simulators stepped at the same time (train); about one per 2 CPU cores")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or auto: where the neural network runs (train)")
    ap.add_argument("--time-penalty", type=float, default=0.1, help="reward lost per step: how much speed matters")
    ap.add_argument("--model", default="corridor_ppo.zip")
    ap.add_argument("--gif", help="watch: save the lap as an animated GIF")
    ap.add_argument("--max-time", type=float, default=20, help="seconds (simulated) before an attempt counts as too slow")
    args = ap.parse_args()
    max_steps = max(1, round(args.max_time / 0.1))        # one step = 0.1 s (CorridorEnv dt)

    if args.command == "baseline":
        env = CorridorEnv(args.robot, args.url, end_on_hit=False, max_steps=max_steps)
        report("Hand-written driver", run_lap(env, lambda obs: baseline_action(env)))
        report("Full throttle, no steering", run_lap(env, lambda obs: [1.0, 0.0]))
        env.close()
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    if args.command == "train":
        venv = VecMonitor(make_vec_env(args.envs, args.robot, args.url, time_penalty=args.time_penalty,
                                       max_steps=max_steps))
        # 2048 steps of experience per update, however many simulators gather them.
        model = PPO("MlpPolicy", venv, n_steps=max(64, 2048 // args.envs), batch_size=256, learning_rate=3e-4,
                    gamma=0.99, ent_coef=0.003, verbose=1, seed=1, device=args.device)
        t0 = time.time()
        model.learn(total_timesteps=args.steps)
        model.save(args.model)
        print(f"Saved {args.model} after {time.time() - t0:.0f} s of training")
        venv.close()
        return

    if not os.path.exists(args.model):
        sys.exit(f"No trained model '{args.model}' in {os.getcwd()}.\n"
                 f"Train one first (it is saved in the folder you run it from):\n"
                 f"    python {sys.argv[0]} train")
    model = PPO.load(args.model)
    env = CorridorEnv(args.robot, args.url, end_on_hit=False, max_steps=max_steps)   # a test lap: count wall hits, don't stop
    policy = lambda obs: model.predict(obs, deterministic=True)[0]   # noqa: E731
    if args.command == "eval":
        for seed in range(3):
            report(f"PPO, run {seed + 1}", run_lap(env, policy, seed=seed))
    else:
        frames = [] if args.gif else None
        lap = run_lap(env, policy, frames=frames, realtime=not args.gif)
        report("PPO", lap)
        if args.gif:
            from io import BytesIO
            from PIL import Image
            def small(jpg):                                      # 640 px wide keeps the GIF small
                im = Image.open(BytesIO(jpg))
                return im.resize((640, round(im.height * 640 / im.width))).convert("P", palette=Image.ADAPTIVE)
            images = [small(f) for f in frames]
            images[0].save(args.gif, save_all=True, append_images=images[1:], duration=int(env.dt * 1000), loop=0)
            print(f"Saved {args.gif} ({len(images)} frames)")
    env.close()


if __name__ == "__main__":
    main()
