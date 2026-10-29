"""Reinforcement learning tutorial 10: the arm learns to touch a point (SAC + Hindsight Experience Replay).

The quadruped's OpenMANIPULATOR-X gets a red ball somewhere in front of / around it and must touch it
with its fingertips (within 1.5 cm). No inverse kinematics: the network moves the four arm joints
itself. The reward is sparse (-1 every step until it touches the ball); Hindsight Experience Replay
(HER) turns every miss into a lesson: "you didn't reach the ball, but you did reach where your hand
ended up".

Uses the Botopia Node API (start it first: node server.mjs) and stable-baselines3
(pip install stable-baselines3 gymnasium).

    python arm_reach_her.py train                     # SAC + HER (60 000 steps, ~20 min), saves reach_sac_her.zip
    python arm_reach_her.py train --algo sac          # the same without HER, to compare
    python arm_reach_her.py train --algo ppo          # PPO (tutorials 6-8), to compare
    python arm_reach_her.py eval                      # success rate on 100 new points
    python arm_reach_her.py watch                     # touch one point after the other (server with --show)

Tutorial: https://robotics.dinno.be/tutorial-rl-arm-reach.html
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

# ---------- The arm (ROBOTIS OpenMANIPULATOR-X, as in indexQuadruped.php) ----------
JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4"]
LIMITS = np.array([[-162, 162], [-103, 90], [-54, 79], [-103, 117]], dtype=np.float64)   # degrees
DT = 0.1                    # one decision every 0.1 s
MAX_DEG = 9.0               # the servos turn at most 90°/s, so 9° per step
TOL = 0.015                 # touching = fingertips within 1.5 cm of the ball
SCALE = 0.4                 # positions (m) are divided by this for the network (about −1 … 1)


def fingertips(q):
    """Forward kinematics: fingertip position (m) in the arm frame for joint angles q (degrees).
    Arm frame: origin on top of the arm's base plate, x forward, y left, z up."""
    q1, q2, q3, q4 = np.radians(q)
    rot = lambda a, x, z: (x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a))  # noqa: E731
    r1, z1 = rot(q2, 0.024, 0.128)                 # upper arm (shoulder → elbow)
    r2, z2 = rot(q2 + q3, 0.124, 0.0)              # forearm (elbow → wrist)
    r3, z3 = rot(q2 + q3 + q4, 0.126, 0.0)         # wrist → fingertips
    r, z = r1 + r2 + r3, z1 + z2 + z3
    return np.array([0.012 + r * math.cos(q1), r * math.sin(q1), 0.017 + 0.0595 + z])


def sample_goal(rng):
    """A point the arm can reach: the fingertips of a random pose, above the base plate and not
    right next to the arm's own column."""
    while True:
        q = [rng.uniform(-100, 100), rng.uniform(-60, 80), rng.uniform(-50, 70), rng.uniform(-80, 100)]
        p = fingertips(q)
        if p[2] > 0.03 and math.hypot(p[0] - 0.012, p[1]) > 0.08:
            return p


# ---------- The environment ----------
class ArmReachEnv(gym.Env):
    """One quadruped standing still; the arm reaches for a new ball every episode.

    Action: change of the 4 joint angles, −1 … 1 = −9 … 9° (one step = 0.1 s).
    Observation (dict, as HER needs it):
      observation   joint angles (/180°) and fingertip position (/0.4 m)
      achieved_goal fingertip position (m)
      desired_goal  ball position (m)
    Reward: −1 every step, 0 when touching (sparse).
    """

    def __init__(self, url="http://localhost:8780", max_steps=50, seed=0, show_speed=0.0):
        super().__init__()
        self.api = BotopiaAPI(url)
        # camview=arm: the 3D view (watch, frames) close to the arm.
        self.id = self.api.create("quadruped", lane="free", seed=seed, options={"camview": "arm"})["id"]
        self.rng = np.random.default_rng(seed)
        self.max_steps, self.show_speed = max_steps, show_speed
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        box = lambda n, lim: spaces.Box(-lim, lim, (n,), np.float32)  # noqa: E731
        self.observation_space = spaces.Dict({"observation": box(7, 5.0), "achieved_goal": box(3, 1.0),
                                              "desired_goal": box(3, 1.0)})
        self.target = None
        self.episodes = 0

    def _obs(self, raw):
        tip = np.array(raw["extra"]["reach"]["tip"], dtype=np.float32)
        joints = np.array([next(m["deg"] for m in raw["motors"] if m["id"] == j) for j in JOINTS], dtype=np.float32)
        return {"observation": np.concatenate([joints / 180.0, tip / SCALE]).astype(np.float32),
                "achieved_goal": tip, "desired_goal": self.goal.astype(np.float32)}, joints

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Works on one goal or a whole batch (HER calls it with many at once)."""
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        return -(d > TOL).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.goal = sample_goal(self.rng)
        if self.episodes % 25 == 0:            # now and then a fresh start: robot standing, arm at home
            self.api.reset(self.id, lane="free")
        self.episodes += 1
        # The arm starts where the last episode left it: varied starting poses for free.
        raw = self.api.step(self.id, {"reach": dict(zip("xyz", map(float, self.goal))), "motors": {"gripper": 0}},
                            dt=DT, frame=1 / 30)["obs"]
        obs, joints = self._obs(raw)
        self.q = joints.astype(np.float64)
        self.steps = 0
        return obs, {}

    def step(self, action):
        self.q = np.clip(self.q + np.clip(action, -1, 1) * MAX_DEG, LIMITS[:, 0], LIMITS[:, 1])
        raw = self.api.step(self.id, {"motors": {j: float(v) for j, v in zip(JOINTS, self.q)}}, dt=DT, frame=1 / 30)["obs"]
        if self.show_speed:
            time.sleep(DT / self.show_speed)
        obs, _ = self._obs(raw)
        self.steps += 1
        d = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))
        return obs, reward, False, self.steps >= self.max_steps, {"is_success": d <= TOL, "distance": d}

    def close(self):
        self.api.close(self.id)


# ---------- Train / evaluate ----------
def make_model(algo, env, device, seed):
    from stable_baselines3 import PPO, SAC, HerReplayBuffer
    if algo == "ppo":
        return PPO("MultiInputPolicy", env, n_steps=1000, batch_size=250, gamma=0.95, learning_rate=3e-4,
                   verbose=0, seed=seed, device=device)
    kw = dict(learning_starts=1000, batch_size=256, gamma=0.95, learning_rate=1e-3, tau=0.05,
              policy_kwargs=dict(net_arch=[256, 256, 256]), verbose=0, seed=seed, device=device)
    if algo == "sac-her":
        # 'future': each step is also stored with 4 goals the hand reached LATER in the same episode.
        kw.update(replay_buffer_class=HerReplayBuffer,
                  replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"))
    return SAC("MultiInputPolicy", env, **kw)


def evaluate(model, env, episodes, log=False):
    """Success rate, final distance and steps until touching, on new random points."""
    ok, dists, steps = 0, [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        for t in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, trunc, info = env.step(action)
            if info["is_success"]:
                ok += 1
                steps.append(t + 1)
                break
        dists.append(info["distance"])
        if log:
            print(f"  point {ep + 1:3d}: {'touched after ' + str(t + 1) + ' steps' if info['is_success'] else 'missed'}"
                  f", distance {info['distance'] * 100:.1f} cm")
    return ok / episodes, float(np.mean(dists)), (float(np.mean(steps)) if steps else float("nan"))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "eval", "watch", "check"])
    ap.add_argument("--algo", default="sac-her", choices=["sac-her", "sac", "ppo"])
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--steps", type=int, default=60000, help="training steps (0.1 s each)")
    ap.add_argument("--device", default="auto", help="cpu, cuda or auto")
    ap.add_argument("--model", help="default: reach_<algo>.zip")
    ap.add_argument("--episodes", type=int, default=100, help="eval: number of points")
    ap.add_argument("--speed", type=float, default=1.0, help="watch: × real time")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    model_path = args.model or f"reach_{args.algo.replace('-', '_')}.zip"

    if args.command == "check":
        # Does our forward kinematics match the simulator? Move to a few poses and compare.
        env = ArmReachEnv(args.url, seed=args.seed)
        env.reset()
        for q in ([0, 0, 0, 0], [30, -20, 40, 10], [-90, 45, -30, 60]):
            for _ in range(40):
                raw = env.api.step(env.id, {"motors": dict(zip(JOINTS, map(float, q)))}, dt=DT, frame=1 / 30)["obs"]
            print(q, "simulator", raw["extra"]["reach"]["tip"], "our FK", np.round(fingertips(q), 4).tolist())
        env.close()
        return

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback

    if args.command == "train":
        env = ArmReachEnv(args.url, seed=args.seed)
        model = make_model(args.algo, env, args.device, args.seed)

        class Progress(BaseCallback):
            """Every 2000 steps: the success rate of the last 100 episodes (while still exploring)."""
            def __init__(self):
                super().__init__()
                self.results, self.t0 = [], time.time()

            def _on_step(self):
                for info, done in zip(self.locals["infos"], self.locals["dones"]):
                    if done:
                        self.results.append(info.get("is_success", False) or info["distance"] <= TOL)
                if self.num_timesteps % 2000 == 0 and self.results:
                    last = self.results[-100:]
                    print(f"step {self.num_timesteps:6d}  episodes {len(self.results):4d}  "
                          f"touched (last 100) {100 * sum(last) / len(last):5.1f} %  {time.time() - self.t0:.0f} s", flush=True)
                return True

        t0 = time.time()
        model.learn(total_timesteps=args.steps, callback=Progress())
        model.save(model_path)
        print(f"Saved {model_path} after {time.time() - t0:.0f} s of training")
        env.close()
        return

    if not os.path.exists(model_path):
        sys.exit(f"No model '{model_path}'. Train one first:  python {sys.argv[0]} train\n"
                 f"or use the trained one:  --model python/models/reach_sac_her.zip")
    cls = PPO if args.algo == "ppo" else SAC
    env = ArmReachEnv(args.url, seed=1000 + args.seed, show_speed=args.speed if args.command == "watch" else 0.0)
    model = cls.load(model_path, env=env if args.algo == "sac-her" else None, device="cpu")
    if args.command == "eval":
        rate, dist, steps = evaluate(model, env, args.episodes)
        print(f"{model_path}: touched {rate * 100:.0f} % of {args.episodes} new points, "
              f"mean final distance {dist * 100:.1f} cm, {steps:.1f} steps of 0.1 s to touch")
    else:
        evaluate(model, env, args.episodes, log=True)
    env.close()


if __name__ == "__main__":
    main()
