"""Reinforcement learning tutorial 4: a sorting job for the quadruped with the OpenMANIPULATOR-X arm.

The robot starts in the door lane (lane 11) with four cubes in front of it. The job: put the cubes on
its back, open the door, walk through, close the door behind it, walk to the table and sort the
cubes into the boxes of their colour. The robot already has the skills (lie down, pick, walk to the
door, ...); it learns WHICH skill to use WHEN, with tabular Q-learning (no extra packages needed).

Uses the Botopia Node API (start it first: node server.mjs).

    python sort_rl.py baseline                 # the right order, written by hand: the score to reach
    python sort_rl.py train                    # Q-learning, saves sort_q.json
    python sort_rl.py eval                     # run the learned policy (no exploring)
    python sort_rl.py watch                    # the same, slowly, to watch (server with --show)

Tutorial: https://robotics.dinno.be/tutorial-rl-sorting.html
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from botopia import BotopiaAPI  # noqa: E402

# The skills of the quadruped page (indexQuadruped.php, SKILLS). Each one first checks whether it can
# start (lying down? at the door? holding a cube?); if not, it fails at once and nothing happens.
SKILLS = ["lie_down", "stand_up", "pick", "to_back", "to_box",
          "walk_to_door", "open_door", "walk_through", "close_door", "walk_to_table"]
ZONES = ["start", "at_door", "doorway", "past_door", "table"]
CUBES = 4

# The hand-written solution: what the learner should discover by itself.
BASELINE = (["lie_down"] + ["pick", "to_back"] * CUBES + ["stand_up", "walk_to_door", "open_door", "walk_through",
            "close_door", "walk_to_table"] + ["pick", "to_box"] * CUBES)


# ---------- The task ----------
class SortTask:
    """One simulator in the door lane, driven skill by skill."""

    def __init__(self, url, door="latch", watch_speed=0.0):
        self.api = BotopiaAPI(url)
        created = self.api.create("quadruped", lane="door", options={"door": door})
        self.id = created["id"]
        self.watch_speed = watch_speed        # > 0: run in small steps at this many × real time (to watch)
        self.sim_time = 0.0

    def reset(self, seed=None):
        obs = self.api.reset(self.id, lane="door", seed=seed)
        self.sim_time = 0.0
        return obs["extra"]["task"]

    def do(self, skill):
        """Run one skill to its end. Returns the task state afterwards (task["last"] = the result)."""
        if self.watch_speed > 0:
            r = self.api.step(self.id, {"skill": skill}, dt=1 / 15, frame=1 / 30)
            while r["obs"]["extra"]["task"]["busy"]:
                time.sleep(1 / 15 / self.watch_speed)
                r = self.api.step(self.id, {}, dt=1 / 15, frame=1 / 30)
        else:
            # One call runs the whole skill (at most 120 s simulated, it stops when the skill ends).
            r = self.api.step(self.id, {"skill": skill}, dt=120, frame=1 / 30, until="idle")
            while r["obs"]["extra"]["task"]["busy"]:
                r = self.api.step(self.id, {}, dt=120, frame=1 / 30, until="idle")
        task = r["obs"]["extra"]["task"]
        self.sim_time = r["obs"]["t"]
        return task

    def close(self):
        self.api.close(self.id)


def state_of(task):
    """The state the learner sees: a small tuple, so the Q-table stays small."""
    return (int(task["lying"]), ZONES.index(task["zone"]), int(task["doorOpen"]), int(task["doorLatched"]),
            int(task["holding"] is not None), task["floor"], task["back"], task["sorted"])


def finished(task):
    return task["sorted"] == CUBES and task["zone"] in ("past_door", "table") and task["doorLatched"]


def hopeless(task):
    """Past the door with cubes left behind: there is no skill to go back, so the job can't be done."""
    return task["zone"] in ("past_door", "table") and task["floor"] > 0


def progress(task):
    """How far the job is (used for the reward): cubes picked up, through the door, door shut behind,
    cubes sorted. Rewarding the CHANGE of this number (reward shaping) helps the learner find its way,
    without rewarding going back and forth."""
    past = task["zone"] in ("past_door", "table")
    return ((CUBES - task["floor"]) * 1.0 + task["sorted"] * 2.0 + past * 2.0
            + (past and task["doorLatched"]) * 2.0 + (task["zone"] == "table") * 0.5)


FINISH_BONUS = 20.0


def reward(before, after, gamma, done, shaping="potential"):
    r = -0.1                                                  # every skill costs a little time
    if not after["last"]["ok"]:
        r -= 0.3                                              # a skill that could not run or did not work
    # Shaping: the progress made. At the end of an attempt the progress counts as 0 again
    # (potential-based shaping): otherwise the robot learns to collect the "through the door" reward
    # and stop there, cubes left behind ("naive" shows that). "none": only the finish bonus.
    if shaping == "potential":
        r += gamma * (0.0 if done else progress(after)) - progress(before)
    elif shaping == "naive":
        r += progress(after) - progress(before)
    if finished(after):
        r += FINISH_BONUS
    return r


# ---------- Q-learning ----------
def greedy(Q, s):
    q = Q.get(s)
    if not q:
        return random.randrange(len(SKILLS))
    best = max(q)
    return random.choice([a for a, v in enumerate(q) if v == best])


def run_episode(env, choose, Q=None, alpha=0.3, gamma=0.97, max_skills=40, seed=None, log=False, shaping="potential"):
    """One attempt at the job. With Q: learn from every skill (Q-learning update).
    Returns (return, number of skills, simulated seconds, finished, the skills used)."""
    task = env.reset(seed)
    total, used = 0.0, []
    for _ in range(max_skills):
        s = state_of(task)
        a = choose(s)
        new = env.do(SKILLS[a])
        done = finished(new) or hopeless(new)
        r = reward(task, new, gamma, done, shaping)
        if Q is not None:
            q = Q.setdefault(s, [0.0] * len(SKILLS))
            nxt = 0.0 if done else max(Q.get(state_of(new), [0.0]))
            q[a] += alpha * (r + gamma * nxt - q[a])
        total += r
        used.append((SKILLS[a], new["last"]["ok"]))
        if log:
            last = new["last"]
            print(f"  {SKILLS[a]:<14} {'ok' if last['ok'] else 'failed: ' + last['why']:<50} "
                  f"floor {new['floor']} back {new['back']} sorted {new['sorted']}  door "
                  f"{'open' if new['doorOpen'] else 'latched' if new['doorLatched'] else 'shut'}  ({new['zone']})")
        task = new
        if done:
            break
    return total, len(used), env.sim_time, finished(task), used


def save(Q, path, info):
    data = {"skills": SKILLS, "info": info, "q": [{"state": list(s), "values": [round(v, 4) for v in q]} for s, q in Q.items()]}
    with open(path, "w") as f:
        json.dump(data, f)


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {tuple(e["state"]): e["values"] for e in data["q"]}


def report(name, result):
    total, n, secs, ok, used = result
    print(f"{name}: {'job done' if ok else 'NOT finished'} in {n} skills, {secs:.0f} s simulated, return {total:.1f}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["baseline", "train", "eval", "watch"])
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--episodes", type=int, default=400, help="training attempts (train)")
    ap.add_argument("--alpha", type=float, default=0.3, help="learning rate of the Q-update")
    ap.add_argument("--gamma", type=float, default=0.97, help="discount: how much later rewards count")
    ap.add_argument("--eps-end", type=float, default=0.05, help="exploring at the end of training (it starts at 1)")
    ap.add_argument("--max-skills", type=int, default=40, help="skills per attempt before it stops")
    ap.add_argument("--shaping", default="potential", choices=["potential", "naive", "none"],
                    help="reward for progress: potential-based (default), naive (the trap), none (finish bonus only)")
    ap.add_argument("--door", default="latch", choices=["latch", "free", "locked"], help="the door's latch mode")
    ap.add_argument("--speed", type=float, default=1.0, help="watch: × real time")
    ap.add_argument("--model", default="sort_q.json", help="the Q-table (JSON)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    env = SortTask(args.url, door=args.door, watch_speed=args.speed if args.command == "watch" else 0.0)
    try:
        if args.command == "baseline":
            plan = iter(BASELINE)
            report("Hand-written order", run_episode(env, lambda s: SKILLS.index(next(plan, "stand_up")),
                                                      max_skills=len(BASELINE), log=True, gamma=args.gamma))
            return

        if args.command == "train":
            Q, t0, recent = {}, time.time(), []
            for ep in range(args.episodes):
                # ε-greedy: explore a lot at first, then more and more use what was learned.
                eps = max(args.eps_end, 1.0 - ep / (0.6 * args.episodes))
                choose = lambda s: random.randrange(len(SKILLS)) if random.random() < eps else greedy(Q, s)  # noqa: E731
                total, n, secs, ok, _ = run_episode(env, choose, Q, args.alpha, args.gamma, args.max_skills, seed=ep,
                                                    shaping=args.shaping)
                recent = (recent + [(total, ok)])[-20:]
                if (ep + 1) % 10 == 0:
                    done = sum(o for _, o in recent)
                    print(f"episode {ep + 1:4d}  ε {eps:.2f}  return {total:6.1f}  last 20: mean return "
                          f"{sum(t for t, _ in recent) / len(recent):6.1f}, job done {done}/{len(recent)}  "
                          f"states {len(Q)}  {time.time() - t0:.0f} s", flush=True)
                if (ep + 1) % 50 == 0:
                    save(Q, args.model, {"episodes": ep + 1})
            save(Q, args.model, {"episodes": args.episodes, "seconds": round(time.time() - t0)})
            print(f"Saved {args.model} ({len(Q)} states) after {time.time() - t0:.0f} s of training")
            return

        if not os.path.exists(args.model):
            sys.exit(f"No Q-table '{args.model}'. Train one first:  python {sys.argv[0]} train\n"
                     f"or use the trained one:  --model python/models/sort_q.json")
        Q = load(args.model)
        report("Learned policy", run_episode(env, lambda s: greedy(Q, s), max_skills=args.max_skills, log=True,
                                             gamma=args.gamma))
    finally:
        env.close()


if __name__ == "__main__":
    main()
