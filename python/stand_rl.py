"""Reinforcement learning tutorial 11: the OpenMANIPULATOR-X stands the tower blocks up in a row.

Six tower-game blocks (75 × 25 × 15 mm) lie in a stack in front of the arm. The job: take them one by
one and stand each one on its small end (15 × 25 mm) in a row, an arc round the arm, in front of it. The
arm already has the two skills (take the top block of the stack; stand the block up at the next free
spot); it learns at WHAT HEIGHT to do them. Its own idea of the right height is off: the simulator has
a hidden calibration error (as the real arm had: it gripped 1 cm too high), so it has to find, by
trying, how many mm to correct for each skill. Tabular Q-learning (no extra packages needed).

Uses the Botopia Node API (start it first: node server.mjs).

    python stand_rl.py baseline                # the page's own heights (no correction): the problem
    python stand_rl.py train                   # Q-learning, saves stand_q.json
    python stand_rl.py eval                    # run the learned policy (no exploring)
    python stand_rl.py watch                   # the same, slowly, to watch (server with --show)

On the real arm (no Node API: the page in your own Chrome, with the OpenCR connected):

    python stand_rl.py real --session ROW1     # page: indexOpenManipulator.php?remote=ROW1

Tutorial: https://robotics.dinno.be/tutorial-rl-stand-row.html
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from botopia import BotopiaAPI, RemoteSession  # noqa: E402

# The actions: a height correction in mm on the page's own idea of the right height.
DZ = [-15, -10, -5, 0, 5, 10, 15]
BLOCKS = 6


# ---------- The task ----------
class StandTask:
    """One OpenMANIPULATOR-X simulator, driven skill by skill (actions rowPick / rowPlace)."""

    def __init__(self, url, zerror=10.0, zsag=4.0, watch_speed=0.0):
        self.api = BotopiaAPI(url)
        # The hidden calibration error goes to the page as URL settings; the learner never sees it.
        created = self.api.create("omx", options={"zerror": zerror, "zsag": zsag})
        self.id = created["id"]
        self.watch_speed = watch_speed        # > 0: run in small steps at this many × real time (to watch)
        self.sim_time = 0.0

    def reset(self, seed=None):
        obs = self.api.reset(self.id, seed=seed)
        self.sim_time = 0.0
        return self.view(obs)

    @staticmethod
    def view(obs):
        row = obs["extra"]["row"]
        return {"source": row["source"], "standing": row["standing"], "holding": obs["extra"]["holding"] is not None,
                "last": row["last"] or {}, "slots": row["slots"]}

    def do(self, action):
        """Run one skill to its end. Returns the task state afterwards (["last"] = the skill's result)."""
        if self.watch_speed > 0:
            r = self.api.step(self.id, action, dt=1 / 15, frame=1 / 30)
            while r["obs"]["extra"]["busy"]:
                time.sleep(1 / 15 / self.watch_speed)
                r = self.api.step(self.id, {}, dt=1 / 15, frame=1 / 30)
        else:
            # One call runs the whole skill (at most 60 s simulated, it stops when the skill ends).
            r = self.api.step(self.id, action, dt=60, frame=1 / 30, until="idle")
            while r["obs"]["extra"]["busy"]:
                r = self.api.step(self.id, {}, dt=60, frame=1 / 30, until="idle")
        # Half a second more: a block that was let go lands (or falls over).
        r = self.api.step(self.id, {}, dt=0.5, frame=1 / 30)
        self.sim_time = r["obs"]["t"]
        return self.view(r["obs"])

    def close(self):
        self.api.close(self.id)


class RealTask:
    """The same two skills, run on the simulator page in YOUR browser (opened with ?remote=CODE), through
    the PHP API. Connect the OpenCR there and tick "Real arm follows the simulator": the real arm then
    makes every move the simulated one makes. The results printed are the simulator's: watch the real arm."""

    def __init__(self, url, session):
        self.remote = RemoteSession(url, session)
        st = self.remote.state()
        if not st.get("active"):
            sys.exit(f"Session {session} is not active: open indexOpenManipulator.php?remote={session} in Chrome "
                     f"(on the same site as --remote-url) and keep that tab open.")
        if (st.get("info") or {}).get("robot") != "omx":
            sys.exit(f"Session {session} is not the OpenMANIPULATOR-X page.")

    def view(self):
        return StandTask.view(self.remote.state()["obs"])

    def reset(self):
        self.remote.command(reset={})
        time.sleep(1.0)
        return self.view()

    def do(self, action, timeout=90):
        # The page numbers its skills (last.n): wait until it reports this one, finished.
        before = (self.view()["last"] or {}).get("n", 0)
        self.remote.command(action=action)
        t0 = time.time()
        while True:
            time.sleep(0.25)                  # the page fetches commands every 0.1 s, reports every 0.2 s
            obs = self.remote.state()["obs"]
            if (obs["extra"]["row"]["last"] or {}).get("n", 0) > before and not obs["extra"]["busy"]:
                break
            if time.time() - t0 > timeout:
                raise RuntimeError("the skill did not end (STOP pressed, or the page was closed?)")
        time.sleep(0.8)                       # a block that was let go lands
        return self.view()


def run_real(args):
    """Run the job on the real arm, one skill at a time, asking before each one."""
    if args.model:
        Q = load(args.model)
        correction = lambda s: DZ[greedy(Q, s)]  # noqa: E731
        print(f"Corrections from {args.model} (learned for the simulator's hidden error: they may be wrong "
              f"for your arm).")
    else:
        correction = lambda s: args.pick_dz if s[0] == "pick" else args.place_dz  # noqa: E731
        print(f"Corrections: pick {args.pick_dz:+g} mm, place {args.place_dz:+g} mm.")
    real = RealTask(args.remote_url, args.session)
    ask = (lambda prompt: "") if args.yes else input
    print("Put the six real blocks in a stack straight in front of the arm, 155 mm from the first servo's axis,\n"
          "lengthwise towards the arm, as in the simulator. Keep a hand near STOP (Esc on the page).")
    if ask("Enter: reset the simulator (the real arm goes to its look-down pose)  q: quit  ").strip().lower() == "q":
        return
    task = real.reset()
    for n in range(1, args.max_skills + 1):
        s = state_of(task)
        dz = correction(s)
        if ask(f"{n:2d}. {s[0]} (stack {task['source']}, standing {task['standing']}) with {dz:+g} mm.  "
               f"Enter: go  q: stop  ").strip().lower() == "q":
            break
        task = real.do({"rowPlace" if s[0] == "place" else "rowPick": dz})
        last = task["last"]
        print(f"    simulator: {last.get('result', '?')}" + (f", {last['offMm']} mm off" if "offMm" in last else ""))
        if finished(task) or hopeless(task):
            break
    print(f"Done: {task['standing']}/{BLOCKS} standing in the simulator. Compare with the real row.")


def state_of(task):
    """The state the learner sees: which skill is next, and how far the job is.
    Not holding: ("pick", blocks left on the stack). Holding: ("place", blocks already standing)."""
    return ("place", task["standing"]) if task["holding"] else ("pick", task["source"])


def action_of(state, a):
    return {"rowPlace" if state[0] == "place" else "rowPick": DZ[a]}


def finished(task):
    return task["standing"] == BLOCKS


def hopeless(task):
    """No blocks left to take and not holding one: blocks that fell over are lost for this attempt."""
    return task["source"] == 0 and not task["holding"]


FINISH_BONUS = 20.0


def reward(after):
    last = after["last"]
    r = -1.0                                                  # every skill costs time
    res = last.get("result", "")
    if last.get("skill") == "pick":
        r += {"grasped": 2.0, "missed": -2.0, "pushed": -6.0, "table": -6.0}.get(res, -2.0)
    else:
        if res in ("standing", "pressed"):
            r += 10.0 - min(10.0, last.get("offMm", 0.0))     # standing: the closer to its spot, the better
            if res == "pressed":
                r -= 4.0                                      # pressed onto the table: it slipped
        else:
            r -= 10.0                                         # fell over
    if finished(after):
        r += FINISH_BONUS
    return r


# ---------- Q-learning ----------
def greedy(Q, s):
    q = Q.get(s)
    if not q:
        return random.randrange(len(DZ))
    best = max(q)
    return random.choice([a for a, v in enumerate(q) if v == best])


def run_episode(env, choose, Q=None, alpha=0.3, gamma=0.95, max_skills=40, seed=None, log=False):
    """One attempt at the job. With Q: learn from every skill (Q-learning update).
    Returns (return, number of skills, simulated seconds, finished, blocks standing)."""
    task = env.reset(seed)
    total, n = 0.0, 0
    for n in range(1, max_skills + 1):
        s = state_of(task)
        a = choose(s)
        new = env.do(action_of(s, a))
        done = finished(new) or hopeless(new)
        r = reward(new)
        if Q is not None:
            q = Q.setdefault(s, [0.0] * len(DZ))
            nxt = 0.0 if done else max(Q.get(state_of(new), [0.0]))
            q[a] += alpha * (r + gamma * nxt - q[a])
        total += r
        if log:
            last = new["last"]
            detail = (f"fingertips {last['tipsMm']:5.1f} mm above the block's bottom" if "tipsMm" in last else
                      f"gap {last['gapMm']:5.1f} mm" + (f", {last['offMm']} mm off" if "offMm" in last else "") if "gapMm" in last else "")
            print(f"  {s[0]:<5} dz {DZ[a]:+3d} mm  {last.get('result', ''):<10} {detail:<45} "
                  f"stack {new['source']}  standing {new['standing']}  reward {r:+6.1f}")
        task = new
        if done:
            break
    return total, n, env.sim_time, finished(task), task["standing"]


def save(Q, path, info):
    data = {"dz": DZ, "info": info, "q": [{"state": list(s), "values": [round(v, 4) for v in q]} for s, q in Q.items()]}
    with open(path, "w") as f:
        json.dump(data, f)


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {tuple(e["state"]): e["values"] for e in data["q"]}


def show_policy(Q):
    """The learned correction per state: what the arm would correct on its own idea of the height."""
    print("Learned height corrections:")
    for phase, levels in (("pick", range(BLOCKS, 0, -1)), ("place", range(BLOCKS))):
        cells = []
        for k in levels:
            q = Q.get((phase, k))
            cells.append(f"{k}: {DZ[max(range(len(DZ)), key=q.__getitem__)]:+d}" if q else f"{k}: ?")
        label = "pick  (blocks on the stack)" if phase == "pick" else "place (blocks already standing)"
        print(f"  {label:<32} " + "   ".join(cells))


def report(name, result):
    total, n, secs, ok, standing = result
    print(f"{name}: {'job done' if ok else 'NOT finished'}: {standing}/{BLOCKS} standing, {n} skills, "
          f"{secs:.0f} s simulated, return {total:.1f}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["baseline", "train", "eval", "watch", "real"])
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--episodes", type=int, default=300, help="training attempts (train)")
    ap.add_argument("--alpha", type=float, default=0.3, help="learning rate of the Q-update")
    ap.add_argument("--gamma", type=float, default=0.95, help="discount: how much later rewards count")
    ap.add_argument("--eps-end", type=float, default=0.05, help="exploring at the end of training (it starts at 1)")
    ap.add_argument("--max-skills", type=int, default=40, help="skills per attempt before it stops")
    ap.add_argument("--zerror", type=float, default=10.0, help="hidden calibration error: the arm is this many mm higher than it thinks")
    ap.add_argument("--zsag", type=float, default=4.0, help="hidden: it hangs this many mm lower per 100 mm further out")
    ap.add_argument("--speed", type=float, default=1.0, help="watch: × real time")
    ap.add_argument("--model", default="stand_q.json", help="the Q-table (JSON)")
    ap.add_argument("--seed", type=int, default=1)
    # real: the page in your browser, through the PHP API
    ap.add_argument("--session", help="real: the remote session code (the page's URL ?remote=CODE)")
    ap.add_argument("--remote-url", default="https://robotics.dinno.be/api/index.php",
                    help="real: the PHP API of the site where the page is open")
    ap.add_argument("--pick-dz", type=float, default=0.0, help="real: height correction for picking (mm)")
    ap.add_argument("--place-dz", type=float, default=0.0, help="real: height correction for placing (mm)")
    ap.add_argument("--yes", action="store_true", help="real: don't ask before each skill")
    args = ap.parse_args()
    random.seed(args.seed)

    if args.command == "real":
        if not args.session:
            sys.exit("real needs --session CODE (open indexOpenManipulator.php?remote=CODE in Chrome first)")
        if "--model" not in sys.argv:
            args.model = None
        run_real(args)
        return

    env = StandTask(args.url, args.zerror, args.zsag, watch_speed=args.speed if args.command == "watch" else 0.0)
    try:
        if args.command == "baseline":
            report("No correction (the page's own heights)",
                   run_episode(env, lambda s: DZ.index(0), max_skills=args.max_skills, log=True, seed=0))
            return

        if args.command == "train":
            Q, t0, recent = {}, time.time(), []
            for ep in range(args.episodes):
                # ε-greedy: explore a lot at first, then more and more use what was learned.
                eps = max(args.eps_end, 1.0 - ep / (0.6 * args.episodes))
                choose = lambda s: random.randrange(len(DZ)) if random.random() < eps else greedy(Q, s)  # noqa: E731
                total, n, secs, ok, standing = run_episode(env, choose, Q, args.alpha, args.gamma, args.max_skills, seed=ep)
                recent = (recent + [(total, ok, standing)])[-20:]
                if (ep + 1) % 10 == 0:
                    print(f"episode {ep + 1:4d}  ε {eps:.2f}  return {total:6.1f}  last 20: mean return "
                          f"{sum(t for t, _, _ in recent) / len(recent):6.1f}, job done {sum(o for _, o, _ in recent)}/{len(recent)}, "
                          f"mean standing {sum(s for _, _, s in recent) / len(recent):.1f}  {time.time() - t0:.0f} s", flush=True)
                if (ep + 1) % 50 == 0:
                    save(Q, args.model, {"episodes": ep + 1})
            save(Q, args.model, {"episodes": args.episodes, "seconds": round(time.time() - t0),
                                 "zerror": args.zerror, "zsag": args.zsag})
            print(f"Saved {args.model} ({len(Q)} states) after {time.time() - t0:.0f} s of training")
            show_policy(Q)
            return

        if not os.path.exists(args.model):
            sys.exit(f"No Q-table '{args.model}'. Train one first:  python {sys.argv[0]} train\n"
                     f"or use the trained one:  --model python/models/stand_q.json")
        Q = load(args.model)
        show_policy(Q)
        report("Learned policy", run_episode(env, lambda s: greedy(Q, s), max_skills=args.max_skills, log=True, seed=123))
    finally:
        env.close()


if __name__ == "__main__":
    main()
