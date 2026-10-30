"""Python client for the Botopia robot simulators (reinforcement learning).

Talks to the Node API (server.mjs, step-by-step training) or to the PHP API (index.php,
remote control of a simulator open in a browser). Standard library only; numpy and gymnasium are
used when installed (BotopiaGymEnv).

    from botopia import BotopiaEnv
    env = BotopiaEnv(robot="skid", lane="step6")
    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step([1.0, 0.0])   # drive axes, see env.action_names

See api/README.md for the JSON formats.
"""
import base64
import http.client
import json
import urllib.parse

__all__ = ["BotopiaAPI", "BotopiaEnv", "BotopiaVecEnv", "RemoteSession", "BotopiaGymEnv"]


class _Http:
    """Small JSON-over-HTTP client that keeps its connection open (much faster per step)."""

    def __init__(self, base_url, timeout=120):          # the server retries a stuck page for up to 60 s
        u = urllib.parse.urlsplit(base_url)
        self.base = u.path.rstrip("/")
        self.conn_args = (u.hostname, u.port or (443 if u.scheme == "https" else 80))
        self.https = u.scheme == "https"
        self.timeout = timeout
        self.conn = None

    def _connect(self):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        self.conn = cls(*self.conn_args, timeout=self.timeout)

    def request(self, method, path, body=None, raw=False):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        for attempt in (1, 2):                      # reconnect once if the server closed the connection
            if self.conn is None:
                self._connect()
            try:
                self.conn.request(method, self.base + path, body=data, headers=headers)
                resp = self.conn.getresponse()
                payload = resp.read()
                break
            except (http.client.HTTPException, ConnectionError, OSError):
                self.conn.close()
                self.conn = None
                if attempt == 2:
                    raise
        if raw:
            return payload
        result = json.loads(payload) if payload else None
        if resp.status >= 400:
            raise RuntimeError(f"{method} {path}: HTTP {resp.status}: {result.get('error') if isinstance(result, dict) else result}")
        return result


class BotopiaAPI:
    """Node API (server.mjs): environments that only move when you call step()."""

    def __init__(self, url="http://localhost:8780"):
        self.http = _Http(url)

    def robots(self):
        return self.http.request("GET", "/api/robots")

    def create(self, robot, lane="free", seed=None, options=None):
        """Open a new environment. Returns {id, info, obs}. options: page settings, passed as URL
        parameters of the simulator page, e.g. {"vmax": 1.8} for the Ackermann's top speed."""
        return self.http.request("POST", "/api/envs", {"robot": robot, "lane": lane, "seed": seed, "options": options or {}})

    def reset(self, env_id, lane=None, seed=None, start=None, offset=0.0, yaw=0.0, image=None, world=None):
        """Back to the start of a lane. On the ε track: start = metres along it, offset = metres beside
        the centre line (+ = left), yaw = degrees turned away from the track direction. image: e.g.
        {"view": "robot", "width": 64} also returns that image, then the answer is (obs, image bytes).
        world: first to another environment ("course", "park", …: info["worlds"]), the robot starts there."""
        body = {"seed": seed}
        if world is not None:
            body["world"] = world
        if lane is not None:
            body["lane"] = lane
        if start is not None:
            body["start"] = start
        if offset:
            body["offset"] = offset
        if yaw:
            body["yaw"] = yaw
        if image:
            body["image"] = image
        r = self.http.request("POST", f"/api/envs/{env_id}/reset", body)
        return (r["obs"], base64.b64decode(r["image"])) if image else r["obs"]

    def step(self, env_id, action, dt=0.05, frame=1 / 60, image=None, until=None):
        """action: {"drive": {"fwd", "turn", "strafe"}} and/or {"motors": {id: value}}, or
        {"skill": name} on pages with skills (quadruped). Returns {obs, events, reward, done}; with
        image (e.g. {"view": "robot", "width": 64}) also "image": the JPEG bytes of that view after
        the step. until="idle": dt is the maximum, the step ends as soon as the skill has finished."""
        body = {"action": action, "dt": dt, "frame": frame}
        if until:
            body["until"] = until
        if image:
            body["image"] = image
        r = self.http.request("POST", f"/api/envs/{env_id}/step", body)
        if "image" in r:
            r["image"] = base64.b64decode(r["image"])
        return r

    def step_many(self, steps):
        """steps: [{"id", "action", "dt", "image"?}] → list of results, all environments in one call
        (each "image" as JPEG bytes)."""
        results = self.http.request("POST", "/api/envs/step", {"steps": steps})["results"]
        for r in results:
            if "image" in r:
                r["image"] = base64.b64decode(r["image"])
        return results

    def observe(self, env_id):
        return self.http.request("GET", f"/api/envs/{env_id}/obs")

    def frame(self, env_id, quality=0.8, view="scene", width=320):
        """JPEG bytes: view "scene" = the 3D view of the simulator, "robot" = what the robot's own
        front camera sees (width × width·3/4 pixels), "external" = a fixed camera beside the robot (xArm)."""
        return self.http.request("GET", f"/api/envs/{env_id}/frame?quality={quality}&view={view}&width={width}", raw=True)

    def close(self, env_id):
        return self.http.request("DELETE", f"/api/envs/{env_id}")


class RemoteSession:
    """PHP API (api/index.php): remote control of a simulator page opened with ?remote=CODE."""

    def __init__(self, url, session):
        self.http = _Http(url)
        self.url = url
        self.session = session
        self.next_event = 0

    @staticmethod
    def sessions(url):
        return _Http(url).request("GET", "?r=sessions")

    def state(self):
        """Latest info, observation and the incidents since the previous call."""
        s = self.http.request("GET", f"?r=state&session={self.session}&since={self.next_event}")
        self.next_event = s["next"]
        return s

    def command(self, action=None, reset=None):
        body = {}
        if action is not None:
            body["action"] = action
        if reset is not None:
            body["reset"] = reset
        return self.http.request("POST", f"?r=command&session={self.session}", body)

    def frame(self, fps=10, width=640):
        """The robot camera's latest picture: JPEG bytes, width × 3/4 (at most 640 × 480), or None while there
        is none yet. Asking keeps the simulator sending pictures, fps a second (it stops 5 s after the last ask)."""
        data = self.http.request("GET", f"?r=frame&session={self.session}&fps={fps}&width={width}", raw=True)
        return data if data[:2] == b"\xff\xd8" else None

    def stream_url(self, fps=10, width=640):
        """The robot camera as an MJPEG stream, e.g. cv2.VideoCapture(session.stream_url())."""
        return f"{self.url}?r=mjpeg&session={self.session}&fps={fps}&width={width}"


# ---------- Observation vector ----------
def features(obs):
    """Fixed-length list of numbers from an observation (same layout for every step of a robot):
    progress, body velocity (vx, vy, wz), roll, pitch, accelerometer (3), gyro (3), stuck flag,
    then per motor: DC → duty and rpm/100; servo → degrees/90."""
    f = [obs["progress"]["distance"], obs["velocity"]["vx"], obs["velocity"]["vy"], obs["velocity"]["wz"],
         obs["pose"]["roll"], obs["pose"]["pitch"], *[a / 9.81 for a in obs["imu"]["accel"]], *obs["imu"]["gyro"],
         1.0 if obs["stuck"] else 0.0]
    for m in obs["motors"]:
        if m["type"] == "dc":
            f += [m["duty"], m["rpm"] / 100.0]
        else:
            f += [m["deg"] / 90.0]
    return f


class BotopiaEnv:
    """One environment with the familiar reset() / step() interface.

    action_mode "drive": the action is one value per drive axis (env.action_names, −1 … 1).
    action_mode "motors": one value per controllable motor: DC duty −1 … 1, servo −1 … 1 (scaled to
    its range). The robot's own controller is bypassed for those motors.
    """

    def __init__(self, robot="skid", lane="free", url="http://localhost:8780", dt=0.05,
                 max_steps=400, action_mode="drive", api=None):
        self.api = api or BotopiaAPI(url)
        self.robot, self.lane, self.dt, self.max_steps, self.action_mode = robot, lane, dt, max_steps, action_mode
        created = self.api.create(robot, lane)
        self.id, self.info = created["id"], created["info"]
        self.last_obs = created["obs"]
        if action_mode == "drive":
            self.action_names = list(self.info["drive"])
        else:
            self.motors = [m for m in self.info["motors"] if m["overridable"]]
            if not self.motors:
                raise ValueError(f"{robot}: no directly controllable motors, use action_mode='drive'")
            self.action_names = [m["id"] for m in self.motors]
        self.steps = 0

    def to_action(self, a):
        a = [max(-1.0, min(1.0, float(v))) for v in a]
        if self.action_mode == "drive":
            return {"drive": dict(zip(self.action_names, a))}
        # Servos: −1 … 1 spans their range (not always symmetric, e.g. the quadruped's gripper 0–70 mm).
        return {"motors": {m["id"]: (v if m["type"] == "dc" else m["range"][0] + (v + 1) / 2 * (m["range"][1] - m["range"][0]))
                           for m, v in zip(self.motors, a)}}

    def reset(self, seed=None, lane=None):
        self.last_obs = self.api.reset(self.id, lane=lane or self.lane, seed=seed)
        self.steps = 0
        return features(self.last_obs), {"obs": self.last_obs}

    def step(self, action):
        r = self.api.step(self.id, self.to_action(action), dt=self.dt)
        self.last_obs = r["obs"]
        self.steps += 1
        terminated = bool(r["done"])
        truncated = self.steps >= self.max_steps and not terminated
        return features(r["obs"]), r["reward"], terminated, truncated, {"obs": r["obs"], "events": r["events"]}

    def frame(self, view="scene", width=320):
        """JPEG bytes of the 3D view, or with view="robot" of what the robot's front camera sees."""
        return self.api.frame(self.id, view=view, width=width)

    def close(self):
        self.api.close(self.id)


class BotopiaVecEnv:
    """Several environments stepped in one HTTP call (much higher throughput)."""

    def __init__(self, robots, lane="free", url="http://localhost:8780", dt=0.05, max_steps=400):
        self.api = BotopiaAPI(url)
        self.envs = [BotopiaEnv(r, lane, dt=dt, max_steps=max_steps, api=self.api) for r in robots]

    def reset(self, seed=None):
        return [e.reset(seed=None if seed is None else seed + i)[0] for i, e in enumerate(self.envs)]

    def step(self, actions):
        results = self.api.step_many([{"id": e.id, "action": e.to_action(a), "dt": e.dt} for e, a in zip(self.envs, actions)])
        out = []
        for e, r in zip(self.envs, results):
            if "error" in r:
                raise RuntimeError(f"env {e.id}: {r['error']}")
            e.last_obs = r["obs"]
            e.steps += 1
            terminated = bool(r["done"])
            truncated = e.steps >= e.max_steps and not terminated
            out.append((features(r["obs"]), r["reward"], terminated, truncated, {"events": r["events"]}))
        return out

    def close(self):
        for e in self.envs:
            e.close()


# ---------- Optional: Gymnasium ----------
try:
    import gymnasium as gym
    import numpy as np

    class BotopiaGymEnv(gym.Env):
        """Gymnasium wrapper: Box action space (−1 … 1) and Box observation space (float32)."""
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self, robot="skid", lane="free", url="http://localhost:8780", dt=0.05, max_steps=400,
                     action_mode="drive", render_mode=None):
            self.env = BotopiaEnv(robot, lane, url, dt, max_steps, action_mode)
            self.render_mode = render_mode
            n_obs = len(features(self.env.last_obs))
            self.action_space = gym.spaces.Box(-1.0, 1.0, (len(self.env.action_names),), np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            obs, info = self.env.reset(seed=seed, lane=(options or {}).get("lane"))
            return np.asarray(obs, np.float32), info

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(np.asarray(action).tolist())
            return np.asarray(obs, np.float32), float(r), term, trunc, info

        def render(self):
            if self.render_mode == "rgb_array":
                from io import BytesIO
                try:
                    from PIL import Image
                except ImportError as e:
                    raise RuntimeError("render needs Pillow") from e
                return np.asarray(Image.open(BytesIO(self.env.frame())).convert("RGB"))

        def close(self):
            self.env.close()
except ImportError:                                  # gymnasium / numpy not installed
    BotopiaGymEnv = None


if __name__ == "__main__":
    # Example: a random driver on the 6 cm step, then the same with 4 environments at once.
    import random
    import time
    env = BotopiaEnv(robot="skid", lane="step6")
    obs, _ = env.reset(seed=1)
    total, t0 = 0.0, time.time()
    for k in range(200):
        obs, r, term, trunc, info = env.step([random.uniform(0, 1), random.uniform(-0.3, 0.3)])
        total += r
        for e in info["events"]:
            print(f"  t={e['t']:.2f}s  {e['type']}  {e.get('reason', '')}")
        if term or trunc:
            break
    print(f"{env.robot}: {k + 1} steps, total reward {total:.2f}, {(k + 1) / (time.time() - t0):.0f} steps/s, "
          f"obs vector length {len(obs)}, actions {env.action_names}")
    env.close()

    vec = BotopiaVecEnv(["skid", "tracked", "ackermann", "ostrich"], lane="step6")
    vec.reset(seed=1)
    t0 = time.time()
    for k in range(100):
        vec.step([[1.0, 0.0] if "strafe" not in e.action_names else [1.0, 0.0, 0.0] for e in vec.envs])
    print(f"4 environments: {400 / (time.time() - t0):.0f} steps/s together")
    for e in vec.envs:
        print(f"  {e.robot:10s} x={e.last_obs['pose']['x']:.2f}  goal={e.last_obs['progress']['goal']}  stuck={e.last_obs['stuck']!r}")
    vec.close()
