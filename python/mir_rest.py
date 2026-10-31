"""The MiR REST API (v2.0.0) from Python, with the same calls for a real MiR robot and for the MiR250 simulator.

Test a program in the simulator, then run it unchanged on the robot: only the line that makes the MiR differs.

    from mir_rest import MiR
    mir = MiR("192.168.12.20", "distributor", "password")           # a real robot (its web interface's user)

    from botopia import BotopiaAPI, RemoteSession
    api = BotopiaAPI("http://localhost:8780")                         # the simulator, step by step (node server.mjs)
    env = api.create("mir250", options={"fleet": 3})                  # 3 robots: 192.168.12.20, .21, .22
    mir = MiR.simulator(api, env["id"], "192.168.12.21")

    mir = MiR.remote(RemoteSession(url, "CODE"), "192.168.12.21")     # the page in a browser, opened with ?remote=CODE

    mir.queue("Move to Station A")         # a mission by name (or its guid)
    mir.ready()                            # state 3: go on (and clear an error); mir.pause(): state 4
    mir.wait(10)                           # seconds: simulated in the step-by-step simulator, real otherwise
    print(mir.status()["state_text"], mir.status()["position"])
    code, data = mir.request("GET", "/mission_queue")                 # any request: (HTTP status, JSON)

Standard library only.
"""
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request

__all__ = ["MiR", "MiRError"]


class MiRError(RuntimeError):
    """An answer with an HTTP status of 400 or more (code, data), or no answer at all (code 0)."""

    def __init__(self, code, data, what):
        self.code, self.data = code, data
        human = data.get("error_human") if isinstance(data, dict) else None
        super().__init__(f"{what}: HTTP {code}" + (f": {human}" if human else ""))


class MiR:
    def __init__(self, ip, user="distributor", password="", timeout=10):
        """A real robot at `ip` (on the robot's own Wi-Fi: 192.168.12.20)."""
        self.ip = ip
        self._send = self._http
        self._sleep = time.sleep
        key = base64.b64encode(f"{user}:{hashlib.sha256(password.encode()).hexdigest()}".encode()).decode()
        self._headers = {"Authorization": "Basic " + key, "Accept-Language": "en_US", "Content-Type": "application/json"}
        self._timeout = timeout

    # ---------- The three ways to reach a MiR ----------
    @classmethod
    def simulator(cls, api, env_id, ip="192.168.12.20", dt=0.1):
        """Robot `ip` of the simulator in the Node API (BotopiaAPI, env_id from api.create("mir250")). Time only runs
        while you step: wait() steps the simulator."""
        m = cls.__new__(cls)
        m.ip = ip

        def send(method, path, body):
            action = {"rest": {"method": method, "path": path, "robot": ip, **({"body": body} if body is not None else {})}}
            r = api.step(env_id, action, dt=0.05)
            for e in r["events"]:
                if e.get("type") == "rest" and e.get("method") == method and e.get("path") == path:
                    return e.get("code", 0), e.get("data")
            return 0, None

        def sleep(seconds):
            n = max(1, round(seconds / dt))
            for _ in range(n):
                api.step(env_id, {}, dt=dt)

        m._send, m._sleep = send, sleep
        return m

    @classmethod
    def remote(cls, session, ip="192.168.12.20", timeout=5.0):
        """Robot `ip` of the simulator page open in a browser with ?remote=CODE (RemoteSession of botopia.py)."""
        m = cls.__new__(cls)
        m.ip = ip
        session.state()                                          # only answers from now on

        def send(method, path, body):
            session.command({"rest": {"method": method, "path": path, "robot": ip, **({"body": body} if body is not None else {})}})
            end = time.time() + timeout
            while time.time() < end:
                for e in session.state().get("events", []):
                    if e.get("type") == "rest" and e.get("method") == method and e.get("path") == path and e.get("robot", ip) == ip:
                        return e.get("code", 0), e.get("data")
                time.sleep(0.1)
            return 0, None

        m._send, m._sleep = send, time.sleep
        return m

    def _http(self, method, path, body):
        req = urllib.request.Request(f"http://{self.ip}/api/v2.0.0{path}", method=method, headers=self._headers,
                                     data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                text = r.read().decode()
                return r.status, json.loads(text) if text else None
        except urllib.error.HTTPError as e:
            text = e.read().decode()
            try:
                return e.code, json.loads(text) if text else None
            except ValueError:
                return e.code, text
        except OSError as e:
            return 0, {"error_human": f"no answer from {self.ip}: {e}"}

    # ---------- Requests ----------
    def request(self, method, path, body=None):
        """Any request, e.g. ("GET", "/status"): (HTTP status, the JSON answer). Status 0: no answer."""
        return self._send(method.upper(), path, body)

    def _ok(self, method, path, body=None):
        code, data = self.request(method, path, body)
        if not 200 <= code < 300:
            raise MiRError(code, data, f"{method} {path} on {self.ip}")
        return data

    def get(self, path):
        return self._ok("GET", path)

    def post(self, path, body):
        return self._ok("POST", path, body)

    def put(self, path, body):
        return self._ok("PUT", path, body)

    def delete(self, path):
        return self._ok("DELETE", path)

    # ---------- Often used ----------
    def status(self):
        """GET /status: state_id, state_text, mission_text, position {x, y, orientation}, battery_percentage, errors …"""
        return self.get("/status")

    def ready(self):
        """PUT /status state 3: go on (runs the queue, clears an error)."""
        return self.put("/status", {"state_id": 3})

    def pause(self):
        return self.put("/status", {"state_id": 4})

    def missions(self):
        return self.get("/missions")

    def positions(self):
        return self.get("/positions")

    def mission_id(self, name):
        """The guid of the mission with this name ("Move to Station A")."""
        for m in self.missions():
            if m["name"] == name:
                return m["guid"]
        raise KeyError(f"no mission {name!r} on {self.ip}")

    def queue(self, mission):
        """POST /mission_queue: the mission by name or guid. Returns the queue entry {id, mission_id, state}."""
        guid = mission if any(m["guid"] == mission for m in self.missions()) else self.mission_id(mission)
        return self.post("/mission_queue", {"mission_id": guid})

    def mission_queue(self):
        return self.get("/mission_queue")

    def clear_queue(self):
        return self._ok("DELETE", "/mission_queue")

    def wait(self, seconds):
        """Let time pass: simulated seconds in the step-by-step simulator, real seconds otherwise."""
        self._sleep(seconds)

    def wait_done(self, entry, timeout=120, every=0.5):
        """Wait until queue entry `entry` (from queue()) is Done, Aborted or Failed; returns its state."""
        t = 0.0
        while t < timeout:
            state = self.get(f"/mission_queue/{entry['id']}")["state"]
            if state in ("Done", "Aborted", "Failed"):
                return state
            self.wait(every)
            t += every
        return "Timeout"
