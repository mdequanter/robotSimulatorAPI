"""Collect what every simulator offers into robots.json, read by the API documentation page
(api.html of the simulator site). Run it again after adding or changing a robot. It needs the
simulator site on disk (--site, default: the folder above this repository, as on the web server):

    node server.mjs --port 8795 --browsers 1 --site ../robotSimulator     (in another terminal)
    python python/make_api_docs.py --url http://localhost:8795 --site ../robotSimulator

For every robot it opens the simulator through the Node API and stores its info (drive axes,
motors, events, skills, cameras and every console button as a ready-made action), the names and
sample values of its telemetry (obs.extra), the card text of index.php and a few extra examples
for actions that have no console button.
"""
import argparse, html, json, os, re, sys
sys.path.append(os.path.dirname(__file__))
from botopia import BotopiaAPI

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # this repository
ROOT = os.path.dirname(HERE)                                                    # the simulator site (--site)

# Interface families, by the key of their messages in an action.
FAMILY = {"webrtc": "Unitree WebRTC", "ros": "ROS 2", "serial": "Serial protocol", "sdk": "Maker SDK",
          "python": "Python library", "xgo": "xgolib (Python)", "booster": "Booster SDK", "ws": "WebSocket API", "tcp": "TCP text protocol",
          "tello": "Tello SDK", "urscript": "URScript", "rest": "REST API (HTTP)"}

# Actions that are not console buttons (page-specific action fields), per robot.
EXTRA = {
    "rbkairos": [("Arm: Cartesian goal", {"arm": {"tcp": {"x": 0.5, "y": 0.0, "z": 0.9, "yaw": 0}}}),
                 ("Arm: pick a block", {"arm": {"skill": "pick"}}), ("Arm: place it", {"arm": {"skill": "place"}}),
                 ("Gripper 40 mm", {"arm": {"grip": 40}})],
    "sespro": [("Servo 2 to 45.0° in 1.5 s", {"serial": "#2D450T1500"}), ("Ask the positions", {"serial": "#1QD#2QD#3QD"}),
               ("All servos limp", {"serial": "#254L"}), ("All servos hold", {"serial": "#254H"}),
               ("Gripper open", {"serial": "#6D900"}), ("Servo 3 LED blue", {"serial": "#3LED3"})],
}
EXTRA["rbvogui"] = EXTRA["rbkairos"]
EXTRA["ur5e"] = [("Home, facing the table", {"urscript": "movej([3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0], a=1.4, v=1.05)"}),
                 ("Straight line down", {"urscript": "movel(p[0.35, 0, 0.2, 3.1416, 0, 0], a=1.2, v=0.25)"}),
                 ("Gripper half closed", {"urscript": "rq_move_and_wait(128)"}),
                 ("Program: wait, then turn the base", {"urscript": "sleep(1.0)\nmovej([d2r(150), -1.5708, 1.5708, -1.5708, -1.5708, 0])"}),
                 ("Read the tool pose", {"urscript": "textmsg(get_actual_tcp_pose())"})]
EXTRA["fr3"] = [("Ready pose", {"python": "robot.move(JointMotion([0, -0.785, 0, -2.356, 0, 1.571, 0.785]))"}),
                ("Straight line, hand down", {"python": "robot.move(CartesianMotion(Affine([0.4, 0.0, 0.2], [1, 0, 0, 0])))"}),
                ("10 cm to the left (relative)", {"python": "robot.move(CartesianMotion(Affine([0.0, 0.1, 0.0]), ReferenceType.Relative))"}),
                ("Slower", {"python": "robot.relative_dynamics_factor = 0.1"}),
                ("Grasp 2 cm", {"python": "gripper.grasp(0.02, 0.1, 20)"}),
                ("Read the joints", {"python": "print(robot.current_joint_state.position)"})]


def cards():
    """index.php cards: page → (title, text, category, image)."""
    s = open(os.path.join(ROOT, "index.php"), encoding="utf-8").read()
    # The groups of robots: <section class="group"> with <h3>Title <span>subtitle</span></h3> (older
    # pages: an <h2> per group).
    out, cat = {}, "Drive types"
    for part in re.split(r'(<div class="group-head">\s*<h3>.*?</h3>|<h2[^>]*>.*?</h2>)', s, flags=re.S):
        m = re.match(r'<div class="group-head">\s*<h3>(.*?)</h3>|<h2[^>]*>(.*?)</h2>', part, re.S)
        if m:
            title = re.sub(r"<span>.*?</span>", "", m.group(1) or m.group(2), flags=re.S)
            cat = html.unescape(re.sub("<[^>]+>", "", title)).strip()
            continue
        for c in re.findall(r'<article class="card">(.*?)</article>', part, re.S):
            h = re.search(r'<h3><a href="([^"]+)">(.*?)</a></h3>', c)
            t = re.search(r"<p>(.*?)</p>", c, re.S)
            im = re.search(r'<img src="([^"]+)"', c)
            if h:
                out[h.group(1)] = (html.unescape(h.group(2)), html.unescape(re.sub("<[^>]+>", "", t.group(1))) if t else "", cat,
                                   im.group(1) if im else "")
    return out


def short(v, depth=0):
    """A sample value, cut down for the documentation."""
    if isinstance(v, list):
        return [short(x, depth + 1) for x in v[:4]] + (["…"] if len(v) > 4 else [])
    if isinstance(v, dict):
        if depth > 2:
            return "{…}"
        return {k: short(x, depth + 1) for k, x in list(v.items())[:12]}
    if isinstance(v, float):
        return round(v, 3)
    return v


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8795")
    ap.add_argument("--out", default=os.path.join(HERE, "robots.json"))
    ap.add_argument("--site", default=ROOT, help="the simulator site folder (index.php, img/)")
    ap.add_argument("--robots", nargs="+", help="only these robots: their entries in --out are replaced or added, the rest stays")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):              # a Windows console (cp1252) can't print → or ε
        sys.stdout.reconfigure(errors="replace")
    ROOT = a.site
    if not os.path.exists(os.path.join(ROOT, "index.php")):
        sys.exit(f"No simulator site in {ROOT}: pass --site path/to/robotSimulator")
    api = BotopiaAPI(a.url)
    card = cards()
    robots, lanes, worlds = [], None, None
    for r in api.robots():
        rid, page = r["robot"], r["page"]
        if a.robots and rid not in a.robots:
            continue
        try:
            e = api.create(rid)
        except Exception as ex:                          # a page that does not load: report and go on
            print(f"{rid:12s} ERROR {ex}")
            continue
        info, obs = e["info"], e["obs"]
        api.close(e["id"])
        lanes = lanes or info.get("lanes")
        worlds = worlds or info.get("worlds")
        base = page.split("?")[0]
        title, text, cat, img = card.get(page) or card.get(base) or (info.get("name", rid), "", "Other", "")
        own = f"img/{rid}.jpg"                           # a variant with its own picture
        if os.path.exists(os.path.join(ROOT, own)):
            img = own
        cmds = info.get("commands", [])
        keys = sorted({k for c in cmds for k in c["action"]} | {k for _, x in EXTRA.get(rid, []) for k in x if k in FAMILY})
        robots.append({
            "robot": rid, "page": page, "name": info.get("name", rid), "title": title, "text": text, "category": cat, "image": img,
            "interfaces": [{"key": k, "family": FAMILY[k]} for k in keys if k in FAMILY],
            "drive": info.get("drive", []), "motors": info.get("motors", []), "events": info.get("events", []),
            "skills": info.get("skills", []), "views": info.get("views", []), "worlds": "worlds" in info,
            "commands": cmds + [{"group": "More actions", "label": l, "action": x} for l, x in EXTRA.get(rid, [])],
            "telemetry": short(obs.get("extra", {})),
        })
        print(f"{rid:12s} {len(cmds):3d} commands  {', '.join(keys) or '-'}")
    if a.robots and os.path.exists(a.out):              # merge: the other robots stay as they were, in their order
        old = json.load(open(a.out, encoding="utf-8"))
        new = {r["robot"]: r for r in robots}
        merged = [new.pop(r["robot"], r) for r in old.get("robots", [])]
        robots = merged + list(new.values())
        lanes, worlds = old.get("lanes") or lanes, old.get("worlds") or worlds
    json.dump({"lanes": lanes, "worlds": worlds, "robots": robots}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(robots)} robots → {a.out}")


if __name__ == "__main__":
    main()
