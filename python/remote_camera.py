"""The robot camera of a simulator in the browser, on your own computer: process the pictures here and
drive the robot back. Through the website API (index.php, ?r=frame and ?r=command), so over https.

Open a simulator with ?remote=1, e.g. https://robotics.dinno.be/indexSkid.php?remote=1 (keep its window
visible: browsers pause hidden tabs), and pass the session code it shows:

    pip install opencv-python numpy
    python remote_camera.py --session AB12CD                  # drive with W A S D, space stops, Esc ends
    python remote_camera.py --session AB12CD --follow         # follows the earth path of the park on its own

--follow is a small example of your own processing: the yellow-brown path (world park: Setup → Environment)
as a mask, steering towards its middle in the lower half of the picture. Start on the path (lane "track").
The same pictures in the browser: https://robotics.dinno.be/stream.html?session=AB12CD&fps=10
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from botopia import RemoteSession  # noqa: E402

KEYS = {ord("w"): (0.6, 0.0), ord("s"): (-0.4, 0.0), ord("a"): (0.3, 0.8), ord("d"): (0.3, -0.8), ord(" "): (0.0, 0.0)}


def path_mask(img):
    """The earth path of the park (yellow-brown) as a 0/255 mask; the grass is greener, the hedges darker."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (8, 80, 40), (32, 255, 255))           # hue ~20; grass and hedges are ~55
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def steer_to(mask):
    """Turn (−1 … 1, + = left) towards the middle of the path in the lower half, or None without a path."""
    h, w = mask.shape
    low = mask[h // 2:]
    xs = np.nonzero(low)[1]
    if len(xs) < 0.01 * low.size:
        return None
    err = (xs.mean() - w / 2) / (w / 2)                    # −1 left … +1 right of the middle
    return float(np.clip(-1.6 * err, -1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="https://robotics.dinno.be/api/index.php", help="the website API")
    ap.add_argument("--session", required=True, help="the session code the simulator shows")
    ap.add_argument("--fps", type=int, default=10, help="pictures a second (1 … 30)")
    ap.add_argument("--width", type=int, default=640, help="picture width (160 … 640, 4:3)")
    ap.add_argument("--follow", action="store_true", help="follow the earth path of the park on its own")
    ap.add_argument("--speed", type=float, default=0.4, help="--follow: forward command (0 … 1)")
    a = ap.parse_args()

    s = RemoteSession(a.url, a.session.upper())
    print(f"Robot: {s.state()['info'].get('name')}  (Esc ends)")
    fwd = turn = 0.0
    sent = None
    try:
        while True:
            t0 = time.time()
            jpeg = s.frame(a.fps, a.width)
            if jpeg is None:                                   # the first picture is on its way
                time.sleep(0.2)
                continue
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if a.follow:
                mask = path_mask(img)
                t = steer_to(mask)
                fwd, turn = (a.speed, t) if t is not None else (0.0, 0.0)
                img = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            cv2.putText(img, f"fwd {fwd:+.2f}  turn {turn:+.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(f"Robot camera {a.session}", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in KEYS and not a.follow:
                fwd, turn = KEYS[key]
            if (fwd, turn) != sent:                            # a command only when it changes
                s.command({"drive": {"fwd": fwd, "turn": turn}})
                sent = (fwd, turn)
            time.sleep(max(0.0, 1 / a.fps - (time.time() - t0)))
    finally:
        s.command({"drive": None})                             # the keys of the page again
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
