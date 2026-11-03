import numpy as np

import io, sys
sys.path.append("python")
from botopia import BotopiaAPI
from PIL import Image
api = BotopiaAPI("http://localhost:8780")

CAM = {"view": "robot", "width": 160}
env = api.create("ackermann", lane="track"); eid = env["id"]
obs, jpg = api.reset(eid, lane="track", image=CAM)

for step in range(20000):
    img = np.asarray(Image.open(io.BytesIO(jpg)).convert("L"), dtype=np.float32) / 255
    road = img[70:, :] > 0.45                  # lower part of the picture: light pixels = road
    if road.sum() < 50:                        # no road in sight: turn on the spot
        action = {"drive": {"fwd": 0.0, "turn": 0.6}}
    else:
        error = (np.nonzero(road)[1].mean() - 80) / 80     # −1 road to the left … +1 road to the right
        action = {"drive": {"fwd": 0.5, "turn": -1.5 * error}}
    r = api.step(eid, action, dt=0.05, image=CAM)
    jpg = r["image"]
api.close(eid)
