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
    road = img[70:, :] > 0.45                  # onderste deel van het beeld: lichte pixels = weg
    if road.sum() < 50:                        # geen weg te zien: draai ter plaatse
        action = {"drive": {"fwd": 0.0, "turn": 0.6}}
    else:
        error = (np.nonzero(road)[1].mean() - 80) / 80     # −1 weg links … +1 weg rechts
        action = {"drive": {"fwd": 0.5, "turn": -1.5 * error}}
    r = api.step(eid, action, dt=0.05, image=CAM)
    jpg = r["image"]
api.close(eid)
