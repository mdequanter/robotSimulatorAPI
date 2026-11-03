"""Tutorial 19: teach YOLO to find ten objects in the house, with the TurtleBot3 and its OpenMANIPULATOR-X.

The robot stands on random free spots in the four rooms of the House with garden; its arm turns the camera on the gripper
to look around. Every picture comes with its labels from the simulator itself (the boxes of the objects in that same view,
what stands in front taken into account), so there is nothing to label by hand.

    python python/house_yolo.py collect              # pictures + labels -> house_data/ (train and val)
    python python/house_yolo.py train                # ultralytics YOLO11n on them -> runs/detect/house/weights/best.pt
    python python/house_yolo.py test                 # new spots (another seed): mAP per class, pictures with the boxes
    python python/house_yolo.py export               # best.pt -> python/models/house_yolo11n.pt (and --onnx)
    python python/house_yolo.py detect               # the model in the simulator: the robot looks around, the boxes on the page

Start the server first: node server.mjs --key rsk_...   (--show to watch). Needs: pip install ultralytics
"""
import argparse
import base64
import io
import os
import random
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from botopia import BotopiaAPI  # noqa: E402

# The ten objects, in the order of their class numbers (the item kinds of the world editor).
CLASSES = ["chair", "table", "desk", "sofa", "bed", "bookcase", "plant", "fridge", "toilet", "person"]
# The rooms of the house (world x0, z0, x1, z1 in metres): the robot is put on free spots in each of them in turn.
ROOMS = {"living room and kitchen": [-5.8, -3.8, -0.2, 3.8], "bedroom": [0.2, 0.2, 5.8, 3.8],
         "bathroom": [0.2, -3.8, 2.8, -0.2], "office": [3.2, -3.8, 5.8, -0.2]}
JOINTS = ["joint1", "joint2", "joint3", "joint4"]
WIDTH = 640                     # pictures of 640 x 480, as YOLO trains them
MIN_BOX, MIN_PX = 8, 60         # smaller boxes (pixels) or fewer visible pixels are left out: too little to learn from
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "house_data")
RUN = os.path.join(HERE, "runs", "detect", "house")
MODEL = os.path.join(HERE, "python", "models", "house_yolo11n.pt")


def arm_to(q, sec=0.8):
    """The arm to joint angles q (radians), as ros2 topic pub /arm_controller/joint_trajectory does."""
    return {"ros": {"kind": "topic", "name": "/arm_controller/joint_trajectory", "type": "trajectory_msgs/msg/JointTrajectory",
                    "msg": {"joint_names": JOINTS, "points": [{"positions": [float(v) for v in q],
                                                                "time_from_start": {"sec": int(sec), "nanosec": int(sec % 1 * 1e9)}}]}}}


def look_pose(rng):
    """A random way to look: turn (joint1) left or right, the arm up or down a little, the camera tilted (joint4)."""
    return [rng.uniform(-1.7, 1.7), rng.uniform(-0.45, 0.25), rng.uniform(-0.3, 0.3), rng.uniform(-0.55, 0.45)]


def open_house(api, seed):
    env = api.create("tb3manipulator")
    eid = env["id"]
    api.reset(eid, world="house", seed=seed)
    return eid


def yolo_lines(labels):
    """The boxes in YOLO's text format: class x_centre y_centre width height, as fractions of the picture."""
    W, H = labels["width"], labels["height"]
    out = []
    for b in labels["boxes"]:
        if b["w"] < MIN_BOX or b["h"] < MIN_BOX or b["px"] < MIN_PX:
            continue
        out.append(f"{b['cls']} {(b['x'] + b['w'] / 2) / W:.6f} {(b['y'] + b['h'] / 2) / H:.6f} {b['w'] / W:.6f} {b['h'] / H:.6f}")
    return out


def collect(api, spots, looks, seed, splits, out):
    """spots per room in turn; at each spot `looks` pictures from the gripper camera and one from the base camera."""
    rng = random.Random(seed)
    eid = open_house(api, seed)
    for s in splits:
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)
    counts, n, rooms = [0] * len(CLASSES), 0, list(ROOMS.items())
    t0 = time.time()
    for i in range(spots):
        room, area = rooms[i % len(rooms)]
        split = splits[0] if len(splits) == 1 else ("val" if i % 6 == 5 else "train")
        r = api.step(eid, {"place": {"random": True, "area": area}, **arm_to(look_pose(rng))}, dt=1.0)
        if not r.get("placed"):
            continue
        for k in range(looks + 1):
            # The first picture: the base camera (the Raspberry Pi camera, low at the front); then the gripper camera.
            view = "robot" if k == 0 else "gripper"
            action = {} if k == 0 else arm_to(look_pose(rng))
            r = api.step(eid, action, dt=1.0, image={"view": view, "width": WIDTH, "quality": 0.9, "labels": CLASSES})
            lines = yolo_lines(r["labels"])
            if not lines and rng.random() > 0.15:     # some pictures without any of the ten: they teach "nothing here"
                continue
            if any(b["w"] > 0.95 * WIDTH and b["h"] > 0.95 * WIDTH * 3 / 4 for b in r["labels"]["boxes"]):
                continue                              # the camera under a table or against a chair: one box is the whole picture
            name = f"{split}_{seed}_{n:05d}"
            with open(os.path.join(out, "images", split, name + ".jpg"), "wb") as f:
                f.write(r["image"])
            with open(os.path.join(out, "labels", split, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            for ln in lines:
                counts[int(ln.split()[0])] += 1
            n += 1
        if i % 20 == 19:
            print(f"  {i + 1}/{spots} spots, {n} pictures, {time.time() - t0:.0f} s", flush=True)
    api.close(eid)
    with open(os.path.join(out, "house.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        for s in ("train", "val", "test"):
            if os.path.isdir(os.path.join(out, "images", s)):
                f.write(f"{s}: images/{s}\n")
        f.write("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"{n} pictures in {out}; boxes per class:")
    for c, k in zip(CLASSES, counts):
        print(f"  {c:9s} {k}")


def draw(jpeg, boxes, names, color=(255, 230, 0)):
    """The picture with boxes [(cls, conf or None, x0, y0, x1, y1)] drawn on it, as JPEG bytes."""
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    d = ImageDraw.Draw(im)
    for cls, conf, x0, y0, x1, y1 in boxes:
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        text = names[cls] + (f" {conf:.2f}" if conf is not None else "")
        d.rectangle([x0, max(0, y0 - 16), x0 + 8 * len(text) + 6, max(16, y0)], fill=color)
        d.text((x0 + 3, max(0, y0 - 15)), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "train", "test", "export", "detect"])
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--spots", type=int, default=320, help="collect: spots (spread over the four rooms)")
    ap.add_argument("--looks", type=int, default=6, help="collect: gripper-camera pictures per spot")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default=None, help="train: 0 = the first GPU, cpu = without")
    ap.add_argument("--model", default=None, help="test / detect: the .pt file (default: the trained one)")
    ap.add_argument("--onnx", action="store_true", help="export: also an ONNX file")
    ap.add_argument("--seconds", type=float, default=60, help="detect: how long")
    a = ap.parse_args()
    api = BotopiaAPI(a.url)

    if a.cmd == "collect":
        collect(api, a.spots, a.looks, a.seed, ["train", "val"], DATA)

    elif a.cmd == "train":
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")                      # pre-trained on COCO: it already knows what objects look like
        model.train(data=os.path.join(DATA, "house.yaml"), epochs=a.epochs, imgsz=WIDTH, batch=16, device=a.device,
                    project=os.path.join(HERE, "runs", "detect"), name="house", exist_ok=True, plots=True)
        print("best weights:", os.path.join(RUN, "weights", "best.pt"))

    elif a.cmd == "test":
        from ultralytics import YOLO
        weights = a.model or os.path.join(RUN, "weights", "best.pt")
        # New spots the model never saw: another seed, its own split.
        shutil.rmtree(os.path.join(DATA, "images", "test"), ignore_errors=True)
        shutil.rmtree(os.path.join(DATA, "labels", "test"), ignore_errors=True)
        collect(api, 60, 4, a.seed + 1000, ["test"], DATA)
        model = YOLO(weights)
        m = model.val(data=os.path.join(DATA, "house.yaml"), split="test", imgsz=WIDTH, plots=True,
                      project=os.path.join(HERE, "runs", "detect"), name="house_test", exist_ok=True)
        print(f"\ntest: mAP50 {m.box.map50:.3f}, mAP50-95 {m.box.map:.3f}")
        for i, c in enumerate(CLASSES):
            if i in m.box.ap_class_index:
                j = list(m.box.ap_class_index).index(i)
                print(f"  {c:9s} AP50 {m.box.ap50[j]:.3f}")

    elif a.cmd == "export":
        from ultralytics import YOLO
        best = os.path.join(RUN, "weights", "best.pt")
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        shutil.copy(best, MODEL)
        print("copied", best, "->", MODEL)
        if a.onnx:
            print("ONNX:", YOLO(MODEL).export(format="onnx", imgsz=WIDTH))

    elif a.cmd == "detect":
        # The model in the simulator: the robot goes from room to room; at every spot the arm turns the camera slowly
        # from left to right. Every picture goes through YOLO, and the picture with its boxes is shown on the page
        # (action show), next to the simulator's own camera picture.
        from PIL import Image
        from ultralytics import YOLO
        model = YOLO(a.model or (MODEL if os.path.exists(MODEL) else os.path.join(RUN, "weights", "best.pt")))
        eid = open_house(api, a.seed + 2000)
        rng, rooms, shown, t_end = random.Random(a.seed), list(ROOMS.values()), None, time.time() + a.seconds
        while time.time() < t_end:
            api.step(eid, {"place": {"random": True, "area": rooms[rng.randrange(len(rooms))]}, **arm_to([-1.5, -0.2, 0, 0.1])}, dt=1.0)
            for pan in [x / 10 for x in range(-15, 16, 2)]:
                r = api.step(eid, {**arm_to([pan, -0.2, 0, 0.1], 0.3), **({"show": shown} if shown else {})}, dt=0.35,
                             image={"view": "gripper", "width": WIDTH, "quality": 0.85})
                res = model.predict(Image.open(io.BytesIO(r["image"])), imgsz=WIDTH, conf=0.4, verbose=False)[0]
                found = [(int(b.cls), float(b.conf), *[float(v) for v in b.xyxy[0]]) for b in res.boxes]
                label = "YOLO: " + (", ".join(f"{CLASSES[c]} {p:.2f}" for c, p, *_ in found) or "nothing")
                shown = {"image": base64.b64encode(draw(r["image"], found, CLASSES)).decode(), "label": label}
                print(label, flush=True)
        api.step(eid, {"show": shown}, dt=0.05)
        input("Press Enter to close the simulator...")
        api.close(eid)


if __name__ == "__main__":
    main()
