"""The same pick-and-place code for the simulator and for the real xArm (6 or 7).

    python xarm_demo.py --robot sim                                   # the xArm 7 simulator (Node API running)
    python xarm_demo.py --robot sim --model 6                         # the xArm 6 simulator
    python xarm_demo.py --robot fake                                  # a pretend arm in Python
    python xarm_demo.py --robot 10.2.172.20 --pick 0.35,0.10 --place 0.35,-0.10
                                                                      # the real arm (4 cm block at --pick)

In the simulator, without --pick, it takes the block nearest to the arm and puts it on another one.
With a depth camera (the simulator, or a real RealSense D435 on the wrist) it first looks at the place
with the depth picture and measures how high the stack there is, and puts the block down on top of it
(--no-depth: trust --under instead). The depth pictures are saved as xarm_*_depth.png.
Positions are metres in the base frame: x forward, y left, from the middle of the arm's base.
Start the real arm slowly (--speed 20) and with nothing else on the table.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xarm_robot import connect, depth_colormap, MIN_Z, TIP, TOOL  # noqa: E402

BLOCK = 0.04                                   # 4 cm blocks
GRASP_Z = TIP - TOOL + MIN_Z + 0.002           # gripper point: the fingertips 7 mm above the table


LOOK_Z = 0.30                                  # gripper point height for a look with the wrist camera


def stack_height(arm, x, y, yaw=0.0, speed=30):
    """Look down at (x, y) with the wrist RealSense and measure how high what lies there is (m). The wrist
    camera is turned so the spot under the gripper is in the middle of the picture, and it looks a little
    slanted: the table's depth changes from the top to the bottom of the picture, but not along a row.
    So: the top in the middle minus the table beside it, left and right, in the same rows (the far
    quarter there: other blocks are nearer). None without a depth camera."""
    arm.move_to(x, y, LOOK_Z, yaw, speed)
    mm = arm.depth("wrist")
    if mm is None:
        return None
    h, w = mm.shape
    # Height above the table per pixel: the table's depth in each row (the far quarter of that row: other
    # blocks are nearer) minus the depth there. The highest spot round the middle is the top (a high
    # stack's top moves away from the middle: the camera looks slanted).
    d = mm.astype(np.float32)
    d[d == 0] = np.nan
    table = np.nanpercentile(d, 75, axis=1)[:, None]
    height = table - d
    win = height[h // 2 - 80:h // 2 + 80, w // 2 - 80:w // 2 + 80]
    if np.all(np.isnan(win)):
        return None
    top = np.nanpercentile(win, 99)                    # the highest spot (a few pixels: not one stray one)
    return max(0.0, float(top) / 1000.0)


def square(yaw):
    """A block looks the same every 90°: the turn between −45° and 45°."""
    return (yaw + 45) % 90 - 45


def pick_and_place(arm, pick, place, yaw_pick=0.0, yaw_place=0.0, under=0.0, speed=30):
    """Pick a block at pick=(x, y) and put it at place=(x, y), on top of `under` metres of blocks.
    yaw: how the block is turned (degrees), so the fingers close on two of its sides."""
    yaw = square(yaw_pick)
    x, y = pick
    arm.move_to(x, y, 0.20, yaw, speed)                    # above the block
    arm.set_gripper(75)
    arm.move_to(x, y, GRASP_Z, yaw, speed / 2)             # down, slowly
    arm.set_gripper(0)                                     # close: stops on the block
    arm.move_to(x, y, 0.20, yaw, speed)                    # lift
    x, y = place
    yaw = square(yaw_place)
    arm.move_to(x, y, 0.20, yaw, speed)
    arm.move_to(x, y, GRASP_Z + under + 0.003, yaw, speed / 2)
    arm.set_gripper(75)                                    # let go
    arm.move_to(x, y, 0.20, yaw, speed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="sim", help="sim, fake, or the arm's IP address")
    ap.add_argument("--url", default="http://localhost:8780", help="the Node API (sim)")
    ap.add_argument("--model", type=int, default=7, choices=[6, 7], help="sim / fake: xArm 6 or 7 (the real arm: asked)")
    ap.add_argument("--pick", help="x,y of the block (m)")
    ap.add_argument("--place", help="x,y where to put it (m)")
    ap.add_argument("--speed", type=float, default=30, help="°/s")
    ap.add_argument("--under", type=float, default=0.0, help="height already at --place (m), without depth")
    ap.add_argument("--no-depth", action="store_true", help="don't measure the stack with the depth camera")
    args = ap.parse_args()
    xy = lambda s: tuple(float(v) for v in s.split(","))  # noqa: E731

    arm = connect(args.robot, model=args.model, **({"url": args.url} if args.robot == "sim" else {}))
    try:
        under, yaw_pick, yaw_place = args.under, 0.0, 0.0
        if args.pick and args.place:
            pick, place = xy(args.pick), xy(args.place)
        elif args.robot == "sim":
            # Only the simulator knows where the blocks are (a real robot would use its camera).
            blocks = sorted(arm.blocks(), key=lambda b: b["x"] ** 2 + b["y"] ** 2)
            a, b = blocks[0], blocks[1]
            pick, place, under = (a["x"], a["y"]), (b["x"], b["y"]), BLOCK
            yaw_pick, yaw_place = a["yawDeg"], b["yawDeg"]
            print(f"Putting the {a['color']} block on the {b['color']} one")
        else:
            sys.exit("Give --pick x,y and --place x,y for this robot")
        if not args.no_depth:
            seen = stack_height(arm, *place, yaw_place, args.speed)
            if seen is not None:
                n = round(seen / BLOCK)
                print(f"Depth camera: {seen * 1000:.0f} mm high at the place ({n} block{'s' if n != 1 else ''})"
                      + (f"; the simulator says {under * 1000:.0f} mm" if args.robot == "sim" and not args.pick else ""))
                under = n * BLOCK
        print("Start:", [round(v, 1) for v in arm.joints()], "gripper point", [round(v, 3) for v in arm.tcp()])
        pick_and_place(arm, pick, place, yaw_pick, yaw_place, under=under, speed=args.speed)
        print("Done: ", [round(v, 1) for v in arm.joints()], "gripper point", [round(v, 3) for v in arm.tcp()])
        for which in ("wrist", "external"):
            jpg = arm.camera(which)
            if jpg:
                open(f"xarm_{which}.jpg", "wb").write(jpg)
                print(f"{which.capitalize()} camera image saved as xarm_{which}.jpg")
            mm = arm.depth(which)
            if mm is not None:
                from PIL import Image
                Image.fromarray(depth_colormap(mm)).save(f"xarm_{which}_depth.png")
                print(f"{which.capitalize()} depth saved as xarm_{which}_depth.png "
                      f"(nearest {mm[mm > 0].min()} mm, farthest {mm.max()} mm)")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
