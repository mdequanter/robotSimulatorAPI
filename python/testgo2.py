import sys, time; sys.path.append("python")
from botopia import BotopiaAPI
api = BotopiaAPI("http://localhost:8780")
env = api.create("go2", lane="stairs"); eid = env["id"]
DT = 0.05
for i in range(80):                                     # 100 × 0.05 s = 5 s
    t0 = time.perf_counter()
    r = api.step(eid, {"drive": {"fwd": 1.0}}, dt=DT)  # drive axes are −1 … 1
    if i % 10 == 0:
        print(i, r["obs"]["pose"], r["events"])
    time.sleep(max(0, DT - (time.perf_counter() - t0)))   # real time: wait out the rest of the step
input("Press Enter to close the simulator...")    # keep the screen open until you are done watching
api.close(eid)
