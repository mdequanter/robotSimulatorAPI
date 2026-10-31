# Botopia simulator API

Control the Botopia robot simulators from your own code: drive a robot, read back what its motors,
encoders, IMU and camera report, and get incidents such as collisions or a fall. Use it to script a
robot, or to train reinforcement-learning models.

The simulators themselves run on [https://robotics.dinno.be/](https://robotics.dinno.be/). This repository holds what you need on your own PC:

| | |
|---|---|
| `server.mjs` | the **API server** (Node.js): runs the simulators step by step in a hidden Chrome |
| `python/botopia.py` | the **Python client** (standard library only) |
| `python/*.py` | example scripts and the scripts of every tutorial |
| `python/models/` | trained models, to try the tutorials without training first |
| `index.php` | the PHP API for remote control; it runs on the web server (https://robotics.dinno.be/api/index.php) |
| `robots.json` | every robot with its commands, read by [api.html](https://robotics.dinno.be/api.html) |

New to all of this? Follow [Getting started](https://robotics.dinno.be/tutorial-getting-started.html) first: installing
the tools, Python with or without a GPU, your first program.

## Contents

1. [Getting started](#getting-started)
2. [Example scripts and tutorials](#example-scripts-and-tutorials)
3. [Two APIs: training or remote control](#two-apis-training-or-remote-control)
4. [The API server](#the-api-server)
5. [The Python client](#the-python-client)
6. [Actions](#actions)
7. [Observation, events and reward](#observation-events-and-reward)
8. [Lanes and the robot camera](#lanes-and-the-robot-camera)
9. [PHP API (remote control)](#php-api-remote-control)
10. [In the page (JavaScript)](#in-the-page-javascript)
11. [Robot reference](#robot-reference)
12. [Notes](#notes)
13. [For maintainers](#for-maintainers)

---

## Getting started

You need **Node.js 22 or newer**, **Google Chrome** (or Chromium / Edge) and **Python 3.10 or newer**.
There is no `npm install`. Run all commands from inside this folder.

```bash
git clone https://github.com/mdequanter/robotSimulatorAPI.git
cd robotSimulatorAPI
node server.mjs --show --key rsk_xxxxxxxx   # terminal 1; leave it running (--show: watch in a Chrome window)
```

The robots on https://robotics.dinno.be/ need an account (a free first day, then a subscription). Sign in
there, open **Account** and make an API key; pass it with `--key` or set the environment variable `ROBOTSIM_KEY`.

```text
Botopia simulator API on http://localhost:8780/api/  (simulators: http://localhost:8780/)
Simulator pages from https://robotics.dinno.be/
```

### A script that drives a robot (no training)

The simulation only moves when your script calls `step`. This drives the Unitree Go2 up the stairs in
real time ([`python/testgo2.py`](python/testgo2.py)):

```python
import sys, time; sys.path.append("python")
from botopia import BotopiaAPI

api = BotopiaAPI("http://localhost:8780")
env = api.create("go2", lane="stairs"); eid = env["id"]
DT = 0.05
for i in range(80):                                     # 80 × 0.05 s = 4 s
    t0 = time.perf_counter()
    r = api.step(eid, {"drive": {"fwd": 1.0}}, dt=DT)  # drive axes are −1 … 1
    if i % 10 == 0:
        print(i, r["obs"]["pose"], r["events"])
    time.sleep(max(0, DT - (time.perf_counter() - t0)))   # real time: wait out the rest of the step
input("Press Enter to close the simulator...")    # keep the screen open until you are done watching
api.close(eid)
```

```bash
python python/testgo2.py      # terminal 2
```

Leave out the `time.sleep` and it runs as fast as your PC can (hundreds of steps per second).

### A training loop

`BotopiaEnv` has the familiar Gymnasium-style `reset()` / `step()`:

```python
import sys; sys.path.append("python")
from botopia import BotopiaEnv

env = BotopiaEnv(robot="ostrich", lane="stairs", dt=0.05, max_steps=400)
obs, info = env.reset(seed=1)              # obs: a list of numbers (see "Observation vector")
for _ in range(400):
    obs, reward, terminated, truncated, info = env.step([0.6, 0.0])   # env.action_names == ['fwd', 'turn']
    if terminated or truncated:
        break
env.close()
```

Plug it into stable-baselines3 with `BotopiaGymEnv` (see [The Python client](#the-python-client)), or follow
[tutorial 6](https://robotics.dinno.be/tutorial-rl-corridor.html).

---

## Example scripts and tutorials

Start the server first (`node server.mjs --key rsk_xxxxxxxx`, your API key from the Account page of the site), then run a script from this folder, e.g.
`python python/corridor_rl.py train`.

| Script | What it does | Needs (`pip install`) |
|---|---|---|
| [`testgo2.py`](python/testgo2.py) | Go2 walks up the stairs in real time | – |
| [`testAckermann.py`](python/testAckermann.py) | Ackermann car follows the track using its camera (hand-written) | `numpy pillow` |
| [`botopia.py`](python/botopia.py) | run it for a demo: a random driver and four robots | – |
| [`camera_view.py`](python/camera_view.py) | live window with the robot's camera; drive with the keys | `numpy opencv-python` |
| [`corridor_rl.py`](python/corridor_rl.py) | [tutorial 6](https://robotics.dinno.be/tutorial-rl-corridor.html): PPO, the fastest lap through the corridor | `stable-baselines3 gymnasium pillow` |
| [`track_vision_rl.py`](python/track_vision_rl.py) | [tutorial 7](https://robotics.dinno.be/tutorial-rl-vision.html): PPO with a CNN, drive the ε track from the camera | `stable-baselines3 gymnasium pillow` |
| [`track_yolo.py`](python/track_yolo.py) | [tutorial 8](https://robotics.dinno.be/tutorial-yolo-track.html): YOLO segmentation finds the path, RL keeps the robot in the middle | `ultralytics stable-baselines3 gymnasium` |
| [`sort_rl.py`](python/sort_rl.py) | [tutorial 9](https://robotics.dinno.be/tutorial-rl-sorting.html): Q-learning over skills, the quadruped sorts cubes | – |
| [`arm_reach_her.py`](python/arm_reach_her.py) | [tutorial 10](https://robotics.dinno.be/tutorial-rl-arm-reach.html): SAC + Hindsight Experience Replay, the arm touches a point | `stable-baselines3 gymnasium` |
| [`stand_rl.py`](python/stand_rl.py) | [tutorial 11](https://robotics.dinno.be/tutorial-rl-stand-row.html): the OpenMANIPULATOR-X stands blocks in a row | – |
| [`go2_track_yolo.py`](python/go2_track_yolo.py) | [tutorial 12](https://robotics.dinno.be/tutorial-rl-go2-path.html): the Unitree Go2 follows the path with the YOLO model of tutorial 8 and a PPO policy, in the simulator and on the real Go2 | `ultralytics stable-baselines3 gymnasium` (`unitree_webrtc_connect` for the real dog) |
| [`xarm_robot.py`](python/xarm_robot.py), [`xarm_demo.py`](python/xarm_demo.py), [`xarm_bridge.py`](python/xarm_bridge.py) | one Python interface for the simulated and the real xArm, see [below](#the-real-xarm-6--7-one-python-interface-digital-twin-follow) | `xarm-python-sdk` (real arm) |
| [`tello_bridge.py`](python/tello_bridge.py) | the Tello EDU simulator page flies a real DJI Tello EDU and shows its video, see [below](#dji-tello-edu) | `websockets av pillow` |
| [`jaguar_bridge.py`](python/jaguar_bridge.py) | the DrRobot Jaguar 4x4 Wheel page shows and drives the real Jaguar, and PC programs (`robot.py --ip 127.0.0.1`) drive the simulated one, see [below](#drrobot-jaguar-4x4-wheel) | `websockets` |
| [`jaguar_cli.py`](python/jaguar_cli.py) | remote control of the Jaguar from a terminal: `status`, `release`, `drive 300 300 2`, `imu`, `gps`, `send "MM0 ?V"`; the real robot or the simulator | – |
| [`remote_camera.py`](python/remote_camera.py) | [tutorial 13](https://robotics.dinno.be/tutorial-camera-stream.html): the robot camera of a simulator in your browser (website API, https) on your own computer: drive with the keys, or `--follow` the earth path of the park on its own, see [below](#the-robot-camera-stream) | `numpy opencv-python` |

The tutorial scripts take a command: `baseline`, `train`, `eval` or `watch` (see `--help`). Training
on a GPU: `--device cuda`, see [Getting started](https://robotics.dinno.be/tutorial-getting-started.html#gpu).

---

## Two APIs: training or remote control

Both take the same actions and return the same observation, so a script moves from one to the other by
changing two lines.

| | **Node API** (`server.mjs`) | **PHP API** (`index.php`) |
|---|---|---|
| For | training, scripting | remote control, monitoring, classroom demos |
| Runs | on your PC; it starts Chrome itself (headless) | on the web server: https://robotics.dinno.be/api/index.php |
| Simulation | **step by step**: time only advances when you call `step` | in real time, in a normal browser tab |
| Reproducible | yes, with a `seed` | no |
| Speed | ~700 steps/s for one environment, ~1000 for several (Python client, measured on a laptop) | ~5 updates/s |
| Several robots at once | yes, one call can step all of them | one session per open browser tab |
| Robot camera | a picture with every step (`image=`), or `frame?view=robot` | a stream over https, up to 30 fps and 640 × 480 ([stream.html](https://robotics.dinno.be/stream.html), [below](#the-robot-camera-stream)) |
| Python | `BotopiaAPI("http://localhost:8780")` | `RemoteSession(url, code)` |

---

## The API server

```bash
node server.mjs [options]       # http://localhost:8780/api/
```

| Option | Meaning |
|---|---|
| `--show` | a visible Chrome window, to watch (one browser, redrawn every step) |
| `--port 8780` | port of the API; Chrome uses the next few ports |
| `--browsers N` | headless Chrome instances that draw in parallel (default: one per 4 CPU cores, at most 4) |
| `--chrome "path"` | where Chrome is, if it is not found (or set the `CHROME` environment variable) |
| `--site URL or folder` | where the simulator pages come from, see below |
| `--key rsk_...` | your API key from the account page of the site (or set `ROBOTSIM_KEY`); needed for the robots from https://robotics.dinno.be/ |

The server also serves the simulator pages, so no web server or PHP is needed. It takes them from
`--site`: by default `https://robotics.dinno.be/` (each page is fetched once and kept in memory, so the first start of a
robot needs internet, and an API key: `--key`), or the folder above this repository when that holds the simulator site (as on
the web server, where this repository is the site's `api/` folder). To try local changes to the
simulators: `node server.mjs --site ../robotSimulator`.

### Endpoints

Base URL `http://localhost:8780/api`. All bodies are JSON.

| Method | Path | Body | Answer |
|---|---|---|---|
| GET | `/robots` | | `[{ robot, page }]` |
| GET | `/envs` | | open environments |
| POST | `/envs` | `{ robot, lane?, seed?, options? }` | `{ id, info, obs }` |
| POST | `/envs/{id}/reset` | `{ lane?, seed?, start?, offset?, yaw?, image? }` | `{ obs, image? }` |
| POST | `/envs/{id}/step` | `{ action, dt?, frame?, image?, until? }` | `{ obs, events, reward, done, image? }` |
| POST | `/envs/step` | `{ steps: [{ id, action, dt?, image? }] }` | `{ results: [...] }` (all in parallel) |
| GET | `/envs/{id}/obs` | | current observation |
| GET | `/envs/{id}/info` | | robot, drive axes, motors, lanes, event types, `commands` |
| GET | `/envs/{id}/frame?view=scene&width=320&quality=0.8` | | JPEG of a view, see below |
| DELETE | `/envs/{id}` | | closes the environment |

| Parameter | Meaning |
|---|---|
| `options` (create) | page settings, passed to the simulator page, e.g. `{ "vmax": 1.8 }` = Ackermann top speed in m/s, `{ "door": "free" }` |
| `dt` | how long one step lasts in simulated seconds (default 0.05 s = 20 Hz control) |
| `frame` | the internal simulation step (default 1/60 s); a step with `dt` 0.05 runs 3 frames |
| `until` | `"idle"`: `dt` is the maximum, the step ends as soon as a running skill has finished |
| `image` | `{ "view": "robot", "width": 64, "quality": 0.9 }`: the answer also has `"image"`, that JPEG in base64, without an extra request (training from pixels) |
| `view` | `scene` (the 3D view), `robot` (the robot's own front camera, 4:3, `width` 32–1280), `mask` (the track band in the robot camera's view, white on black PNG: a ready-made label for segmentation), `external` (a fixed camera, arms), `real` (the real USB camera, OpenMANIPULATOR-X page); robots with a RealSense D435 (the xArms' wrist camera): `depth` (depth in mm as a lossless PNG, mm = red · 256 + green, 0 = no depth) and `depthcolor` (the same in colour) |
| `start`, `offset`, `yaw` (reset, ε track) | metres along the track, metres beside the centre line (+ = left), degrees off the track direction |

`info.commands` lists every button of a robot's console as a ready-made action `{group, label, line, action}`.

```bash
curl -X POST localhost:8780/api/envs -d '{"robot":"skid","lane":"step6","seed":1}'
curl -X POST localhost:8780/api/envs/1/step -d '{"action":{"drive":{"fwd":1}},"dt":0.1}'
curl localhost:8780/api/envs/1/frame -o view.jpg
curl "localhost:8780/api/envs/1/frame?view=robot&width=160" -o robot.jpg    # the robot's own camera
```

---

## The Python client

`python/botopia.py`, standard library only (`numpy`, `gymnasium` and `pillow` are used when installed).
Add `sys.path.append("python")` and import from `botopia`.

**`BotopiaAPI(url="http://localhost:8780")`**: the endpoints one to one.

| Method | Returns |
|---|---|
| `robots()` | `[{robot, page}]` |
| `create(robot, lane="free", seed=None, options=None)` | `{id, info, obs}` |
| `reset(id, lane=None, seed=None, start=None, offset=0, yaw=0, image=None)` | `obs`, or `(obs, jpeg bytes)` with `image` |
| `step(id, action, dt=0.05, frame=1/60, image=None, until=None)` | `{obs, events, reward, done}` (+ `image` as JPEG bytes) |
| `step_many([{id, action, dt, image?}])` | list of results, all environments in one call |
| `observe(id)` | current observation |
| `frame(id, quality=0.8, view="scene", width=320)` | JPEG bytes |
| `close(id)` | |

**For training:**

* `BotopiaEnv(robot, lane, url, dt=0.05, max_steps=400, action_mode="drive")`: `reset()` / `step()` as in
  Gymnasium, observations as a list of numbers (`features`, below). `action_mode="motors"` controls the
  motors directly (PWM duty per DC motor, angle per servo) instead of the robot's drive commands.
* `BotopiaVecEnv(["skid", "tracked", "ostrich"], lane="step6")`: several environments, one HTTP call per
  step. Use this for speed.
* `BotopiaGymEnv(...)`: Gymnasium wrapper (`Box` spaces, passes `gymnasium.utils.env_checker`);
  `render_mode="rgb_array"` returns the camera image (needs Pillow).
* `RemoteSession(url, code)`: the PHP API, see [below](#php-api-remote-control); `frame()` and `stream_url()` give
  its robot camera ([the robot camera stream](#the-robot-camera-stream)).

### Observation vector

`botopia.features(obs)` turns an observation into a fixed-length list:
`[progress, vx, vy, wz, roll, pitch, accel_x/g, accel_y/g, accel_z/g, gyro_x, gyro_y, gyro_z,
stuck(0/1)]`, then per motor `duty, rpm/100` (DC) or `degrees/90` (servo).

---

## Actions

```json
{
  "drive":  { "fwd": 0.8, "turn": -0.2, "strafe": 0 },
  "motors": { "FL": 0.5, "steer": 12.0 }
}
```

* **`drive`**: −1 … 1 per axis, exactly like holding the buttons / keys: the robot's own
  controller turns it into motor commands (balancing included). Axes per robot in the table below
  (`info.drive`). Stays active until you send another one; `"drive": null` gives control back to
  the keyboard (remote mode).
* **`motors`**: direct control that bypasses the robot's controller for that motor. DC motors:
  duty −1 … 1 (sign = direction, |duty| × 255 = PWM). Servos: angle in degrees. `null` releases a
  motor again. Only motors marked controllable (`info.motors[].overridable`).

Several fields can be combined. Robots with a real-robot interface (ROS 2, WebRTC, serial, SDK…) also
take the messages of that interface, with the names and fields of the real robot: see the
[robot reference](#robot-reference). The answer to such a message comes back as an event with the same
name.

| Robot | `drive` axes | Controllable motors (° = servo) | Read-only motors | Own events | `extra` telemetry |
|---|---|---|---|---|---|
| `wheeled` | fwd, turn | – (it must balance) | L, R | fell, bottle_fell | tiltDeg, fallen, balancing, bottleOn, bottleRisk, waterLitres |
| `bike` | fwd, turn | drive, steer° | rw (reaction wheel) | fell | leanDeg, steerDeg, reactionWheelRpm, speed, fallen, mode |
| `synchro` | fwd, strafe | drive, steer | – | – | steerDeg |
| `mecanum` | fwd, turn, strafe | FL, FR, RL, RR | – | – | – |
| `kiwi` | fwd, turn, strafe | M1, M2, M3 | – | – | – |
| `swerve` | fwd, turn, strafe | D1–D4 (drive), S1–S4 (steer) | – | – | – |
| `tracked` | fwd, turn | L, R | – | – | chi (slip factor) |
| `skid` | fwd, turn | FL, FR, RL, RR | – | – | chi |
| `ackermann` | fwd, turn | drive, steer° | – | – | radius |
| `4wd` | fwd, turn | FL, FR, RL, RR, steerF°, steerR° | – | – | – |
| `whegs` | fwd, turn | L, R | – | – | – |
| `quadruped` | fwd, turn, strafe | 12 joints° (`FL_hip`, `FL_thigh`, `FL_calf`, …), arm `arm_joint1`–`arm_joint4`°, `gripper` (opening in mm, 0–70) | – | door | bodyHeight, walking, message, arm {holding, busy, grasp {x, y, z}}, cubes [{id, x, y, z, held, onBack, inBox}], reach {tip [x, y, z] (fingertips in the arm frame), target, dist}, door {angleDeg, handleDeg, mode, latched, handle {x, y, z}, script} (door lane) |
| `mini` | fwd, turn, strafe | 12 joints° | – | – | bodyHeight, walking, message |
| `wheellegs` | fwd, turn | FLw, FRw, RLw, RRw (wheels) | 8 leg joints | – | bodyHeight, lifted, level |
| `ostrich` | fwd, turn | – (it must balance) | wL, wR, 4 leg joints | fell, jump | tiltDeg, speed, legLength, mode, fallen, autoJump |
| `humanoid` | fwd, turn, strafe | – | 12 joints | fell | comInside, comMarginMm, feetOnFloor, stepHeight, stepLength, sway |
| `snake` | fwd, turn, strafe (sidewinding) | J1–J6° (J1–Jn with `?servos=4…12`; Dynamixel, also `position` 0–4095) | – | – | gait, servos, underside, touching, blocked, waveHz, speed |
| `create3` | fwd, turn | L, R | – | bump, cliff | bumpers {left, right} (as the Python SDK), bumpZones {left, front_left, front_center, front_right, right}, irIntensity {side_left … right: 0–4095}, cliff {side_left, front_left, front_right, side_right}, wheelDrop {left, right} |
| `go2` | fwd, turn, strafe (the app's joystick) | – | 12 joints (`FL_hip`, `FL_thigh`, `FL_calf`, …) | webrtc | sportmodestate {mode, gait_type, body_height, foot_raise_height, velocity, yaw_speed, range_obstacle [front, left, right, back]}, state, motionMode, avoid, webrtc (last responses) |
| `g1` | fwd, turn, strafe (the app's joystick) | – | 23 joints (`left_hip_pitch` … `right_wrist_roll`, `waist_yaw`) | webrtc, fell | fsm_id, fsm_name, fsm_mode, posture, arm_action, velocity, webrtc (last responses) |
| `turtlebot4` | fwd, turn (teleop keys) | L, R | – | ros, bump, cliff | model, topics {`/odom`, `/scan` (360 ranges, null = no echo), `/hazard_detection`, `/ir_intensity`, `/battery_state`, `/dock_status`, `/wheel_vels`}, goal, estop, safety_override, ros (last answers) |
| `go2w` | fwd, turn (the app's joystick) | – | 4 wheels (`FL_wheel` … `RR_wheel`), 12 joints (`FL_hip`, `FL_thigh`, `FL_calf`, …) | webrtc | sportmodestate {mode, gait_type, body_height, wheel_speed [FL, FR, RL, RR rad/s], velocity, yaw_speed, range_obstacle}, state, motionMode, avoid, lifting [legs lifting their wheel], webrtc |
| `r1` | fwd, turn, strafe (the app's joystick) | – | 26 joints (`left_hip_pitch` … `right_wrist_roll`, `waist_yaw`, `waist_pitch`, `head_yaw`, `head_pitch`); R1 Air (`?variant=air`): 20 (no waist, head or wrists) | webrtc, fell | model, fsm_id, fsm_name, fsm_reported, reporting, posture, arm_action, velocity, webrtc (last responses) |
| `rosbotxl` | fwd, turn, strafe (mecanum; `?variant=wheels`: fwd, turn) | FL, FR, RL, RR | – | ros | variant, topics {`/odometry/filtered`, `/rosbot_xl_base_controller/odom`, `/scan` (360 ranges, null = no echo), `/joint_states`, `/battery_state`}, leds, ros (last answers) |
| `leorover` | fwd, turn (teleop keys) | FL, RL, FR, RR | – | ros | topics {`/wheel_odom`, `/merged_odom`, `/imu/rpy`, `/firmware/battery`, `/joint_states`, `/scan` (only with the LiDAR accessory)}, lidar, ros (last answers) |
| `phoenix` | fwd, turn, strafe (the Phoenix code walks) | `RR_coxa` … `LF_tibia` (18)° | – | serial | mode (gait / serial), gait, walking, bodyHeight, legsUp, pulses {SSC-32 channel: µs}, joints {RR … LF: [coxa, femur, tibia]°}, blocked, serial (last replies) |
| `bittle` | fwd, turn (the app's joystick: kwkF / kbk / kwkL / kwkR) | `j0` (head), `j8`–`j11` (shoulders), `j12`–`j15` (knees)° | – | serial | skill, gait, paused, gyro, joints {0, 8…15}, bodyHeight, blocked, serial (last replies) |
| `roboworks` | fwd, turn (teleop keys; turn steers) | L, R (rear wheels), steer° (servo) | – | ros | model, topics {`/odom`, `/scan` (360 ranges, null = no echo), `/PowerVoltage`}, steerDeg, ros (last answers) |
| `mecabot` | fwd, turn, strafe | FL, FR, RL, RR | – | ros | model, topics {`/odom`, `/scan`, `/PowerVoltage`}, ros (last answers) |
| `ugvrover` | fwd, turn (teleop keys) | L, R (one channel per side) | – | ros, serial | topics {`/odom`, `/odom/odom_raw`, `/voltage`, `/scan` (360 ranges, null = no echo)}, feedback {T:1001 …}, gimbal {pan, tilt}, leds, mode, ros, serial |
| `jetbot` | fwd, turn (gamepad keys) | left, right | – | python | motors {left, right: {value, alpha, beta}}, script, python |
| `pirateshooter` | fwd, turn (the game controller: A and T lines) | `esc` (dc), `steer` (servo), read only | – | serial | opencr {angle, throttle, servo_deg, servo_us, esc_us, offsets {center, left, right}, test}, speed_kmh, wheel_deg, on_stand, max_throttle, real {connected, follow, angle, throttle}, serial (last answers) |
| `driftx` | fwd, turn (a joystick: A and T lines, the watchdog fed) | `motor` (dc), `steer` (servo), read only | – | serial | xiao {steer, throttle, trim, max, watchdog_ms, braking, pulse_us, duty}, speed_kmh, slip_deg, yaw_rate, wheel_deg, tyres, on_stand, max_throttle, real {connected, follow, steer, throttle}, serial (last answers) |
| `jetracer` | fwd, turn (gamepad keys) | throttle (dc), steering (servo) | – | python | car {steering, throttle, steering_gain, steering_offset, throttle_gain}, script, python |
| `tello` | fwd, turn, strafe, up (the app's sticks) | `m1` … `m4` (read only) | – | tello | phase (landed / takeoff / flying / landing / flip / falling), sdk, height_cm, battery, flight_s, flight_left_s, max_height_m, stream, downvision, mission_pads, pad, busy, state (the SDK state packet), tello (last answers) |
| `xgo` / `xgomini` | fwd, turn, strafe (the app's joystick) | – | 15 servos (`m11` … `m43` legs, `m51` gripper, `m52` `m53` arm) | xgo | model, action, actionName, move {x, y, turn}, gait, pace, markTime, height_mm, attitude {r, p, y}, imu, lying, arm {x, z, claw}, cube {x, y, z, held}, motors, blocked, xgo (last answers) |
| `a2` / `as2` | fwd, turn, strafe (the remote) | – | 12 joints (`FL_hip`, `FL_thigh`, `FL_calf`, …) | sdk | model, sportmodestate {mode (FSM 0–15), mode_name, progress, position, velocity, yaw_speed, body_height}, gait, speed_level, trajectory, sdk (last answers) |
| `spot` | fwd, turn, strafe (the tablet's joysticks, while it stands with power on) | – | 12 joints (`FL_hip`, `FL_thigh`, `FL_calf`, …) | ros | lease, power, estop, standing, stairMode, mode, topics {`/odometry`, `/status/feet`, `/status/battery_states`, `/status/power_state`}, ros |
| `agibotd1` | fwd, turn, strafe (the remote) | – | 12 joints | sdk | ctrl_mode (0 / 1 / 18 / 21 / 51), position, body_velocity, battery, move, two_leg, sdk (last answers) |
| `h2` | fwd, turn, strafe (the remote's joystick) | – | 31 joints (`left_hip_pitch` … `right_wrist_yaw`, `waist_yaw/roll/pitch`, `head_yaw/pitch`) | sdk, fell | fsm_id, fsm_name, fsm_mode, posture, arm_action, velocity, balance_mode, speed_mode, swing_height, stand_height, sdk (last answers) |
| `boosterk1` | fwd, turn, strafe (the app's joystick) | – | 22 joints (`head_yaw`, `head_pitch`, `left_shoulder_pitch` … `right_crank_down`) | booster, fell | mode, mode_name, posture, action, gait, head {pitch, yaw}, odometer {x, y, theta}, velocity, booster (last answers) |
| `tron1` | fwd, turn, strafe (the handheld remote) | – | 6 joints (`abad_L_Joint`, `hip_L_Joint`, `knee_L_Joint`, … `_R_`); sole feet (`options={"variant": "sole"}`): + `ankle_L_Joint`, `ankle_R_Joint` | ws, fell | variant, robot_info {accid, sw_version, imu, camera, motor, battery, status}, imu (after request_enable_imu), marktime, baseHeightOffset, blocked, notify, ws (last answers) |
| `tron1w` | fwd, turn (the handheld remote) | – | 6 leg joints + `wheel_L_Joint`, `wheel_R_Joint` | ws, fell | as `tron1`, plus stair, odom (after request_enable_odom) |
| `panther` | fwd, turn (mecanum `?variant=mecanum`: + strafe) | FL, FR, RL, RR | – | ros | variant, topics {`/odometry/wheels`, `/odometry/filtered`, `/battery/battery_status`, `/hardware/e_stop`, `/joint_states`, `/scan` (accessory)}, estop, lights, lidar, ros |
| `jaguar` | fwd, turn (the gamepad: `PING` + `MMW !M` lines) | FL, FR, RL, RR (channel sign: the right ones − = forward) | – | tcp, watchdog | estop, watchdog_ok, power [[FL, FR], [RL, RR]], encoders {FL, FR, RL, RR, m_per_count}, imu {raw {gyro, acc, mag}, gyro_dps, acc_g, compass_deg, yaw}, gps {fix, lat, lon, speed_kn, course, sats, hdop}, battery_v, temps, amps, surface_mu, connections, real {connected, robot, mode, lines, battery_v, gps}, tcp (last answers) |
| `jackal` | fwd, turn (teleop keys) | FL, FR, RL, RR | – | ros | topics {`/j100_0000/platform/odom`, `/j100_0000/platform/odom/filtered`, `/j100_0000/platform/emergency_stop`, `/j100_0000/platform/mcu/status/power`, `/j100_0000/sensors/lidar2d_0/scan` (271 ranges, 270°)}, estop, ros |
| `husky` | fwd, turn (teleop keys) | FL, FR, RL, RR | – | ros | as `jackal`, namespace `/a200_0000` |
| `summitxl` | fwd, turn (mecanum `?variant=mecanum`: + strafe) | FL, FR, RL, RR | – | ros | variant, topics {`/robot/robotnik_base_control/odom`, `/robot/front_laser/scan` (271 ranges, 270°), `/robot/joint_states`, `/robot/battery_estimator/data`, `/robot/robotnik_base_hw/emergency_stop`}, motorsEnabled, ros |
| `limo` | fwd, turn (mecanum `options={"mode": "mecanum"}`: + strafe; Ackermann: turn steers) | FL, FR, RL, RR (hub motors); Ackermann: + `steer`° | – | ros | mode, motionMode (0 differential / tracked, 1 Ackermann, 2 mecanum), steerDeg (Ackermann), topics {`/odom`, `/limo_status`, `/scan` (360 ranges)}, ros |
| `scoutmini` | fwd, turn (mecanum `options={"variant": "mecanum"}`: + strafe) | FL, FR, RL, RR | – | ros | variant, topics {`/odom`, `/scout_status`, `/scan` (360 ranges)}, ros |
| `hunterse` | fwd, turn (turn steers) | RL, RR (rear wheels), `steer`° (central angle) | – | ros | steerDeg, topics {`/odom`, `/hunter_status` (with `steering_angle`), `/scan`}, ros |
| `bunker` | fwd, turn (teleop keys) | FL, FR, RL, RR (the sprockets of the two tracks) | – | ros | variant (mini / pro), topics {`/odom`, `/bunker_status`, `/scan`}, ros |
| `mir250` | fwd, turn (manual control while held) | L, R (drive wheels) | – | rest | status (as GET /status: state_id, state_text, mission_text, position {x, y, orientation°}, battery, errors), queue [{id, mission_id, state}], rest (last answers) |
| `rangermini` | fwd, turn, strafe (the mode follows: fwd + turn dual Ackermann, turn alone spin, strafe side slip, fwd + strafe parallel) | FL, FR, RL, RR (drive), SFL, SFR, SRL, SRR (steering) | – | ros | motionMode (0 dual Ackermann, 1 parallel, 2 spin, 3 side slip), motionModeName, steeringDeg [FL, FR, RL, RR], topics {`/odom`, `/motion_state`, `/system_state`, `/battery_state`, `/scan` (360 ranges)}, ros |
| `rbkairos` | fwd, turn, strafe (teleop keys) | FL, FR, RL, RR (wheels), `shoulder_pan` … `wrist_3`° (arm) | – | ros | model, arm_type, topics {`/robot/robotnik_base_control/odom`, `/robot/front_laser/scan`, `/robot/rear_laser/scan` (241 ranges), `/robot/joint_states`}, arm {q, tcp, grip, holding, busy, message}, blocks [{id, color, x, y, z, held}], goal, ros (last answers) |
| `rbvogui` | fwd, turn, strafe (teleop keys) | FL, FR, RL, RR (drive), `shoulder_pan` … `wrist_3`° (arm) | SFL, SFR, SRL, SRR (steering) | ros | as `rbkairos` |
| `xarm5` | – (it stays on its table) | `joint1`–`joint5`°, `gripper` (opening in mm, 0–85) | – | – | as `xarm6` |
| `xarm850` | – | `joint1`–`joint6`°, `gripper` | – | – | as `xarm6` |
| `ur5e` | – (it stays on its table) | `joint1`–`joint6`°, `gripper` (opening in mm, 0–85) | – | urscript | as `xarm6`, plus urscript (last programs: `program`, `reply`, `note`) |
| `fr3` | – (it stays on its table) | `joint1`–`joint7`°, `gripper` (opening in mm, 0–80) | – | python | as `xarm6`, plus python (last scripts: `code`, `reply`, `note`) |
| `sespro` | – | `joint1`–`joint5`°, `gripper` (mm, 0–85) | – | serial | as `xarm6`, plus lss {limp [servo ids], status [LSS Q code per servo 1–6], led [0–7]}, serial (last answers) |
| `xarm6` | – (it stays on its table) | `joint1`–`joint6`°, `gripper` (opening in mm, 0–85) | – | – | tcp {x, y, z, yawDeg} (point between the finger pads, base frame), holding, busy, blocks [{id, color, x, y, z, yawDeg, held}] |
| `xarm7` | – | `joint1`–`joint7`°, `gripper` | – | – | as `xarm6` |
| `omx` | – | `joint1`–`joint4`°, `gripper` (opening in mm, 13–65) | – | – | tcp {x, y, z, yawDeg, pitchDeg}, holding, busy, blocks [{id, color, x, y, z, yawDeg, held}] |

---

## Observation, events and reward

### Observation

```json
{
  "t": 1.25, "robot": "skid", "lane": "step6",
  "pose":     { "x": 1.31, "y": 0.0, "z": -3.5, "yaw": 0.0, "pitch": 0.0, "roll": 0.0 },
  "velocity": { "vx": 0.0, "vy": 0.0, "wz": 0.0 },
  "imu":      { "accel": [0.0, 0.0, 9.81], "gyro": [0.0, 0.0, 0.0] },
  "motors": [
    { "id": "FL", "name": "Front left", "type": "dc", "pwm": 171, "dir": "FWD", "duty": 0.671,
      "rpm": 20.6, "encoder": { "angle": 38.7, "ticks": 6308 }, "current": 1.07 },
    { "id": "steer", "type": "servo", "deg": 12.0, "target": 12.0, "us": 1700 }
  ],
  "progress": { "distance": 0.41, "goal": false },
  "stuck": "6.0 cm edge: a driven wheel can climb 3.0 cm",
  "extra": { "chi": 1.69 },
  "done": false
}
```

| Field | Meaning |
|---|---|
| `t` | simulated time since the reset (s) |
| `pose` | position in the world (m; x along the lanes, y up, z sideways), heading `yaw` (+ = left), body `pitch` (+ = nose up), `roll` (+ = left side up), radians |
| `velocity` | body frame: `vx` forward, `vy` left (m/s), `wz` turn rate (rad/s, + = left) |
| `imu.accel` | accelerometer, body frame (x forward, y left, z up), m/s², includes gravity (≈ 9.81 up when still) |
| `imu.gyro` | roll, pitch and yaw rate (rad/s) |
| `motors[]` | per motor: H-bridge `pwm` 0–255, `dir` FWD/REV/STOP, `duty` −1…1, `rpm`, `encoder.angle` (rad) and `ticks` (1024 per output revolution), `current` (A, estimated from duty and back-EMF; ~2 A when stalled at full power). Servos: `deg`, `target`, `us` (pulse) |
| `progress` | metres driven along the lane from its start (the corridor along its bend) and whether the robot is past the obstacle |
| `stuck` | why the robot can't go on right now (empty if it can) |
| `extra` | robot-specific values (table above) |
| `done` | true after `fell`, `bottle_fell` or `goal`: end the episode |

### Events (incidents)

Each step returns the events that happened during that step:
`{ "t": 1.02, "type": "blocked", "reason": "6.0 cm edge: a driven wheel can climb 3.0 cm" }`.

| Type | When | Ends the episode |
|---|---|---|
| `collision` | the robot drives (or turns) into a wall | no |
| `blocked` | an edge it can't climb, a slope that is too steep, a step it can't take | no |
| `stall` | drive motors at more than 40 % power but the robot hasn't moved for 0.5 s | no |
| `goal` | past the obstacle of the lane | yes |
| `fell` | balancing robots / bike / ostrich tip over; humanoid: centre of mass outside its feet | yes |
| `bottle_fell` | `wheeled`: the water bottle falls off the plateau | yes |
| `jump` | `ostrich` takes off | no |
| `bump` | `create3`: a bumper zone closes (`zones`: which ones) | no |
| `cliff` | `create3`: a cliff sensor sees a drop (`sensors`: which ones) | no |
| `webrtc` | `go2`, `go2w`, `g1`, `r1`: the answer to a WebRTC message (`topic`, `api_id`, `code`, `data`, `msg`) | no |
| `ros` | `turtlebot4`, `rosbotxl`, `leorover`, `roboworks`, `mecabot`, `ugvrover`, `panther`, `summitxl`, `jackal`, `husky`, `rangermini`, `scoutmini`, `hunterse`, `bunker`, `limo`, `spot`, `rbkairos`, `rbvogui`: the answer to a ROS 2 message (`kind`, `name`, `code`, `text`; actions also `status` SUCCEEDED / ABORTED and the result in `data` when the goal ends) | no |
| `serial` | `bittle`: the answer to an OpenCat serial token (`line`, `reply`); `pirateshooter`: what `rccar.ino` prints for a line (`line`, `code`, `reply`); `driftx`: what `xiao_drift.ino` prints (`line`, `code`, `reply`); `ugvrover`: to an ESP32 JSON command; `sespro`: to an LSS command (`line`, `reply`, `note`); `phoenix`: to an SSC-32U line (`line`, `reply`: empty for a move) | no |
| `rest` | `mir250`: the answer to a REST request (`method`, `path`, `code` = HTTP status, `data`); a mission that ends blocked (`method`: EVENT, `data` {state_text, errors}) | no |
| `urscript` | `ur5e`: a URScript program ran (`program`, `reply` from textmsg / get_actual_…, `note`); a move that failed while it ran (`program`: null, `note`) | no |
| `python` | `jetbot`, `jetracer`: the result of a Python line (`out` or the error); `fr3`: a franky script ran (`code`, `reply` from print, `note`) | no |
| `xgo` | `xgo`, `xgomini`: the answer to an xgolib call (`line`, `code`, `reply`) | no |
| `tello` | `tello`: the answer of the drone to a Tello SDK command (`command`, `code`, `reply`) | no |
| `tcp` | `jaguar`: what the motor boards answer to a line (`line`, `code`, `reply`) | no |
| `watchdog` | `jaguar`: no line for 0.3 s while the motors had a command: the controller set them to 0 (`reason`) | no |
| `sdk` | `a2`, `as2`, `agibotd1`, `h2`: the answer of the maker's SDK call (code, data) | no |
| `booster` | `boosterk1`: the status of a B1LocoClient call (0, 400, 409, 501, 502) | no |
| `ws` | `tron1`, `tron1w`: an answer or push of the TRON 1 WebSocket (`title` response_… / notify_…, `data.result`) | no |
| `door` | `quadruped` in the door lane: `state` = `unlatched`, `opened`, `closed` or `latched` | no |

### Reward

The Node API adds a suggested `reward` to every step (use it or compute your own from `obs` and
`events`):

```
reward = 10 × (progress gained in this step, m)
       − 1 per collision − 0.2 per blocked − 0.5 per stall
       − 10 for fell / bottle_fell + 10 for goal
```

---

## Lanes and the robot camera

### Lanes (the obstacle course)

`free` (start square), `cable` (1 cm), `threshold` (3 cm), `step6` (6 cm step), `curb12` (12 cm),
`stairs` (3 × 7 cm), `ramp15`, `ramp30`, `gaps` (5 and 12 cm), `gravel`, `corridor` (60 cm wide,
sharp right-hand bend), `door` (lane 11: a wall with a 60 cm doorway and a door that swings away from
the robot, lever handles on both sides). `reset` puts the robot upright, standing still, just before
the obstacle, facing it.

The door has three modes, set with the page option `door` (`options: { "door": "free" }`, or `?door=free`
in the browser): `latch` (default: the latch holds it shut, press the handle down to open it), `free`
(not latched: a push opens it) and `locked` (the handle does nothing). Every robot can push an
unlatched door open; only the quadruped can work the handle, with the gripper of its arm (press the
lever down from above with the fingertips; while you hold it down, moving the hand drags the door
along). Let go and the door swings on, slowed by its hinges; in the last 6° a catch pulls it shut and
the latch clicks in. The quadruped page has two scripted sequences: *Open and walk through* and
*Close it behind*, and *Walk to the table*: behind the door stands a table (40 × 60 cm, 28 cm high, a
wall for driving) within reach of the standing quadruped's arm. On it stand four open sorting boxes (blue, red, green,
yellow; 7.5 cm, 3.5 cm high) in a row along the near edge, each with the cube of its colour inside
(`table_blue`, `table_red`, `table_green`, `table_yellow`). *Place* puts the held cube on a free spot,
*Into the box of its colour* (key K) sorts it; Pick takes cubes that are already in their own box last.
`extra.cubes[].inBox` is the colour of the box a cube is in (or null).

`track`: the **ε track** behind the start square, a closed loop of 42.6 m with bends of 1.2 m
radius both ways and small hills (2–4 cm, at most 6°). It is painted for camera-based track
following: a light 70 cm band with white edge lines and a dashed centre line on the dark floor.
Every robot can drive it (tested: all 17 stay on the band lap after lap with the autopilot,
`?auto=track` in the page URL). It has no goal; `progress` gives instead:

| Field | Meaning |
|---|---|
| `progress.distance` | metres along the track from the chequered start line (0 … 42.56, starts again every lap) |
| `progress.offset` | metres beside the centre line, + = left (`null` when far from the track) |
| `progress.onTrack` | `true` while the robot's centre is on the painted band (±0.35 m) |

The robot camera (`frame?view=robot`) shows only what a real camera would see: path trail, turning
circle, slip arrows and other helper drawings are left out of it.

### Robot camera

Every robot has a small camera at its front, fixed to the part of the robot that tilts (so on a ramp
or when a quadruped pitches, the view tilts with it). In the simulator pages its picture is shown in
the top-right corner of the 3D view (✕ hides it, “📷 Robot camera” brings it back, `?robotcam=0`
starts without it). Through the API, `frame?view=robot` returns that image; in Python
`env.frame(view="robot", width=84)`. Use it to train from pixels, for example with a `CnnPolicy`.

---

## PHP API (remote control)

1. Open a simulator with `?remote=1`, e.g. `https://robotics.dinno.be/indexSkid.php?remote=1`
   (or `?remote=MYCODE` to choose the code). The page shows **Remote session ABC123** in its
   bottom-right corner and keeps running normally (keys still work until a drive command comes in).
2. Talk to `https://robotics.dinno.be/api/index.php` (this repository's `index.php`):

| Method | Query | Body | Answer |
|---|---|---|---|
| GET | `?r=sessions` | | active sessions `[{ session, robot, name, lastSeenSeconds }]` |
| GET | `?r=state&session=ABC123&since=0` | | `{ info, obs, events, next, active }`; pass `next` as `since` next time to get only new events |
| POST | `?r=command&session=ABC123` | `{ "action": { … } }` and/or `{ "reset": { "lane": "stairs" } }` | `{ seq }` |
| GET | `?r=frame&session=ABC123&fps=10&width=640` | | the robot camera's latest picture (JPEG); 404 until the first one is there |
| GET | `?r=mjpeg&session=ABC123&fps=10&width=640` | | the robot camera as an MJPEG stream (`multipart/x-mixed-replace`) |

The page itself uses `register`, `pull` (every 100 ms), `telemetry` (every 200 ms) and, while someone watches
its camera, `POST ?r=frame` (a JPEG).

```python
from botopia import RemoteSession
s = RemoteSession("https://robotics.dinno.be/api/index.php", "ABC123")
s.command(reset={"lane": "stairs"}, action={"drive": {"fwd": 0.6}})
print(s.state()["obs"]["pose"])
```

Sessions are stored as small JSON files in `data/` next to `index.php` (made on first use, closed to the web
with `.htaccess`; on nginx deny `/api/data/` yourself). The session code is the only
key: anyone who knows it can drive that browser tab's simulated robot and see its camera, nothing else.
Sessions older than a day are removed.

### The robot camera stream

What the robot's camera sees, over https, to process on your own computer while you drive the robot
through the same session:

* **In the browser:** `https://robotics.dinno.be/stream.html?session=ABC123&fps=10&width=640`, with the
  addresses and a Python example for your own code.
* **One picture:** `GET ?r=frame&session=ABC123&fps=10&width=640` (JPEG; the header `X-Frame-Age` says how
  old it is, in whole seconds).
* **A stream:** `GET ?r=mjpeg&session=ABC123&fps=10&width=640`: `cv2.VideoCapture(url)` opens it, as do VLC
  and an `<img>` in a browser. A connection lasts at most 10 minutes; connect again after that.

`fps` is 1 … 30 (default 10), `width` 160 … 640 (default 640; the pictures are 4:3, so at most 640 × 480).
The simulator only sends pictures while someone asks for them (a `frame` or `mjpeg` request in the last 5
seconds): its badge then says **camera 10 fps**. The first `frame` request answers 404 *no picture yet*;
ask again a moment later. Keep the simulator window visible: browsers pause hidden tabs.

Limits, so the web server keeps up: at most **5 cameras stream at the same time**, of all sessions and users
together, and **each camera has at most 2 receivers** at the same time. More is refused: `503 too many camera
streams at once`, or `429 this camera already has 2 receivers`. A receiver is one program or browser polling
`?r=frame` (the same address and browser or program counts once) or one open `?r=mjpeg` connection; it frees its
place 5 seconds after its last request (an MJPEG connection: as soon as it closes). `RemoteSession.frame()`
raises a `RuntimeError` with that message.

```python
import cv2, numpy as np
from botopia import RemoteSession
s = RemoteSession("https://robotics.dinno.be/api/index.php", "ABC123")
jpeg = s.frame(fps=10, width=640)                  # bytes, or None while there is no picture yet
img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
s.command({"drive": {"fwd": 0.4, "turn": 0.2}})    # what your processing decided
cam = cv2.VideoCapture(s.stream_url(fps=10))       # or as a stream
```

[`python/remote_camera.py`](python/remote_camera.py) does all of it: a window with the camera, drive with
W A S D, or `--follow` to steer along the earth path of the park from a colour mask
(`python python/remote_camera.py --session ABC123 --follow`).

---

## In the page (JavaScript)

Every simulator page exposes the same interface as `window.botopiaEnv` (made by
`common/env.js`), handy in the browser console:

```js
botopiaEnv.info()
botopiaEnv.reset({ lane: 'gravel', seed: 1 })
botopiaEnv.step({ drive: { fwd: 1 } }, { dt: 0.1 })   // in normal mode the page also keeps running
```

Environments instead of the obstacle course (Setup → Environment, `?world=` in the page, `options={"world": …}` in
the API): `botopia` (the Kyosho Mini-Z 50x50 Race track), `corridor` (the Corridor Environment) and `park` (the ε
track as an earth path in the grass, with pebbles, puddles, a hedge in the middle of the loop, bushes and trees; the
lane `track` works there as on the course, so `track_yolo.py` and `go2_track_yolo.py` take `--world park`). The
examples of the [world editor](https://robotics.dinno.be/design-your-world.html) are built in too: `house` (a house with
a garden), `hospital` (a hospital ward with lifts), `warehouse` (pallet racks and aisles) and `maze` (the maze of
tutorial 14). Worlds that users build themselves stay in their own browser, so the API does not have them.

```python
env = api.create("go2", "track", options={"world": "park"})
api.reset(env["id"], world="corridor")        # a reset can change it too (and back: world="course")
```

Every robot that drives or walks has them (`info["worlds"]`, `obs["world"]`); the robot arms stay at their
table. In the website API the same reset is `{"reset": {"world": "park", "lane": "track"}}`.

URL switches: `?env=1` lock-step (used by the Node API: the page stands still between steps),
`?remote=1` remote mode (PHP API), `?lane=step6` start in a lane, `?auto=corridor` autopilot through
the corridor, `?door=free` door mode in the door lane, `?fixedfps=30` fixed time step, `?shot` 3D view only.

---

## Robot reference

Details per robot: extra commands, interfaces of the real robots, page options.

### Snake and quadruped joints

Snake joints: odd J1, J3, … pitch, even J2, J4, … yaw. The page URL sets the number of servos with `?servos=4` … `12` (default 6). A commanded angle replaces the gait for that joint,
so with all six you can learn your own gait; the body moves by friction alone (no drive motors).
Add `?feet=skin` (rubber pads) or `?feet=scales` (snake skin with scales) to the page URL instead of passive wheels.

Quadruped joints: a commanded angle puts the leg in that pose (walking stops, the body rests on
whatever touches the floor). Walking itself is a `drive` command: the simulators move the body
kinematically, legs do not push it along.

### Unitree Go2 and G1

Unitree Go2 / G1: these two are driven the way the real robots are, over the WebRTC data channel
(what the Unitree apps and the Python driver [unitree_webrtc_connect](https://github.com/legion1581/unitree_webrtc_connect)
send). Put the messages in the action as `webrtc` (one or a list), with the same topic and fields as on the robot:

```python
api.step(env_id, {"webrtc": {"topic": "rt/api/sport/request", "api_id": 1016}})          # Go2: Hello
api.step(env_id, {"webrtc": {"topic": "rt/api/sport/request", "api_id": 1008,
                             "parameter": {"x": 0.5, "y": 0, "z": 0}}})                   # Go2: Move (stops after 1 s)
api.step(env_id, {"webrtc": {"topic": "rt/api/sport/request", "api_id": 7101, "parameter": {"data": 801}}})  # G1: run
api.step(env_id, {"webrtc": {"topic": "rt/api/arm/request", "api_id": 7106, "parameter": {"data": 26}}})     # G1: high wave
api.step(env_id, {"webrtc": {"topic": "rt/wirelesscontroller", "data": {"lx": 0, "ly": 0.5, "rx": 0, "ry": 0, "keys": 0}}})
```

On the real robot the same messages are `await conn.datachannel.pub_sub.publish_request_new(topic, {"api_id": …,
"parameter": …})` and `conn.datachannel.pub_sub.publish_without_callback(topic, data)`. Every answer comes back as a
`webrtc` event (`code` 0 = done). The G1 refuses a state change it can't make with code 1002 and an unknown state with
1003, as the robot does; other refusals of the simulator have code −1 and say why in `msg`.
Go2: the API ids changed with firmware 1.1.7 (MCF, the default here: e.g. HandStand 2044, BackFlip 2043); pick the
older firmware under Setup or with `options={"fw": "classic"}` (`indexGo2.php?fw=classic`); with the older firmware the AI tricks need
`rt/api/motion_switcher/request` `api_id 1002` `{"name": "ai"}` first. Also simulated: `rt/api/vui/request` (LED colour
1007, brightness 1005 / 1006) and `rt/api/obstacles_avoid/request` (1001 on / off, 1003 move with avoidance).
G1: FSM states with 7101 (0 zero torque, 1 damp, 2 squat, 3 sit, 4 locked stand, 200 / 500 / 501 walk, 702 lie → stand
up, 706 squat ↔ stand, 801 run), velocity with 7105 `{"velocity": [vx, vy, ω], "duration": s}`, 7001 / 7002 read the
state, gestures with 7106 on `rt/api/arm/request` (12, 15, 17–27; 99 lowers the arms). On the start square (lane `free`) the G1 starts
sitting on a chair (state 3): send 4 to stand up, then 200 to walk; in front of the chair, facing away from it, 3 sits
it down on the chair again. In the obstacle lanes it starts standing, in state 200.
G1 skin (looks only, the robot behaves the same): `options={"skin": "technopolis"}` (`indexG1.php?skin=technopolis`, or *Skin*
under Setup) gives it silver shells and the light-blue TECHNOPOLIS T-shirt, `"dinno"` silver shells and a T-shirt in the
Dinno logo's colours (black to blue) with the Dinno logo and robotics.dinno.be; the default is `unitree` (white shells).

Go2 GPS map (`common/gpsmap.js`): every metre the dog walks, also in rounds on a track, is a metre along a route of
waypoints on a map. Set the route with `{"gps": {"route": [[lat, lon], …], "start": true}}` (also `stop`, `mode`:
`exact` / `estimated`, `rate` 1 … 10 Hz); `extra.gps` gives the position on the map, the events `waypoint` and
`route_end` say when it gets there. The API's page has no map panel; `options={"map": 1}` shows it anyway (in the
`--show` window). `python go2_track_yolo.py watch --route "50.8417,4.3229 50.8425,4.3240"` follows the path with
YOLO + PPO until the end of the route, with the map in view.

### Unitree Go2-W

Unitree Go2-W (`go2w`, `indexGo2W.php`): the Go2 with a hub-motor wheel on every leg. It drives on its wheels with
skid steering (up to 2.5 m/s, not sideways); over an edge higher than about 4 cm a leg lifts its wheel onto it (up to
17 cm; `extra.lifting` names the legs doing so). Controlled like the Go2 with `{"webrtc": …}`: the Go2 sport API without
the walking gaits and flips, plus VUI lights, obstacle avoidance and the joystick topic; `?fw=classic` selects the API
ids of firmware before 1.1.7. Which sport commands the real Go2-W firmware accepts is not publicly documented.

### Unitree R1

Unitree R1 (`r1`, `indexR1.php`; R1 Air with `options={"variant": "air"}`): a WebRTC robot like the G1, with its own
state machine (`R1_FSM` in unitree_webrtc_connect). It starts standing in Lock (4), the hub: SitDown 7, LieDown 702, the
dances 601–604, KungFu 607, JeetKuneDo 608 and Run 811 are reached from Lock; a refused switch answers 1001 (1003 = no
such state). Damping (1) is accepted from everywhere and drops the robot (event `fell`); StandUp (701) gets it up. It
walks in 811 (the Air reports 830), 813, 814, 815, 816, 830 and 831, driven by `7105` or the joystick. Gestures:
`7106` on `rt/api/arm/request` (not the G1-only hearts 20 and 21). `7108` (SET_MOTION) is not simulated.

### Unitree A2 / AS2

Unitree A2 / AS2 (`a2`, `as2`; `indexA2.php`, `?variant=as2`): industrial quadrupeds driven with the SportClient of
`unitree_sdk2`: `{"sdk": {"api_id": 1008, "parameter": {"x": 0.5, "y": 0, "z": 0}}}` (Move stops after 1 s). Shared:
Damp, BalanceStand, StopMove, StandUp / Down, RecoveryStand, Euler, Move, BodyPosition, SwitchGait, BodyHeight, SpeedLevel,
Left / RightSideGait, HandStand, BipedStand, Front / BackFlip, GetState, SetAutoRecovery; the A2 adds Trajectory 1035 and
ResetEstimator 1111, the AS2 the tricks (Greeting, Heart, Content, Dance1 / 2, Handshake, Stretch, Sit, FrontJump, PushUp,
UpJump) and SwitchJoystick.

### Unitree H2

Unitree H2 (`h2`, `indexH2.php`): the high-level API of `unitree_sdk2` (H2 EDU): `{"sdk": {"service": "sport", "api_id":
7101, "parameter": {"data": 601}}}` or `{"sdk": {"service": "arm", "api_id": 7106, "parameter": {"action_id": 26}}}`. FSM 0
zero torque, 1 damp, 2 squat, 3 sit, 4 stand up, 601 start (walks with SetVelocity 7105); SDK error codes 7302, 7303, 7401,
7402, 7404. It starts seated on a chair.

### AgiBot D1

AgiBot D1 (`agibotd1`, `indexAgiBotD1.php`): the HighLevel functions of AgiBot's D1 SDK, `{"sdk": {"fn": "move", "args":
[0.5, 0, 0]}}`; answers with code 0, 0x3007 (state switch refused) or 0x3013 (speed too large) and the getters' data.

### Booster K1

Booster K1 (`boosterk1`, `indexBoosterK1.php`): the Booster Robotics SDK's `B1LocoClient`: `{"booster": {"method":
"ChangeMode", "args": ["kWalking"]}}`, `{"booster": {"method": "Move", "args": [0.3, 0, 0]}}`, `{"booster": {"method":
"WholeBodyDance", "args": ["kRoundhouseKick"]}}`. Modes Damping 0, Prepare 1 (start), Walking 2, Custom 3, Soccer 4.

### LimX TRON 1

LimX TRON 1 (`tron1`, `indexTron1.php`; wheel feet `tron1w` = `?variant=wheel`; soles with `options={"variant": "sole"}`):
LimX's remote-control interface, JSON over a WebSocket, as `ws` messages: `{"ws": {"title": "request_walk_mode"}}`,
`{"ws": {"title": "request_twist", "data": {"x": 0.5, "y": 0, "z": 0}}}` (resend at 30 Hz; it stops 0.25 s after the last),
`request_stand_mode`, `request_sitdown`, `request_base_height`, `request_emgy_stop`, `request_recover`,
`request_stair_mode` (wheel feet), `request_marktime_mode`, `request_enable_imu` / `_odom`, `request_light_effect`. Point
feet can't stand still: they keep stepping.

### TurtleBot 4

TurtleBot 4 (`turtlebot4`, `indexTurtleBot4.php`, `?variant=lite` for the Lite): driven the ROS 2 way. Put messages
in the action as `ros` (one or a list); names and types as on the real robot (ROS 2 Humble), a namespace in front
(`/robot1/cmd_vel`) is fine:

```python
api.step(env_id, {"ros": {"kind": "topic", "name": "/cmd_vel", "msg": {"linear": {"x": 0.2}, "angular": {"z": 0.0}}}})  # drives 0.5 s
api.step(env_id, {"ros": {"kind": "action", "name": "/undock"}})
api.step(env_id, {"ros": {"kind": "action", "name": "/drive_distance", "msg": {"distance": 0.5, "max_translation_speed": 0.2}}})
api.step(env_id, {"ros": {"kind": "service", "name": "/e_stop", "msg": {"e_stop_on": True}}})
api.step(env_id, {"ros": {"kind": "param", "node": "/motion_control", "name": "safety_override", "value": "full"}})
```

Topics in: `/cmd_vel` (stops 0.5 s after the last message, like a teleop that is let go), `/cmd_lightring`,
`/cmd_audio`, `/hmi/display/message` (standard model). Actions (`irobot_create_msgs`): `/dock`, `/undock`,
`/drive_distance`, `/rotate_angle`, `/drive_arc`, `/navigate_to_position` (odom frame), `/wall_follow`; a new goal,
teleop or `/cmd_vel` aborts the running one, so does a bump or a cliff. Services: `/e_stop`, `/reset_pose`. Parameter:
`/motion_control safety_override` (`none`: cliff stop and 0.306 m/s, `full`: 0.46 m/s, no cliff stop). What the robot
publishes is in `extra.topics` (or `{"kind": "get", "name": "/scan"}`); `/odom` comes from the wheels and drifts when they
slip. The OAK-D picture (`/oakd/rgb/preview/image_raw`) is `frame?view=robot`. The dock stands behind the start square.
The page also has a **Terminal** (button, or `indexTurtleBot4.php?terminal=1`): a browser command line that
understands the common `ros2` commands (`topic list / echo / pub / info / hz`, `action list / send_goal`, `service call`,
`param get / set`, `node list`, `interface show`, `ros2 run teleop_twist_keyboard teleop_twist_keyboard`) with the
output ROS 2 Humble prints, so students can practise the CLI without installing ROS 2. Type `help` for the list.
It also has tabs (＋), `rqt_graph`, `rqt_console` (/rosout with log levels, `--log-level`), `ros2 param dump / load`,
`--params-file`, `ros2 launch` (also your own launch files, written with `nano`), `ros2 bag record / info / play`, the
demo nodes `demo_nodes_cpp talker / listener` and `botopia_demos square_driver / scan_monitor`, and a small file system
in the browser (`ls`, `cd`, `cat`, `cp`, `mkdir`, `> file`, `&&`). `tutorial-ros2-cli.html` follows the ROS 2 Humble
"Beginner: CLI tools" tutorials step by step on it.
Python nodes run in it too (Pyodide, `common/rospython.js`, this simulator's rclpy in `common/pyros/`):
`ros2 pkg create`, `colcon build / test`, `source install/setup.bash`, `ros2 run`, `ros2 launch`, custom `.msg` / `.srv`,
`git clone` of ros2/examples, `wget`, `rosdep`, `ros2 doctor` (`common/rosdev.js`). `tutorial-ros2-python.html` follows the
ROS 2 Humble "Beginner: Client libraries" tutorials in Python on it.

### Husarion ROSbot XL

Husarion ROSbot XL (`rosbotxl`, `indexROSbotXL.php`; `?variant=wheels` for rubber wheels with skid steering): ROS 2 like
the TurtleBot 4, with `ros` messages. In: `/cmd_vel` (geometry_msgs/msg/Twist; `linear.y` drives sideways on mecanum
wheels; stops 0.5 s after the last message), `/leds` (std_msgs/msg/UInt8, bit 0 / 1 = LED1 / LED2), `/set_pose` (resets
the EKF). Out: `/scan` (RPLIDAR S3), `/odometry/filtered` (EKF: wheels plus IMU heading),
`/rosbot_xl_base_controller/odom` (wheels only: it drifts because the mecanum rollers slip), `/imu_broadcaster/imu`,
`/joint_states`, `/battery_state`; service `/controller_manager/list_controllers`. The robot itself has no actions (Nav2
adds them). Same Terminal as the TurtleBot 4.

### Leo Rover

Leo Rover (`leorover`, `indexLeoRover.php`, `?lidar=1` adds the LiDAR accessory): Fictionlab's outdoor rover with a rocker
suspension, driven the ROS 2 way with `ros` messages. In: `/cmd_vel`, `/heading_controller/cmd_vel` (holds the IMU
heading driving straight), `/firmware/wheel_{FL,RL,FR,RR}/cmd_velocity` and `cmd_pwm_duty` (all stop after
`controller.input_timeout`, 500 ms). Out: `/wheel_odom` (wheels only: drifts when the tyres slide), `/merged_odom`
(heading from the IMU), `/imu/data`, `/imu/rpy`, `/firmware/battery`, `/joint_states`, the camera topics (picture:
`frame?view=robot`). Services (`std_srvs/srv/Trigger`): `/reset_odometry`, `/firmware/get_board_type`,
`/firmware/get_firmware_version`, `/imu_filter/reset_calibration`, `/system/reboot`. Parameters on
`/firmware_parameter_bridge` (`controller.*`) and `/heading_controller`. Same Terminal as the TurtleBot 4.

### Roboworks Rosbot 2 and Mecabot 2

Roboworks Rosbot 2 (`roboworks`, `indexRoboworks.php`) and Mecabot 2 (`mecabot`, `?variant=mecabot`): ROS 2 education
robots with a Jetson Orin, an LSLiDAR N10 and an Orbbec Astra camera, running Wheeltec's `turn_on_wheeltec_robot` (ROS 2
Humble). The Rosbot 2 steers like a car (smallest turning radius 0.77 m), the Mecabot 2 has mecanum wheels. `/cmd_vel`
(Twist; on the Rosbot 2 `angular.z` sets the steering angle, the Mecabot also uses `linear.y`) keeps the robot going until
the next message. Published: `/odom`, `/imu`, `/PowerVoltage`, `/scan`, the camera images. Terminal user `wheeltec`;
`ros2 run wheeltec_robot_keyboard wheeltec_keyboard` works as teleop.

### Waveshare UGV Rover

Waveshare UGV Rover (`ugvrover`, `indexUGVRover.php`): two ways in, as on the real rover. `{"serial": {"json": {"T": 1, "L":
0.2, "R": 0.2}}}` sends the JSON commands of the ESP32 sub-controller (T 1 / 11 / 13 motion, 133–137 / 141 gimbal, 132 LEDs,
3 OLED, 130 / 126 feedback, 136 heartbeat: the wheels stop 3 s after the last motion command). `{"ros": …}` is ROS 2
(ugv_ws): in `/cmd_vel`, `/ugv/joint_states` (pan, tilt), `/ugv/led_ctrl`; out `/scan`, `/odom`, `/odom/odom_raw`,
`/imu/data_raw`, `/imu/mag`, `/voltage`, `/image_raw`.

### Husarion Panther and Robotnik SUMMIT-XL

Husarion Panther (`panther`, `indexPanther.php`; `?variant=mecanum`, `?lidar=0`): `/cmd_vel` is TwistStamped (stops 0.5 s
after the last message; teleop: `--ros-args -p stamped:=true`); out `/odometry/wheels`, `/odometry/filtered`, `/imu/data`,
`/joint_states`, `/battery/battery_status`, `/hardware/e_stop`, `/scan` (accessory); services for the e-stop, power, lights
(`/lights/set_animation`) and localization. Robotnik SUMMIT-XL (`summitxl`, `indexSummitXL.php`; `?variant=mecanum`):
namespace `/robot`: `/robot/cmd_vel` (Twist; teleop: `--ros-args -r cmd_vel:=/robot/cmd_vel`) or
`/robot/robotnik_base_control/cmd_vel` (TwistStamped); out odom, IMU, the 270° front laser, joint states, e-stop, battery
and the PTZ camera image. Both are too big for the 60 cm corridor.

### Boston Dynamics Spot

Boston Dynamics Spot (`spot`, `indexSpot.php`, [documentation](https://robotics.dinno.be/docs-boston-dynamics-spot.html)):
the ROS 2 interface of the `spot_ros2` driver. It starts as with `auto_claim`, `auto_power_on` and `auto_stand` (standing,
lease held). Services (`std_srvs/srv/Trigger`, answer `success` + `message`): `/claim`, `/release` (power off first),
`/power_on` (needs the lease, no e-stop), `/power_off` (sits first), `/stand`, `/sit`, `/stop`, `/self_right`,
`/estop/gentle` (sits, then power off), `/estop/hard` (power off at once), `/estop/release`; `/stair_mode`
(`std_srvs/srv/SetBool`: higher, slower steps). Topics in: `/cmd_vel` (Twist, up to 1.6 m/s, stops 0.5 s after the last
message; only with lease, power and standing, else the answer says why) and `/body_pose` (Pose: body tilt and height while
it stands). Out: `/odometry`, `/joint_states`, `/status/feet`, `/status/battery_states`, `/status/power_state`,
`/status/estop`, `/status/leases` and the front camera image.

```python
for srv in ["/sit", "/power_off", "/release", "/claim", "/power_on", "/stand"]:
    api.step(env_id, {"ros": {"kind": "service", "name": srv}})
```

### AgileX Scout Mini, Hunter SE and Bunker

The interface of AgileX's drivers (`scout_ros2`, `hunter_ros2`, `bunker_ros2`, over CAN): in `/cmd_vel` (Twist, stops
0.5 s after the last message), out `/odom`, `/<robot>_status` (speed, battery voltage, motors) and `/scan` (a 360° LiDAR as
in AgileX's research kits). Scout Mini (`scoutmini`, `indexScoutMini.php`,
[documentation](https://robotics.dinno.be/docs-agilex-scout-mini.html)): skid steering up to 2.7 m/s, or mecanum wheels
with `options={"variant": "mecanum"}` (`linear.y` drives sideways, up to 1.4 m/s).

Hunter SE (`hunterse`, `indexHunterSE.php`, [documentation](https://robotics.dinno.be/docs-agilex-hunter-se.html)):
Ackermann steering up to 4.8 m/s with the parameters of `hunter_ros2` (track 0.52 m, wheelbase 0.55 m, central angle at
most 0.384 rad). As the driver it turns `/cmd_vel` into a steering angle: radius `|linear.x| / |angular.z|`, the largest
angle below the wheelbase; `angular.z` alone only turns the wheels.

Bunker (`bunker`, `indexBunker.php`, [documentation](https://robotics.dinno.be/docs-agilex-bunker.html)): the tracked
Bunker Mini 2.0, or the Bunker Pro with `options={"model": "pro"}`: it turns on the spot with track slip (its wheel
odometry drifts in turns) and climbs higher edges than wheels.

### MiR250

MiR250 (`mir250`, `indexMiR250.php`, [documentation](https://robotics.dinno.be/docs-mir250.html)): an industrial AMR,
starting in the warehouse world on its dock. `rest` takes a request of the MiR REST API (`/api/v2.0.0`): `{"rest": {"method":
"GET", "path": "/status"}}`, `PUT /status {"state_id": 3 | 4}` (Ready: go on and clear an error / Pause), `GET /missions`,
`GET /missions/{guid}`, `GET|POST /positions` (a new position also gets a mission "Move to …"), `GET /positions/{guid}`,
`POST /mission_queue {"mission_id": …}`, `GET|DELETE /mission_queue`, `GET /mission_queue/{id}`. The answer comes back as a
`rest` event with the HTTP status in `code`. The map frame is where it starts: x forward, y left, orientation in degrees;
positions Dock (0, 0), Station A (3.5, 0), Station B (3.5, 2, 90°), Station C (1, −2, −90°). Missions run one after the
other while it is Ready; a path blocked for 5 s aborts the mission (state 12, Error). The drive axes drive it by hand.

```python
missions = api.step(env_id, {"rest": {"method": "GET", "path": "/missions"}})["events"][0]["data"]
api.step(env_id, {"rest": {"method": "POST", "path": "/mission_queue", "body": {"mission_id": missions[1]["guid"]}}})
```

### AgileX LIMO

AgileX LIMO / LIMO Pro (`limo`, `indexLimo.php`, [documentation](https://robotics.dinno.be/docs-agilex-limo.html)): a 4.2 kg
ROS 2 education robot with four motion modes on one chassis, chosen with `options={"mode": …}` (`?mode=` on the page):
`diff` (four-wheel differential, the default), `ackermann` (the front wheels steer: `angular.z / linear.x` sets the angle,
inner wheel at most 28°), `tracked` (differential with tracks: more slip, worse odometry) and `mecanum` (`linear.y` drives
sideways). In `/cmd_vel` (Twist, stops 0.5 s after the last message); out `/odom` (wheel odometry), `/imu`, `/scan` (360°),
`/limo_status` (`motion_mode` 0 differential or tracked, 1 Ackermann, 2 mecanum) and the depth camera's colour image.

### Clearpath Jackal and Husky

Clearpath Jackal J100 (`jackal`, `indexJackal.php`, [documentation](https://robotics.dinno.be/docs-clearpath-jackal.html)):
Clearpath's ROS 2 interface (Humble), everything in the namespace of the robot's serial (here `/j100_0000`): in
`/j100_0000/cmd_vel` (Twist, stops 0.5 s after the last message; Jazzy robots take a TwistStamped); out
`platform/odom` (wheels), `platform/odom/filtered` (wheels + IMU), `sensors/imu_0/data`, `platform/joint_states`,
`platform/emergency_stop`, `platform/mcu/status/power` and `sensors/lidar2d_0/scan` (a 270° 2D LiDAR). The e-stop is
under Setup.

Clearpath Husky A200 (`husky`, `indexHusky.php`, [documentation](https://robotics.dinno.be/docs-clearpath-husky.html)): the
same interface in the namespace `/a200_0000` (`/a200_0000/cmd_vel`, `/a200_0000/platform/odom/filtered`, …), 50 kg, up
to 1 m/s; too big for the 60 cm corridor.

### AgileX Ranger Mini 3.0

AgileX Ranger Mini 3.0 (`rangermini`, `indexRangerMini.php`, [documentation](https://robotics.dinno.be/docs-agilex-ranger-mini.html)):
a 75 kg base with four wheel modules, each with a drive and a steering motor. As AgileX's `ranger_ros2` driver, it picks
the motion mode from `/cmd_vel` (Twist, stops 0.5 s after the last message): `linear.y` with `linear.x` → parallel (crab,
`angular.z` ignored), `linear.y` alone → side slip (wheels at 90°), `angular.z` alone or a curve tighter than 0.48 m → spin,
else dual Ackermann (inner wheel at most 40°). The wheels first turn to the new mode's angles. Out `/odom`, `/motion_state`
(`motion_mode` 0–3), `/system_state`, `/actuator_state` (8 motors: rpm, current, pulses), `/battery_state` (48 V) and `/scan`
(a 360° LiDAR on the deck, as in AgileX's research kits).

```python
api.step(env_id, {"ros": {"kind": "topic", "name": "/cmd_vel", "msg": {"linear": {"x": 0.43, "y": 0.25}}}})   # crab 30° left
```

### DrRobot Jaguar 4x4 Wheel

DrRobot Jaguar 4x4 Wheel (`jaguar`, `indexJaguar.php`, [documentation](https://robotics.dinno.be/docs-drrobot-jaguar-4x4.html)):
a 20 kg outdoor robot with four 80 W wheel motors, without a computer on board: a PC on its Wi-Fi drives it with text lines over
TCP (192.168.0.60:10001). It starts in the park. The simulated controller speaks the real protocol, line for line:

| Line | What |
|---|---|
| `MMW !MG` / `MMW !EX` | release / set the E-Stop (it starts in E-Stop, `FF=16`) |
| `MMW !M 300 -300` | open-loop power −1000 … 1000, left then right channel; forward is negative on the right one |
| `PING` | at least every 0.2 s (the simulator: 0.3 s), else the motors stop (event `watchdog`) |
| `MM0 ?V`, `?C`, `?A`, `?T`, `?FF` · `MM0 ~KP` · `MM0 ^KP 1 255` | ask a value, read a setting, set one: the echo, then the answer or `+` |

It streams, unasked: `#seq,YAW,y,GYRO,gx,gy,gz,ACC,ax,ay,az,COMP,mx,my,mz,ADC,0` (IMU, 50 Hz, raw: ÷ 14.375 = °/s, ÷ 256 = g),
`MM0 C=fl:fr` / `MM1 C=rl:rr` (encoders, 5 Hz, 300 counts a wheel turn of Ø 270 mm), `MMx V= / FF= / T= / A= / P= / S=` and
`$GPRMC` / `$GPGGA` (GPS, 1–5 Hz). In an action a line is `{"tcp": "MMW !MG"}`; `{"tcp": {"line": "MMW !M 300 -300", "hold": 2}}`
sends PINGs for 2 s after it. `drive` works like the keys: PING and `MMW !M` ten times a second (the first one releases the E-Stop).
The page's terminal has `nc 192.168.0.60 10001`, the `jaguar` commands and Python with `socket`.

**PC programs and the real Jaguar:** `python/jaguar_bridge.py` connects the page (WebSocket `ws://localhost:8772`) to the real
robot (TCP 192.168.0.60:10001), and opens 127.0.0.1:10001 on your PC for programs that should drive the simulated one. On the
page (card *Real Jaguar*) you then see the real robot's IMU, encoders and GPS, let it follow the simulator, or drive it with the
3D Jaguar following its encoders and gyro. `python/jaguar_cli.py` has the terminal's `jaguar` commands for your PC.

```bash
pip install websockets
python python/jaguar_bridge.py                 # PC on the DriJaguar Wi-Fi (192.168.0.104): the real robot at 192.168.0.60
python python/jaguar_bridge.py --no-robot      # only PC programs <-> the simulator
python python/jaguar_bridge.py --fake          # a pretend robot, to try the page's connector
python python/jaguar_cli.py --ip 127.0.0.1 status          # the simulated Jaguar (through the bridge)
python python/jaguar_cli.py drive 300 300 2                # the real one
python robot.py --ip 127.0.0.1                             # a program written for the real robot, on the simulator
```

### Robotnik RB-KAIROS+ and RB-VOGUI+

Robotnik RB-KAIROS+ (`rbkairos`, `indexRBKairos.php`) and RB-VOGUI+ (`rbvogui`, `?variant=vogui`; `?arm=ur10e` for a UR10e):
mobile manipulators with a UR5e arm and a Robotiq 2F-85 gripper, and a table with blocks in front of the start square. ROS 2
under `/robot`: `robotnik_base_control/cmd_vel` (TwistStamped) or `cmd_vel_unstamped` (Twist), stopping 2 s after the last
message; the actions `joint_trajectory_controller/follow_joint_trajectory` and `robotiq_gripper_controller/gripper_cmd`;
out odom, two lasers, IMU, joint states, the RGB-D camera. For Cartesian moves use `{"arm": {"tcp": {"x", "y", "z",
"yaw"}}}` or `{"arm": {"skill": "pick"}}`.

### Petoi Bittle X

Petoi Bittle X (`bittle`, `indexBittle.php`): driven with OpenCat serial tokens, as over USB / Bluetooth (115200 baud).
Put them in the action as `serial` (one string or a list): `{"serial": "kwkF"}`, `{"serial": "ksit"}`,
`{"serial": "m0 45"}`, `{"serial": "i 8 -70 9 -70"}`, `{"serial": "j"}`. The answer comes back as a `serial` event
(`line`, `reply`): the token letter, or the joint table for `j`. All 56 skills of the real robot are there with its own
joint data (OpenCat, MIT); gaits keep going until the next token. Joints: 0 head, 8–11 shoulders, 12–15 knees (left
front, right front, right back, left back). The page has a Serial monitor (`indexBittle.php?serial=1`) to type tokens.

### Lynxmotion Phoenix hexapod

Lynxmotion Phoenix 3DOF (`phoenix`, `indexPhoenix.php`, [documentation](https://robotics.dinno.be/docs-lynxmotion-phoenix.html)):
six legs with a coxa, femur and tibia servo each, on an SSC-32U. The drive axes walk it as the Phoenix code does (inverse
kinematics, `fwd` / `strafe` / `turn`); the gait (tripod, triple tripod, ripple, wave) and the body height are under Setup.
`serial` takes SSC-32U lines: `{"serial": "#24P1600 #25P2000T500"}` (a group move, pulses 500–2500 µs), `Q` (`+` moving,
`.` done), `QP 24` (pulse / 10), `STOP 24`, `VER`. A servo line (or a servo angle in `motors`) takes the servos away from the
Phoenix code until the next drive command. Channels: right rear 0 1 2, right middle 4 5 6, right front 8 9 10, left rear
16 17 18, left middle 20 21 22, left front 24 25 26; the right side is mirrored (a longer pulse turns the other way). The body
moves over the feet that stand on the floor, so your own gait walks as well as it is written.

### JetBot and JetRacer

JetBot (`jetbot`) and JetRacer (`jetracer`): driven with the Python lines of their Jupyter notebooks, as
`{"python": {"code": "robot.forward(0.3)"}}` or `{"python": {"code": "car.throttle = 0.25"}}`; a script with `time.sleep()`
runs in simulation time and the answer comes back as a `python` event. JetBot: `Robot()` with forward / backward / left /
right / stop / set_motors and `left_motor` / `right_motor` `.value`. JetRacer: `NvidiaRacecar()` with steering, throttle,
steering_gain (−0.65), steering_offset and throttle_gain (0.8).

### T2M Pirate Shooter

T2M Pirate Shooter (`pirateshooter`, `indexPirateShooter.php`): a 1/10 4x4 RC buggy with the students' AI upgrade, a Jetson Orin
and an OpenCR board. A simulated OpenCR takes the serial lines of `rccar.ino` and answers as the sketch does:
`{"serial": "A,120"}` (steering 50–140: 90 straight, 120 full left, 60 full right), `{"serial": "T,20"}` (throttle −100…100 %),
`C`, `STOP`, `S`, `O`, `OC/OL/OR,<n>` (trim), `TEST`. `drive` works like the keys: `turn` gives `A,90+turn·30`, `fwd` gives
`T,fwd·max throttle` (`?maxt=20`, Setup). `?stand=1` puts the car on a stand (its wheels turn, it stays put). The page also
drives the real OpenCR over USB (Web Serial, 57600 baud) when *Wheels follow the simulator* is ticked, with the car on a stand.

### HIMOTO Drift X

HIMOTO Drift X (`driftx`, `indexDriftX.php`, [documentation](https://robotics.dinno.be/docs-himoto-drift-x.html)): a 1/18 4WD
drift car made self-driving with a Seeed XIAO ESP32S3 Sense (camera, Wi-Fi) and an L298N H-bridge. A simulated XIAO takes the lines of
`xiao_drift.ino` and answers as the sketch does: `{"serial": "A,120"}` (steering 60–120: 90 straight, 120 full left, 60 full right),
`{"serial": "T,20"}` (throttle −100…100 %), `C`, `STOP` (brake), `S`, `IP`, `OC,<n>` (trim), `MAX,<n>` (throttle limit), `W,<ms>`
(watchdog, 0 = off). The **watchdog** stops the motor 1 s after the last line: send `T` again and again, or `W,0` first. `drive`
works like the keys: `turn` gives `A,90+turn·30`, `fwd` gives `T,fwd·max throttle` (`?maxt=30`). The hard drift tyres slide
(`?tyres=rubber` grips): `slip_deg` in the telemetry is how sideways the car goes. It starts on the Kyosho Mini-Z race track (`botopia`); `options={"world": "course"}` (or `?world=course`) puts it on the obstacle course. `?stand=1` puts it on a stand, `?nobody=1` takes the
body off. The page drives the real XIAO over USB (Web Serial, 115200) when *Wheels follow the simulator* is ticked. The sketch and
`drift_wifi.py` (the car over Wi-Fi: its camera at `http://192.168.4.1:81/stream`, commands at `/cmd?l=A,110`) are on the
documentation page.

### DJI Tello EDU

DJI Tello EDU (`tello`, `indexTello.php`): a mini drone flown with the Tello SDK, the text commands it takes over UDP:
`{"tello": "command"}`, `{"tello": ["takeoff", "forward 100", "cw 90"]}`, `{"tello": "go 0 0 60 40 m1"}`. The answer
(`ok`, `error …`, a value) comes back as a `tello` event when the drone is done; `extra.state` is its state packet. The drive
axes include `up` (`{"drive": {"up": 0.5}}` climbs, like the keys R / F). The page has a Python bar with real Python
(Pyodide) and djitellopy (`?python=1`), two cameras (`view=robot`: the front camera, `view=bottom`: the black-and-white
bottom camera), four mission pads on the floor, a height limit (`?maxh=2`, Setup) and 10 minutes of flight on a battery
(`?flight=540` starts with 9 minutes used).

**A real Tello EDU:** `python/tello_bridge.py` runs on a PC on the Tello's Wi-Fi and connects the page to the drone
(WebSocket `ws://localhost:8771` ⇄ UDP 8889 / 8890 / 11111). The page then shows its battery, height and video, and with
*Fly the real Tello* ticked every command of the page goes to the real drone as well.

```bash
pip install websockets av pillow
python tello_bridge.py                       # the Tello at 192.168.10.1 (its own Wi-Fi)
python tello_bridge.py --ip 192.168.1.42     # a Tello EDU that joined your network
python tello_bridge.py --fake                # try the page without a drone
```

### XGO-Lite3 / XGO-Mini2

XGO-Lite3 / XGO-Mini2 (`xgo`, `xgomini`, `indexXGO.php[?variant=mini]`): desktop robot dogs with an arm and gripper,
programmed with the xgolib Python calls: `{"xgo": "dog.move_x(15)"}`, `{"xgo": "dog.action(13)"}`, `{"xgo": ["dog.arm(130, 0)",
"dog.claw(255)"]}`. Actions 1–24, the grabs 128–130 and reset 255; a red cube in front of the dog can be picked up (action
130). The page has a Python bar (`?python=1`).

### Quadruped arm

Quadruped arm (OpenMANIPULATOR-X on the top plate): `arm_joint1` turns it (−162…162°), `arm_joint2`–`arm_joint4`
pitch shoulder, elbow and wrist (+ = forward / down; −103…90, −54…79, −103…117°). A commanded value is the servo's
goal: it moves there at its own speed (90°/s), so read `deg` to see where it is. `gripper` is the opening between the
fingers in mm; closing on one of the 3 cm cubes stops at the cube and holds it (`extra.arm.holding`). The cubes
lie 30 cm in front of the robot after a reset; the arm reaches the floor when the robot lies down. A cube released above the top plate lands on the back and rides along (`onBack`).
`extra.reach.tip` is the fingertip position in the arm frame (m; origin on top of the arm's base plate, x forward,
y left, z up). An action `{ "reach": { "x", "y", "z" } }` (same frame) shows a red target ball there and adds
`extra.reach.target` and `dist`; `{ "reach": null }` hides it (tutorial 10). Page option `camview: "arm"` puts the
3D view close to the arm.

### Universal Robots UR5e

Universal Robots UR5e (`ur5e`, `indexXArm6.php?model=ur5e` or `indexUR5e.php`,
[documentation](https://robotics.dinno.be/docs-universal-robots-ur5e.html)): the arm page (table, blocks, wrist D435,
external camera, pick and place) with UR's own DH parameters and a Robotiq 2F-85. `urscript` takes a URScript program (a
string, lines or `;`), as sent to port 30002: a new program stops the running one. Simulated: `movej(q | p[…], a, v, t)`,
`movel(p[…] | q, a, v)` (a straight line from where the tool is when it starts), `movep`, `sleep(t)`, `stopj`, `stopl`,
`rq_open()`, `rq_close()`, `rq_move_and_wait(0…255)`, `textmsg(…)`, `get_actual_joint_positions()`, `get_actual_tcp_pose()`,
`d2r`, `r2d`, `pi`. Poses: metres and a rotation vector in the table's base frame (x forward, y left, z up);
`p[0.3, 0, 0.2, 3.1416, 0, 0]` points the gripper down. With all joints 0 the arm lies along −x, as on the robot.

```python
api.step(env_id, {"urscript": "movel(p[0.35, 0.1, 0.25, 3.1416, 0, 0], a=1.2, v=0.25)
rq_close()"})
```

### Franka Research 3

Franka Research 3 (`fr3`, `indexXArm6.php?model=fr3` or `indexFR3.php`,
[documentation](https://robotics.dinno.be/docs-franka-research-3.html)): the arm page with Franka's kinematics (seven
joints; joint 4 always bent, joint 6 between 31° and 259°), the Franka Hand (0–80 mm) and the wrist D435. `python` takes
lines of a **franky** script; moves run one after the other, as `robot.move()` waits: `robot.move(JointMotion([7 × rad]))`
(refused outside the limits, as libfranka does), `robot.move(CartesianMotion(Affine([x, y, z], [qx, qy, qz, qw])))` (a
straight line; `[1, 0, 0, 0]` points the hand down; without a quaternion it keeps its orientation; `ReferenceType.Relative`
adds to the present pose in the base frame), `robot.relative_dynamics_factor = 0.1`, `gripper.move(0.04, 0.1)`,
`gripper.grasp(…)`, `gripper.open(…)`, `time.sleep(…)`, `print(robot.current_joint_state.position)`,
`print(robot.current_cartesian_state.pose.end_effector_pose)`, `gripper.width`. Imports and `Robot(...)` are accepted.

```python
api.step(env_id, {"python": "robot.relative_dynamics_factor = 0.2\nrobot.move(CartesianMotion(Affine([0.4, 0.1, 0.25], [1, 0, 0, 0])))"})
```

### xArm 5 Lite, UFACTORY 850 and SES-PRO

xArm 5 Lite (`xarm5`, `indexXArm6.php?model=5`) and UFACTORY 850 (`xarm850`, `indexXArm6.php?model=850`): the same table and
API as the xArm 6; the xArm 5 Lite has 5 joints (the gripper points down and turns, but cannot tilt), the 850 has 6 joints
and 85 cm reach. Lynxmotion SES-PRO (`sespro`, `indexSESPro.php`): a 900 mm 5DoF arm driven over the LSS serial protocol:
`{"serial": "#2D450T1500"}` moves servo 2 to 45.0° in 1.5 s, `#1QD` asks a position, `#254L` makes all servos limp (the arm
sags), `#254H` holds. IDs 1 base, 2 shoulder, 3 elbow, 4 wrist pitch, 5 wrist rotate, 6 gripper.

### xArm 6 / xArm 7

xArm 6 / xArm 7 (UFACTORY, on a table with 12 coloured 4 cm blocks; the xArm 7 page is `indexXArm6.php?model=7` or
`indexXArm7.php`): the joints are servos (degrees, limits as on the real arm: xArm 6 J2 −118…120, J3 −225…11, J5 −97…180;
xArm 7 J2 −118…120, J4 −11…225, J6 −97…180; the others ±360); a commanded value is the goal, the joints move there
together at up to 120°/s. Base frame: origin on the table under the arm, x forward, y left, z up (m). `reset` puts the
arm in its look-down pose and scatters the blocks (with a `seed`: always the same). The robot camera
(`view: "robot"`) is the RealSense D435 on the wrist; the gripper's fingers show at the bottom of its picture. A second
D435 on a tripod beside the table sees the whole arm (`view: "external"`, also shown on the page).
The wrist D435 also gives **depth**, aligned to its colour picture as a real D435 with `rs.align`: z-depth (along
the optical axis) in millimetres, nothing nearer than 10.5 cm or further than 10 m (0). `view: "depth"` is a
lossless PNG (mm = red · 256 + green), `"depthcolor"` the same in colour. The external camera is a normal camera
(no depth). On the page, *RealSense depth view* shows the wrist picture-in-picture as depth (also `?depth` in the
address), with a cross on what lies under the gripper: along the gripper's axis from the fingertips, the first table
or block surface, with the depth the D435 measures there, how far it is under the fingertips and what it is. The
cross sits where the camera sees that point, which is not the middle of the picture when the gripper is low (the
camera is beside the gripper). The same in the telemetry: `extra.below = { depth_mm, gap_mm, height_mm, block, u, v }`
(u, v: the pixel in a 640 × 480 wrist picture). The simulated cameras are pinhole cameras with a 55° vertical field
of view (4 : 3).

### The real xArm 6 / 7: one Python interface, digital twin, follow

`python/xarm_robot.py` gives the simulator and the real arm the same Python interface, so one script runs on both:

```python
from xarm_robot import connect, deproject
arm = connect("sim", model=7)     # the xArm 7 simulator · "10.2.172.20": the real arm (6 or 7 axes: it asks) · "fake": a pretend arm
arm.move_to(0.35, 0.0, 0.20)      # gripper point (m, base frame), gripper pointing down, inverse kinematics in Python
arm.set_gripper(75); arm.move_to(0.35, 0.0, 0.027); arm.set_gripper(0)
print(arm.joints(), arm.tcp())
jpeg = arm.camera()               # wrist camera: sim image or the real RealSense D435 over USB
jpeg = arm.camera("external")     # the fixed camera that sees the whole arm
mm = arm.depth()                  # wrist depth: numpy uint16, millimetres, 0 = none (sim or the real D435)
print(arm.below())                # simulator: what lies under the gripper (depth_mm, gap_mm, height_mm, block, u, v)
intr = arm.intrinsics()           # fx, fy, cx, cy (pixels) of those pictures
x, y, z = deproject(intr, 320, 240, mm[240, 320])   # a pixel → a point in the camera frame (m, RealSense axes)
```

The real wrist RealSense streams colour and depth and aligns the depth to the colour picture (`rs.align`), so
`camera()` and `depth()` match pixel for pixel, as in the simulator. `depth_colormap(mm)` makes a coloured picture of
it. The real external camera is a normal USB webcam: `connect(ip, wrist_serial="…", external_camera=1)` (its OpenCV
index; `pip install opencv-python`); `depth("external")` is None.

Every move is checked first (joint limits, fingertips at least 5 mm above the table). For the xArm 7 the inverse
kinematics keeps the extra joint (upper arm roll, joint 3) near 0° and the base and flange within ±270°. The real arm needs
`pip install xarm-python-sdk` (and `pyrealsense2` for the camera); `python/xarm_demo.py --robot sim | fake | <ip>`
is a pick-and-place example. With a depth camera it first looks at the place and measures how high the stack there
is (the table beside it in the same picture rows minus the highest spot round the middle), then puts the block on
top: it builds a tower without knowing where the blocks are (`--no-depth`: use `--under` instead).

A web page can't reach the arm's control box itself (plain TCP, port 502), so `python/xarm_bridge.py` runs on a PC
in the robot's network and connects the simulator page (WebSocket `ws://localhost:8770`) to the arm (SDK) and the
camera (USB):

```
pip install xarm-python-sdk websockets pyrealsense2 pillow numpy
python xarm_bridge.py --list-cameras         # the RealSense serial numbers (two cameras on one PC)
python xarm_bridge.py --ip 10.2.172.20 --wrist-camera <serial> --external-camera <serial>
python xarm_bridge.py --fake --model 7       # try the page without a robot (pretend arm and cameras)
```

On the page of the same model, *Real robot → Connect* (the bridge refuses to follow a page of the other model). The real arm then shows as a blue see-through **digital twin**; tick *Real arm
follows the simulator* to make it follow (joint online trajectory planning, mode 6; at most `--max-speed` °/s, 30 by
default). It stops on STOP / Esc, when the page closes or goes quiet for 1 s, and when a target would bring the
fingertips below `--min-z`. *Copy real pose* puts the simulated arm where the real one is. Chrome asks once to allow
the page to reach `localhost` (local network access). Set the arm's own collision sensitivity and safety boundary
in UFACTORY Studio and keep the emergency stop within reach: the simulator doesn't know what else is on the table.

### OpenMANIPULATOR-X

OpenMANIPULATOR-X (`omx`, `indexOpenManipulator.php`, ROBOTIS): the same table scene, smaller (reach 38 cm), with
six tower-game blocks (75 × 25 × 15 mm, lying flat; `yawDeg` is the direction of their length), stacked at
the start (and on reset) straight in front of the arm at x = 155 mm. The gripper takes a block
across its 25 mm width, so it can only pick one that lies lengthwise towards the arm (within about 15°); blocks stack.
Joint 1 turns the arm (−162…162°), joints 2–4 pitch it (+ = forward / down; −103…90, −54…79, −103…117°), up to 90°/s;
`gripper` 13–65 mm (13 = closed). With four joints the gripper always points the way the base turns: `tcp.pitchDeg` is how steeply it
points down (90 = straight down). Views: `robot` (wrist camera) and `external` (a camera on a tripod, the whole arm);
`real`: the USB camera on the real gripper, when it is switched on (*Camera on* in the Real robot panel) in that same
browser window (a page the API opens itself in headless Chrome has no camera: there it is a black "camera off" picture).

Camera vision (the page's *Camera vision* panel): a fixed camera on a pole in the middle of the plate's front border (in the
simulator 400 mm up by default: URL `camx`, `camy`, `camh` in mm; `frame?view=external` in the API) looks at the
calibration sheet. *Look* stands the arm up out of its way (leaning back, the base does not turn), finds the blocks lying flat in the picture (colour
against the background, the size of a block); only blocks whose middle is on the sheet count. The simulated blocks go
there, in the colour it saw; the ones it didn't see are taken away (hidden). The analysed picture shows the found pixels
orange and the sheet's outline blue. The blocks are plain wood (as the real ones; `?colors=1`: six colours): wood looks
like the wooden plate, so it only looks on the sheet, with the white paper as the background. Keep the blocks wholly on
the sheet and a little apart (touching ones look like one big block). *Pick (camera)* looks, then picks the block
nearest to the gripper. With the real camera on it uses the real picture, and waits until the following real arm is out
of the way. Picture → table: through the simulated camera, or for the real camera after *Calibrate* with the A4
calibration sheet (`img/calibration-sheet-a4.pdf`, print at 100 %; `img/calibration-sheet.svg`, also on the simulated
plate): landscape in front of the arm, its top edge flush with the plate's front border (the plate: 46 × 34 cm, its
front border 27 cm in front of the bottom servo), its dashed centre line on the robot's centre line. The page finds its
black squares by itself (at least 5 of the 6; else you click them, right-click: not in the picture), tries every way
they can be named and keeps the best, unmirrored fit (a least-squares homography of the table plane), and works out
where the camera is from it (for the blocks' height); kept in the browser, shift-click forgets it.
It sees single blocks lying flat, not stacks.

Standing row (tutorial 11, `python/stand_rl.py`): `{ "rowPick": dz }` takes the top block of the stack (gripper straight
down), `{ "rowPlace": dz }` stands the held block upright on the next free spot of an arc 240 mm from the arm, 40° left to 5° right
(the orange marks). dz is a height correction in mm on the page's own idea of the height; use `until: "idle"`.
`extra.row`: `source` (blocks left on the stack), `standing`, `slots` [{x, y, block, mm, deg}] and `last` (the last
skill: `result` grasped / missed / pushed / table, standing / fell over / pressed; `tipsMm`, `gapMm`, `offMm`). Page
settings (create `options`): `zerror` (mm: a hidden calibration error, the arm is higher than it thinks), `zsag` (mm
lower per 100 mm further out), `rowr` (the arc's radius, mm).

Coordinates in mm (the page's *Go to* panel, and the action `{ "moveTo": { "x": 200, "y": 0, "z": 0, "pitch": 90 } }`):
x from the first servo's axis (+ forward, − behind), y + right / − left (seen from behind the arm), z the lowest point of
the fingers above the base plate (0 = touching it); `pitch` is optional (degrees down, 90 = straight down; left out: as
steep as it can). The arm goes there in a straight line when it can, else with a joint move; points it can't reach (also
the ±18° straight behind it: the base turns ±162°) are refused. `extra.pos_mm` gives where the fingers are now.

Collisions: every move (keys, sliders, *Go to*, *Pick*, the API, and so the real arm when it follows) is checked
against the base plate, the arm's own foot and the upper arm (folding back onto itself); not against the blocks (the
gripper has to reach in among them to pick one up). A move that would hit something stops just before it, with the
reason (e.g. "the forearm would hit the arm's own foot"). *Go to* and *Pick* plan around obstacles: straight, else a joint move,
else up, across and down again, else by way of the look-down pose; with no collision-free way the move is refused.
The simulator only knows its own table, not what stands next to the real arm.

Real OpenMANIPULATOR-X (digital twin): upload [`opencr/botopia_omx/botopia_omx.ino`](https://robotics.dinno.be/opencr/botopia_omx/botopia_omx.ino) to the OpenCR board (Arduino IDE,
board "OpenCR", library "Dynamixel2Arduino"; Dynamixel IDs 11–15, 1 Mbps). On the page, *Real robot → Connect OpenCR
(USB)* opens the board's USB port from the browser (Web Serial: Chrome or Edge, over https or localhost). The real arm
then shows as a blue see-through twin; *Torque off* lets you move it by hand; *Real arm follows the simulator* (ticked
by you) sends the joint angles 20×/s, the board moves the arm at most 30°/s and stops by itself when no goal comes for
0.5 s. Protocol (text lines): browser → board `HELLO`, `TORQUE 0|1`, `FOLLOW 0|1`, `STOP`, `SPEED <°/s>`,
`GOAL j1 j2 j3 j4 gripper_mm`; board → browser every 50 ms `{"q":[…],"g":mm,"graw":raw,"torque":0|1,"follow":0|1,…}`.
The gripper moves only after you have calibrated it in the sketch (`GRIP_RAW_CLOSED`, `GRIP_RAW_OPEN`).

---

## Notes

* The models are made for comparing drive types and for learning, not to be exact copies of
  real robots: wheels and tracks have climb limits per wheel size and drive type, slopes have a
  maximum, legs are kinematic (they place the body, they do not push it).
* Same `seed` + same actions = the same result, for every robot (checked).
* One environment per browser tab; Chrome renders nothing between steps, so many environments fit
  on one PC. `frame` renders on demand.

---

## For maintainers

**Regenerate `robots.json`** (the robot list of api.html) after adding or changing a robot. It needs the
simulator site on disk:

```bash
node server.mjs --port 8795 --browsers 1 --site ../robotSimulator      # in another terminal
python python/make_api_docs.py --url http://localhost:8795 --site ../robotSimulator
python python/make_api_docs.py --url http://localhost:8795 --site ../robotSimulator --robots jaguar   # only one robot, merged in
```

A new robot page also needs its id in `ROBOTS` of `server.mjs` (e.g. `jaguar: 'indexJaguar.php'`).

**On the web server** this repository is the `api/` folder of the site, so `index.php` is reached at
`https://robotics.dinno.be/api/index.php`:

```bash
cd /var/www/robotics
git clone https://github.com/mdequanter/robotSimulatorAPI.git api
sudo mkdir -p api/data && sudo chown www-data:www-data api/data     # PHP keeps the sessions here
```
