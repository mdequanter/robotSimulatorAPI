# Botopia simulator API

Control every Botopia robot simulator from code and read back what its motors, encoders and IMU
report, plus incidents such as collisions or a fall. Made for training reinforcement-learning
models.

This repository holds what you need on your own PC: the API server (`server.mjs`), the Python client
and tutorial scripts (`python/`) and trained models (`python/models/`). The simulators themselves are
on [https://robotics.dinno.be/](https://robotics.dinno.be/); the server fetches their pages from there (see `--site` below).

```bash
git clone https://github.com/mdequanter/robotSimulatorAPI.git
cd robotSimulatorAPI
node server.mjs
```

All commands below are run from inside this folder.

**New here?** Start with [Getting started](https://robotics.dinno.be/tutorial-getting-started.html): install the tools, run the simulators
locally, your first API program, with or without a GPU.

**Tutorials:** [1. train a robot to drive the corridor as fast as possible](https://robotics.dinno.be/tutorial-rl-corridor.html)
(PPO, stable-baselines3, `python/corridor_rl.py`) · [2. learn to drive the ε track from the robot's
camera](https://robotics.dinno.be/tutorial-rl-vision.html) (PPO with a CNN, `python/track_vision_rl.py`) · [3. YOLO segmentation
finds the path, RL keeps the robot in the middle](https://robotics.dinno.be/tutorial-yolo-track.html) (Ultralytics, `python/track_yolo.py`) · [4. a sorting job: the quadruped learns which skill to use
when](https://robotics.dinno.be/tutorial-rl-sorting.html) (Q-learning over skills, `python/sort_rl.py`) · [5. the arm learns to touch a
point](https://robotics.dinno.be/tutorial-rl-arm-reach.html) (SAC + Hindsight Experience Replay, `python/arm_reach_her.py`).

There are two ways in. Both use the same JSON formats.

| | **Node API** (`server.mjs`) | **PHP API** (`index.php`) |
|---|---|---|
| For | training | remote control / monitoring |
| Runs | on your PC; it starts Chrome itself (headless) | on the web server, next to the site (https://robotics.dinno.be/api/index.php) |
| Simulation | **step by step**: time only advances when you call `step` | in real time, in a normal browser tab |
| Reproducible | yes, with a `seed` | no |
| Speed | ~700 steps/s for one environment, ~1000 for several (Python client, measured on a laptop) | ~5 updates/s |
| Several robots at once | yes, one call can step all of them | one session per open browser tab |

**On the website:** [api.html](https://robotics.dinno.be/api.html) documents every robot with every command (made from the simulators
themselves: `robots.json`, regenerate it with `python python/make_api_docs.py --url http://localhost:8795 --site ../robotSimulator`
while a server runs; it needs the simulator site on disk) and drives them live through the PHP API. `info` now also lists `commands`: every button of a robot's
console as a ready-made action `{group, label, line, action}`.

All simulators take part: `wheeled`, `bike`, `synchro`, `mecanum`, `kiwi`, `swerve`, `tracked`,
`skid`, `ackermann`, `4wd`, `whegs`, `quadruped`, `mini`, `wheellegs`, `ostrich`, `humanoid`, `snake`, `create3`,
the Unitree robots `go2`, `go2w`, `g1`, `r1`, `a2`, `as2` and `h2`, the ROS 2 robots `turtlebot4`, `rosbotxl`, `leorover`,
`roboworks`, `mecabot`, `ugvrover`, `panther`, `summitxl`, `rbkairos` and `rbvogui`, the Petoi `bittle`, `jetbot`, `jetracer`,
`xgo`, `xgomini`, `agibotd1`, `boosterk1`, `tron1`, `tron1w`, the arms `xarm5`, `xarm850` and `sespro`,
and the robot arms `xarm6`, `xarm7` and `omx` (OpenMANIPULATOR-X).

---

## Quick start (training)

Needs **Node.js 22 or newer** and **Google Chrome** (or Chromium / Edge). No `npm install`.

```bash
node server.mjs          # http://localhost:8780/api/
node server.mjs --show   # same, with a visible Chrome window to watch
```

Options: `--port 8780`, `--chrome "path/to/chrome"` (or the `CHROME` environment variable).
The server also serves the simulator pages, so no PHP is needed for training. It takes them from
`--site`: by default `https://robotics.dinno.be/` (fetched once per page and kept in memory, so the
first start of a robot needs internet), or the folder above this repository when that holds the
simulator site (as on the web server, where this repository is the site's `api/` folder). To try local
changes to the simulators: `node server.mjs --site ../robotSimulator`.

Python (standard library only; `numpy` + `gymnasium` are optional):

```python
import sys; sys.path.append("python")
from botopia import BotopiaEnv

env = BotopiaEnv(robot="ostrich", lane="stairs", dt=0.05, max_steps=400)
obs, info = env.reset(seed=1)              # obs: list of numbers (see "Observation vector")
for _ in range(400):
    obs, reward, terminated, truncated, info = env.step([0.6, 0.0])   # env.action_names == ['fwd', 'turn']
    if terminated or truncated:
        break
env.close()
```

* `BotopiaVecEnv(["skid", "tracked", "ostrich"], lane="step6")`: several environments, one HTTP
  call per step. Use this for speed.
* `BotopiaGymEnv(...)`: Gymnasium wrapper (`Box` spaces, passes `gymnasium.utils.env_checker`),
  `render_mode="rgb_array"` returns the camera image (needs Pillow).
* `action_mode="motors"`: control the motors directly (PWM duty per DC motor, angle per servo)
  instead of the robot's own drive commands. See the table below for which motors.

Run `python python/botopia.py` for a small demo (random driver + 4 robots).

---

## Node API

Base URL `http://localhost:8780/api`. All bodies are JSON.

| Method | Path | Body | Answer |
|---|---|---|---|
| GET | `/robots` | | `[{ robot, page }]` |
| GET | `/envs` | | open environments |
| POST | `/envs` | `{ robot, lane?, seed?, options? }` | `{ id, info, obs }` (`options`: page settings, e.g. `{ "vmax": 1.8 }` = Ackermann top speed in m/s) |
| POST | `/envs/{id}/reset` | `{ lane?, seed? }` | `{ obs }` |
| POST | `/envs/{id}/step` | `{ action, dt?, frame? }` | `{ obs, events, reward, done }` |
| POST | `/envs/step` | `{ steps: [{ id, action, dt? }] }` | `{ results: [...] }` (all in parallel) |
| GET | `/envs/{id}/obs` | | current observation |
| GET | `/envs/{id}/info` | | robot, actions, motors, lanes, event types |
| GET | `/envs/{id}/frame?quality=0.8` | | JPEG image of the 3D view |
| GET | `/envs/{id}/frame?view=robot&width=320` | | JPEG of what the robot's own front camera sees (4:3, `width` 32–1280) |

`step` (also in the batch call) and `reset` accept `"image": { "view": "robot", "width": 64, "quality": 0.9 }`:
the answer then also has `"image"`, that JPEG in base64, without an extra request (training from
pixels); `"view": "mask"` gives the track band in the robot camera's view, white on black (PNG): a
ready-made label for segmentation models. `reset` on the ε track also takes `"start"` (metres along the
track), `"offset"` (metres beside the centre line, + = left) and `"yaw"` (degrees off the track direction).
`node server.mjs --browsers N` starts N headless Chrome instances that draw in parallel (default:
one per 4 CPU cores, at most 4).
| DELETE | `/envs/{id}` | | closes the environment |

`dt` is how long one step lasts in simulated seconds (default 0.05 s = 20 Hz control);
`frame` is the internal simulation step (default 1/60 s). A step with `dt` 0.05 runs 3 frames.

```bash
curl -X POST localhost:8780/api/envs -d '{"robot":"skid","lane":"step6","seed":1}'
curl -X POST localhost:8780/api/envs/1/step -d '{"action":{"drive":{"fwd":1}},"dt":0.1}'
curl localhost:8780/api/envs/1/frame -o view.jpg
curl "localhost:8780/api/envs/1/frame?view=robot&width=160" -o robot.jpg    # the robot's own camera
```

### Robot camera

Every robot has a small camera at its front, fixed to the part of the robot that tilts (so on a ramp
or when a quadruped pitches, the view tilts with it). In the simulator pages its picture is shown in
the top-right corner of the 3D view (✕ hides it, “📷 Robot camera” brings it back, `?robotcam=0`
starts without it). Through the API, `frame?view=robot` returns that image; in Python
`env.frame(view="robot", width=84)`. Use it to train from pixels, for example with a `CnnPolicy`.

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

### Action

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
| `bittle` | fwd, turn (the app's joystick: kwkF / kbk / kwkL / kwkR) | `j0` (head), `j8`–`j11` (shoulders), `j12`–`j15` (knees)° | – | serial | skill, gait, paused, gyro, joints {0, 8…15}, bodyHeight, blocked, serial (last replies) |
| `roboworks` | fwd, turn (teleop keys; turn steers) | L, R (rear wheels), steer° (servo) | – | ros | model, topics {`/odom`, `/scan` (360 ranges, null = no echo), `/PowerVoltage`}, steerDeg, ros (last answers) |
| `mecabot` | fwd, turn, strafe | FL, FR, RL, RR | – | ros | model, topics {`/odom`, `/scan`, `/PowerVoltage`}, ros (last answers) |
| `ugvrover` | fwd, turn (teleop keys) | L, R (one channel per side) | – | ros, serial | topics {`/odom`, `/odom/odom_raw`, `/voltage`, `/scan` (360 ranges, null = no echo)}, feedback {T:1001 …}, gimbal {pan, tilt}, leds, mode, ros, serial |
| `jetbot` | fwd, turn (gamepad keys) | left, right | – | python | motors {left, right: {value, alpha, beta}}, script, python |
| `jetracer` | fwd, turn (gamepad keys) | throttle (dc), steering (servo) | – | python | car {steering, throttle, steering_gain, steering_offset, throttle_gain}, script, python |
| `xgo` / `xgomini` | fwd, turn, strafe (the app's joystick) | – | 15 servos (`m11` … `m43` legs, `m51` gripper, `m52` `m53` arm) | xgo | model, action, actionName, move {x, y, turn}, gait, pace, markTime, height_mm, attitude {r, p, y}, imu, lying, arm {x, z, claw}, cube {x, y, z, held}, motors, blocked, xgo (last answers) |
| `a2` / `as2` | fwd, turn, strafe (the remote) | – | 12 joints (`FL_hip`, `FL_thigh`, `FL_calf`, …) | sdk | model, sportmodestate {mode (FSM 0–15), mode_name, progress, position, velocity, yaw_speed, body_height}, gait, speed_level, trajectory, sdk (last answers) |
| `agibotd1` | fwd, turn, strafe (the remote) | – | 12 joints | sdk | ctrl_mode (0 / 1 / 18 / 21 / 51), position, body_velocity, battery, move, two_leg, sdk (last answers) |
| `h2` | fwd, turn, strafe (the remote's joystick) | – | 31 joints (`left_hip_pitch` … `right_wrist_yaw`, `waist_yaw/roll/pitch`, `head_yaw/pitch`) | sdk, fell | fsm_id, fsm_name, fsm_mode, posture, arm_action, velocity, balance_mode, speed_mode, swing_height, stand_height, sdk (last answers) |
| `boosterk1` | fwd, turn, strafe (the app's joystick) | – | 22 joints (`head_yaw`, `head_pitch`, `left_shoulder_pitch` … `right_crank_down`) | booster, fell | mode, mode_name, posture, action, gait, head {pitch, yaw}, odometer {x, y, theta}, velocity, booster (last answers) |
| `tron1` | fwd, turn, strafe (the handheld remote) | – | 6 joints (`abad_L_Joint`, `hip_L_Joint`, `knee_L_Joint`, … `_R_`); sole feet (`options={"variant": "sole"}`): + `ankle_L_Joint`, `ankle_R_Joint` | ws, fell | variant, robot_info {accid, sw_version, imu, camera, motor, battery, status}, imu (after request_enable_imu), marktime, baseHeightOffset, blocked, notify, ws (last answers) |
| `tron1w` | fwd, turn (the handheld remote) | – | 6 leg joints + `wheel_L_Joint`, `wheel_R_Joint` | ws, fell | as `tron1`, plus stair, odom (after request_enable_odom) |
| `panther` | fwd, turn (mecanum `?variant=mecanum`: + strafe) | FL, FR, RL, RR | – | ros | variant, topics {`/odometry/wheels`, `/odometry/filtered`, `/battery/battery_status`, `/hardware/e_stop`, `/joint_states`, `/scan` (accessory)}, estop, lights, lidar, ros |
| `summitxl` | fwd, turn (mecanum `?variant=mecanum`: + strafe) | FL, FR, RL, RR | – | ros | variant, topics {`/robot/robotnik_base_control/odom`, `/robot/front_laser/scan` (271 ranges, 270°), `/robot/joint_states`, `/robot/battery_estimator/data`, `/robot/robotnik_base_hw/emergency_stop`}, motorsEnabled, ros |
| `rbkairos` | fwd, turn, strafe (teleop keys) | FL, FR, RL, RR (wheels), `shoulder_pan` … `wrist_3`° (arm) | – | ros | model, arm_type, topics {`/robot/robotnik_base_control/odom`, `/robot/front_laser/scan`, `/robot/rear_laser/scan` (241 ranges), `/robot/joint_states`}, arm {q, tcp, grip, holding, busy, message}, blocks [{id, color, x, y, z, held}], goal, ros (last answers) |
| `rbvogui` | fwd, turn, strafe (teleop keys) | FL, FR, RL, RR (drive), `shoulder_pan` … `wrist_3`° (arm) | SFL, SFR, SRL, SRR (steering) | ros | as `rbkairos` |
| `xarm5` | – (it stays on its table) | `joint1`–`joint5`°, `gripper` (opening in mm, 0–85) | – | – | as `xarm6` |
| `xarm850` | – | `joint1`–`joint6`°, `gripper` | – | – | as `xarm6` |
| `sespro` | – | `joint1`–`joint5`°, `gripper` (mm, 0–85) | – | serial | as `xarm6`, plus lss {limp [servo ids], status [LSS Q code per servo 1–6], led [0–7]}, serial (last answers) |
| `xarm6` | – (it stays on its table) | `joint1`–`joint6`°, `gripper` (opening in mm, 0–85) | – | – | tcp {x, y, z, yawDeg} (point between the finger pads, base frame), holding, busy, blocks [{id, color, x, y, z, yawDeg, held}] |
| `xarm7` | – | `joint1`–`joint7`°, `gripper` | – | – | as `xarm6` |
| `omx` | – | `joint1`–`joint4`°, `gripper` (opening in mm, 13–65) | – | – | tcp {x, y, z, yawDeg, pitchDeg}, holding, busy, blocks [{id, color, x, y, z, yawDeg, held}] |

Snake joints: odd J1, J3, … pitch, even J2, J4, … yaw. The page URL sets the number of servos with `?servos=4` … `12` (default 6). A commanded angle replaces the gait for that joint,
so with all six you can learn your own gait; the body moves by friction alone (no drive motors).
Add `?feet=skin` (rubber pads) or `?feet=scales` (snake skin with scales) to the page URL instead of passive wheels.

Quadruped joints: a commanded angle puts the leg in that pose (walking stops, the body rests on
whatever touches the floor). Walking itself is a `drive` command: the simulators move the body
kinematically, legs do not push it along.

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

Unitree Go2-W (`go2w`, `indexGo2W.php`): the Go2 with a hub-motor wheel on every leg. It drives on its wheels with
skid steering (up to 2.5 m/s, not sideways); over an edge higher than about 4 cm a leg lifts its wheel onto it (up to
17 cm; `extra.lifting` names the legs doing so). Controlled like the Go2 with `{"webrtc": …}`: the Go2 sport API without
the walking gaits and flips, plus VUI lights, obstacle avoidance and the joystick topic; `?fw=classic` selects the API
ids of firmware before 1.1.7. Which sport commands the real Go2-W firmware accepts is not publicly documented.

Unitree R1 (`r1`, `indexR1.php`; R1 Air with `options={"variant": "air"}`): a WebRTC robot like the G1, with its own
state machine (`R1_FSM` in unitree_webrtc_connect). It starts standing in Lock (4), the hub: SitDown 7, LieDown 702, the
dances 601–604, KungFu 607, JeetKuneDo 608 and Run 811 are reached from Lock; a refused switch answers 1001 (1003 = no
such state). Damping (1) is accepted from everywhere and drops the robot (event `fell`); StandUp (701) gets it up. It
walks in 811 (the Air reports 830), 813, 814, 815, 816, 830 and 831, driven by `7105` or the joystick. Gestures:
`7106` on `rt/api/arm/request` (not the G1-only hearts 20 and 21). `7108` (SET_MOTION) is not simulated.

Husarion ROSbot XL (`rosbotxl`, `indexROSbotXL.php`; `?variant=wheels` for rubber wheels with skid steering): ROS 2 like
the TurtleBot 4, with `ros` messages. In: `/cmd_vel` (geometry_msgs/msg/Twist; `linear.y` drives sideways on mecanum
wheels; stops 0.5 s after the last message), `/leds` (std_msgs/msg/UInt8, bit 0 / 1 = LED1 / LED2), `/set_pose` (resets
the EKF). Out: `/scan` (RPLIDAR S3), `/odometry/filtered` (EKF: wheels plus IMU heading),
`/rosbot_xl_base_controller/odom` (wheels only: it drifts because the mecanum rollers slip), `/imu_broadcaster/imu`,
`/joint_states`, `/battery_state`; service `/controller_manager/list_controllers`. The robot itself has no actions (Nav2
adds them). Same Terminal as the TurtleBot 4.

Leo Rover (`leorover`, `indexLeoRover.php`, `?lidar=1` adds the LiDAR accessory): Fictionlab's outdoor rover with a rocker
suspension, driven the ROS 2 way with `ros` messages. In: `/cmd_vel`, `/heading_controller/cmd_vel` (holds the IMU
heading driving straight), `/firmware/wheel_{FL,RL,FR,RR}/cmd_velocity` and `cmd_pwm_duty` (all stop after
`controller.input_timeout`, 500 ms). Out: `/wheel_odom` (wheels only: drifts when the tyres slide), `/merged_odom`
(heading from the IMU), `/imu/data`, `/imu/rpy`, `/firmware/battery`, `/joint_states`, the camera topics (picture:
`frame?view=robot`). Services (`std_srvs/srv/Trigger`): `/reset_odometry`, `/firmware/get_board_type`,
`/firmware/get_firmware_version`, `/imu_filter/reset_calibration`, `/system/reboot`. Parameters on
`/firmware_parameter_bridge` (`controller.*`) and `/heading_controller`. Same Terminal as the TurtleBot 4.

Petoi Bittle X (`bittle`, `indexBittle.php`): driven with OpenCat serial tokens, as over USB / Bluetooth (115200 baud).
Put them in the action as `serial` (one string or a list): `{"serial": "kwkF"}`, `{"serial": "ksit"}`,
`{"serial": "m0 45"}`, `{"serial": "i 8 -70 9 -70"}`, `{"serial": "j"}`. The answer comes back as a `serial` event
(`line`, `reply`): the token letter, or the joint table for `j`. All 56 skills of the real robot are there with its own
joint data (OpenCat, MIT); gaits keep going until the next token. Joints: 0 head, 8–11 shoulders, 12–15 knees (left
front, right front, right back, left back). The page has a Serial monitor (`indexBittle.php?serial=1`) to type tokens.

Roboworks Rosbot 2 (`roboworks`, `indexRoboworks.php`) and Mecabot 2 (`mecabot`, `?variant=mecabot`): ROS 2 education
robots with a Jetson Orin, an LSLiDAR N10 and an Orbbec Astra camera, running Wheeltec's `turn_on_wheeltec_robot` (ROS 2
Humble). The Rosbot 2 steers like a car (smallest turning radius 0.77 m), the Mecabot 2 has mecanum wheels. `/cmd_vel`
(Twist; on the Rosbot 2 `angular.z` sets the steering angle, the Mecabot also uses `linear.y`) keeps the robot going until
the next message. Published: `/odom`, `/imu`, `/PowerVoltage`, `/scan`, the camera images. Terminal user `wheeltec`;
`ros2 run wheeltec_robot_keyboard wheeltec_keyboard` works as teleop.

Waveshare UGV Rover (`ugvrover`, `indexUGVRover.php`): two ways in, as on the real rover. `{"serial": {"json": {"T": 1, "L":
0.2, "R": 0.2}}}` sends the JSON commands of the ESP32 sub-controller (T 1 / 11 / 13 motion, 133–137 / 141 gimbal, 132 LEDs,
3 OLED, 130 / 126 feedback, 136 heartbeat: the wheels stop 3 s after the last motion command). `{"ros": …}` is ROS 2
(ugv_ws): in `/cmd_vel`, `/ugv/joint_states` (pan, tilt), `/ugv/led_ctrl`; out `/scan`, `/odom`, `/odom/odom_raw`,
`/imu/data_raw`, `/imu/mag`, `/voltage`, `/image_raw`.

JetBot (`jetbot`) and JetRacer (`jetracer`): driven with the Python lines of their Jupyter notebooks, as
`{"python": {"code": "robot.forward(0.3)"}}` or `{"python": {"code": "car.throttle = 0.25"}}`; a script with `time.sleep()`
runs in simulation time and the answer comes back as a `python` event. JetBot: `Robot()` with forward / backward / left /
right / stop / set_motors and `left_motor` / `right_motor` `.value`. JetRacer: `NvidiaRacecar()` with steering, throttle,
steering_gain (−0.65), steering_offset and throttle_gain (0.8).

XGO-Lite3 / XGO-Mini2 (`xgo`, `xgomini`, `indexXGO.php[?variant=mini]`): desktop robot dogs with an arm and gripper,
programmed with the xgolib Python calls: `{"xgo": "dog.move_x(15)"}`, `{"xgo": "dog.action(13)"}`, `{"xgo": ["dog.arm(130, 0)",
"dog.claw(255)"]}`. Actions 1–24, the grabs 128–130 and reset 255; a red cube in front of the dog can be picked up (action
130). The page has a Python bar (`?python=1`).

Unitree A2 / AS2 (`a2`, `as2`; `indexA2.php`, `?variant=as2`): industrial quadrupeds driven with the SportClient of
`unitree_sdk2`: `{"sdk": {"api_id": 1008, "parameter": {"x": 0.5, "y": 0, "z": 0}}}` (Move stops after 1 s). Shared:
Damp, BalanceStand, StopMove, StandUp / Down, RecoveryStand, Euler, Move, BodyPosition, SwitchGait, BodyHeight, SpeedLevel,
Left / RightSideGait, HandStand, BipedStand, Front / BackFlip, GetState, SetAutoRecovery; the A2 adds Trajectory 1035 and
ResetEstimator 1111, the AS2 the tricks (Greeting, Heart, Content, Dance1 / 2, Handshake, Stretch, Sit, FrontJump, PushUp,
UpJump) and SwitchJoystick.

AgiBot D1 (`agibotd1`, `indexAgiBotD1.php`): the HighLevel functions of AgiBot's D1 SDK, `{"sdk": {"fn": "move", "args":
[0.5, 0, 0]}}`; answers with code 0, 0x3007 (state switch refused) or 0x3013 (speed too large) and the getters' data.

Unitree H2 (`h2`, `indexH2.php`): the high-level API of `unitree_sdk2` (H2 EDU): `{"sdk": {"service": "sport", "api_id":
7101, "parameter": {"data": 601}}}` or `{"sdk": {"service": "arm", "api_id": 7106, "parameter": {"action_id": 26}}}`. FSM 0
zero torque, 1 damp, 2 squat, 3 sit, 4 stand up, 601 start (walks with SetVelocity 7105); SDK error codes 7302, 7303, 7401,
7402, 7404. It starts seated on a chair.

Booster K1 (`boosterk1`, `indexBoosterK1.php`): the Booster Robotics SDK's `B1LocoClient`: `{"booster": {"method":
"ChangeMode", "args": ["kWalking"]}}`, `{"booster": {"method": "Move", "args": [0.3, 0, 0]}}`, `{"booster": {"method":
"WholeBodyDance", "args": ["kRoundhouseKick"]}}`. Modes Damping 0, Prepare 1 (start), Walking 2, Custom 3, Soccer 4.

LimX TRON 1 (`tron1`, `indexTron1.php`; wheel feet `tron1w` = `?variant=wheel`; soles with `options={"variant": "sole"}`):
LimX's remote-control interface, JSON over a WebSocket, as `ws` messages: `{"ws": {"title": "request_walk_mode"}}`,
`{"ws": {"title": "request_twist", "data": {"x": 0.5, "y": 0, "z": 0}}}` (resend at 30 Hz; it stops 0.25 s after the last),
`request_stand_mode`, `request_sitdown`, `request_base_height`, `request_emgy_stop`, `request_recover`,
`request_stair_mode` (wheel feet), `request_marktime_mode`, `request_enable_imu` / `_odom`, `request_light_effect`. Point
feet can't stand still: they keep stepping.

Husarion Panther (`panther`, `indexPanther.php`; `?variant=mecanum`, `?lidar=0`): `/cmd_vel` is TwistStamped (stops 0.5 s
after the last message; teleop: `--ros-args -p stamped:=true`); out `/odometry/wheels`, `/odometry/filtered`, `/imu/data`,
`/joint_states`, `/battery/battery_status`, `/hardware/e_stop`, `/scan` (accessory); services for the e-stop, power, lights
(`/lights/set_animation`) and localization. Robotnik SUMMIT-XL (`summitxl`, `indexSummitXL.php`; `?variant=mecanum`):
namespace `/robot`: `/robot/cmd_vel` (Twist; teleop: `--ros-args -r cmd_vel:=/robot/cmd_vel`) or
`/robot/robotnik_base_control/cmd_vel` (TwistStamped); out odom, IMU, the 270° front laser, joint states, e-stop, battery
and the PTZ camera image. Both are too big for the 60 cm corridor.

Robotnik RB-KAIROS+ (`rbkairos`, `indexRBKairos.php`) and RB-VOGUI+ (`rbvogui`, `?variant=vogui`; `?arm=ur10e` for a UR10e):
mobile manipulators with a UR5e arm and a Robotiq 2F-85 gripper, and a table with blocks in front of the start square. ROS 2
under `/robot`: `robotnik_base_control/cmd_vel` (TwistStamped) or `cmd_vel_unstamped` (Twist), stopping 2 s after the last
message; the actions `joint_trajectory_controller/follow_joint_trajectory` and `robotiq_gripper_controller/gripper_cmd`;
out odom, two lasers, IMU, joint states, the RGB-D camera. For Cartesian moves use `{"arm": {"tcp": {"x", "y", "z",
"yaw"}}}` or `{"arm": {"skill": "pick"}}`.

xArm 5 Lite (`xarm5`, `indexXArm6.php?model=5`) and UFACTORY 850 (`xarm850`, `indexXArm6.php?model=850`): the same table and
API as the xArm 6; the xArm 5 Lite has 5 joints (the gripper points down and turns, but cannot tilt), the 850 has 6 joints
and 85 cm reach. Lynxmotion SES-PRO (`sespro`, `indexSESPro.php`): a 900 mm 5DoF arm driven over the LSS serial protocol:
`{"serial": "#2D450T1500"}` moves servo 2 to 45.0° in 1.5 s, `#1QD` asks a position, `#254L` makes all servos limp (the arm
sags), `#254H` holds. IDs 1 base, 2 shoulder, 3 elbow, 4 wrist pitch, 5 wrist rotate, 6 gripper.

xArm 6 / xArm 7 (UFACTORY, on a table with 12 coloured 4 cm blocks; the xArm 7 page is `indexXArm6.php?model=7` or
`indexXArm7.php`): the joints are servos (degrees, limits as on the real arm: xArm 6 J2 −118…120, J3 −225…11, J5 −97…180;
xArm 7 J2 −118…120, J4 −11…225, J6 −97…180; the others ±360); a commanded value is the goal, the joints move there
together at up to 120°/s. Base frame: origin on the table under the arm, x forward, y left, z up (m). `reset` puts the
arm in its look-down pose and scatters the blocks (with a `seed`: always the same). The robot camera
(`view: "robot"`) is the RealSense D435 on the wrist; the gripper's fingers show at the bottom of its picture. A second
D435 on a tripod beside the table sees the whole arm (`view: "external"`, also shown on the page).

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

Standing row (tutorial 6, `python/stand_rl.py`): `{ "rowPick": dz }` takes the top block of the stack (gripper straight
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

#### The real xArm 6 / 7: one Python interface, digital twin, follow

`python/xarm_robot.py` gives the simulator and the real arm the same Python interface, so one script runs on both:

```python
from xarm_robot import connect
arm = connect("sim", model=7)     # the xArm 7 simulator · "10.2.172.20": the real arm (6 or 7 axes: it asks) · "fake": a pretend arm
arm.move_to(0.35, 0.0, 0.20)      # gripper point (m, base frame), gripper pointing down, inverse kinematics in Python
arm.set_gripper(75); arm.move_to(0.35, 0.0, 0.027); arm.set_gripper(0)
print(arm.joints(), arm.tcp())
jpeg = arm.camera()               # wrist camera: sim image or the real RealSense D435 over USB
jpeg = arm.camera("external")     # the fixed camera that sees the whole arm
```

Every move is checked first (joint limits, fingertips at least 5 mm above the table). For the xArm 7 the inverse
kinematics keeps the extra joint (upper arm roll, joint 3) near 0° and the base and flange within ±270°. The real arm needs
`pip install xarm-python-sdk` (and `pyrealsense2` for the camera); `python/xarm_demo.py --robot sim | fake | <ip>`
is a pick-and-place example.

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

Quadruped arm (OpenMANIPULATOR-X on the top plate): `arm_joint1` turns it (−162…162°), `arm_joint2`–`arm_joint4`
pitch shoulder, elbow and wrist (+ = forward / down; −103…90, −54…79, −103…117°). A commanded value is the servo's
goal: it moves there at its own speed (90°/s), so read `deg` to see where it is. `gripper` is the opening between the
fingers in mm; closing on one of the 3 cm cubes stops at the cube and holds it (`extra.arm.holding`). The cubes
lie 30 cm in front of the robot after a reset; the arm reaches the floor when the robot lies down. A cube released above the top plate lands on the back and rides along (`onBack`).
`extra.reach.tip` is the fingertip position in the arm frame (m; origin on top of the arm's base plate, x forward,
y left, z up). An action `{ "reach": { "x", "y", "z" } }` (same frame) shows a red target ball there and adds
`extra.reach.target` and `dist`; `{ "reach": null }` hides it (tutorial 5). Page option `camview: "arm"` puts the
3D view close to the arm.

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
| `ros` | `turtlebot4`, `rosbotxl`, `leorover`, `roboworks`, `mecabot`, `ugvrover`, `panther`, `summitxl`, `rbkairos`, `rbvogui`: the answer to a ROS 2 message (`kind`, `name`, `code`, `text`; actions also `status` SUCCEEDED / ABORTED and the result in `data` when the goal ends) | no |
| `serial` | `bittle`: the answer to an OpenCat serial token (`line`, `reply`); `ugvrover`: to an ESP32 JSON command; `sespro`: to an LSS command (`line`, `reply`, `note`) | no |
| `python` | `jetbot`, `jetracer`: the result of a Python line (`out` or the error) | no |
| `xgo` | `xgo`, `xgomini`: the answer to an xgolib call (`line`, `code`, `reply`) | no |
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

### Observation vector (Python client)

`botopia.features(obs)` turns an observation into a fixed-length list:
`[progress, vx, vy, wz, roll, pitch, accel_x/g, accel_y/g, accel_z/g, gyro_x, gyro_y, gyro_z,
stuck(0/1)]`, then per motor `duty, rpm/100` (DC) or `degrees/90` (servo).

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

The page itself uses `register`, `pull` (every 100 ms) and `telemetry` (every 200 ms).

```python
from botopia import RemoteSession
s = RemoteSession("https://robotics.dinno.be/api/index.php", "ABC123")
s.command(reset={"lane": "stairs"}, action={"drive": {"fwd": 0.6}})
print(s.state()["obs"]["pose"])
```

Sessions are stored as small JSON files in `data/` next to `index.php` (made on first use, closed to the web
with `.htaccess`; on nginx deny `/api/data/` yourself). The session code is the only
key: anyone who knows it can drive that browser tab's simulated robot, nothing else.
Sessions older than a day are removed.

---

## In the page (JavaScript)

Every simulator page exposes the same interface as `window.botopiaEnv` (made by
`common/env.js`), handy in the browser console:

```js
botopiaEnv.info()
botopiaEnv.reset({ lane: 'gravel', seed: 1 })
botopiaEnv.step({ drive: { fwd: 1 } }, { dt: 0.1 })   // in normal mode the page also keeps running
```

URL switches: `?env=1` lock-step (used by the Node API: the page stands still between steps),
`?remote=1` remote mode (PHP API), `?lane=step6` start in a lane, `?auto=corridor` autopilot through
the corridor, `?door=free` door mode in the door lane, `?fixedfps=30` fixed time step, `?shot` 3D view only.

## Notes

* The models are made for comparing drive types and for learning, not to be exact copies of
  real robots: wheels and tracks have climb limits per wheel size and drive type, slopes have a
  maximum, legs are kinematic (they place the body, they do not push it).
* Same `seed` + same actions = the same result, for every robot (checked).
* One environment per browser tab; Chrome renders nothing between steps, so many environments fit
  on one PC. `frame` renders on demand.
