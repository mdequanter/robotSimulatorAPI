"""Remote control of the DrRobot Jaguar 4x4 Wheel from a terminal: the same commands as `jaguar` in the
terminal of the Jaguar simulator page. Standard library only.

    python jaguar_cli.py status                      # the real Jaguar (PC on the DriJaguar Wi-Fi, IP 192.168.0.104)
    python jaguar_cli.py --ip 127.0.0.1 status       # the simulator, through jaguar_bridge.py
    python jaguar_cli.py release                     # MMW !MG: release the E-Stop
    python jaguar_cli.py drive 300 300 2             # power left, right (-1000 ... 1000, + = forward) for 2 s
    python jaguar_cli.py forward 300 2 · back 300 2 · turn 400 1.5 (+ = left) · stop · estop
    python jaguar_cli.py imu | enc | gps | watch     # live values, Ctrl+C stops
    python jaguar_cli.py send "MM0 ?V" "MM0 ~MMOD"   # lines as they are, and what the boards answer
    python jaguar_cli.py nc                          # the raw stream; type lines to send them

The protocol (TCP 192.168.0.60:10001, lines ending with CRLF): MMW !M <left> <-right> drives (the right motors
count the other way round), MMW !MG / !EX release / set the E-Stop, a PING at least every 0.2 s keeps the motors on.
The robot streams the IMU (#seq,YAW,y,GYRO,gx,gy,gz,ACC,ax,ay,az,COMP,mx,my,mz,...: 50 Hz, raw counts),
the encoders (MM0 C=fl:fr, MM1 C=rl:rr: 5 Hz, 300 counts per wheel turn) and the GPS ($GPRMC, $GPGGA).
"""
import argparse
import math
import socket
import sys
import threading
import time

ROBOT_IP, ROBOT_PORT = "192.168.0.60", 10001
CPR, WHEEL_R = 300, 0.135
M_PER_CNT = 2 * math.pi * WHEEL_R / CPR
GYRO_LSB, ACC_LSB = 14.375, 256.0         # ITG3205 counts per deg/s, ADXL345 counts per g
ESTOP = 16
FLAGS = ["overheat", "overvoltage", "undervoltage", "short circuit", "E-STOP"]


class Jaguar:
    """A TCP connection that reads the stream in a thread and keeps the latest values."""

    def __init__(self, ip, port, echo=None):
        self.sock = socket.create_connection((ip, port), timeout=3)
        self.sock.settimeout(0.5)
        self.lock = threading.Lock()
        self.echo = echo                    # function(line) for every line (nc, send)
        self.imu = self.gps = None
        self.enc, self.volts, self.ff, self.temps, self.amps = {}, {}, {}, {}, {}
        self.running = True
        threading.Thread(target=self._read, daemon=True).start()

    def send(self, line):
        with self.lock:
            self.sock.sendall((line + "\r\n").encode("ascii"))

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass

    def _read(self):
        buf = b""
        while self.running:
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buf += data
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                line = raw.decode("ascii", "replace").strip()
                if line:
                    try:
                        self._parse(line)
                    except (ValueError, IndexError):
                        pass
                    if self.echo:
                        self.echo(line)

    def _parse(self, line):
        if line.startswith("#"):
            p = line.split(",")
            if len(p) >= 15 and p[3] == "GYRO" and p[7] == "ACC":
                self.imu = {"yaw": float(p[2]), "gyro": [int(v) / GYRO_LSB for v in p[4:7]], "acc": [int(v) / ACC_LSB for v in p[8:11]],
                            "mag": [int(v) for v in p[12:15]] if p[11] == "COMP" else None}
        elif line.startswith("$"):
            p = line.split("*")[0].split(",")
            if p[0].endswith("RMC"):
                g = self.gps or {}
                g["fix"] = p[2] == "A"
                if g["fix"]:
                    g.update(lat=_deg(p[3], p[4]), lon=_deg(p[5], p[6]), speed=float(p[7] or 0) * 0.514444,
                             course=float(p[8]) if p[8] else None)
                self.gps = g
            elif p[0].endswith("GGA") and self.gps is not None:
                self.gps.update(sats=int(p[7] or 0), hdop=p[8])
        elif line[:3] in ("MM0", "MM1") and "=" in line:
            board, rest = line.split(" ", 1)
            key, _, val = rest.partition("=")
            nums = [int(v) for v in val.split(":")]
            {"C": self.enc, "V": self.volts, "FF": self.ff, "T": self.temps, "A": self.amps}.get(key, {})[board] = nums

    def estop_active(self):
        return any(v[0] & ESTOP for v in self.ff.values())


def _deg(v, hemi):
    if not v:
        return None
    i = v.index(".") - 2
    d = float(v[:i]) + float(v[i:]) / 60
    return -d if hemi in ("S", "W") else d


def _flags(ff):
    return "OK" if not ff else ", ".join(n for i, n in enumerate(FLAGS) if ff & (1 << i))


def drive(j, left, right, seconds):
    if j.estop_active():
        print("E-Stop is active: the wheels won't turn (python jaguar_cli.py release)")
    e0 = None
    t0 = time.time()
    heading, last = 0.0, time.time()
    print(f"MMW !M {left} {-right} + PING every 0.1 s" + (f" for {seconds} s" if seconds else " (Ctrl+C stops)"))
    try:
        while seconds is None or time.time() - t0 < seconds:
            j.send("PING")
            j.send(f"MMW !M {left} {-right}")
            if e0 is None and "MM0" in j.enc and "MM1" in j.enc:
                e0 = j.enc["MM0"] + j.enc["MM1"]
            time.sleep(0.1)
            if j.imu:
                heading += j.imu["gyro"][2] * (time.time() - last)
            last = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        j.send("MMW !M 0 0")
    time.sleep(0.4)
    if e0 and "MM0" in j.enc and "MM1" in j.enc:
        e1 = j.enc["MM0"] + j.enc["MM1"]
        d = [(b - a) * (-1 if i % 2 else 1) * M_PER_CNT for i, (a, b) in enumerate(zip(e0, e1))]
        print(f"{time.time() - t0:.1f} s · left {(d[0] + d[2]) / 2:.3f} m · right {(d[1] + d[3]) / 2:.3f} m · "
              f"turned {heading:.1f}° (gyro) · FL {d[0]:.3f} FR {d[1]:.3f} RL {d[2]:.3f} RR {d[3]:.3f} m")


def live(j, what):
    try:
        while True:
            time.sleep(0.25)
            i, g, e = j.imu, j.gps, j.enc
            imu = (f"gyro °/s {i['gyro'][0]:6.1f}{i['gyro'][1]:6.1f}{i['gyro'][2]:6.1f} · acc g {i['acc'][0]:5.2f}{i['acc'][1]:5.2f}{i['acc'][2]:5.2f}"
                   + (f" · compass {math.degrees(math.atan2(i['mag'][1], i['mag'][0])) % 360:3.0f}°" if i.get("mag") else "")) if i else "no IMU data"
            enc = (f"FL {e['MM0'][0]} FR {e['MM0'][1]} RL {e['MM1'][0]} RR {e['MM1'][1]} · left {(e['MM0'][0] + e['MM1'][0]) / 2 * M_PER_CNT:7.3f} m "
                   f"right {-(e['MM0'][1] + e['MM1'][1]) / 2 * M_PER_CNT:7.3f} m") if "MM0" in e and "MM1" in e else "no encoder data"
            gps = ((f"{g['lat']:.7f}, {g['lon']:.7f} · {g['speed']:.2f} m/s · {g.get('sats', '?')} sats" if g.get("fix") else "no fix") if g else "no GPS data")
            print({"imu": imu, "enc": enc, "gps": gps}.get(what, f"{enc} | {imu} | {gps}"))
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description="Remote control of the DrRobot Jaguar 4x4 Wheel (or its simulator) over TCP")
    ap.add_argument("--ip", default=ROBOT_IP, help=f"{ROBOT_IP} = the real Jaguar, 127.0.0.1 = the simulator through jaguar_bridge.py")
    ap.add_argument("--port", type=int, default=ROBOT_PORT)
    ap.add_argument("command", choices=["status", "release", "estop", "stop", "drive", "forward", "back", "turn", "imu", "enc", "gps", "watch", "send", "nc"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()

    # send: a board echoes the command, then answers on its next line (+, -, NAME=value).
    want, waiting = set(), set()

    def echo(line):
        if a.command == "nc":
            print(line)
        elif a.command == "send":
            if line in want:
                want.discard(line)
                waiting.add(line[:3])
                print("<", line)
            elif line[:3] in waiting:
                waiting.discard(line[:3])
                print("<", line)
    try:
        j = Jaguar(a.ip, a.port, echo)
    except OSError as e:
        sys.exit(f"Can't connect to {a.ip}:{a.port}: {e}" + (" (is jaguar_bridge.py running, and the page connected?)" if a.ip.startswith("127.") else
                                                              " (is this PC on the DriJaguar Wi-Fi with IP 192.168.0.104?)"))
    try:
        n = [float(x) for x in a.args] if a.command in ("drive", "forward", "back", "turn") else []
        if a.command == "status":
            j.send("PING")
            t0 = time.time()                              # voltages and faults come twice a second: wait for both boards
            while time.time() - t0 < 3 and not (len(j.volts) == 2 and len(j.ff) == 2 and j.gps is not None):
                time.sleep(0.1)
            for b, name in (("MM0", "front"), ("MM1", "rear")):
                v = j.volts.get(b)
                print(f"{b} ({name}): " + (f"main {v[1] / 10:.1f} V, 12V {v[0] / 10:.1f} V" if v else "no voltage")
                      + f" · {_flags(j.ff[b][0]) if b in j.ff else '?'} · temp {j.temps.get(b, '?')} °C · current {[x / 10 for x in j.amps.get(b, [])]} A")
            e = j.enc
            print("Encoders: " + (f"FL {e['MM0'][0]} FR {e['MM0'][1]} RL {e['MM1'][0]} RR {e['MM1'][1]}" if "MM0" in e and "MM1" in e else "none"))
            print("IMU: " + (f"acc z {j.imu['acc'][2]:.2f} g, gyro z {j.imu['gyro'][2]:.1f} °/s" if j.imu else "none"))
            print("GPS: " + ((f"{j.gps['lat']:.7f}, {j.gps['lon']:.7f}" if j.gps.get("fix") else "no fix") if j.gps else "none"))
            if "MM0" in j.volts and j.volts["MM0"][1] / 10 < 22.2:
                print("Battery below 22.2 V: charge it before driving (LiPo 6S)")
        elif a.command == "release":
            j.send("PING"); j.send("MMW !MG"); time.sleep(0.7)
            print("E-Stop still active: check the robot" if j.estop_active() else "E-Stop released")
        elif a.command == "estop":
            j.send("MMW !M 0 0"); j.send("MMW !EX"); time.sleep(0.2); print("Motors 0, E-Stop active")
        elif a.command == "stop":
            j.send("MMW !M 0 0"); time.sleep(0.1)
        elif a.command == "drive":
            if len(n) < 2:
                sys.exit("usage: jaguar_cli.py drive LEFT RIGHT [SECONDS]")
            drive(j, int(n[0]), int(n[1]), n[2] if len(n) > 2 else None)
        elif a.command in ("forward", "back", "turn"):
            if len(n) != 2:
                sys.exit(f"usage: jaguar_cli.py {a.command} POWER SECONDS")
            p = int(n[0])
            left, right = {"forward": (p, p), "back": (-p, -p), "turn": (-p, p)}[a.command]
            drive(j, left, right, n[1])
        elif a.command in ("imu", "enc", "gps", "watch"):
            live(j, a.command)
        elif a.command == "send":
            for line in a.args:
                head, _, cmd = line.strip().partition(" ")
                for b in (["MM0", "MM1"] if head.upper() == "MMW" else [head.upper()]):
                    want.add(f"{b} {cmd.strip()}")
                print(">", line)
                j.send(line)
            time.sleep(0.5)
        elif a.command == "nc":
            print("(type a line and Enter to send it; Ctrl+C or Ctrl+D ends)", file=sys.stderr)
            for line in sys.stdin:
                j.send(line.strip())
    except KeyboardInterrupt:
        pass
    finally:
        j.close()


if __name__ == "__main__":
    main()
