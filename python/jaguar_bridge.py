"""Bridge between the DrRobot Jaguar 4x4 Wheel simulator page in your browser and the real Jaguar, and your PC programs.

A web page can't open a TCP socket, and the Jaguar only talks TCP, so this program does:

    browser (indexJaguar.php)  <-- WebSocket ws://localhost:8772 -->  jaguar_bridge.py  <-- TCP -->  real Jaguar 192.168.0.60:10001
                                                                            ^
    your PC programs (robot.py --ip 127.0.0.1, jaguar_cli.py)  -- TCP 127.0.0.1:10001 --+   (they reach the page's Jaguar)

What it is for:
  * the page shows the real Jaguar's IMU, encoders and GPS (card "Sensors", the map), and when you choose so on the
    page, drives it: "The real Jaguar follows the simulator", or "Commands drive the real Jaguar" (the 3D Jaguar then
    moves as its encoders and gyro report);
  * a program on your PC that talks to the Jaguar over TCP drives the simulated one when you give it 127.0.0.1
    instead of 192.168.0.60: the page's Jaguar answers with the same lines as the real robot.

    pip install websockets
    python jaguar_bridge.py                     # PC on the DriJaguar Wi-Fi (IP 192.168.0.104), robot at 192.168.0.60
    python jaguar_bridge.py --no-robot          # only your PC programs <-> the simulator (no real robot)
    python jaguar_bridge.py --fake              # a pretend Jaguar, to try the page's connector without one

Then open indexJaguar.php, card "Real Jaguar" -> Connect.

Safety: the bridge never sends PING by itself, so the Jaguar's watchdog stops it when the page stops sending
(page closed, tab hidden, connection lost). When the page goes away the bridge also sends MMW !M 0 0.
"""
import argparse
import asyncio
import json
import math
import random
import sys
import time

ROBOT_IP, ROBOT_PORT = "192.168.0.60", 10001
WS_PORT = 8772
BATCH = 0.05            # s: lines to the page in batches


class Bridge:
    def __init__(self, args):
        self.args = args
        self.page = None                    # the WebSocket of the page (one at a time: the newest)
        self.robot = None                   # asyncio StreamWriter to the real Jaguar (or the fake one)
        self.robot_state = "off" if args.no_robot else "connecting"
        self.to_page = {"real": [], "sim": []}
        self.clients = set()                # PC programs on the simulator port

    # ---------- the page ----------
    async def page_handler(self, ws):
        if self.page is not None:
            await self.page.close()
        self.page = ws
        print(f"Page connected ({getattr(ws, 'remote_address', '')})")
        await self.status()
        try:
            async for raw in ws:
                try:
                    m = json.loads(raw)
                except ValueError:
                    continue
                if "real" in m:
                    for line in _lines(m["real"]):
                        await self.to_robot(line)
                if "sim" in m:
                    data = "".join(line + "\r\n" for line in _lines(m["sim"])).encode("ascii", "replace")
                    for w in list(self.clients):
                        try:
                            w.write(data)
                        except (ConnectionError, OSError):
                            self.clients.discard(w)
        except Exception as e:      # noqa: BLE001 - a dropped page connection must not stop the bridge
            print(f"Page connection: {e!r}")
        finally:
            if self.page is ws:
                self.page = None
                print("Page gone: MMW !M 0 0 to the robot")
                await self.to_robot("MMW !M 0 0")

    async def status(self):
        if self.page is None:
            return
        msg = {"status": {"robot": self.robot_state, "ip": f"{self.args.ip}:{self.args.port}", "fake": self.args.fake,
                          "sim_port": self.args.sim_port, "sim_clients": len(self.clients)}}
        try:
            await self.page.send(json.dumps(msg))
        except Exception:           # noqa: BLE001
            pass

    async def pump(self):
        """Batches to the page, and the status once a second."""
        last = 0.0
        while True:
            await asyncio.sleep(BATCH)
            if self.page is None:
                self.to_page = {"real": [], "sim": []}
                continue
            out = {k: v for k, v in self.to_page.items() if v}
            self.to_page = {"real": [], "sim": []}
            try:
                if out:
                    await self.page.send(json.dumps(out))
                if time.monotonic() - last > 1:
                    last = time.monotonic()
                    await self.status()
            except Exception:       # noqa: BLE001
                pass

    # ---------- the real Jaguar ----------
    async def to_robot(self, line):
        if self.robot is None:
            return
        try:
            self.robot.write((line + "\r\n").encode("ascii", "replace"))
            await self.robot.drain()
        except (ConnectionError, OSError) as e:
            print(f"Robot: {e}")

    async def robot_loop(self):
        while True:
            try:
                if self.args.fake:
                    reader, writer = await FakeJaguar().open()
                    where = "fake Jaguar"
                else:
                    reader, writer = await asyncio.wait_for(asyncio.open_connection(self.args.ip, self.args.port), 3)
                    where = f"{self.args.ip}:{self.args.port}"
                self.robot, self.robot_state = writer, f"connected ({where})"
                print(f"Robot connected: {where}")
                await self.status()
                while True:
                    raw = await reader.readline()
                    if not raw:
                        break
                    line = raw.decode("ascii", "replace").strip()
                    if line and self.page is not None:
                        self.to_page["real"].append(line)
                self.robot_state = "connection closed by the robot"
            except (OSError, asyncio.TimeoutError) as e:
                self.robot_state = f"not reachable ({e.__class__.__name__}): PC on the DriJaguar Wi-Fi? robot on?"
            self.robot = None
            await self.status()
            await asyncio.sleep(2)

    # ---------- PC programs: TCP 127.0.0.1:sim_port to the page's Jaguar ----------
    async def sim_client(self, reader, writer):
        peer = writer.get_extra_info("peername")
        self.clients.add(writer)
        print(f"PC program connected {peer} -> the page's Jaguar" + ("" if self.page else " (no page connected yet: open the simulator and Connect)"))
        await self.status()
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode("ascii", "replace").strip()
                if line and self.page is not None:
                    self.to_page["sim"].append(line)
        except (ConnectionError, OSError):
            pass
        finally:
            self.clients.discard(writer)
            writer.close()
            print(f"PC program gone {peer}")
            if self.page is not None:
                self.to_page["sim"].append("MMW !M 0 0")
            await self.status()


def _lines(v):
    return [str(x) for x in (v if isinstance(v, list) else [v])]


# ---------- a pretend Jaguar (--fake) ----------
class FakeJaguar:
    """Streams like the real robot (IMU 50 Hz, encoders 5 Hz, GPS 5 Hz) and drives on a flat field: open loop
    !M, E-Stop !MG / !EX, the PING watchdog. Enough to try the page's connector, not a simulation."""

    def __init__(self):
        self.estop, self.cmd, self.last_rx = True, [0, 0, 0, 0], 0.0
        self.x = self.y = self.th = 0.0
        self.counts = [0.0] * 4
        self.seq = 0

    async def open(self):
        self.out = asyncio.StreamReader()
        asyncio.get_running_loop().create_task(self.run())
        return self.out, _FakeWriter(self.command)

    def command(self, line):
        self.last_rx = time.monotonic()
        w = line.split()
        if not w or w[0] not in ("MMW", "MM0", "MM1") or len(w) < 2:
            return
        boards = [0, 1] if w[0] == "MMW" else [int(w[0][2])]
        if w[1] == "!MG":
            self.estop = False
        elif w[1] == "!EX":
            self.estop, self.cmd = True, [0, 0, 0, 0]
        elif w[1] == "!M" and len(w) >= 4:
            for b in boards:
                self.cmd[2 * b], self.cmd[2 * b + 1] = int(w[2]), int(w[3])

    def emit(self, line):
        self.out.feed_data((line + "\r\n").encode())

    async def run(self):
        t, dt = 0.0, 0.02
        while True:
            await asyncio.sleep(dt)
            t += dt
            alive = time.monotonic() - self.last_rx < 0.3 and not self.estop
            p = self.cmd if alive else [0, 0, 0, 0]
            w = [v / 1000 * 23.0 for v in p]                      # wheel rad/s, channel sign
            for i in range(4):
                self.counts[i] += w[i] * dt * 300 / (2 * math.pi)
            vl, vr = (w[0] + w[2]) / 2 * 0.135, -(w[1] + w[3]) / 2 * 0.135
            v, wz = (vl + vr) / 2, (vr - vl) / (1.4 * 0.463)
            self.th += wz * dt
            self.x += v * math.cos(self.th) * dt
            self.y += v * math.sin(self.th) * dt
            g = [round(random.gauss(0, 2)), round(random.gauss(0, 2)), round(math.degrees(wz) * 14.375 + random.gauss(0, 2))]
            a = [round(random.gauss(0, 3)), round(random.gauss(0, 3)), round(256 + random.gauss(0, 3))]
            hd = (math.pi / 2 - self.th) % (2 * math.pi)
            m = [round(212 * math.cos(hd)), round(212 * math.sin(hd)), -478]
            self.emit(f"#{self.seq},YAW,{hd:.3f},GYRO,{g[0]},{g[1]},{g[2]},ACC,{a[0]},{a[1]},{a[2]},COMP,{m[0]},{m[1]},{m[2]},ADC,0")
            self.seq = (self.seq + 1) % 256
            k = round(t / dt)
            if k % 10 == 0:
                c = [round(x) for x in self.counts]
                self.emit(f"MM0 C={c[0]}:{c[1]}")
                self.emit(f"MM1 C={c[2]}:{c[3]}")
                lat, lon = 50.8417 + self.y / 111320, 4.3229 + self.x / (111320 * math.cos(math.radians(50.8417)))
                for s in _nmea(lat, lon, abs(v) / 0.514444, math.degrees(hd) % 360):
                    self.emit(s)
            if k % 25 == 0:
                for b in ("MM0", "MM1"):
                    self.emit(f"{b} V=121:{round(246 + random.gauss(0, 1))}:4980")
                    self.emit(f"{b} FF={16 if self.estop else 0}")
                    self.emit(f"{b} T=27:27")


class _FakeWriter:
    def __init__(self, on_line):
        self.on_line, self.buf = on_line, b""

    def write(self, data):
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            self.on_line(line.decode("ascii", "replace").strip())

    async def drain(self):
        pass


def _nmea(lat, lon, kn, course):
    def dm(deg, w):
        d = int(abs(deg))
        return f"{d:0{w}d}{(abs(deg) - d) * 60:07.4f}"

    def wrap(body):
        cs = 0
        for ch in body:
            cs ^= ord(ch)
        return f"${body}*{cs:02X}"
    t = time.gmtime()
    hms, date = time.strftime("%H%M%S.00", t), time.strftime("%d%m%y", t)
    pos = f"{dm(lat, 2)},{'N' if lat >= 0 else 'S'},{dm(lon, 3)},{'E' if lon >= 0 else 'W'}"
    return [wrap(f"GPRMC,{hms},A,{pos},{kn:.2f},{course:.1f},{date},,,D"), wrap(f"GPGGA,{hms},{pos},2,09,0.9,35.0,M,46.0,M,,")]


async def main():
    ap = argparse.ArgumentParser(description="Bridge between the Jaguar simulator page, the real Jaguar and your PC programs")
    ap.add_argument("--ip", default=ROBOT_IP, help=f"the real Jaguar (default {ROBOT_IP})")
    ap.add_argument("--port", type=int, default=ROBOT_PORT)
    ap.add_argument("--ws-port", type=int, default=WS_PORT, help=f"WebSocket port for the page (default {WS_PORT})")
    ap.add_argument("--sim-port", type=int, default=ROBOT_PORT,
                    help=f"TCP port on 127.0.0.1 where PC programs reach the page's Jaguar (default {ROBOT_PORT}, 0 = off)")
    ap.add_argument("--fake", action="store_true", help="a pretend Jaguar instead of the real one")
    ap.add_argument("--no-robot", action="store_true", help="don't connect a real Jaguar (only PC programs <-> the simulator)")
    args = ap.parse_args()
    try:
        import websockets
    except ImportError:
        sys.exit("pip install websockets")
    b = Bridge(args)
    async with websockets.serve(b.page_handler, "localhost", args.ws_port, max_size=2 ** 22):
        print(f"Page: ws://localhost:{args.ws_port}  (indexJaguar.php, card Real Jaguar -> Connect)")
        tasks = [asyncio.create_task(b.pump())]
        if args.sim_port:
            server = await asyncio.start_server(b.sim_client, "127.0.0.1", args.sim_port)
            print(f"PC programs: 127.0.0.1:{args.sim_port} is the page's Jaguar (python robot.py --ip 127.0.0.1)")
            tasks.append(asyncio.create_task(server.serve_forever()))
        if not args.no_robot:
            print("Robot: " + ("a pretend Jaguar (--fake)" if args.fake else f"{args.ip}:{args.port}"))
            tasks.append(asyncio.create_task(b.robot_loop()))
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
