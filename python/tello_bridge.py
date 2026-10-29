"""Bridge between the Tello EDU simulator page in your browser and a real DJI Tello EDU.

A web page can't send UDP, and the Tello only talks UDP, so this program does, on a PC that is connected
to the Tello's Wi-Fi (TELLO-xxxxxx), or in the same network when the Tello EDU joined yours (ap command):

    browser (indexTello.php)  <-- WebSocket ws://localhost:8771 -->  tello_bridge.py  <-- UDP -->  Tello EDU
                                                                        8889: SDK commands and their answers
                                                                        8890: state packets, 10 per second
                                                                       11111: video (H.264), decoded here

What the page can do through it:
  * see the real drone: its battery, height, ToF, flight time, and its video (front camera, or the bottom
    camera after "downvision 1") in the card "Real Tello EDU";
  * fly it (only when "Fly the real Tello" is ticked on the page): every SDK command of the page (the Python
    script, the buttons, the keys as rc) goes to the real drone too, and its answer goes back to the script.

    pip install websockets av pillow
    python tello_bridge.py                      # the Tello at 192.168.10.1 (its own Wi-Fi)
    python tello_bridge.py --ip 192.168.1.42    # a Tello EDU that joined your network
    python tello_bridge.py --fake               # no drone: a pretend Tello, to try the page

Then open indexTello.php, "Real Tello EDU" -> Connect.

Safety: the drone only gets commands when "Fly the real Tello" is ticked; the page never sends commands above
its height limit and lands it after 10 minutes of flight. This bridge lands it when the page closes or loses the
connection while it flies, and stops an rc movement that the page stopped sending for 1 s. While it flies it
sends "keepalive" every 10 s, so the Tello doesn't land by itself after 15 s without a command
(--no-keepalive: it does, as with djitellopy). Fly in a free space, and keep Esc (land) within reach.
"""
import argparse
import asyncio
import base64
import io
import json
import math
import sys
import threading
import time

TELLO_IP = "192.168.10.1"
CMD_PORT, STATE_PORT, VIDEO_PORT = 8889, 8890, 11111
RC_WATCHDOG = 1.0       # s without an rc command after a moving one -> rc 0 0 0 0
KEEPALIVE = 10.0        # s between keepalives while flying
INT_FIELDS = ("mid", "x", "y", "z", "pitch", "roll", "yaw", "vgx", "vgy", "vgz", "templ", "temph", "tof", "h", "bat", "time")
FLOAT_FIELDS = ("baro", "agx", "agy", "agz")


def parse_state(raw):
    """'pitch:0;roll:0;...' -> {'pitch': 0, ...} (as djitellopy does)."""
    out = {}
    for field in raw.strip().split(";"):
        k, _, v = field.partition(":")
        if not k or not _:
            continue
        try:
            out[k] = int(v) if k in INT_FIELDS else float(v) if k in FLOAT_FIELDS else v
        except ValueError:
            out[k] = v
    return out


# ---------- the real drone ----------
class _Udp(asyncio.DatagramProtocol):
    def __init__(self, on_packet):
        self.on_packet = on_packet

    def datagram_received(self, data, addr):
        self.on_packet(data, addr)


class RealTello:
    name = "Tello EDU"

    def __init__(self, ip):
        self.ip = ip
        self.replies = asyncio.Queue()
        self.state_raw, self.state_t = "", 0.0
        self.lock = asyncio.Lock()
        self.cmd = None
        self.jpeg, self.jpeg_lock = None, threading.Lock()
        self.video_on = False

    async def open(self):
        loop = asyncio.get_running_loop()
        self.cmd, _ = await loop.create_datagram_endpoint(
            lambda: _Udp(lambda d, a: self.replies.put_nowait(d.decode("utf-8", "replace").strip())), local_addr=("0.0.0.0", CMD_PORT))
        await loop.create_datagram_endpoint(lambda: _Udp(self._state), local_addr=("0.0.0.0", STATE_PORT))

    def _state(self, data, addr):
        self.state_raw, self.state_t = data.decode("utf-8", "replace").strip(), time.time()

    @property
    def answering(self):
        return time.time() - self.state_t < 2

    def send(self, text):
        """A command without an answer (rc, emergency)."""
        self.cmd.sendto(text.encode(), (self.ip, CMD_PORT))

    async def command(self, text, timeout=7.0):
        """A command and the drone's answer ('ok', 'error ...', a value); None when none came in time."""
        async with self.lock:                        # one at a time, as the Tello wants
            while not self.replies.empty():
                self.replies.get_nowait()            # an answer that came too late for an earlier command
            self.cmd.sendto(text.encode(), (self.ip, CMD_PORT))
            try:
                return await asyncio.wait_for(self.replies.get(), timeout)
            except asyncio.TimeoutError:
                return None

    def start_video(self, width):
        """Decode the H.264 stream (UDP 11111) in a thread; keep the newest picture as a JPEG."""
        if self.video_on:
            return
        self.video_on = True
        threading.Thread(target=self._video, args=(width,), daemon=True).start()

    def _video(self, width):
        import av
        while True:
            try:
                container = av.open(f"udp://@0.0.0.0:{VIDEO_PORT}", timeout=(5, None))
                for frame in container.decode(video=0):
                    im = frame.to_image()
                    if im.width > width:
                        im = im.resize((width, round(im.height * width / im.width)))
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=70)
                    with self.jpeg_lock:
                        self.jpeg = buf.getvalue()
            except Exception as e:                   # no stream yet (streamoff), or it broke off: try again
                print(f"video: {e.__class__.__name__}: {e}")
                time.sleep(1)

    def picture(self):
        with self.jpeg_lock:
            jpeg, self.jpeg = self.jpeg, None
        return jpeg


# ---------- a pretend drone (--fake) ----------
class FakeTello:
    """Answers like a Tello, without flying anything: to try the page and this bridge without a drone."""
    name = "pretend Tello (--fake)"

    def __init__(self):
        self.h, self.flying, self.bat, self.t0, self.flown, self.yaw = 0, False, 100, time.time(), 0.0, 0
        self.stream, self.down, self.n = False, False, 0
        self.answering = True

    async def open(self):
        pass

    @property
    def state_raw(self):
        if self.flying:
            self.flown = time.time() - self.t0
        return (f"mid:-2;x:-200;y:-200;z:-200;mpry:0,0,0;pitch:0;roll:0;yaw:{self.yaw};vgx:0;vgy:0;vgz:0;templ:60;temph:62;"
                f"tof:{self.h + 10};h:{self.h};bat:{max(0, 100 - int(self.flown / 6))};baro:42.10;time:{int(self.flown)};agx:0.00;agy:0.00;agz:-1000.00;")

    def send(self, text):
        if text == "emergency":
            self.flying, self.h = False, 0

    async def command(self, text, timeout=7.0):
        c, *a = text.split()
        n = [int(x) for x in a if x.lstrip("-").isdigit()]
        if c == "takeoff":
            await asyncio.sleep(2)
            self.flying, self.h, self.t0 = True, 80, time.time() - self.flown
        elif c == "land":
            await asyncio.sleep(2)
            self.flying, self.h = False, 0
        elif c in ("up", "down", "forward", "back", "left", "right", "go", "curve", "jump", "cw", "ccw", "flip"):
            if not self.flying:
                return "error Motor stop"
            await asyncio.sleep(min(timeout, 1 + (abs(n[0]) / 100 if n else 0)))
            self.h += n[0] if c == "up" else -n[0] if c == "down" else 0
            self.yaw = (self.yaw + (n[0] if c == "cw" else -n[0] if c == "ccw" else 0) + 180) % 360 - 180
        elif c == "streamon":
            self.stream = True
        elif c == "streamoff":
            self.stream = False
        elif c == "downvision":
            self.down = a == ["1"]
        elif c.endswith("?"):
            st = parse_state(self.state_raw)
            return {"battery?": str(st["bat"]), "height?": f"{self.h // 10}dm", "time?": f"{st['time']}s", "speed?": "100.0",
                    "tof?": f"{(self.h + 10) * 10}mm", "sdk?": "20", "sn?": "0TQZFAKE000000", "wifi?": "90",
                    "temp?": "60~62C", "baro?": "42.10", "attitude?": f"pitch:0;roll:0;yaw:{self.yaw};"}.get(c, "ok")
        return "ok"

    def start_video(self, width):
        pass

    def picture(self):
        """A pretend picture: a moving horizon, so the page shows something."""
        if not self.stream:
            return None
        from PIL import Image, ImageDraw
        self.n += 1
        w, h = (320, 240) if self.down else (480, 360)
        im = Image.new("RGB", (w, h), (90, 90, 90) if self.down else (40, 90, 160))
        d = ImageDraw.Draw(im)
        if not self.down:
            y = h // 2 + int(10 * math.sin(self.n / 10))
            d.rectangle([0, y, w, h], fill=(70, 70, 60))
        d.text((10, 10), f"{self.name}  h {self.h} cm  {'bottom' if self.down else 'front'} camera", fill=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
        return buf.getvalue()


# ---------- the bridge ----------
class Bridge:
    def __init__(self, tello, args):
        self.tello, self.args = tello, args
        self.clients = set()
        self.flying, self.down, self.msg = False, False, ""
        self.rc_moving, self.last_rc, self.last_cmd = False, 0.0, 0.0
        self.sdk = False

    async def broadcast(self, message):
        data = json.dumps(message)
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except Exception:
                self.clients.discard(ws)

    async def command(self, text, timeout=7.0):
        """A command with an answer; keeps track of flying, the camera and SDK mode."""
        self.last_cmd = time.time()
        answer = await self.tello.command(text, timeout)
        c = text.split()[0].lower() if text.split() else ""
        if answer and "ok" in answer.lower():
            if c == "command":
                self.sdk = True
            elif c in ("takeoff", "throwfly"):
                self.flying = True
            elif c == "land":
                self.flying = False
            elif c == "downvision":
                self.down = text.split()[1:] == ["1"]
            elif c == "streamon":
                self.tello.start_video(self.args.video_width)
        print(f"{text!r:32} -> {answer!r}")
        return answer

    def send(self, text):
        self.last_cmd = time.time()
        parts = text.split()
        if parts and parts[0] == "rc":
            self.last_rc = time.time()
            self.rc_moving = any(p != "0" for p in parts[1:])
        if parts and parts[0] == "emergency":
            self.flying = False
            print("EMERGENCY: motors off")
        self.tello.send(text)

    async def handle(self, ws, m):
        cmd = m.get("cmd")
        if cmd == "hello":
            await ws.send(json.dumps({"type": "hello", "name": self.tello.name}))
        elif cmd == "sdk":
            text = str(m.get("text", "")).strip()
            answer = await self.command(text, float(m.get("timeout", 7)))
            await ws.send(json.dumps({"type": "reply", "id": m.get("id"), "text": answer}))
        elif cmd == "send":
            self.send(str(m.get("text", "")).strip())

    async def client(self, ws):
        self.clients.add(ws)
        print(f"Page connected ({len(self.clients)} open)")
        from websockets.exceptions import ConnectionClosed
        try:
            async for raw in ws:
                try:
                    asyncio.create_task(self.handle(ws, json.loads(raw)))   # a long move doesn't hold up the rest (rc, land)
                except (ValueError, KeyError, TypeError) as e:
                    print(f"bad message: {e}")
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            print(f"Page disconnected ({len(self.clients)} open)")
            if self.flying and not self.clients:
                print("The page went away while the Tello flies: landing")
                self.send("rc 0 0 0 0")
                await self.command("land")

    async def state_loop(self):
        """10 times a second: the state to every page; the rc watchdog and the keepalive."""
        while True:
            now = time.time()
            if self.rc_moving and now - self.last_rc > RC_WATCHDOG:
                print("No rc from the page for 1 s: stop")
                self.send("rc 0 0 0 0")
            if self.flying and self.args.keepalive and now - self.last_cmd > KEEPALIVE:
                asyncio.create_task(self.command("keepalive"))
            raw = self.tello.state_raw
            if self.clients:
                await self.broadcast({"type": "state", "name": self.tello.name, "connected": bool(self.tello.answering),
                                      "raw": raw, "fields": parse_state(raw) if raw else None, "flying": self.flying,
                                      "down": self.down, "msg": self.msg})
            await asyncio.sleep(0.1)

    async def camera_loop(self):
        while True:
            if self.clients:
                jpeg = self.tello.picture()
                if jpeg:
                    await self.broadcast({"type": "camera", "jpeg": base64.b64encode(jpeg).decode(), "down": self.down})
            await asyncio.sleep(1 / self.args.camera_fps)

    async def start(self):
        """SDK mode, then the video (unless --no-video): retried until the Tello answers."""
        while True:
            if await self.command("command", 3) == "ok":
                break
            self.msg = f"no answer from {getattr(self.tello, 'ip', 'the Tello')}: is this PC on the Tello's Wi-Fi?"
            print(self.msg)
            await asyncio.sleep(2)
        self.msg = ""
        bat = await self.command("battery?", 3)
        print(f"SDK mode; battery {bat} %")
        if not self.args.no_video:
            await self.command("streamon", 5)


async def main_async(args):
    import websockets
    tello = FakeTello() if args.fake else RealTello(args.ip)
    await tello.open()
    bridge = Bridge(tello, args)
    async with websockets.serve(bridge.client, args.host, args.port, max_size=2 ** 22):
        print(f"Bridge ready on ws://{args.host}:{args.port}  ->  open indexTello.php, 'Real Tello EDU' -> Connect")
        print("The Tello only flies when 'Fly the real Tello' is ticked on the page.  Ctrl+C to quit.")
        await asyncio.gather(bridge.start(), bridge.state_loop(), bridge.camera_loop())


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=TELLO_IP, help="the Tello's IP address (192.168.10.1 on its own Wi-Fi)")
    ap.add_argument("--fake", action="store_true", help="no drone: a pretend Tello (to try the page)")
    ap.add_argument("--host", default="127.0.0.1", help="where the page connects (keep 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--no-video", action="store_true", help="don't start the video stream")
    ap.add_argument("--video-width", type=int, default=480, help="width of the pictures sent to the page")
    ap.add_argument("--camera-fps", type=float, default=10)
    ap.add_argument("--no-keepalive", dest="keepalive", action="store_false",
                    help="no keepalive while flying: the Tello lands after 15 s without a command")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Bye")


if __name__ == "__main__":
    main()
