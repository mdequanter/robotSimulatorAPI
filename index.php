<?php
// Botopia simulator REST API (PHP): remote control of a simulator that is open in a browser.
//
// Open a simulator with ?remote=1 (e.g. https://robotics.dinno.be/indexSkid.php?remote=1). The page shows a
// session code, registers itself here, fetches commands every 100 ms and sends its telemetry every
// 200 ms. Clients use the same session code to send commands and read telemetry. The simulator
// runs in real time in the browser; for fast, step-by-step training use server.mjs instead.
// Same JSON formats as the Node API; see README.md.
//
//   Page:   POST ?r=register&session=ID        body: info              → register / re-register
//           GET  ?r=pull&session=ID&since=N                             → commands with seq > N (+ stream: { fps, width }
//                                                                          while someone watches the camera)
//           POST ?r=telemetry&session=ID       body: { obs, events }   → latest observation
//           POST ?r=frame&session=ID           body: a JPEG            → the robot camera's latest picture
//   Client: GET  ?r=sessions                                           → active sessions
//           GET  ?r=state&session=ID&since=N                            → info, obs, events after N
//           POST ?r=command&session=ID         body: { action } or { reset: { lane, seed } }
//           GET  ?r=frame&session=ID&fps=10&width=640                  → the latest picture of the robot camera (JPEG)
//           GET  ?r=mjpeg&session=ID&fps=10&width=640                  → the robot camera as an MJPEG stream
//
// The camera: the page only sends pictures while someone asks for them (frame or mjpeg in the last
// STREAM_SECONDS), at the fps and width asked (at most 30 fps and 640 × 480).

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Headers: Content-Type');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Cache-Control: no-store');
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(204); exit; }

const DATA_DIR = __DIR__ . '/data';
const MAX_BODY = 262144;          // bytes
const MAX_EVENTS = 1000;          // kept per session
const MAX_COMMANDS = 200;
const ACTIVE_SECONDS = 120;       // a session counts as active this long after the page's last message
const KEEP_SECONDS = 86400;       // session files older than this are removed
const STREAM_SECONDS = 5;         // the page sends camera pictures this long after the last request for one
const MJPEG_SECONDS = 600;        // one MJPEG connection lasts at most this long (then connect again)

function reply($code, $data) {
    http_response_code($code);
    echo json_encode($data, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    exit;
}
function body() {
    $raw = file_get_contents('php://input', false, null, 0, MAX_BODY + 1);
    if ($raw === false || strlen($raw) > MAX_BODY) reply(413, ['error' => 'body too large']);
    if ($raw === '') return [];
    $data = json_decode($raw, true);
    if (!is_array($data)) reply(400, ['error' => 'invalid JSON']);
    return $data;
}

// Data folder, shut off from the web (Apache); the PHP built-in server never serves it either.
if (!is_dir(DATA_DIR)) {
    @mkdir(DATA_DIR, 0775, true);
    @file_put_contents(DATA_DIR . '/.htaccess', "Require all denied\nDeny from all\n");
    @file_put_contents(DATA_DIR . '/index.html', '');
}
// Without a writable data folder no session can be stored: say so, instead of a bare error 500 later.
if (!is_dir(DATA_DIR) || !is_writable(DATA_DIR)) {
    $user = function_exists('posix_geteuid') ? (posix_getpwuid(posix_geteuid())['name'] ?? '') : get_current_user();
    reply(500, ['error' => 'the data folder is missing or not writable for PHP (user ' . $user . '): ' . DATA_DIR]);
}

$route = $_GET['r'] ?? '';
// On the simulator site (auth.php one folder up) only a signed-in page with access may open a session;
// clients then reach the robot through that session's code. Without auth.php the API stays open.
if ($route === 'register' && is_file(__DIR__ . '/../auth.php')) {
    require __DIR__ . '/../auth.php';
    try { $allowed = hasAccess(); }
    catch (Throwable $e) { error_log('robotsim API: account check failed: ' . $e->getMessage()); reply(500, ['error' => 'the account check failed']); }
    if (session_status() === PHP_SESSION_ACTIVE) session_write_close();
    if (!$allowed) reply(403, ['error' => 'sign in on the simulator site first: your free day is over or you are not signed in']);
}
$session = strtoupper(preg_replace('/[^A-Za-z0-9]/', '', $_GET['session'] ?? ''));
if ($session !== '' && (strlen($session) < 4 || strlen($session) > 16)) reply(400, ['error' => 'session code: 4 to 16 letters or digits']);
$file = DATA_DIR . '/' . $session . '.json';

// Read-modify-write a session file under an exclusive lock.
function withSession($file, $fn) {
    $h = fopen($file, 'c+');
    if (!$h) reply(500, ['error' => 'cannot open session store']);
    flock($h, LOCK_EX);
    $raw = stream_get_contents($h);
    $s = $raw ? json_decode($raw, true) : null;
    [$s, $out] = $fn($s);
    if ($s !== null) {
        ftruncate($h, 0);
        rewind($h);
        fwrite($h, json_encode($s, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
    }
    flock($h, LOCK_UN);
    fclose($h);
    return $out;
}
function readSession($file) {
    if (!is_file($file)) return null;
    $h = fopen($file, 'r');
    flock($h, LOCK_SH);
    $s = json_decode(stream_get_contents($h), true);
    flock($h, LOCK_UN);
    fclose($h);
    return $s;
}
function needSession($session) { if ($session === '') reply(400, ['error' => 'missing session']); }

// ---------- Camera ----------
// What the viewers ask for: fps 1 … 30 (default 10), width 160 … 640 (default 640, 4:3 so at most 640 × 480).
function streamAsk() {
    return ['fps' => max(1, min(30, (int)($_GET['fps'] ?? 10))), 'width' => max(160, min(640, (int)($_GET['width'] ?? 640)))];
}
function streamWant($session, $ask) {
    @file_put_contents(DATA_DIR . '/' . $session . '.want', json_encode($ask + ['t' => microtime(true)]), LOCK_EX);
}
function streamWanted($session) {
    $w = @json_decode((string)@file_get_contents(DATA_DIR . '/' . $session . '.want'), true);
    return is_array($w) && ($w['t'] ?? 0) > microtime(true) - STREAM_SECONDS ? ['fps' => $w['fps'], 'width' => $w['width']] : null;
}
function framePath($session) { return DATA_DIR . '/' . $session . '.jpg'; }

switch ($route) {
    case '':
        reply(200, ['name' => 'Botopia simulator API (PHP, remote control)', 'version' => 1, 'docs' => 'api/README.md',
            'routes' => ['sessions', 'state', 'command', 'frame', 'mjpeg', 'register', 'pull', 'telemetry']]);

    // ---------- Page side ----------
    case 'register':
        needSession($session);
        $info = body();
        foreach (glob(DATA_DIR . '/*.{json,jpg,want}', GLOB_BRACE) as $old) {          // tidy up old sessions
            if (filemtime($old) < time() - KEEP_SECONDS) @unlink($old);
        }
        reply(200, withSession($file, function ($s) use ($session, $info) {
            $now = microtime(true);
            $s = $s ?: ['session' => $session, 'created' => $now, 'seq' => 0, 'commands' => [], 'events' => [], 'eventBase' => 0, 'obs' => null];
            $s['info'] = $info;
            $s['seen'] = $now;
            return [$s, ['session' => $session, 'seq' => $s['seq']]];
        }));

    case 'pull':
        needSession($session);
        $since = (int)($_GET['since'] ?? 0);
        $s = readSession($file);
        if (!$s) reply(404, ['error' => 'unknown session']);
        $cmds = array_values(array_filter($s['commands'], fn($c) => $c['seq'] > $since));
        $stream = streamWanted($session);
        reply(200, $stream ? ['commands' => $cmds, 'stream' => $stream] : ['commands' => $cmds]);

    case 'telemetry':
        needSession($session);
        if (!is_file($file)) reply(404, ['error' => 'unknown session: register first']);
        $msg = body();
        reply(200, withSession($file, function ($s) use ($msg) {
            if (!$s) return [null, ['error' => 'unknown session']];
            $s['obs'] = $msg['obs'] ?? $s['obs'];
            $s['seen'] = microtime(true);
            foreach (($msg['events'] ?? []) as $e) $s['events'][] = $e;
            $over = count($s['events']) - MAX_EVENTS;
            if ($over > 0) { $s['events'] = array_slice($s['events'], $over); $s['eventBase'] += $over; }
            return [$s, ['ok' => true]];
        }));

    // The robot camera: the page POSTs a JPEG, a client GETs the latest one (and so asks for more).
    case 'frame':
        needSession($session);
        if (!is_file($file)) reply(404, ['error' => 'unknown session']);
        if ($_SERVER['REQUEST_METHOD'] === 'POST') {
            $jpeg = file_get_contents('php://input', false, null, 0, MAX_BODY + 1);
            if ($jpeg === false || strlen($jpeg) > MAX_BODY) reply(413, ['error' => 'picture too large']);
            if (strncmp($jpeg, "\xFF\xD8", 2) !== 0) reply(400, ['error' => 'send a JPEG']);
            $tmp = framePath($session) . '.' . getmypid() . '.tmp';
            if (@file_put_contents($tmp, $jpeg) === false || !@rename($tmp, framePath($session))) { @unlink($tmp); reply(500, ['error' => 'cannot store the picture']); }
            reply(200, ['ok' => true]);
        }
        streamWant($session, streamAsk());
        $jpeg = @file_get_contents(framePath($session));
        if (!$jpeg) reply(404, ['error' => 'no picture yet: the simulator sends its camera once someone asks; try again in a second (and keep its window visible)']);
        header('Content-Type: image/jpeg');
        header('X-Frame-Age: ' . max(0, time() - filemtime(framePath($session))));      // whole seconds
        echo $jpeg;
        exit;

    // The same pictures as one MJPEG stream (multipart/x-mixed-replace): a browser shows it as an image,
    // OpenCV opens it with cv2.VideoCapture(url). Each new picture is sent once, at most fps a second; the
    // same one again after 2 s without a new one, so a viewer that has gone is noticed (PHP only sees that
    // when it writes) and the connection does not hold a server process for nothing.
    case 'mjpeg':
        needSession($session);
        if (!is_file($file)) reply(404, ['error' => 'unknown session']);
        $ask = streamAsk();
        set_time_limit(MJPEG_SECONDS + 30);
        ignore_user_abort(false);
        header('Content-Type: multipart/x-mixed-replace; boundary=frame');
        header('X-Accel-Buffering: no');                       // nginx: pass every picture on at once
        while (ob_get_level()) ob_end_flush();
        $last = null;
        $sentAt = 0;
        $end = microtime(true) + MJPEG_SECONDS;
        while (!connection_aborted() && microtime(true) < $end) {
            $t0 = microtime(true);
            streamWant($session, $ask);
            $jpeg = @file_get_contents(framePath($session));
            if ($jpeg && (($sum = crc32($jpeg)) !== $last || $t0 - $sentAt > 2)) {
                $last = $sum;
                $sentAt = $t0;
                echo "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " . strlen($jpeg) . "\r\n\r\n", $jpeg, "\r\n";
                flush();
            } elseif (!$jpeg && $t0 - $sentAt > 2) {            // no picture yet: an empty line keeps the check going
                $sentAt = $t0;
                echo "\r\n";
                flush();
            }
            usleep(max(0, (int)((1 / $ask['fps'] - (microtime(true) - $t0)) * 1e6)));
        }
        exit;

    // ---------- Client side ----------
    case 'sessions':
        $list = [];
        foreach (glob(DATA_DIR . '/*.json') as $f) {
            $s = readSession($f);
            if (!$s || ($s['seen'] ?? 0) < microtime(true) - ACTIVE_SECONDS) continue;
            $list[] = ['session' => $s['session'], 'robot' => $s['info']['robot'] ?? null, 'name' => $s['info']['name'] ?? null,
                'lastSeenSeconds' => round(microtime(true) - $s['seen'], 1)];
        }
        reply(200, $list);

    case 'state':
        needSession($session);
        $s = readSession($file);
        if (!$s) reply(404, ['error' => 'unknown session']);
        // Events have a running number; ask for the ones after `since` (start with 0).
        $since = (int)($_GET['since'] ?? 0);
        $from = max(0, $since - $s['eventBase']);
        reply(200, ['session' => $s['session'], 'info' => $s['info'], 'obs' => $s['obs'],
            'events' => array_slice($s['events'], $from), 'next' => $s['eventBase'] + count($s['events']),
            'lastSeenSeconds' => round(microtime(true) - ($s['seen'] ?? 0), 2),
            'active' => ($s['seen'] ?? 0) >= microtime(true) - ACTIVE_SECONDS]);

    case 'command':
        needSession($session);
        if ($_SERVER['REQUEST_METHOD'] !== 'POST') reply(405, ['error' => 'use POST']);
        if (!is_file($file)) reply(404, ['error' => 'unknown session']);
        $cmd = body();
        if (!isset($cmd['action']) && !isset($cmd['reset'])) reply(400, ['error' => 'send { "action": … } or { "reset": { "lane": … } }']);
        reply(200, withSession($file, function ($s) use ($cmd) {
            if (!$s) return [null, ['error' => 'unknown session']];
            $c = ['seq' => ++$s['seq'], 't' => microtime(true)];
            if (isset($cmd['action'])) $c['action'] = $cmd['action'];
            if (isset($cmd['reset'])) $c['reset'] = $cmd['reset'];
            $s['commands'][] = $c;
            if (count($s['commands']) > MAX_COMMANDS) $s['commands'] = array_slice($s['commands'], -MAX_COMMANDS);
            return [$s, ['seq' => $c['seq']]];
        }));

    default:
        reply(404, ['error' => 'unknown route', 'routes' => ['sessions', 'state', 'command', 'frame', 'mjpeg', 'register', 'pull', 'telemetry']]);
}
