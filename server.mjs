// Botopia simulator REST API for training (reinforcement learning).
//
//   node server.mjs [--port 8780] [--show] [--browsers 4] [--chrome "path/to/chrome"] [--site URL or folder] [--key rsk_...]
//
// Needs only Node.js 22+ (built-in fetch and WebSocket) and Google Chrome / Chromium. The server
// serves the simulator pages itself, starts Chrome headless (--show: visible, redrawn every step) and runs every
// environment in its own tab, step by step: the simulation only advances when you call /step,
// so results are reproducible (with a seed) and as fast as the CPU allows. See README.md.
//
// The simulator pages come from --site: a folder with the simulator site, or a URL whose pages are
// fetched once and kept in memory. Default: the folder above this one when it holds the site (as on
// the web server, where this repository is the site's api/ folder), else https://robotics.dinno.be/.
// That site needs an account: --key (or ROBOTSIM_KEY) is the API key from its account page.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const API_DIR = path.dirname(fileURLToPath(import.meta.url));
const arg = (name, def) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : def; };
const isSite = dir => fs.existsSync(path.join(dir, 'index.php')) && fs.existsSync(path.join(dir, 'common', 'sim.js'));
const SITE_ARG = arg('site', isSite(path.join(API_DIR, '..')) ? path.join(API_DIR, '..') : 'https://robotics.dinno.be/');
const SITE_URL = /^https?:\/\//.test(SITE_ARG) ? SITE_ARG.replace(/\/?$/, '/') : null;
const ROOT = SITE_URL ? null : path.resolve(SITE_ARG);                         // the simulator site on disk
if (ROOT && !isSite(ROOT)) { console.error(`--site ${ROOT}: no simulator site there (index.php, common/sim.js)`); process.exit(1); }
const KEY = arg('key', process.env.ROBOTSIM_KEY || '');
const PORT = Number(arg('port', 8780));
const SHOW = process.argv.includes('--show');
// Headless Chrome draws all its tabs in ONE (software) GPU process, so several browsers render in
// parallel: --browsers N (default one per 4 CPU cores, at most 4). --show uses one visible browser.
const BROWSERS = SHOW ? 1 : Math.max(1, Number(arg('browsers', Math.min(4, Math.round(os.cpus().length / 4)))));

// Robot id → simulator page.
const ROBOTS = {
  wheeled: 'index2Wheels.php', bike: 'indexBike.php', synchro: 'indexSynchro.php',
  mecanum: 'indexOmni.php?variant=mecanum', kiwi: 'indexOmni.php?variant=kiwi', swerve: 'indexOmni.php?variant=swerve',
  tracked: 'indexTracked.php', skid: 'indexSkid.php', ackermann: 'indexAckermann.php', '4wd': 'index4WD.php',
  whegs: 'indexWhegs.php', quadruped: 'indexQuadruped.php', mini: 'indexSmall.php', wheellegs: 'indexWheelLegs.php',
  ostrich: 'indexOstrich.php', humanoid: 'indexHumanoid.php', snake: 'indexSnake.php', create3: 'indexCreate3.php',
  go2: 'indexGo2.php', g1: 'indexG1.php', turtlebot4: 'indexTurtleBot4.php',
  go2w: 'indexGo2W.php', r1: 'indexR1.php', rosbotxl: 'indexROSbotXL.php', leorover: 'indexLeoRover.php', bittle: 'indexBittle.php',
  roboworks: 'indexRoboworks.php', mecabot: 'indexRoboworks.php?variant=mecabot', ugvrover: 'indexUGVRover.php',
  jetbot: 'indexJetBot.php', jetracer: 'indexJetRacer.php', xarm5: 'indexXArm6.php?model=5', xarm850: 'indexXArm6.php?model=850',
  sespro: 'indexSESPro.php', xgo: 'indexXGO.php', xgomini: 'indexXGO.php?variant=mini', a2: 'indexA2.php', as2: 'indexA2.php?variant=as2',
  agibotd1: 'indexAgiBotD1.php', h2: 'indexH2.php', boosterk1: 'indexBoosterK1.php', tron1: 'indexTron1.php', tron1w: 'indexTron1.php?variant=wheel',
  panther: 'indexPanther.php', jaguar: 'indexJaguar.php', summitxl: 'indexSummitXL.php', rbkairos: 'indexRBKairos.php', rbvogui: 'indexRBKairos.php?variant=vogui',
  xarm6: 'indexXArm6.php', xarm7: 'indexXArm6.php?model=7', omx: 'indexOpenManipulator.php', driftx: 'indexDriftX.php',
  rangermini: 'indexRangerMini.php', limo: 'indexLimo.php', phoenix: 'indexPhoenix.php', spot: 'indexSpot.php', jackal: 'indexJackal.php', husky: 'indexHusky.php', scoutmini: 'indexScoutMini.php', hunterse: 'indexHunterSE.php',
};

// Suggested reward (the client may compute its own from obs + events): progress along the lane,
// small penalties for pushing against things, big ones for falling, a bonus at the goal.
const REWARD = { progress: 10, collision: -1, blocked: -0.2, stall: -0.5, fell: -10, bottle_fell: -10, goal: 10 };
function reward(prevObs, res) {
  let r = REWARD.progress * (res.obs.progress.distance - (prevObs ? prevObs.progress.distance : res.obs.progress.distance));
  for (const e of res.events) r += REWARD[e.type] || 0;
  return +r.toFixed(4);
}

// ---------- Static files (the simulator pages) ----------
const TYPES = { '.php': 'text/html', '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.md': 'text/plain; charset=utf-8', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml', '.mp4': 'video/mp4' };
function serveFile(res, file) {
  fs.readFile(file, (err, data) => {
    if (err) return send(res, 404, { error: 'not found' });
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
}
const remoteCache = new Map();        // path → Promise of { status, type, data, why }
// A file of the site at SITE_URL, fetched once. The robot pages are only served with an account's API key
// (--key); the scripts, models and images are open. Without a valid key the site answers 401 / 403 with its
// reason as plain text, or redirects a request without a key to the sign-in page.
function remoteFile(pathname) {
  if (!remoteCache.has(pathname)) {
    const p = fetch(new URL('.' + pathname, SITE_URL), { headers: KEY ? { 'X-Robotsim-Key': KEY } : {}, redirect: 'manual' }).then(async r => {
      const data = Buffer.from(await r.arrayBuffer());
      const plain = (r.headers.get('content-type') || '').startsWith('text/plain');
      return { status: r.status, type: TYPES[path.extname(pathname)] || r.headers.get('content-type') || 'application/octet-stream', data,
        why: r.status === 302 ? 'sign in needed: start the API with --key and the API key from the account page'
          : plain ? data.toString('utf8').slice(0, 300) : `HTTP ${r.status}` };
    });
    p.then(r => { if (r.status !== 200) remoteCache.delete(pathname); }, () => remoteCache.delete(pathname));
    remoteCache.set(pathname, p);
  }
  return remoteCache.get(pathname);
}
const remoteError = r => `${SITE_URL}: ${r.why}`;
function serveStatic(req, res, pathname) {
  if (pathname === '/') pathname = '/index.php';
  if (/\/\./.test(pathname)) return send(res, 403, { error: 'forbidden' });       // .git, .htaccess, …
  if (SITE_URL) {
    return remoteFile(pathname).then(r => {
      if (r.status !== 200) return send(res, r.status === 302 ? 401 : r.status, { error: remoteError(r) });
      res.writeHead(200, { 'Content-Type': r.type, 'Content-Length': r.data.length });
      res.end(r.data);
    }, err => send(res, 502, { error: `${SITE_URL} not reachable: ${err.message}` }));
  }
  const file = path.resolve(ROOT, '.' + decodeURIComponent(pathname));
  if (!file.startsWith(ROOT + path.sep) || file.startsWith(path.join(ROOT, 'api') + path.sep)) return send(res, 403, { error: 'forbidden' });
  serveFile(res, file);
}

// ---------- Chrome ----------
function findChrome() {
  const given = arg('chrome', process.env.CHROME);
  const candidates = given ? [given] : {
    win32: ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
      path.join(os.homedir(), 'AppData/Local/Google/Chrome/Application/chrome.exe'), 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'],
    darwin: ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium'],
  }[process.platform] || ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
  const found = candidates.find(p => fs.existsSync(p));
  if (!found) throw new Error('Chrome not found: start with --chrome "path/to/chrome" or set CHROME');
  return found;
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
const browsers = [];                  // { proc, profile, port, tabs }
// --show: first tab of the visible Chrome, until an environment opens its own tab.
const START_PAGE = `<title>Botopia API</title><body style="margin:0;display:grid;place-items:center;height:100vh;
  background:#061d34;color:#f7fbff;font:16px system-ui,sans-serif;text-align:center"><div>
  <h1 style="color:#ff9d1c">Botopia simulator API is running</h1>
  <p>http://localhost:${PORT}/api/</p>
  <p>Waiting for an environment: every environment your code creates opens in its own tab here.</p></div>`;
async function startChrome(i) {
  const port = PORT + 1 + i, profile = fs.mkdtempSync(path.join(os.tmpdir(), 'botopia-chrome-'));
  const proc = spawn(findChrome(), [
    // Headless: software WebGL (SwiftShader), works without a GPU. Visible (--show): the normal GPU,
    // because with SwiftShader a visible Chrome window on Windows stays blank.
    ...(SHOW ? [] : ['--headless=new', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
    `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--window-size=960,600',
    // A fresh profile would otherwise open Chrome's welcome / search-engine screens over our tabs.
    '--no-first-run', '--no-default-browser-check', '--disable-search-engine-choice-screen',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    SHOW ? 'data:text/html;charset=utf-8,' + encodeURIComponent(START_PAGE) : 'about:blank'], { stdio: 'ignore' });
  const b = { proc, profile, port, tabs: 0 };
  browsers.push(b);
  for (let k = 0; k < 100; k++) {
    try { if ((await fetch(`http://127.0.0.1:${port}/json/version`)).ok) return b; } catch { /* not up yet */ }
    await sleep(100);
  }
  throw new Error('Chrome did not start');
}

// One tab = one environment, driven over the DevTools protocol.
// A new tab goes to the browser with the fewest tabs.
async function openTab(url) {
  const b = browsers.reduce((m, x) => (x.tabs < m.tabs ? x : m));
  const t = await (await fetch(`http://127.0.0.1:${b.port}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' })).json();
  b.tabs++;
  if (SHOW) await fetch(`http://127.0.0.1:${b.port}/json/activate/${t.id}`).catch(() => {});   // show the newest
  const tab = { targetId: t.id, browser: b, ws: null, errors: [] };
  let id = 0, pending = new Map();
  const failAll = why => { for (const r of pending.values()) r({ error: { message: why } }); pending = new Map(); };
  // (Re)connect to the tab. A DevTools connection can stop delivering answers while the page itself
  // is fine; then a new connection to the same tab takes over.
  async function connect() {
    try { tab.ws?.close(); } catch { /* already closed */ }
    failAll('connection replaced');
    const ws = new WebSocket(t.webSocketDebuggerUrl);
    await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = fail; });
    ws.onmessage = e => {
      const m = JSON.parse(e.data);
      if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
      if (m.method === 'Runtime.exceptionThrown') tab.errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    };
    ws.onclose = () => { if (tab.ws === ws) failAll('connection closed'); };
    tab.ws = ws;
    await send('Runtime.enable');
  }
  const send = (method, params = {}) => new Promise(r => { pending.set(++id, r); tab.ws.send(JSON.stringify({ id, method, params })); });
  // A call with a time limit: no answer within `timeout` ms → reconnect and try once more.
  tab.call = async (method, params = {}, timeout = 30000) => {
    for (let attempt = 0; ; attempt++) {
      let timer;
      const m = await Promise.race([send(method, params), new Promise(r => { timer = setTimeout(() => r(null), timeout); })]);
      clearTimeout(timer);
      if (m && !m.error) return m;
      if (attempt > 0) throw new Error(`no answer from the simulator page (${method}): ${m ? m.error.message : 'timed out'}`);
      console.log(`tab ${t.id.slice(0, 8)}: ${m ? m.error.message : 'no answer in ' + timeout / 1000 + ' s'}, reconnecting`);
      await connect();
    }
  };
  // Evaluate JS in the page, return its (JSON-able) value or throw the page's error.
  tab.evaluate = async expr => {
    const m = await tab.call('Runtime.evaluate', { expression: expr, returnByValue: true });
    if (m.result?.exceptionDetails) throw new Error(m.result.exceptionDetails.exception?.description || 'page error');
    return m.result?.result?.value;
  };
  await connect();
  return tab;
}

// ---------- Environments ----------
const envs = new Map();
let nextId = 1;
// options: page settings passed as URL parameters, e.g. { vmax: 1.8 } (Ackermann top speed).
async function createEnv({ robot, lane = 'free', seed = null, options = {} }) {
  if (!ROBOTS[robot]) throw Object.assign(new Error(`unknown robot "${robot}"`), { status: 400 });
  const page = ROBOTS[robot];
  const extra = new URLSearchParams(Object.entries(options || {}).map(([k, v]) => [k, String(v)])).toString();
  const url = `http://127.0.0.1:${PORT}/${page}${page.includes('?') ? '&' : '?'}env=1&shot${extra ? '&' + extra : ''}`;
  // From the web site: is this robot open with our key? Else say why, instead of a tab that never starts.
  if (SITE_URL) {
    const r = await remoteFile('/' + page.split('?')[0]);
    if (r.status !== 200) throw Object.assign(new Error(remoteError(r)), { status: r.status === 403 ? 403 : 401 });
  }
  const tab = await openTab(url);
  for (let i = 0; i < 200 && !(await tab.evaluate('!!window.botopiaEnv').catch(() => false)); i++) await sleep(100);
  if (!(await tab.evaluate('!!window.botopiaEnv').catch(() => false))) throw new Error(`page did not start: ${tab.errors.join('; ') || url}`);
  const id = String(nextId++);
  const env = { id, robot, tab, lastObs: null, steps: 0, created: new Date().toISOString() };
  envs.set(id, env);
  env.info = await tab.evaluate('window.botopiaEnv.info()');
  env.lastObs = await tab.evaluate(`window.botopiaEnv.reset(${JSON.stringify({ lane, seed })})`);
  await drawIfShown(env);
  return env;
}
// Lock-step pages only draw when asked: with --show, draw after create / reset (steps draw themselves).
async function drawIfShown(env) {
  if (SHOW) await env.tab.evaluate('void window.botopiaEnv.frame(0.1)').catch(() => {});
}
async function closeEnv(env) {
  envs.delete(env.id);
  env.tab.browser.tabs--;
  try { env.tab.ws.close(); await fetch(`http://127.0.0.1:${env.tab.browser.port}/json/close/${env.tab.targetId}`); } catch { /* already gone */ }
}

// One step: { action: { drive: { fwd, turn, strafe }, motors: { id: value }, skill? }, dt: 0.05, frame: 1/60, until? }
// image: { view: 'robot' | 'scene', width, quality } adds a JPEG (base64) of that view after the step,
// in the same call (training from camera images).
// mask: the track band in the robot's view (PNG); external: a fixed camera (xArm); real: the real USB camera (OM-X page);
// depth / depthcolor: the depth of a RealSense D435 (the xArm's wrist camera), mm as a PNG (mm = R·256 + G) or coloured.
const VIEWS = ['scene', 'robot', 'mask', 'external', 'real', 'depth', 'depthcolor'];
const imageExpr = image => image ? `window.botopiaEnv.frame(${Number(image.quality ?? 0.9)}, ${JSON.stringify(VIEWS.includes(image.view) ? image.view : 'robot')}, ${Math.max(16, Math.min(1280, Number(image.width) || 84))}).split(',')[1]` : 'undefined';
// until: 'idle' ends the step early when a running skill has finished (dt is then the maximum).
async function stepEnv(env, { action = {}, dt = 0.05, frame = 1 / 60, image = null, until = null }) {
  usedBy = env.robot; usedAt = Date.now();
  const r = await env.tab.evaluate(`(() => { const r = window.botopiaEnv.step(${JSON.stringify(action)}, ${JSON.stringify({ dt, frame, render: SHOW, until })});
    r.image = ${imageExpr(image)}; return r; })()`);
  const out = { obs: r.obs, events: r.events, reward: reward(env.lastObs, r), done: r.obs.done };
  if (r.image) out.image = r.image;
  env.lastObs = r.obs;
  env.steps++;
  if (env.tab.errors.length) out.pageErrors = env.tab.errors.splice(0);
  return out;
}

// ---------- Hours of a class ----------
// The key of a class member (the class environment on the site): every minute in which an environment was
// stepped counts as an hour of "API" in the progress of the class. Other keys: the site answers 401 and
// nothing more is reported.
let usedAt = 0, usedBy = '';
if (SITE_URL && KEY) {
  const beat = setInterval(() => {
    if (Date.now() - usedAt > 60000) return;
    fetch(new URL('classtrack.php', SITE_URL), { method: 'POST', body: new URLSearchParams({ kind: 'api', page: usedBy }),
      headers: { 'X-Robotsim-Key': KEY } }).then(r => { if (r.status === 401) clearInterval(beat); }, () => { /* offline: the next minute */ });
  }, 60000);
  beat.unref();
}

// ---------- HTTP ----------
function send(res, status, body, type = 'application/json') {
  const data = type === 'application/json' ? Buffer.from(JSON.stringify(body)) : body;
  res.writeHead(status, { 'Content-Type': type, 'Content-Length': data.length, 'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS' });
  res.end(data);
}
const readJson = req => new Promise((ok, fail) => {
  let data = '';
  req.on('data', c => { data += c; if (data.length > 1e6) { fail(Object.assign(new Error('body too large'), { status: 413 })); req.destroy(); } });
  req.on('end', () => { try { ok(data ? JSON.parse(data) : {}); } catch { fail(Object.assign(new Error('invalid JSON'), { status: 400 })); } });
});

const server = http.createServer(async (req, res) => {
  const { pathname, searchParams } = new URL(req.url, 'http://x');
  if (req.method === 'OPTIONS') return send(res, 204, '');
  if (!pathname.startsWith('/api/')) return serveStatic(req, res, pathname);
  try {
    const parts = pathname.split('/').filter(Boolean);           // ['api', 'envs', id, action]
    if (pathname === '/api/' || pathname === '/api') return send(res, 200, { name: 'Botopia simulator API', version: 1, docs: 'api/README.md', robots: Object.keys(ROBOTS) });
    // This repository's own files that the site's pages read (api.html: robots.json, README.md).
    if (req.method === 'GET' && ['robots.json', 'README.md'].includes(parts[1]) && parts.length === 2) return serveFile(res, path.join(API_DIR, parts[1]));
    if (parts[1] === 'robots') return send(res, 200, Object.entries(ROBOTS).map(([robot, page]) => ({ robot, page })));
    if (parts[1] !== 'envs') return send(res, 404, { error: 'unknown endpoint' });

    // Batch: step several environments in one call (in parallel), for vectorised training.
    // { steps: [{ id, action, dt, frame }] } → { results: [{ id, obs, events, reward, done } | { id, error }] }
    if (parts[2] === 'step' && req.method === 'POST') {
      const { steps = [] } = await readJson(req);
      const results = await Promise.all(steps.map(async st => {
        const e = envs.get(String(st.id));
        if (!e) return { id: st.id, error: 'no such environment' };
        try { return { id: e.id, ...(await stepEnv(e, st)) }; } catch (err) { return { id: e.id, error: err.message }; }
      }));
      return send(res, 200, { results });
    }
    if (parts.length === 2) {
      if (req.method === 'GET') return send(res, 200, [...envs.values()].map(e => ({ id: e.id, robot: e.robot, steps: e.steps, created: e.created })));
      if (req.method === 'POST') {
        const env = await createEnv(await readJson(req));
        return send(res, 201, { id: env.id, info: env.info, obs: env.lastObs });
      }
    }
    const env = envs.get(parts[2]);
    if (!env) return send(res, 404, { error: `no environment ${parts[2]}` });
    const action = parts[3] || '';
    if (req.method === 'DELETE' && !action) { await closeEnv(env); return send(res, 200, { closed: env.id }); }
    if (req.method === 'GET' && action === 'info') return send(res, 200, env.info);
    if (req.method === 'GET' && action === 'obs') return send(res, 200, env.lastObs = await env.tab.evaluate('window.botopiaEnv.observe()'));
    if (req.method === 'GET' && action === 'frame') {
      const quality = Number(searchParams.get('quality') || 0.8);
      const view = VIEWS.includes(searchParams.get('view')) ? searchParams.get('view') : 'scene';   // robot: its camera, mask: the band
      const width = Math.max(32, Math.min(1280, Number(searchParams.get('width')) || 320));
      const data = await env.tab.evaluate(`window.botopiaEnv.frame(${quality}, '${view}', ${width})`);
      return send(res, 200, Buffer.from(data.split(',')[1], 'base64'), data.startsWith('data:image/png') ? 'image/png' : 'image/jpeg');
    }
    if (req.method === 'POST' && action === 'reset') {
      const { image = null, ...body } = await readJson(req);      // body: { lane, seed, start, offset, yaw, world }
      env.steps = 0;
      const r = await env.tab.evaluate(`(() => ({ obs: window.botopiaEnv.reset(${JSON.stringify(body)}), image: ${imageExpr(image)} }))()`);
      env.lastObs = r.obs;
      await drawIfShown(env);
      return send(res, 200, r.image ? { obs: r.obs, image: r.image } : { obs: r.obs });
    }
    if (req.method === 'POST' && action === 'step') return send(res, 200, await stepEnv(env, await readJson(req)));
    return send(res, 404, { error: 'unknown endpoint' });
  } catch (err) {
    return send(res, err.status || 500, { error: err.message });
  }
});

for (let i = 0; i < BROWSERS; i++) await startChrome(i);
server.listen(PORT, () => {
  console.log(`Botopia simulator API on http://localhost:${PORT}/api/  (simulators: http://localhost:${PORT}/)`);
  console.log(`Simulator pages from ${SITE_URL || ROOT}`);
  if (SITE_URL && !KEY) console.log(`No --key: ${SITE_URL} only opens the robots with the API key from ${SITE_URL}account.php`);
  console.log(`Chrome ${SHOW ? 'visible' : `headless × ${BROWSERS}`}, robots: ${Object.keys(ROBOTS).join(', ')}`);
});
const stop = async () => {
  for (const b of browsers) try { b.proc.kill(); } catch { /* ignore */ }
  await sleep(500);                                     // let Chrome let go of its profile folder
  for (const b of browsers) try { fs.rmSync(b.profile, { recursive: true, force: true }); } catch { /* ignore */ }
  process.exit(0);
};
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
