// Part of alive-nca, MIT license: https://github.com/ultran1te/alive-nca
/* alive-worker.js: runs one colony of organisms off the main thread. See alive-core.js.
 * Works hard while something is growing or healing, then settles to a slow resting
 * rate so an idle page costs almost nothing. Stops entirely while the tab is hidden.
 */
importScripts('alive-core.js');
let colony = null, rate = 45, rest = 6, busyUntil = 0, timer = 0, paused = false;
const GROW = 520, HEAL = 700;   // steps of full speed after starting, and after each cut

function send(extra) {
  const orgs = colony.snapshot(), move = [];
  for (let i = 0; i < orgs.length; i++) move.push(orgs[i].rgba.buffer, orgs[i].hd.buffer);
  postMessage(Object.assign({ type: 'state', ticks: colony.ticks, ms: colony.ms, resting: colony.ticks >= busyUntil && colony.healthy, restarts: colony.restarts, orgs: orgs }, extra || {}), move);
}
let cost = 0;   // running average of what one step (and its snapshot) costs on this device, in ms
function loop() {
  const t = performance.now();
  colony.tick(); send();
  const spent = performance.now() - t; cost = cost ? cost * 0.9 + spent * 0.1 : spent;
  const busy = colony.ticks < busyUntil || !colony.healthy;   // never rest while something is still mending
  // On a slow phone, leave the CPU some room: at most ~80% of a core while mending, ~10% at rest.
  const period = busy ? Math.max(1000 / rate, cost / 0.8) : Math.max(1000 / rest, cost / 0.1);
  timer = setTimeout(loop, Math.max(0, period - spent));
}
function wake() { busyUntil = colony.ticks + HEAL; if (!paused) { clearTimeout(timer); timer = setTimeout(loop, 0); } }

onmessage = function (e) {
  const m = e.data;
  if (m.type === 'init') {
    colony = new Alive.Colony(m.cells, m.orgs); rate = m.rate || 45; rest = m.rest || 6;
    let last = 0; for (let i = 0; i < m.orgs.length; i++) last = Math.max(last, m.orgs[i].delay || 0);
    busyUntil = last + GROW; loop();
  }
  else if (!colony) return;
  else if (m.type === 'wound') { colony.wound(m.id, m.r, m.c, m.rad); wake(); }
  else if (m.type === 'pause') { paused = true; clearTimeout(timer); }
  else if (m.type === 'resume') { if (paused) { paused = false; loop(); } }
  else if (m.type === 'burst') { clearTimeout(timer); for (let i = 0; i < m.n; i++) colony.tick(); send({ burst: m.token }); if (!paused) timer = setTimeout(loop, 1000 / rate); }
};
