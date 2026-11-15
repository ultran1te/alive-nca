// Part of alive-nca, MIT license: https://github.com/ultran1te/alive-nca
/* alive-core.js
 * Runs the neural cellular automaton trained by alive/nca.py.
 * Every cell holds C numbers. Each step, a cell looks at itself and its 8 neighbours
 * (its own values plus two gradients), runs one small network, and nudges its values.
 * The same network runs in every cell; there is no global plan.
 * Each seed gets a dish the size of the one the cells were trained in; nothing lives outside a dish.
 * Cells trained with an anchor keep their seed cell alive for good: it stops the shape creeping
 * over time and lets it grow back from nothing.
 * No dependencies. window.Alive in the browser, module.exports in Node.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Alive = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function NCA(wts, rows, cols) {
    const C = this.C = wts.C, HID = this.HID = wts.HID;
    this.rows = rows; this.cols = cols; this.fire = wts.fire == null ? 0.5 : wts.fire;
    // stored as [3C][HID] and [HID][C]; regroup w1 so each hidden unit's weights sit together
    this.w1 = new Float32Array(HID * 3 * C);
    for (let i = 0; i < 3 * C; i++) for (let j = 0; j < HID; j++) this.w1[j * 3 * C + i] = wts.w1[i * HID + j];
    this.b1 = Float32Array.from(wts.b1); this.w2 = Float32Array.from(wts.w2);
    this.x = new Float32Array(rows * cols * C); this.y = new Float32Array(rows * cols * C);
    this.pre = new Uint8Array(rows * cols); this.post = new Uint8Array(rows * cols);
    this.p = new Float32Array(3 * C); this.dx = new Float32Array(C);
    this.dish = new Uint8Array(rows * cols);
    this.anchored = !!wts.anchor; this.pins = [];
    this.frame = { h: wts.H || rows, w: wts.W || cols, r: wts.seed ? wts.seed[0] : rows >> 1, c: wts.seed ? wts.seed[1] : cols >> 1 };
    this.age = 0; this.living = 0;
  }
  NCA.prototype.clear = function () { this.x.fill(0); this.dish.fill(0); this.pins.length = 0; this.age = 0; this.living = 0; };
  // start over from the anchor cell(s): everything dies, the seed stays
  NCA.prototype.restart = function () { this.x.fill(0); this._pin(this.x); this.age = 0; this.living = 0; };
  NCA.prototype._pin = function (a) {
    for (let i = 0; i < this.pins.length; i++) { const o = this.pins[i] * this.C; for (let k = 3; k < this.C; k++) a[o + k] = 1; }
  };
  NCA.prototype.seed = function (r, c) {
    if (r < 0 || c < 0 || r >= this.rows || c >= this.cols) return;
    const o = (r * this.cols + c) * this.C; for (let k = 3; k < this.C; k++) this.x[o + k] = 1;
    if (this.anchored) this.pins.push(r * this.cols + c);
    const f = this.frame, r0 = Math.max(0, r - f.r), r1 = Math.min(this.rows, r - f.r + f.h), c0 = Math.max(0, c - f.c), c1 = Math.min(this.cols, c - f.c + f.w);
    for (let i = r0; i < r1; i++) this.dish.fill(1, i * this.cols + c0, i * this.cols + c1);
  };
  NCA.prototype.wound = function (r, c, rad) {
    const C = this.C, r0 = Math.max(0, Math.floor(r - rad)), r1 = Math.min(this.rows - 1, Math.ceil(r + rad));
    const c0 = Math.max(0, Math.floor(c - rad)), c1 = Math.min(this.cols - 1, Math.ceil(c + rad));
    for (let i = r0; i <= r1; i++) for (let j = c0; j <= c1; j++) {
      if ((i - r) * (i - r) + (j - c) * (j - c) > rad * rad) continue;
      const o = (i * this.cols + j) * C; for (let k = 0; k < C; k++) this.x[o + k] = 0;
    }
    if (this.pins.length) this._pin(this.x);
  };
  NCA.prototype.alphaAt = function (r, c) { return (r < 0 || c < 0 || r >= this.rows || c >= this.cols) ? 0 : this.x[(r * this.cols + c) * this.C + 3]; };
  // a cell is "living" if it or any neighbour has alpha above 0.1
  NCA.prototype._life = function (src, out) {
    const R = this.rows, K = this.cols, C = this.C; let n = 0;
    for (let r = 0; r < R; r++) for (let c = 0; c < K; c++) {
      let on = 0;
      for (let dr = -1; dr <= 1 && !on; dr++) { const rr = r + dr; if (rr < 0 || rr >= R) continue; for (let dc = -1; dc <= 1; dc++) { const cc = c + dc; if (cc < 0 || cc >= K) continue; if (src[(rr * K + cc) * C + 3] > 0.1) { on = 1; break; } } }
      out[r * K + c] = on & this.dish[r * K + c]; n += on;
    }
    return n;
  };
  // rnd: a function returning 0..1 (pass () => 0 to make every cell update, for testing)
  NCA.prototype.step = function (rnd) {
    const R = this.rows, K = this.cols, C = this.C, HID = this.HID, C3 = 3 * C;
    const x = this.x, y = this.y, p = this.p, dx = this.dx, w1 = this.w1, b1 = this.b1, w2 = this.w2, fire = this.fire;
    rnd = rnd || Math.random;
    this._life(x, this.pre);
    for (let r = 0; r < R; r++) for (let c = 0; c < K; c++) {
      const o = (r * K + c) * C;
      if (!this.pre[r * K + c]) { for (let k = 0; k < C; k++) y[o + k] = 0; continue; }
      if (rnd() >= fire) { for (let k = 0; k < C; k++) y[o + k] = x[o + k]; continue; }
      const up = r > 0, dn = r < R - 1, lf = c > 0, rt = c < K - 1, row = K * C;
      for (let k = 0; k < C; k++) {
        const a = up && lf ? x[o - row - C + k] : 0, b = up ? x[o - row + k] : 0, cc = up && rt ? x[o - row + C + k] : 0;
        const d = lf ? x[o - C + k] : 0, f = rt ? x[o + C + k] : 0;
        const g = dn && lf ? x[o + row - C + k] : 0, h = dn ? x[o + row + k] : 0, i = dn && rt ? x[o + row + C + k] : 0;
        p[k] = x[o + k];
        p[C + k] = (cc + 2 * f + i - a - 2 * d - g) * 0.125;          // left-right gradient
        p[2 * C + k] = (g + 2 * h + i - a - 2 * b - cc) * 0.125;      // up-down gradient
      }
      dx.fill(0);
      for (let j = 0; j < HID; j++) {
        let s = b1[j]; const wo = j * C3;
        for (let q = 0; q < C3; q++) s += w1[wo + q] * p[q];
        if (s > 0) { const vo = j * C; for (let k = 0; k < C; k++) dx[k] += s * w2[vo + k]; }
      }
      for (let k = 0; k < C; k++) y[o + k] = x[o + k] + dx[k];
    }
    this._life(y, this.post);
    let living = 0;
    for (let q = 0; q < R * K; q++) {
      if (!(this.pre[q] && this.post[q])) { const o = q * C; for (let k = 0; k < C; k++) y[o + k] = 0; }
      else if (y[q * C + 3] > 0.5) living++;
    }
    if (this.pins.length) this._pin(y);
    this.x = y; this.y = x; this.age++; this.living = living;
    return living;
  };

  // A colony is a handful of organisms stepped together. The site runs several colonies,
  // each in its own worker, and only ever sees the bytes it needs to draw them.
  // Health check: each organism knows roughly how many cells it should have when whole.
  // One that stays well off that number (a letter that withered, a growth that ran wild) for
  // about ten seconds is restarted from its anchor, so the page always mends itself.
  const SETTLE = 420, PATIENCE = 450;
  function Colony(cells, list) {
    this.ticks = 0; this.ms = 0; this.healthy = false; this.restarts = 0;
    this.orgs = list.map(function (o) {
      const w = cells[o.name];
      return { id: o.id, delay: o.delay || 0, seed: w.seed, expect: o.expect || 0, off: 0, nca: new NCA(w, w.H, w.W), begun: false };
    });
  }
  Colony.prototype.tick = function () {
    const t = typeof performance !== 'undefined' ? performance.now() : 0;
    let healthy = true;
    for (let i = 0; i < this.orgs.length; i++) {
      const o = this.orgs[i];
      if (!o.begun) { healthy = false; if (this.ticks < o.delay) continue; o.nca.seed(o.seed[0], o.seed[1]); o.begun = true; }
      const n = o.nca; n.step();
      if (!o.expect) continue;
      const whole = n.living > 0.9 * o.expect && n.living < 1.12 * o.expect;
      if (whole) o.off = 0;
      else {
        healthy = false;
        if (n.age > SETTLE && ++o.off > PATIENCE && n.anchored) { n.restart(); o.off = 0; this.restarts++; }
      }
    }
    this.healthy = healthy; this.ticks++;
    if (typeof performance !== 'undefined') this.ms = this.ms * 0.9 + (performance.now() - t) * 0.1;
  };
  Colony.prototype.wound = function (id, r, c, rad) {
    for (let i = 0; i < this.orgs.length; i++) if (this.orgs[i].id === id && this.orgs[i].begun) this.orgs[i].nca.wound(r, c, rad);
  };
  // colour and alpha as bytes, plus one hidden channel for the flicker inside each cell
  Colony.prototype.snapshot = function () {
    return this.orgs.map(function (o) {
      const n = o.nca, C = n.C, x = n.x, N = n.rows * n.cols, rgba = new Uint8Array(N * 4), hd = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        const q = i * C, p = i * 4;
        rgba[p] = x[q] <= 0 ? 0 : x[q] >= 1 ? 255 : x[q] * 255; rgba[p + 1] = x[q + 1] <= 0 ? 0 : x[q + 1] >= 1 ? 255 : x[q + 1] * 255;
        rgba[p + 2] = x[q + 2] <= 0 ? 0 : x[q + 2] >= 1 ? 255 : x[q + 2] * 255; rgba[p + 3] = x[q + 3] <= 0 ? 0 : x[q + 3] >= 1 ? 255 : x[q + 3] * 255;
        hd[i] = 127.5 + 127.5 * Math.tanh(x[q + 5]);
      }
      return { id: o.id, rgba: rgba, hd: hd, living: n.living, age: n.age };
    });
  };

  return { NCA: NCA, Colony: Colony };
});
