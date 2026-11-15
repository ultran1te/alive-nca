// Part of alive-nca, MIT license: https://github.com/ultran1te/alive-nca
/* alive-dish.js
 * Draws organisms (see alive-core.js) on a full-window WebGL 2 canvas, as wet tissue on a dotted
 * dish, and lets people cut them with a mouse or their fingers. Everything grows back.
 *
 * Each organism lives in its own small dish inside a worker (alive-worker.js); the page only says
 * where that dish is drawn. So a resize, or a phone turned on its side, moves and rescales everything
 * without killing anything.
 *
 *   <script src="alive-core.js"></script>
 *   <script src="alive-dish.js"></script>
 *   const dish = AliveDish.start({
 *     canvas,              a <canvas> covering the window
 *     cells,               { name: trained weights }   (from `python -m alive train`)
 *     arrange,             ({ W, H }) => ({ layers, orgs }); called at start and after every resize
 *     handles,             optional: other elements to cut through when dragged across (a link over an organism)
 *     colors,              optional: { paper, dots, tissue, nucleus, edge } as '#rrggbb' or [r, g, b] in 0..1,
 *                          learn: how much of the colour the cells learned shows through (default 0.45),
 *                          dotsAlpha: how strongly the dots show on a transparent dish (default 0.5; 0 hides them)
 *     transparent,         optional: true to draw only the tissue, its shadow and the dots, so whatever is behind
 *                          the canvas shows through (the paper colour is then not used)
 *     worker,              optional: URL of alive-worker.js (default 'alive-worker.js')
 *     onFail(error),       optional: called if the GPU is taken away and cannot be got back
 *   });
 *
 * arrange returns:
 *   layers  [{ s, oy }]    lattices. s: size of one cell in px. oy: top of the lattice in px.
 *                          Layer 0 is drawn over the whole window. Later layers are blended on top, from oy
 *                          down; use one for things that should stay small while layer 0 grows with the window.
 *   orgs    [{ name, cells, layer, r0, c0, delay, colony, expect }]   in the same order every time
 *                          r0, c0: where the organism's dish starts, in its layer's cells.
 *                          cells: which trained weights to use (defaults to name).
 *                          delay: steps to wait before seeding it. colony: organisms with the same colony
 *                          share a worker. expect: solid cells when whole, for the health check (0 = none).
 *
 * start() throws if WebGL 2 is not available, so a page can show something plain instead.
 * It returns { orgs, layers, arrange(), advance(n), status(), cut(a, b), colors(c) } for pages and tests;
 * cut takes two points on the page in px. colors(c) changes the palette while it runs (same keys as above).
 */
(function (root) {
  'use strict';

  const VS = '#version 300 es\nvoid main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }';
  const FS = `#version 300 es
precision highp float;
uniform sampler2D uS; uniform sampler2D uHd; uniform sampler2D uM; uniform vec2 uGrid; uniform vec2 uMask; uniform vec2 uRes; uniform vec2 uOrigin; uniform float uCell; uniform float uDot; uniform float uCalm; uniform float uOverlay;
uniform vec3 uPaper; uniform vec3 uDots; uniform vec3 uTissue; uniform vec3 uNucleus; uniform vec3 uEdge; uniform float uLearn; uniform float uDotA;
out vec4 o;
vec4 sAt(ivec2 p) {
  if (p.x < 0 || p.y < 0 || p.x >= int(uGrid.x) || p.y >= int(uGrid.y)) return vec4(0.0);
  return texelFetch(uS, p, 0);
}
vec4 bs(float t) {
  float t2 = t * t, t3 = t2 * t;
  return vec4(1.0 - 3.0 * t + 3.0 * t2 - t3, 4.0 - 6.0 * t2 + 3.0 * t3, 1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3, t3) / 6.0;
}
vec4 smoothState(vec2 g) {
  g -= 0.5; vec2 f = fract(g); ivec2 i = ivec2(floor(g));
  vec4 wx = bs(f.x), wy = bs(f.y), s = vec4(0.0);
  for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) s += wx[x] * wy[y] * sAt(i + ivec2(x - 1, y - 1));
  return s;
}
void main() {
  // uOverlay 0: the main lattice, drawn with the dish under it. 1: drawn without paper, premultiplied: a finer
  // lattice blended on top, or the main one on a transparent dish (then uDotA > 0 keeps the dots).
  vec2 g = (vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y) - uOrigin) / uCell;
  // the dish: paper, with a faint dot lattice
  float ld = length(fract(g / uDot) - 0.5) * uDot, px = 1.0 / uCell;
  float dotc = 1.0 - smoothstep(0.75 * px, 1.75 * px, ld);
  vec3 col = mix(uPaper, uDots, dotc);
  vec4 under = vec4(uDots, 1.0) * dotc * uDotA;
  // Most of the window is bare dish, so skip the tissue there. Decided per 2x2 block of pixels,
  // so every pixel in a block takes the same branch and the edge smoothing below stays valid.
  vec2 q = floor(gl_FragCoord.xy * 0.5) * 2.0;
  ivec2 cb = ivec2(floor((vec2(q.x, uRes.y - q.y) - uOrigin) / uCell)), mb = cb >> 2;
  bool near = cb.x >= 0 && cb.y >= 0 && mb.x < int(uMask.x) && mb.y < int(uMask.y) && texelFetch(uM, mb, 0).r > 0.5;
  if (!near) { o = uOverlay > 0.5 ? under : vec4(col, 1.0); return; }
  vec4 s = smoothState(g);
  float a = s.a, aa = fwidth(a) * 0.9 + 1e-4;
  float body = smoothstep(0.42 - aa, 0.42 + aa, a);
  float shadow = 0.16 * smoothstep(0.12, 0.7, smoothState(g - vec2(0.6, 1.0)).a);
  col *= 1.0 - shadow * (1.0 - body);
  // the tissue: the colour the cells learned, pulled toward one colour, with a nucleus in every cell
  vec3 learned = clamp(s.rgb / max(a, 0.25), 0.0, 1.0);
  vec3 tissue = mix(uTissue, learned, uLearn);
  ivec2 ci = ivec2(floor(g));
  float act = mix(abs(texelFetch(uHd, clamp(ci, ivec2(0), ivec2(uGrid) - 1), 0).r * 2.0 - 1.0), 0.45, uCalm);
  float dd = length(fract(g) - 0.5);
  float nucleus = (1.0 - smoothstep(0.15, 0.32, dd)) * smoothstep(0.35, 0.8, sAt(ci).a);
  tissue = mix(tissue, uNucleus, nucleus * (0.2 + 0.6 * act));
  tissue = mix(tissue, uEdge, (1.0 - smoothstep(0.42, 0.66, a)) * 0.6);
  // without paper: tissue where there is tissue, a soft shadow around it, the dots (if any) elsewhere (premultiplied)
  float sh = shadow * (1.0 - body);
  under = vec4(under.rgb * (1.0 - sh), sh + under.a * (1.0 - sh));
  o = uOverlay > 0.5 ? vec4(tissue * body, body) + under * (1.0 - body) : vec4(mix(col, tissue, body), 1.0);
}`;

  // '#rrggbb' or [r, g, b] in 0..1
  function rgb(v) {
    if (typeof v !== 'string') return v;
    const n = parseInt(v.replace('#', ''), 16);
    return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
  }
  // The defaults are mowalker.ai's: cool white paper, one signal red.
  function palette(c) {
    c = c || {};
    const p = {
      paper: c.paper ? rgb(c.paper) : [0.957, 0.961, 0.965],
      tissue: c.tissue ? rgb(c.tissue) : [1.0, 0.165, 0.07],
      learn: c.learn != null ? c.learn : 0.45,
    };
    const lum = 0.2126 * p.paper[0] + 0.7152 * p.paper[1] + 0.0722 * p.paper[2];
    p.dots = c.dots ? rgb(c.dots) : c.paper ? p.paper.map(function (v) { return lum > 0.5 ? v * 0.85 : v + (1 - v) * 0.18; }) : [0.80, 0.81, 0.84];
    p.nucleus = c.nucleus ? rgb(c.nucleus) : c.tissue ? p.tissue.map(function (v) { return 0.52 * v * v; }) : [0.52, 0.02, 0.02];
    p.edge = c.edge ? rgb(c.edge) : c.tissue ? p.tissue.map(function (v) { return 0.6 * Math.pow(v, 1.8); }) : [0.6, 0.04, 0.03];
    p.dotsAlpha = c.dotsAlpha != null ? c.dotsAlpha : 0.5;
    return p;
  }

  function start(opts) {
    const cv = opts.canvas, cellsFor = opts.cells, handles = opts.handles || [], workerUrl = opts.worker || 'alive-worker.js';
    const calm = opts.calm != null ? opts.calm : matchMedia('(prefers-reduced-motion: reduce)').matches;
    const clear = !!opts.transparent;
    let pal = palette(opts.colors);
    const fail = opts.onFail || function (e) { console.error(e); };
    let gl = null, U = {}, lost = false, layers = [], orgs = [], colonies = [], fresh = true, seen = '';

    // Builds everything the GPU holds. Runs at start, and again if a phone takes the canvas away
    // (iOS drops WebGL when you switch apps for a while) and later gives it back.
    function initGL() {
      gl = cv.getContext('webgl2', { antialias: false, alpha: clear, premultipliedAlpha: true });
      if (!gl) throw new Error('WebGL 2 is not available');
      const sh = function (type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(prog); gl.useProgram(prog);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      ['uS', 'uHd', 'uM'].forEach(function (name, unit) { gl.uniform1i(gl.getUniformLocation(prog, name), unit); });
      U = {}; ['uGrid', 'uMask', 'uRes', 'uOrigin', 'uCell', 'uDot', 'uCalm', 'uOverlay', 'uDotA', 'uPaper', 'uDots', 'uTissue', 'uNucleus', 'uEdge', 'uLearn'].forEach(function (n) { U[n] = gl.getUniformLocation(prog, n); });
      setColors();
      layers.forEach(function (L) { L.tex = null; });   // textures are made again on the next draw
      fresh = true;
    }
    function setColors() {
      [['uPaper', pal.paper], ['uDots', pal.dots], ['uTissue', pal.tissue], ['uNucleus', pal.nucleus], ['uEdge', pal.edge]].forEach(function (u) { gl.uniform3fv(U[u[0]], u[1]); });
      gl.uniform1f(U.uLearn, pal.learn);
      fresh = true;
    }
    cv.addEventListener('webglcontextlost', function (e) { e.preventDefault(); lost = true; });
    cv.addEventListener('webglcontextrestored', function () { try { initGL(); lost = false; } catch (e) { fail(e); } });

    // ---------- lattices: each organism in its own patch of one ----------
    function sizeLayer(L, s, oy, w, h) {
      L.s = s; L.oy = oy; L.cols = Math.ceil(w / s); L.rows = Math.max(1, Math.ceil(h / s)); L.mc = Math.ceil(L.cols / 4); L.mr = Math.ceil(L.rows / 4);
      L.S = new Uint8Array(L.rows * L.cols * 4); L.Hd = new Uint8Array(L.rows * L.cols * 4); L.M = new Uint8Array(L.mc * L.mr); L.G = new Uint8Array(L.mc * L.mr);
      L.sized = false;
    }
    function texturesFor(L) {
      if (!L.tex) {
        L.tex = [0, 1, 2].map(function () {
          const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          return t;
        });
        L.sized = false;
      }
      if (!L.sized) {
        for (let i = 0; i < 2; i++) { gl.bindTexture(gl.TEXTURE_2D, L.tex[i]); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, L.cols, L.rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, null); }
        gl.bindTexture(gl.TEXTURE_2D, L.tex[2]); gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, L.mc, L.mr, 0, gl.RED, gl.UNSIGNED_BYTE, null);
        L.sized = true;
      }
    }

    // Each colony steps its organisms about 45 times a second: in a worker where possible, on this thread otherwise.
    function startColonies() {
      colonies.forEach(function (c) { c.stop(); }); colonies = [];
      const groups = {};
      orgs.forEach(function (o) { (groups[o.colony] = groups[o.colony] || []).push({ id: o.id, name: o.cells, delay: o.delay, expect: o.expect }); });
      Object.keys(groups).forEach(function (key) {
        const list = groups[key], cells = {}; list.forEach(function (o) { cells[o.name] = cellsFor[o.name]; });
        const c = { ticks: 0, ms: 0, waiting: {}, worker: null, local: null, timer: 0 };
        const take = function (m) {
          m.orgs.forEach(function (st) { const o = orgs[st.id]; if (o && o.run === c) { o.rgba = st.rgba; o.hd = st.hd; o.living = st.living; } });
          c.ticks = m.ticks; c.ms = m.ms; c.resting = !!m.resting; c.restarts = m.restarts || 0; fresh = true;
          if (m.burst != null && c.waiting[m.burst]) { c.waiting[m.burst](); delete c.waiting[m.burst]; }
        };
        const local = function () {
          c.local = new Alive.Colony(cells, list);
          c.timer = setInterval(function () { if (!document.hidden) { c.local.tick(); take({ orgs: c.local.snapshot(), ticks: c.local.ticks, ms: c.local.ms, restarts: c.local.restarts }); } }, 1000 / 30);
        };
        try {
          c.worker = new Worker(workerUrl);
          c.worker.onmessage = function (e) { if (e.data.type === 'state') take(e.data); };
          c.worker.onerror = function (e) { console.warn('worker failed, stepping on the page instead', e.message); c.worker.terminate(); c.worker = null; local(); };
          c.worker.postMessage({ type: 'init', cells: cells, orgs: list, rate: 45 });
        } catch (e) { c.worker = null; local(); }
        c.wound = function (id, r, cc, rad) { if (c.worker) c.worker.postMessage({ type: 'wound', id: id, r: r, c: cc, rad: rad }); else if (c.local) c.local.wound(id, r, cc, rad); };
        c.burst = function (n) {
          return new Promise(function (done) {
            if (c.worker) { const token = String(Math.random()); c.waiting[token] = done; c.worker.postMessage({ type: 'burst', n: n, token: token }); }
            else { for (let i = 0; i < n; i++) c.local.tick(); take({ orgs: c.local.snapshot(), ticks: c.local.ticks, ms: c.local.ms, restarts: c.local.restarts }); done(); }
          });
        };
        c.stop = function () { if (c.worker) c.worker.terminate(); clearInterval(c.timer); };
        colonies.push(c); list.forEach(function (o) { orgs[o.id].run = c; });
      });
    }

    // Asks the page where everything goes, for the window as it is now.
    function arrange(force) {
      const W = innerWidth, H = innerHeight, key = W + 'x' + H;
      if (key === seen && !force) return;
      seen = key;
      const spec = opts.arrange({ W: W, H: H });
      spec.layers.forEach(function (ls, i) {
        const L = layers[i] || (layers[i] = { ox: 0, tex: null });
        sizeLayer(L, ls.s, ls.oy || 0, W, H - (ls.oy || 0));
      });
      layers.length = spec.layers.length;
      const want = spec.orgs.filter(function (w) { return cellsFor[w.cells || w.name]; });
      if (!orgs.length) {
        want.forEach(function (w) {
          const c = cellsFor[w.cells || w.name];
          orgs.push({ id: orgs.length, name: w.name, cells: w.cells || w.name, layer: w.layer || 0, expect: w.expect || 0, H: c.H, W: c.W, r0: Math.round(w.r0), c0: Math.round(w.c0), delay: w.delay || 0, colony: w.colony || 'all', rgba: null, hd: null, living: 0 });
        });
        startColonies();
      } else want.forEach(function (w, i) { if (orgs[i]) { orgs[i].r0 = Math.round(w.r0); orgs[i].c0 = Math.round(w.c0); orgs[i].layer = w.layer || 0; } });
      fresh = true;
    }

    function compose(L, which) {
      const S = L.S, Hd = L.Hd, M = L.M, G = L.G, cols = L.cols, rows = L.rows, mc = L.mc, mr = L.mr;
      S.fill(0); Hd.fill(127); M.fill(0);
      for (let n = 0; n < orgs.length; n++) {
        const org = orgs[n], src = org.rgba, hd = org.hd; if (!src || org.layer !== which) continue;
        for (let r = 0; r < org.H; r++) {
          const gr = org.r0 + r; if (gr < 0 || gr >= rows) continue;
          for (let c = 0; c < org.W; c++) {
            const gc = org.c0 + c; if (gc < 0 || gc >= cols) continue;
            const o = r * org.W + c, p = o * 4, q = (gr * cols + gc) * 4;
            if (src[p + 3] <= S[q + 3]) continue;
            S[q] = src[p]; S[q + 1] = src[p + 1]; S[q + 2] = src[p + 2]; S[q + 3] = src[p + 3]; Hd[q] = hd[o];
            M[(gr >> 2) * mc + (gc >> 2)] = 1;
          }
        }
      }
      // Which 4x4 blocks of the lattice have tissue within reach. Grown by one block each way,
      // because the smooth edge and the shadow reach about three cells past the tissue.
      G.fill(0);
      for (let y = 0; y < mr; y++) for (let x = 0; x < mc; x++) {
        if (!M[y * mc + x]) continue;
        for (let dy = -1; dy <= 1; dy++) { const yy = y + dy; if (yy < 0 || yy >= mr) continue; for (let dx = -1; dx <= 1; dx++) { const xx = x + dx; if (xx >= 0 && xx < mc) G[yy * mc + xx] = 255; } }
      }
    }
    function drawLayer(L, overlay, dpr) {
      texturesFor(L);
      [[L.S, gl.RGBA, L.cols, L.rows], [L.Hd, gl.RGBA, L.cols, L.rows], [L.G, gl.RED, L.mc, L.mr]].forEach(function (u, i) {
        gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, L.tex[i]); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, u[2], u[3], u[1], gl.UNSIGNED_BYTE, u[0]);
      });
      gl.uniform2f(U.uGrid, L.cols, L.rows); gl.uniform2f(U.uMask, L.mc, L.mr); gl.uniform2f(U.uOrigin, L.ox * dpr, L.oy * dpr);
      gl.uniform1f(U.uCell, L.s * dpr); gl.uniform1f(U.uOverlay, overlay); gl.uniform1f(U.uDotA, overlay && L === layers[0] ? pal.dotsAlpha : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    function draw() {
      if (lost || !gl || !layers.length) return;
      const dpr = Math.min(devicePixelRatio || 1, 2), w = Math.round(innerWidth * dpr), h = Math.round(innerHeight * dpr);
      const resized = cv.width !== w || cv.height !== h;
      if (resized) { cv.width = w; cv.height = h; }
      if (!fresh && !resized) return;
      fresh = false; layers.forEach(compose);
      gl.viewport(0, 0, w, h); gl.uniform2f(U.uRes, w, h); gl.uniform1f(U.uDot, layers[0].s < 7 ? 2 : 1); gl.uniform1f(U.uCalm, calm ? 1 : 0);
      gl.disable(gl.BLEND); gl.disable(gl.SCISSOR_TEST);
      drawLayer(layers[0], clear ? 1 : 0, dpr);   // covers every pixel, so nothing needs clearing first
      // finer lattices: blended over the part of the window they cover
      for (let i = 1; i < layers.length; i++) {
        gl.enable(gl.SCISSOR_TEST); gl.scissor(0, 0, w, Math.max(0, h - Math.floor(layers[i].oy * dpr)));
        gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        drawLayer(layers[i], 1, dpr);
      }
      gl.disable(gl.BLEND); gl.disable(gl.SCISSOR_TEST);
    }
    let need = false;
    function frame() { requestAnimationFrame(frame); if (need) { need = false; arrange(); } draw(); }

    // ---------- hands on: drag to cut anything; it grows back. Each finger cuts its own path. ----------
    const hands = new Map(); let lastWasCut = false;
    const at = function (e) { return [e.clientX, e.clientY]; };
    function cut(a, b) {   // a, b: points on the page, in px
      const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / Math.min.apply(null, layers.map(function (L) { return L.s; }))));
      for (let i = 0; i <= n; i++) {
        const x = a[0] + (b[0] - a[0]) * i / n, y = a[1] + (b[1] - a[1]) * i / n;
        for (let k = 0; k < orgs.length; k++) {
          const o = orgs[k], L = layers[o.layer], rad = Math.max(2.4, 9 / L.s), lr = (y - L.oy) / L.s - o.r0, lc = (x - L.ox) / L.s - o.c0;
          if (lr > -rad && lc > -rad && lr < o.H + rad && lc < o.W + rad) o.run.wound(o.id, lr, lc, rad);
        }
      }
    }
    function onDown(e) {
      // a fingertip wobbles more than a mouse, so a tap on a link is allowed to move a little and still open it
      hands.set(e.pointerId, { x: e.clientX, y: e.clientY, last: at(e), dragged: false, slop: e.pointerType === 'mouse' ? 4 : 12 });
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch (x) {}
    }
    function onMove(e) {
      const h = hands.get(e.pointerId); if (!h) return;
      if (!h.dragged && Math.hypot(e.clientX - h.x, e.clientY - h.y) < h.slop) return;
      h.dragged = true; const p = at(e); cut(h.last, p); h.last = p;
    }
    function onUp(e) {
      const h = hands.get(e.pointerId); if (!h) return;
      hands.delete(e.pointerId); lastWasCut = h.dragged;
      if (!h.dragged && e.currentTarget === cv) { const p = at(e); cut(p, p); }
    }
    [cv].concat(handles).forEach(function (el) { el.addEventListener('pointerdown', onDown); el.addEventListener('pointermove', onMove); el.addEventListener('pointerup', onUp); el.addEventListener('pointercancel', function (e) { hands.delete(e.pointerId); }); });
    handles.forEach(function (el) {
      el.addEventListener('click', function (e) { if (lastWasCut) e.preventDefault(); lastWasCut = false; });   // a drag across a link is a cut, not a visit
      el.addEventListener('dragstart', function (e) { e.preventDefault(); });
    });

    document.addEventListener('visibilitychange', function () { colonies.forEach(function (c) { if (c.worker) c.worker.postMessage({ type: document.hidden ? 'pause' : 'resume' }); }); });
    // Phones report their new size a beat after turning, so look again shortly after.
    addEventListener('resize', function () { need = true; setTimeout(function () { need = true; }, 350); });

    initGL();
    arrange(true); draw();
    requestAnimationFrame(frame);

    function status() {
      return { workers: colonies.filter(function (c) { return c.worker; }).length, ticks: colonies.map(function (c) { return c.ticks; }), ms: colonies.map(function (c) { return +c.ms.toFixed(1); }), resting: colonies.map(function (c) { return !!c.resting; }), restarts: colonies.map(function (c) { return c.restarts || 0; }), living: orgs.map(function (o) { return o.living; }) };
    }
    return {
      get orgs() { return orgs; }, get layers() { return layers; },
      arrange: function () { arrange(true); draw(); },
      advance: async function (n) { await Promise.all(colonies.map(function (c) { return c.burst(n); })); fresh = true; draw(); return status(); },   // for tests: run n steps now
      status: status,
      cut: function (a, b) { cut(a, b); },
      colors: function (c) { pal = palette(c); if (gl && !lost) setColors(); },
    };
  }

  root.AliveDish = { start: start, palette: palette };
})(typeof self !== 'undefined' ? self : this);
