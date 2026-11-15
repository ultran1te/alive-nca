"""Train a neural cellular automaton to grow one shape, and test that it grows back reliably.

Every cell holds 12 numbers and runs the same small network on what it can sense of its
8 neighbours. Trained to grow the target from one seed cell, to hold it, and to grow it
back after being bitten or sliced in half. The seed cell is an anchor: it is held alive
for good, which stops the whole shape creeping across the dish over time and lets it
regrow from nothing. Pure numpy; the backward pass through time is written out by hand.

Usually run for you by `python -m alive train`. To work on one shape directly:

  python -m alive.nca gradcheck                       check the hand-written gradients on a tiny grid
  python -m alive.nca fit TARGET.json OUT.json        train, test, train more if needed, keep the best
  python -m alive.nca test TARGET.json CELLS.json     the reliability test on its own
"""
import argparse, json, os, shutil, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # one sample per process is faster than threaded maths here
import multiprocessing as mp
import numpy as np

C, HID, FIRE = 12, 96, 0.5
CFG = {'H': 0, 'W': 0, 'seed': (0, 0), 'anchor': True, 'T': None}
TOP, DROP = (1.0, 0.22, 0.14), (0.16, 0.16, 0.04)   # the colour the cells learn: red at the top of a shape, a little darker at the bottom


def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def configure(target, anchor=True):
    CFG.update(H=target['H'], W=target['W'], seed=tuple(target['seed']), anchor=anchor)
    a = np.array(target['alpha'], np.float32).reshape(CFG['H'], CFG['W'])
    rows = np.argwhere(a.max(1) > 0.05).ravel()
    g = np.clip((np.arange(CFG['H'], dtype=np.float32)[:, None] - rows.min()) / max(1, rows.max() - rows.min()), 0, 1)
    top = np.array(hex_rgb(target['color']) if target.get('color') else TOP, np.float32)
    rgb = np.clip(top - np.array(DROP, np.float32) * g[..., None], 0, 1) * np.ones((1, CFG['W'], 1), np.float32)
    CFG['T'] = np.concatenate([rgb * a[..., None], a[..., None]], -1).astype(np.float32)   # premultiplied RGBA


def _pad(x):
    p = np.zeros((x.shape[0], x.shape[1] + 2, x.shape[2] + 2, x.shape[3]), x.dtype)
    p[:, 1:-1, 1:-1] = x
    return p


def sobel_x(x):
    p = _pad(x)
    d = p[:, :, 2:] - p[:, :, :-2]
    return (d[:, :-2] + 2 * d[:, 1:-1] + d[:, 2:]) * 0.125


def sobel_y(x):
    p = _pad(x)
    d = p[:, 2:] - p[:, :-2]
    return (d[:, :, :-2] + 2 * d[:, :, 1:-1] + d[:, :, 2:]) * 0.125


def living(x):
    """A cell is alive if it or a neighbour has alpha > 0.1."""
    h, w = x.shape[1], x.shape[2]
    a = np.pad(x[..., 3], ((0, 0), (1, 1), (1, 1)), constant_values=-1.0)
    m = a[:, 1:-1, 1:-1].copy()
    for di in range(3):
        for dj in range(3):
            np.maximum(m, a[:, di:di + h, dj:dj + w], out=m)
    return (m > 0.1)[..., None]


def pin(x):
    if CFG['anchor']: x[..., CFG['seed'][0], CFG['seed'][1], 3:] = 1.0
    return x


def step(x, P, rng, keep=False):
    c = x.shape[-1]
    pre = living(x)
    rows = pre.reshape(-1)                       # dead neighbourhoods end up zero anyway, so skip them
    p = np.concatenate([x, sobel_x(x), sobel_y(x)], -1).reshape(-1, 3 * c)[rows]
    hid = np.maximum(p @ P['w1'] + P['b1'], 0)
    dx = np.zeros((rows.size, c), x.dtype)
    dx[rows] = hid @ P['w2']
    fire = (rng.random(x.shape[:3] + (1,)) < FIRE)
    x2 = x + dx.reshape(x.shape) * fire
    life = pre & living(x2)
    out = pin(x2 * life)
    return (out, (p, hid, rows, fire, life)) if keep else (out, None)


def back(g, cache, P, G):
    """g: dL/d(out of this step). Accumulates weight grads in G, returns dL/d(in)."""
    p, hid, rows, fire, life = cache
    c = g.shape[-1]
    if CFG['anchor']: g[:, CFG['seed'][0], CFG['seed'][1], 3:] = 0      # the anchor was overwritten, nothing flows through it
    g2 = g * life
    gdx = (g2 * fire).reshape(-1, c)[rows]
    G['w2'] += hid.T @ gdx
    gh = (gdx @ P['w2'].T) * (hid > 0)
    G['w1'] += p.T @ gh
    G['b1'] += gh.sum(0)
    gp = np.zeros((rows.size, 3 * c), g.dtype)
    gp[rows] = gh @ P['w1'].T
    gp = gp.reshape(g.shape[:3] + (3 * c,))
    return g2 + gp[..., :c] - sobel_x(gp[..., c:2 * c]) - sobel_y(gp[..., 2 * c:])


def seed_state(n, dtype=np.float32):
    x = np.zeros((n, CFG['H'], CFG['W'], C), dtype)
    x[:, CFG['seed'][0], CFG['seed'][1], 3:] = 1.0
    return x


def init_params(rng, c=C, hid=HID, dtype=np.float32):
    return {'w1': (rng.standard_normal((3 * c, hid)) * np.sqrt(1.0 / (3 * c))).astype(dtype),
            'b1': np.zeros(hid, dtype), 'w2': np.zeros((hid, c), dtype)}


def load(path):
    d = json.load(open(path))
    return {'w1': np.array(d['w1'], np.float32).reshape(3 * C, HID), 'b1': np.array(d['b1'], np.float32), 'w2': np.array(d['w2'], np.float32).reshape(HID, C)}, d


def loss_and_grad(x, T):
    d = x[..., :4] - T
    per = (d * d).mean((1, 2, 3))
    g = np.zeros_like(x)
    g[..., :4] = 2 * d / d[0].size / x.shape[0]
    return per, g


def gradient_check():
    rng = np.random.default_rng(1)
    h, w, c, hid = 7, 8, 5, 6
    CFG.update(H=h, W=w, seed=(3, 4), anchor=True)
    P = init_params(rng, c, hid, np.float64)
    P['w2'] = rng.standard_normal((hid, c)) * 0.3
    T = rng.random((h, w, 4))
    x0 = rng.random((2, h, w, c))

    def run(Pq, grads):
        r = np.random.default_rng(7)
        x, caches = x0.copy(), []
        for _ in range(4):
            x, cch = step(x, Pq, r, keep=True); caches.append(cch)
        per, g = loss_and_grad(x, T)
        if grads:
            G = {k: np.zeros_like(v) for k, v in Pq.items()}
            for cch in reversed(caches): g = back(g, cch, Pq, G)
            return per.mean(), G
        return per.mean(), None

    _, G = run(P, True)
    worst = 0.0
    for k in P:
        for _ in range(12):
            idx = tuple(rng.integers(0, s) for s in P[k].shape)
            old = P[k][idx]; eps = 1e-6
            P[k][idx] = old + eps; lp, _ = run(P, False)
            P[k][idx] = old - eps; lm, _ = run(P, False)
            P[k][idx] = old
            num = (lp - lm) / (2 * eps)
            err = abs(num - G[k][idx]) / max(1e-9, abs(num) + abs(G[k][idx]))
            worst = max(worst, err if abs(num) > 1e-9 else 0)
    print('gradient check, worst relative error: %.2e' % worst)
    return worst < 1e-4


def ascii_art(a):
    chars = ' .:-=+*#%@'
    return '\n'.join(''.join(chars[int(np.clip(v, 0, 0.999) * 10)] for v in row) for row in a[::2])


def save(P, path, meta):
    out = {'H': CFG['H'], 'W': CFG['W'], 'C': C, 'HID': HID, 'seed': list(CFG['seed']), 'fire': FIRE, 'anchor': CFG['anchor'], **meta,
           'w1': np.round(P['w1'], 5).ravel().tolist(), 'b1': np.round(P['b1'], 5).tolist(), 'w2': np.round(P['w2'], 5).ravel().tolist()}
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(out, f, separators=(',', ':'))
    os.replace(tmp, path)


def _worker(target, anchor):
    configure(target, anchor)


def rollout(job):
    """One sample: run it forward, score it, and send the gradients back."""
    P, x, n, seed, scale = job
    rng = np.random.default_rng(seed)
    x = x[None]; caches = []
    for _ in range(n):
        x, cch = step(x, P, rng, keep=True); caches.append(cch)
    per, g = loss_and_grad(x, CFG['T'])
    g *= scale
    G = {k: np.zeros_like(v) for k, v in P.items()}
    for cch in reversed(caches): g = back(g, cch, P, G)
    return G, x[0], float(per[0])


def wound(x, kind, rng, yy, xx):
    """Damage in the proportions of the frame: a bite, a big bite, or a clean slice with one side thrown away."""
    H, W = CFG['H'], CFG['W']
    if kind < 2:
        cy, cx, rad = rng.uniform(0.25, 0.75) * H, rng.uniform(0.12, 0.88) * W, rng.uniform(0.1, 0.27) * H
        x[(yy - cy) ** 2 + (xx - cx) ** 2 < rad * rad] = 0
    elif kind == 2:
        cy, cx, rad = rng.uniform(0.3, 0.7) * H, rng.uniform(0.18, 0.82) * W, rng.uniform(0.25, 0.42) * H
        x[(yy - cy) ** 2 + (xx - cx) ** 2 < rad * rad] = 0
    elif rng.random() < 0.65:
        c = rng.uniform(0.25, 0.75) * W; x[(xx < c) if rng.random() < 0.5 else (xx > c)] = 0
    else:
        r = rng.uniform(0.38, 0.62) * H; x[(yy < r) if rng.random() < 0.5 else (yy > r)] = 0


def default_span(target):
    """Steps per training rollout: long enough for growth to reach the far side of the shape."""
    far = target.get('far', 30)
    return max(32, int(far * 2.0)), max(48, int(far * 3.0))


def train(target, steps, out, batch=8, pool_n=256, resume=None, span=None, anchor=True, quiet=False):
    configure(target, anchor)
    span = span or default_span(target)
    H, W, T = CFG['H'], CFG['W'], CFG['T']
    rng = np.random.default_rng(0)
    P = init_params(rng)
    if resume and os.path.exists(resume):
        P, _ = load(resume)
        print('starting from', resume)
    M = {k: np.zeros_like(v) for k, v in P.items()}; V = {k: np.zeros_like(v) for k, v in P.items()}
    pool = seed_state(pool_n)
    yy, xx = np.mgrid[0:H, 0:W]
    workers = mp.Pool(batch, initializer=_worker, initargs=(target, anchor))
    t0 = time.time(); hist = []
    for it in range(1, steps + 1):
        f = it / steps
        lr = (1.5e-3 if f < 0.45 else 7e-4 if f < 0.8 else 3e-4) if resume else (2e-3 if f < 0.4 else 1e-3 if f < 0.75 else 4e-4)
        idx = rng.choice(pool_n, batch, replace=False)
        x = pool[idx].copy()
        per, _ = loss_and_grad(x, T)
        order = np.argsort(-per); idx = idx[order]; x = x[order]
        if it % 2: x[0] = seed_state(1)[0]           # every other round the worst one starts over; otherwise it has to recover
        if it > (40 if resume else 300):             # and the best ones get wounded
            for j, k in enumerate(range(batch - 4, batch)): wound(x[k], j, rng, yy, xx)
        pin(x)
        n = int(rng.integers(span[0], span[1] + 1))
        res = workers.map(rollout, [(P, x[k], n, int(rng.integers(1 << 30)), 1.0 / batch) for k in range(batch)])
        G = {k: sum(r[0][k] for r in res) for k in P}
        xs = np.stack([r[1] for r in res]); loss = float(np.mean([r[2] for r in res]))
        if not all(np.isfinite(v).all() for v in G.values()) or not np.isfinite(xs).all():
            print('step %d: non-finite, resetting pool' % it); pool = seed_state(pool_n); continue
        for k in P:
            gk = G[k] / (np.linalg.norm(G[k]) + 1e-8)
            M[k] = 0.9 * M[k] + 0.1 * gk; V[k] = 0.999 * V[k] + 0.001 * gk * gk
            P[k] -= (lr * (M[k] / (1 - 0.9 ** it)) / (np.sqrt(V[k] / (1 - 0.999 ** it)) + 1e-8)).astype(P[k].dtype)
        pool[idx] = xs
        hist.append(loss)
        if it % 50 == 0:
            print('step %5d/%d  loss %.5f  %.2fs/step' % (it, steps, np.mean(hist[-50:]), (time.time() - t0) / it), flush=True)
        if it % 250 == 0 or it == steps:
            save(P, out, {'name': target.get('name', ''), 'trained_steps': it, 'loss': float(np.mean(hist[-50:]))})
            if quiet: continue
            r = np.random.default_rng(5); z = seed_state(1)
            for _ in range(250): z, _ = step(z, P, r)
            grown = float(((z[0, ..., :4] - T) ** 2).mean())
            z[0][xx > W * 0.5] = 0; pin(z)
            for _ in range(250): z, _ = step(z, P, r)
            healed = float(((z[0, ..., :4] - T) ** 2).mean())
            print('  from seed: %.5f | right half sliced off, then 250 steps: %.5f' % (grown, healed))
            print(ascii_art(z[0, ..., 3]), flush=True)
    workers.close()
    return P


# ---------- does it actually grow back? ----------
# Loss is a poor judge: the last checkpoint is not always the best. So each shape is grown many
# times from scratch and after each kind of damage, and counted: how many runs end up whole.
SCENARIOS = ['grow from seed', 'lose left half', 'lose right half', 'wiped out', 'cut while growing']


def reliability(P, target, runs=24, n=450, seed=0):
    """Out of `runs`, how many end up with between 93% and 108% of the cells the shape should have."""
    configure(target, anchor=True)
    H, W = CFG['H'], CFG['W']
    expect = target.get('solid') or int((np.array(target['alpha']) > 0.5).sum())
    lo, hi = 0.93 * expect, 1.08 * expect
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    def run(x, k):
        for _ in range(k): x, _ = step(x, P, rng)
        return x

    def whole(x):
        cnt = (x[..., 3] > 0.5).sum((1, 2))
        return int(((cnt >= lo) & (cnt <= hi)).sum())

    grown = run(seed_state(runs), n)
    left = grown.copy(); left[:, xx < W / 2] = 0
    right = grown.copy(); right[:, xx >= W / 2] = 0
    gone = np.zeros_like(grown)
    young = run(seed_state(runs), 60)
    for i in range(runs): young[i][(xx < W / 2) if i % 2 else (yy < H / 2)] = 0
    out = {'grow from seed': whole(grown)}
    for name, x in (('lose left half', left), ('lose right half', right), ('wiped out', gone), ('cut while growing', young)):
        out[name] = whole(run(pin(x), n))
    return out


def score(rel):
    return min(rel.values()), sum(rel.values())


def fit(target_path, out, steps=3000, more=1500, rounds=3, need=23, runs=24, batch=8, warm=None):
    """Train; test; while any kind of damage leaves fewer than `need` of `runs` whole, train more.
    Keeps whichever checkpoint tested best, and writes a report next to the cells."""
    target = json.load(open(target_path))
    work = out + '.work.json'
    best, report = None, []
    for rnd in range(rounds):
        resume = work if rnd else warm
        k = steps if rnd == 0 else more
        print('%s: round %d, %d steps%s' % (target.get('name'), rnd + 1, k, ' from ' + resume if resume else ''), flush=True)
        P = train(target, k, work, batch=batch, resume=resume)
        rel = reliability(P, target, runs=runs)
        report.append({'round': rnd + 1, 'steps': k, **rel})
        print('test, whole out of %d: %s' % (runs, ', '.join('%s %d' % kv for kv in rel.items())), flush=True)
        if best is None or score(rel) > best[0]:
            best = (score(rel), rel); shutil.copyfile(work, out)
        if best[0][0] >= need: break
    d = json.load(open(out)); d['test'] = {'runs': runs, **best[1]}
    with open(out, 'w') as f: json.dump(d, f, separators=(',', ':'))
    if os.path.exists(work): os.remove(work)
    ok = best[0][0] >= need
    print('%s: %s (worst case %d of %d whole)' % (target.get('name'), 'reliable' if ok else 'NOT fully reliable', best[0][0], runs), flush=True)
    return ok, best[1]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('gradcheck')
    f = sub.add_parser('fit', help='train one shape until it grows back reliably')
    f.add_argument('target'); f.add_argument('out')
    f.add_argument('--steps', type=int, default=3000); f.add_argument('--more', type=int, default=1500)
    f.add_argument('--rounds', type=int, default=3); f.add_argument('--batch', type=int, default=8)
    f.add_argument('--warm', default=None, help='start from these trained cells instead of from scratch')
    t = sub.add_parser('test', help='the reliability test for trained cells')
    t.add_argument('target'); t.add_argument('cells'); t.add_argument('--runs', type=int, default=24)
    a = ap.parse_args()
    if a.cmd == 'gradcheck':
        sys.exit(0 if gradient_check() else 1)
    if a.cmd == 'fit':
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        ok, _ = fit(a.target, a.out, steps=a.steps, more=a.more, rounds=a.rounds, batch=a.batch, warm=a.warm)
        sys.exit(0 if ok else 3)
    if a.cmd == 'test':
        P, _ = load(a.cells)
        rel = reliability(P, json.load(open(a.target)), runs=a.runs)
        print(', '.join('%s %d/%d' % (k, v, a.runs) for k, v in rel.items()))
