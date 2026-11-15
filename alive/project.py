"""A project folder holds one living page:

  alive.json      what to grow: text, font, colours, marks (edit it, then run draw / train / build again)
  marks/          copies of the images used for marks
  targets/        the shapes, one per organism (from draw)
  layout.json     where every organism sits (from draw)
  preview.png     the organisms, each in its own shade, with its seed
  cells/          the trained organisms (from train)
  site/           the finished page (from build): upload this folder anywhere that serves static files
"""
import hashlib, html, json, os, re, shutil, subprocess, sys, time
import numpy as np
from . import shapes

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, 'web')
RUNTIME = ['alive-core.js', 'alive-worker.js', 'alive-dish.js']
DEFAULTS = {'text': '', 'title': '', 'font': None, 'size': 20, 'spacing': 1.0, 'weight': None,
            'color': '#ff2a12', 'paper': '#f4f5f6', 'marks': []}
TOO_BIG = 64        # cells: beyond this, one seed struggles to organise a piece


def config(folder):
    path = os.path.join(folder, 'alive.json')
    if not os.path.exists(path): raise SystemExit('%s has no alive.json; start one with: python -m alive new %s "your text"' % (folder, folder))
    return {**DEFAULTS, **json.load(open(path, encoding='utf-8'))}


def save_config(folder, cfg):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'alive.json'), 'w', encoding='utf-8') as f: json.dump(cfg, f, indent=2, ensure_ascii=False)


def _label(ch):
    return ch if ch.isascii() and ch.isalnum() else 'u%04x' % ord(ch)


def _key(label, t):
    """Organisms with the same shape, seed and colour share one trained file."""
    h = hashlib.sha1(json.dumps([t['H'], t['W'], t['seed'], t.get('color'), t['alpha']]).encode()).hexdigest()[:8]
    return '%s-%s' % (label, h)


def draw(folder, log=print):
    """Turn alive.json into targets/, layout.json and preview.png."""
    cfg = config(folder)
    tdir = os.path.join(folder, 'targets'); os.makedirs(tdir, exist_ok=True)
    pieces, targets, warn, sheet = [], {}, [], []

    def add(label, frame, r, c, group, extra):
        t = shapes.make_target(label, frame, color=cfg['color'])
        key = _key(label, t); targets[key] = t
        pieces.append({'name': '%s%d' % (group[0], len(pieces)), 'cells': key, 'group': group, 'x': int(c), 'y': int(r),
                       'H': t['H'], 'W': t['W'], 'seed': t['seed'], 'solid': t['solid'], **extra})
        sheet.append((frame, r, c, t['seed'], group, extra.get('mark', 0)))
        if max(t['H'], t['W']) - 2 * shapes.MARGIN > TOO_BIG:
            warn.append('%s is %dx%d cells; one seed organises about 50 across, so it may not hold together. Try a smaller size, or a font whose letters do not join.' % (label, t['W'], t['H']))

    text = None
    if cfg['text'].strip():
        glyphs, stroke = shapes.render_text(cfg['text'], cfg['font'], cfg['size'], cfg['spacing'], cfg['weight'])
        B, br, bc = shapes.board([(a, r, c) for _, a, r, c in glyphs])
        on = np.argwhere(B > 0.02); top, left = br + on[:, 0].min(), bc + on[:, 1].min()
        text = {'width': int(on[:, 1].max() - on[:, 1].min() + 1), 'height': int(on[:, 0].max() - on[:, 0].min() + 1), 'stroke': round(float(stroke), 1)}
        for ch, a, r, c in glyphs:
            for frame, fr, fc in shapes.organisms(a):
                add(_label(ch), frame, r + fr - top, c + fc - left, 'text', {})
        if stroke < 3.5: warn.append('strokes are only %.1f cells thick; thin shapes heal badly. Raise "weight" or "size".' % stroke)

    marks = []
    for k, m in enumerate(cfg['marks']):
        path = m['image'] if os.path.isabs(m['image']) else os.path.join(folder, m['image'])
        a = shapes.render_image(path, m.get('size', 36), m.get('thicken', 0), m.get('invert'))
        on = np.argwhere(a > 0.02); top, left = on[:, 0].min(), on[:, 1].min()
        th = shapes.thickness(a > 0.5)
        if th < 3.5: warn.append('mark %d (%s): lines are only about %.1f cells thick; raise its "size" or "thicken".' % (k, m['image'], th))
        for frame, fr, fc in shapes.organisms(a):
            add('mark%d' % k, frame, fr - top, fc - left, 'mark', {'mark': k})
        link = m.get('link') or ''
        marks.append({'width': int(on[:, 1].max() - left + 1), 'height': int(on[:, 0].max() - top + 1), 'link': link,
                      'label': m.get('label') or re.sub(r'^https?://(www\.)?', '', link).rstrip('/')})

    if not pieces: raise SystemExit('nothing to grow: give some text or a mark')
    for key, t in targets.items():
        with open(os.path.join(tdir, key + '.json'), 'w') as f: json.dump(t, f, separators=(',', ':'))
    layout = {'version': 1, 'text': text, 'marks': marks, 'pieces': pieces}
    with open(os.path.join(folder, 'layout.json'), 'w', encoding='utf-8') as f: json.dump(layout, f, indent=1, ensure_ascii=False)
    _preview(os.path.join(folder, 'preview.png'), sheet, text)

    trained = [k for k in targets if os.path.exists(os.path.join(folder, 'cells', k + '.json'))]
    log('%d organisms, %d different shapes to train%s.' % (len(pieces), len(targets), ' (%d already trained)' % len(trained) if trained else ''))
    if text: log('text: %dx%d cells, strokes %.1f cells thick' % (text['width'], text['height'], text['stroke']))
    for w in warn: log('warning: ' + w)
    log('look at %s to see how it was split.' % os.path.join(folder, 'preview.png'))
    return layout


def _preview(path, sheet, text):
    """Every organism in its own shade of red, its seed in black; marks drawn below the text."""
    from PIL import Image
    rows_t = (text['height'] + 12) if text else 0
    items = [(f, r + (rows_t if g == 'mark' else 0), c + (0 if g == 'text' else mk * 60), s) for f, r, c, s, g, mk in sheet]
    r0 = min(r for _, r, _, _ in items); c0 = min(c for _, _, c, _ in items)
    H = max(r + f.shape[0] for f, r, _, _ in items) - r0; W = max(c + f.shape[1] for f, _, c, _ in items) - c0
    img = np.full((H, W, 3), 245.0)
    for i, (f, r, c, s) in enumerate(items):
        tint = np.array([255, 42, 18]) if i % 2 == 0 else np.array([190, 20, 60])
        a = f[..., None]; sub = img[r - r0: r - r0 + f.shape[0], c - c0: c - c0 + f.shape[1]]
        sub[:] = sub * (1 - a) + tint * a
        img[r - r0 + s[0], c - c0 + s[1]] = 0
    k = max(2, min(8, 1600 // max(W, 1)))
    Image.fromarray(img.astype(np.uint8)).resize((W * k, H * k), Image.NEAREST).save(path)


def train(folder, steps=3000, jobs=None, again=False, log=print):
    """Train every shape that is not trained yet, several at a time, each in its own process."""
    layout = json.load(open(os.path.join(folder, 'layout.json'), encoding='utf-8'))
    tdir, cdir = os.path.join(folder, 'targets'), os.path.join(folder, 'cells')
    ldir = os.path.join(cdir, 'logs'); os.makedirs(ldir, exist_ok=True)
    keys = list(dict.fromkeys(p['cells'] for p in layout['pieces']))
    todo = [k for k in keys if again or not os.path.exists(os.path.join(cdir, k + '.json'))]
    if not todo: log('everything is trained.'); return True
    jobs = jobs or max(1, (os.cpu_count() or 8) // 8)
    log('training %d shapes, %d at a time (each uses 8 processes). Logs: %s' % (len(todo), jobs, ldir))
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    queue, running, results, t0, shown = list(todo), {}, {}, time.time(), 0
    while queue or running:
        while queue and len(running) < jobs:
            k = queue.pop(0); lf = open(os.path.join(ldir, k + '.log'), 'w')
            cmd = [sys.executable, '-m', 'alive.nca', 'fit', os.path.join(tdir, k + '.json'), os.path.join(cdir, k + '.json'), '--steps', str(steps)]
            running[k] = (subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf)
        for k, (p, lf) in list(running.items()):
            if p.poll() is None: continue
            lf.close(); del running[k]; results[k] = p.returncode
            say = {0: 'grows back reliably', 3: 'trained, but does not always grow back whole (see its log)'}.get(p.returncode, 'FAILED, see ' + os.path.join(ldir, k + '.log'))
            log('  %s: %s  [%d of %d done, %s]' % (k, say, len(results), len(todo), _clock(time.time() - t0)))
        if running and time.time() - shown > 60:
            shown = time.time()
            log('  ... ' + ', '.join('%s %s' % (k, _progress(os.path.join(ldir, k + '.log'))) for k in running))
        time.sleep(1)
    bad = [k for k, rc in results.items() if rc not in (0,)]
    if bad: log('%d of %d shapes are not fully reliable. The page still works: its health check restarts any organism that stays broken. To try harder: python -m alive train %s --again --steps %d' % (len(bad), len(todo), folder, steps * 2))
    return not bad


def _progress(path):
    try:
        lines = open(path).read().splitlines()
        for line in reversed(lines):
            m = re.match(r'step\s+(\d+)/(\d+)', line)
            if m: return '%d%%' % (100 * int(m.group(1)) // int(m.group(2)))
            if line.startswith('test,'): return 'testing'
    except OSError: pass
    return 'starting'


def _clock(s):
    return '%dm%02ds' % (s // 60, s % 60)


def test(folder, log=print):
    from . import nca
    layout = json.load(open(os.path.join(folder, 'layout.json'), encoding='utf-8'))
    for k in dict.fromkeys(p['cells'] for p in layout['pieces']):
        cp = os.path.join(folder, 'cells', k + '.json')
        if not os.path.exists(cp): log('%s: not trained' % k); continue
        P, _ = nca.load(cp)
        rel = nca.reliability(P, json.load(open(os.path.join(folder, 'targets', k + '.json'))))
        log('%s: %s' % (k, ', '.join('%s %d/24' % kv for kv in rel.items())))


def copy_runtime(out):
    os.makedirs(out, exist_ok=True)
    for f in RUNTIME: shutil.copyfile(os.path.join(WEB, f), os.path.join(out, f))


def _lum(hexcolor):
    r, g, b = (int(hexcolor.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build(folder, out=None, log=print):
    """Write the page: index.html, the runtime, layout.json and the trained cells."""
    cfg = config(folder)
    layout = json.load(open(os.path.join(folder, 'layout.json'), encoding='utf-8'))
    out = out or os.path.join(folder, 'site')
    os.makedirs(os.path.join(out, 'cells'), exist_ok=True)
    copy_runtime(out)
    shutil.copyfile(os.path.join(folder, 'layout.json'), os.path.join(out, 'layout.json'))
    missing = []
    for k in dict.fromkeys(p['cells'] for p in layout['pieces']):
        src = os.path.join(folder, 'cells', k + '.json')
        if os.path.exists(src): shutil.copyfile(src, os.path.join(out, 'cells', k + '.json'))
        else: missing.append(k)
    name = cfg['text'] or cfg['title'] or 'alive'
    links = ''.join('<a class="mark" data-mark="%d" href="%s" target="_blank" rel="noopener" aria-label="%s" data-label="%s" draggable="false"></a>\n'
                    % (k, html.escape(m['link']), html.escape(m['label']), html.escape(m['label'])) for k, m in enumerate(layout['marks']) if m['link'])
    ink = '#0c0d10' if _lum(cfg['paper']) > 0.5 else '#f4f5f6'
    page = open(os.path.join(WEB, 'page.html'), encoding='utf-8').read()
    for k, v in {'{{TITLE}}': html.escape(cfg['title'] or name), '{{TEXT}}': html.escape(name), '{{PAPER}}': cfg['paper'], '{{INK}}': ink,
                 '{{TISSUE}}': cfg['color'], '{{TISSUE_URL}}': cfg['color'].replace('#', '%23'), '{{LINKS}}': links,
                 '{{CONFIG}}': json.dumps({'colors': {'paper': cfg['paper'], 'tissue': cfg['color']}}).replace('</', '<\\/')}.items():
        page = page.replace(k, v)
    with open(os.path.join(out, 'index.html'), 'w', encoding='utf-8') as f: f.write(page)
    if missing: log('warning: %d shapes are not trained yet, so they will not appear: run python -m alive train %s' % (len(missing), folder))
    log('wrote %s' % out)
    return out


def serve(folder, port=8000):
    """Serve the built page locally (workers need http://, not file://)."""
    import http.server, functools, webbrowser
    root = os.path.join(folder, 'site') if os.path.exists(os.path.join(folder, 'site', 'index.html')) else folder

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.js': 'text/javascript', '.json': 'application/json'}

    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Handler, directory=root))
    url = 'http://127.0.0.1:%d/' % port
    print('serving %s at %s (Ctrl+C to stop)' % (root, url))
    try: webbrowser.open(url)
    except Exception: pass
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
