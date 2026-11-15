"""Turn text and images into the shapes cells are trained to grow.

A shape is grown as one or more organisms, each from its own seed. Cells only talk to their
8 neighbours, and one seed can organise a region about 50 cells across, so:
  - text is split letter by letter, and a dot over an i is its own organism;
  - anything held together only by a thin neck or a single point (two letters touching) is split
    there, because growth cannot cross a one-cell bridge reliably;
  - strokes are made at least about 4.4 cells thick, or a shape falls apart while it heals.

Pen draws shapes from round strokes instead, which is how mowalker.ai's letters were made.
"""
import numpy as np

SS = 8             # rasterise at 8x and average down, for smooth edges
MARGIN = 5         # empty cells around each organism's frame, room to heal into
MIN_STROKE = 4.4   # thinnest stroke, in cells, that holds up well


class Pen:
    """Draws shapes as a signed distance field: seg, arc, taper and box add ink, carve removes it."""
    def __init__(self, h, w):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        self.x, self.y = xx + 0.5, yy + 0.5
        self.d = np.full((h, w), 1e3)

    def seg(self, ax, ay, bx, by, r):
        px, py, dx, dy = self.x - ax, self.y - ay, bx - ax, by - ay
        L = dx * dx + dy * dy
        t = np.clip((px * dx + py * dy) / L, 0, 1) if L > 0 else 0
        self.d = np.minimum(self.d, np.hypot(px - t * dx, py - t * dy) - r)

    def arc(self, cx, cy, rad, a0, a1, r):
        """Part of a circle from angle a0 to a1 in degrees: 0 = right, 90 = down (y grows downward)."""
        ang = np.degrees(np.arctan2(self.y - cy, self.x - cx)) % 360
        on = (ang >= a0) & (ang <= a1) if a1 <= 360 else (ang >= a0) | (ang <= a1 - 360)
        ring = np.abs(np.hypot(self.x - cx, self.y - cy) - rad)
        ends = np.minimum(*[np.hypot(self.x - (cx + rad * np.cos(np.radians(a))), self.y - (cy + rad * np.sin(np.radians(a)))) for a in (a0, a1)])
        self.d = np.minimum(self.d, np.where(on, ring, ends) - r)

    def taper(self, ax, ay, bx, by, r0, r1, bend=1.35):
        """A stroke that thins from r0 at a to r1 at b, curving inward (sparkle arms)."""
        px, py, dx, dy = self.x - ax, self.y - ay, bx - ax, by - ay
        t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
        self.d = np.minimum(self.d, np.hypot(px - t * dx, py - t * dy) - (r1 + (r0 - r1) * (1 - t) ** bend))

    def box(self, x0, y0, x1, y1, rr):
        """A filled rectangle with corners rounded to rr."""
        cx, cy, bx, by = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 - rr, (y1 - y0) / 2 - rr
        qx, qy = np.abs(self.x - cx) - bx, np.abs(self.y - cy) - by
        self.d = np.minimum(self.d, np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rr)

    def carve(self, ax, ay, bx, by, r):
        """Cut a round-ended hole out of what is already drawn."""
        px, py, dx, dy = self.x - ax, self.y - ay, bx - ax, by - ay
        L = dx * dx + dy * dy
        t = np.clip((px * dx + py * dy) / L, 0, 1) if L > 0 else 0
        self.d = np.maximum(self.d, -(np.hypot(px - t * dx, py - t * dy) - r))

    def alpha(self):
        return np.clip(0.5 - self.d, 0, 1)


def closest(da, db):
    """Smallest gap between two shapes, from their distance fields."""
    return float((np.maximum(da, 0) + np.maximum(db, 0)).min())


def reach(alpha, seed):
    """Flood out from the seed through cells that can carry a signal. Returns steps to each cell (-1 = cut off)."""
    h, w = alpha.shape
    p = np.pad(alpha, 1)
    live = np.max([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0) > 0.1
    dist = np.full((h, w), -1); dist[seed] = 0; front = [seed]
    while front:
        nxt = []
        for r, c in front:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and live[rr, cc] and dist[rr, cc] < 0:
                        dist[rr, cc] = dist[r, c] + 1; nxt.append((rr, cc))
        front = nxt
    return dist


def pick_seed(alpha, prefer=None):
    """The solid cell nearest the centre from which growth reaches every part soonest.
    Returns ((row, col), steps to the farthest cell), or None if some part is out of reach of every seed."""
    h, w = alpha.shape
    solid = np.argwhere(alpha > 0.99)
    if not len(solid): solid = np.argwhere(alpha > 0.5)
    cy, cx = prefer if prefer else (h / 2, w / 2)
    best = None
    for r, c in solid[np.argsort(((solid[:, 0] - cy) ** 2 + (solid[:, 1] - cx) ** 2))][:400]:
        d = reach(alpha, (int(r), int(c)))
        if (d[alpha > 0.5] < 0).any(): continue
        far = int(d[alpha > 0.5].max())
        if best is None or far < best[1]: best = ((int(r), int(c)), far)
        if best and far <= best[1] and len(solid) > 0 and best[1] < 40: break
    return best


def make_target(name, alpha, prefer=None, color=None):
    """What the trainer needs for one organism."""
    best = pick_seed(alpha, prefer)
    if not best: raise ValueError(name + ': part of the shape is out of signalling range of every seed')
    seed, far = best
    t = {'name': name, 'H': int(alpha.shape[0]), 'W': int(alpha.shape[1]), 'seed': list(seed), 'far': far,
         'solid': int((alpha > 0.5).sum()), 'alpha': np.round(alpha, 4).ravel().tolist()}
    if color: t['color'] = color
    return t


def ascii_art(alpha, seed=None):
    chars = ' .:-=+*#%@'
    art = [''.join(chars[int(min(v, 0.999) * 10)] for v in row) for row in alpha[::2]]
    if seed:
        art[seed[0] // 2] = art[seed[0] // 2][:seed[1]] + 'S' + art[seed[0] // 2][seed[1] + 1:]
    return '\n'.join(a for a in art if a.strip())


def crop(alpha, margin=MARGIN):
    """Cut a frame out of a drawing, with `margin` empty cells all round.
    Returns the frame and where its top left sat in the drawing (can be negative)."""
    a = np.pad(alpha, margin)
    on = np.argwhere(a > 0.02)
    r0, c0 = on[:, 0].min() - margin, on[:, 1].min() - margin
    return a[r0: on[:, 0].max() + 1 + margin, c0: on[:, 1].max() + 1 + margin], int(r0) - margin, int(c0) - margin


def thickness(on):
    """Typical stroke width of a mask, in its own pixels: twice the area over the outline length."""
    area = on.sum()
    edge = (on[:, 1:] != on[:, :-1]).sum() + (on[1:] != on[:-1]).sum()
    return 2.0 * area / max(1, edge)


# ---------- splitting a shape into organisms ----------

def _components(mask, eight):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32); n = 0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if eight else [])
    for r, c in np.argwhere(mask):
        if lab[r, c]: continue
        n += 1; lab[r, c] = n; todo = [(r, c)]
        while todo:
            y, x = todo.pop()
            for dy, dx in nbrs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n; todo.append((yy, xx))
    return lab, n


def _spread(region, into):
    """Grow labelled cells outward, one ring at a time, through the cells marked in `into`."""
    h, w = region.shape
    while True:
        grown = region.copy(); p = np.pad(region, 1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nb = p[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
                take = (grown == 0) & into & (nb > 0)
                grown[take] = nb[take]
        if (grown == region).all(): return region
        region = grown


def split(alpha, min_core=6):
    """The organisms a shape should be grown as, each as an alpha array the size of the input.
    The shape's core (solid cells with solid all round) is split into pieces wherever it is pinched to
    nothing; each piece then takes back the rest of the shape nearest to it."""
    solid = alpha > 0.5
    h, w = alpha.shape
    p = np.pad(solid, 1)
    core = solid.copy()
    for di in range(3):
        for dj in range(3): core &= p[di:di + h, dj:dj + w]
    lab, n = _components(core, eight=False)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = [k for k in range(1, n + 1) if sizes[k] >= min_core]
    if not keep:   # strokes thinner than 3 cells have no core: fall back to the solid parts themselves
        lab, n = _components(solid, eight=True)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = [k for k in range(1, n + 1) if sizes[k] >= 2]
    region = _spread(np.where(np.isin(lab, keep), lab, 0), alpha > 0.1)
    region = _spread(region, alpha > 0.02)
    parts = [np.where(region == k, alpha, 0.0) for k in keep]
    return [a for a in parts if (a > 0.5).sum() >= 4]


def organisms(alpha, margin=MARGIN):
    """Split a shape and cut out each organism's frame: [(frame, row, col)], row and col being where the
    frame's top left sits in the input."""
    return [crop(part, margin) for part in split(alpha)]


def board(items):
    """Lay [(alpha, row, col)] onto one array. Returns it and the row and col of its top left."""
    r0 = min(r for a, r, c in items); c0 = min(c for a, r, c in items)
    r1 = max(r + a.shape[0] for a, r, c in items); c1 = max(c + a.shape[1] for a, r, c in items)
    B = np.zeros((r1 - r0, c1 - c0))
    for a, r, c in items:
        np.maximum(B[r - r0: r - r0 + a.shape[0], c - c0: c - c0 + a.shape[1]], a, out=B[r - r0: r - r0 + a.shape[0], c - c0: c - c0 + a.shape[1]])
    return B, r0, c0


# ---------- text and images ----------

def _font(path, px):
    from PIL import ImageFont
    if path: return ImageFont.truetype(path, px)
    try:
        return ImageFont.load_default(size=px)   # Pillow 10.1+ ships a scalable font (Aileron)
    except TypeError:
        raise SystemExit('The built-in font needs Pillow 10.1 or newer: pip install -U pillow (or pass --font)')


def _cells(im):
    """A Pillow 'L' image whose sides are multiples of SS, averaged down to cells (0..1)."""
    from PIL import Image
    return np.asarray(im.resize((im.width // SS, im.height // SS), Image.BOX), np.float32) / 255.0


def _up(v):
    return -(-int(v) // SS) * SS


def render_text(text, font=None, size=20, spacing=1.0, weight=None):
    """Rasterise text into cells, one glyph at a time.
    size: height of a lowercase x, in cells. spacing: extra cells between letters.
    weight: cells to add to every stroke (None: thicken thin fonts up to MIN_STROKE).
    Returns ([(char, alpha, row, col)], stroke width in cells). Each glyph is drawn the same way wherever it
    falls, so repeated letters can share one trained organism."""
    from PIL import Image, ImageDraw
    probe = _font(font, 1000)
    b = probe.getbbox('x')
    xh = (b[3] - b[1]) / 1000.0 or 0.5
    f = _font(font, max(8, int(round(size * SS / xh))))
    asc, desc = f.getmetrics()

    plain = Image.new('L', (_up(f.getlength(text)) + 8 * SS, _up(asc + desc) + 8 * SS), 0)
    ImageDraw.Draw(plain).text((4 * SS, 4 * SS), text, font=f, fill=255)
    base = thickness(np.asarray(plain) > 127) / SS
    extra = max(0.0, MIN_STROKE - base) if weight is None else float(weight)
    stroke = int(round(extra * SS / 2))

    out = []
    for i, ch in enumerate(text):
        if ch.isspace(): continue
        l, t, r, btm = f.getbbox(ch, stroke_width=stroke)
        pad = _up(stroke + 2 * SS)
        ox = _up(pad - min(0, l))
        im = Image.new('L', (_up(ox + max(r, f.getlength(ch)) + pad), _up(max(btm, asc + desc) + 2 * pad)), 0)
        ImageDraw.Draw(im).text((ox, pad), ch, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        x = f.getlength(text[:i]) / SS + i * spacing
        out.append((ch, _cells(im), -(pad // SS), int(round(x)) - ox // SS))
    return out, base + 2 * stroke / SS


def render_image(path, size=36, thicken=0.0, invert=None):
    """An image file as alpha in cells, its longest side `size` cells.
    A PNG with transparency is used as it is. Otherwise the shapes are whatever stands out from the
    background: dark on light or light on dark, judged from the border (or forced with invert).
    thicken: cells to add to every stroke, for thin lines."""
    from PIL import Image, ImageFilter, ImageOps
    im = ImageOps.exif_transpose(Image.open(path))
    a = None
    if 'A' in im.getbands() or (im.mode == 'P' and 'transparency' in im.info):
        a = im.convert('RGBA').getchannel('A')
        if a.getextrema()[0] >= 250: a = None   # opaque everywhere: the alpha channel says nothing
    if a is None:
        g = np.asarray(im.convert('L'), np.float32)
        border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]).mean()
        light = border > 127 if invert is None else bool(invert)
        a = Image.fromarray((255 - g if light else g).astype(np.uint8))
    box = a.point(lambda v: 255 if v > 24 else 0).getbbox()
    if not box: raise ValueError(path + ': the image is empty')
    a = a.crop(box)
    k = size * SS / max(a.size)
    a = a.resize((max(1, round(a.width * k)), max(1, round(a.height * k))), Image.LANCZOS)
    pad = (int(np.ceil(thicken)) + 2) * SS
    canvas = Image.new('L', (_up(a.width + 2 * pad), _up(a.height + 2 * pad)), 0)
    canvas.paste(a, (pad, pad))
    r = int(round(thicken * SS / 2))
    if r > 0: canvas = canvas.filter(ImageFilter.MaxFilter(2 * r + 1))
    return _cells(canvas)
