# alive-nca

Grow your name and your logo out of living cells that heal when you cut them. This is the library behind [mowalker.ai](https://mowalker.ai).

Every letter on a page made with it is a living organism. It is a grid of cells that all run the same tiny neural network, and each cell only talks to its 8 neighbours. Nothing is drawn: each shape grows from a single seed cell. Drag across it with a mouse or a finger and you cut it. It grows back.

You give it text, a font, logos and colours. It trains one organism per shape and writes a finished static page you can host anywhere.

## Make your own

You need Python 3.9 or newer.

```bash
git clone https://github.com/ultran1te/alive-nca.git
cd alive-nca
python3 -m venv .venv && . .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt                    # numpy and Pillow, nothing else
python -m alive make mysite "your name"
python -m alive serve mysite
```

`make` draws your text, trains one organism per letter, and writes a finished page to `mysite/site/`. Training takes a few minutes per letter shape on a fast desktop, and much longer on a laptop. Repeated letters are only trained once. Spelling "hello" plus a logo took about 15 minutes on a 32-thread desktop.

To see a finished one without training anything:

```bash
python -m alive serve examples/hello
```

You can also install it as a package, which adds an `alive` command that does the same as `python -m alive`:

```bash
pip install git+https://github.com/ultran1te/alive-nca.git
```

### Options

```bash
python -m alive make mysite "ada" --font Inter-Bold.ttf --color "#2457ff" --paper "#0e1014"
python -m alive make mysite "ada" --mark github.png=https://github.com/you
```

| Option | What it does |
| --- | --- |
| `--font PATH` | Any `.ttf` or `.otf` font. The default is Aileron, which comes with Pillow. |
| `--size N` | Height of a lowercase x, in cells (default 20). Bigger looks finer but trains slower. |
| `--weight N` | Cells to add to every stroke. By default, thin fonts are thickened automatically. |
| `--spacing N` | Extra cells between letters (default 1). |
| `--color '#rrggbb'` | The living tissue (default `#ff2a12`). |
| `--paper '#rrggbb'` | The dish behind it (default `#f4f5f6`). A dark paper works too. |
| `--mark IMAGE[=URL]` | A logo grown in a row under the text, optionally as a link. Repeat for more. |
| `--title TEXT` | The browser tab title. |

### What grows well

- **Bold, simple, filled shapes.** Strokes need to be about 4 cells thick or they fall apart while healing. Text is thickened for you. For a logo, raise `size` or `thicken` in `alive.json`.
- **Letters that stand apart.** One seed can organise a region about 50 cells across. A script font where every letter joins makes one huge organism that won't hold together.
- **Logos as PNG.** A transparent background is best. Otherwise use dark shapes on a light background, or the reverse. If a logo comes out in pieces, that's on purpose: wherever two parts only touch at a point, it's split into separate organisms, because growth can't cross a one-cell bridge reliably.

After drawing, open `mysite/preview.png`. Each shade of red is one organism and the black dot is its seed.

### Step by step

`make` is these commands in order. Run them yourself to change things between steps.

```bash
python -m alive new mysite "ada" [options]   # writes mysite/alive.json and draws the shapes
python -m alive draw mysite                  # draw again after editing mysite/alive.json
python -m alive train mysite                 # train every shape that isn't trained yet
python -m alive test mysite                  # how reliably each one grows back
python -m alive build mysite                 # write the page to mysite/site
python -m alive serve mysite                 # open it in your browser
```

To change the text, a colour or a logo later, edit `mysite/alive.json` and run `draw`, `train` and `build` again. Only shapes that changed are trained again. Changing `color` retrains everything, because the cells learn their colour.

A project folder looks like this:

| Path | What it holds |
| --- | --- |
| `alive.json` | What to grow: text, font, size, colours, marks and their links |
| `marks/` | Copies of your logo images |
| `targets/`, `layout.json`, `preview.png` | The shapes, and where each one sits |
| `cells/` | The trained organisms, with each one's test results |
| `site/` | The finished page. Upload this folder. |

### Putting it online

`site/` is plain static files. Any static host works: Cloudflare, GitHub Pages, Netlify, or your own server. Serve it over `http(s)://`, because the cells run in web workers, and those don't start from a `file://` page.

On Cloudflare:

```bash
npx wrangler deploy --assets mysite/site --name my-living-page --compatibility-date 2026-11-01
```

### A page of your own design

The generated page puts the text in the middle with the marks under it. To lay things out yourself, copy the three browser files next to your own HTML:

```bash
python -m alive runtime mypage/
```

```html
<canvas id="gl"></canvas>
<script src="alive-core.js"></script>
<script src="alive-dish.js"></script>
<script type="module">
  const cells = { h: await (await fetch('cells/h-c124f444.json')).json() };
  AliveDish.start({
    canvas: document.getElementById('gl'),
    cells,
    // called at start and on every resize: lattices, and where each organism's dish starts in cells
    arrange: ({ W, H }) => ({ layers: [{ s: 4, oy: 0 }], orgs: [{ name: 'h', r0: 10, c0: 10, expect: 301 }] }),
  });
</script>
```

`alive-dish.js` explains every option at the top of the file. [mowalker.ai](https://mowalker.ai) is a page laid out this way; its source is at [ultran1te/mowalker.ai](https://github.com/ultran1te/mowalker.ai) if you want to see how it works. It has a second, finer lattice for small icons, a link laid over an organism, and text that the organisms make room for. To draw shapes from strokes instead of a font, as the mowalker.ai wordmark was, see `Pen` in [alive/shapes.py](alive/shapes.py).

## How it works

**The cells.** Each cell holds 12 numbers: colour, alpha, and 8 hidden values the cells use to signal each other. On every step, each cell looks at its 3×3 neighbourhood through two Sobel filters. That gives it its own 12 values, plus how each one changes left to right and top to bottom: 36 numbers. It runs them through one small network (36 → 96 → 12) and adds the result to its state. About half the cells update on each step, chosen at random, so there's no global clock. A cell counts as alive if it or a neighbour has alpha above 0.1. Every other cell is set to zero, so growth can only spread outward from living tissue.

**The anchor.** Each organism's seed cell is held alive permanently. Without it, a shape slowly creeps across its dish over thousands of steps, and once it's wiped out completely it can never come back.

**Training** ([alive/nca.py](alive/nca.py)) is pure numpy. The backward pass through time is written out by hand and checked against numerical gradients (`python -m alive.nca gradcheck`).
- A pool of 256 states is kept.
- On each step, 8 of them run forward in parallel processes, for 2 to 3 times as many steps as growth needs to reach the far edge of the shape.
- The worst of the 8 starts over from a single seed.
- The best few are bitten, or sliced in half with one side thrown away.

So the network learns three things at once: grow, hold the shape, and heal.

**The test.** Loss is a poor judge of whether a shape will actually grow back, and the last checkpoint isn't always the best one. So every shape is grown 24 times in each of five situations:
- from its seed;
- after losing its left half;
- after losing its right half;
- after being wiped out;
- after being cut while still growing.

Training continues until at least 23 of 24 runs end up whole in every situation, and the best checkpoint is kept.

**In the browser.** Organisms run in web workers ([alive-core.js](alive/web/alive-core.js), [alive-worker.js](alive/web/alive-worker.js)):
- **Speed:** about 45 steps a second while growing or healing, and 6 at rest.
- **Hidden tab:** they stop completely.
- **Slow phones:** they never use more than about 80% of one core.
- **Health check:** any organism that stays broken for about ten seconds is restarted from its anchor.

[alive-dish.js](alive/web/alive-dish.js) draws everything in one WebGL 2 pass:
- the cell states are smoothed with a cubic B-spline and cut at an edge;
- a soft shadow sits under the tissue;
- each cell gets a nucleus that flickers with one of its hidden channels;
- only the parts of the window near living tissue are shaded.

A resize or a turned phone only moves where each organism is drawn, so nothing dies.

## This repository

```
alive/              the library
  nca.py            training and the reliability test
  shapes.py         fonts, images and pen strokes into shapes, split into organisms
  project.py        project folders: draw, train, build, serve
  web/              alive-core.js, alive-worker.js, alive-dish.js, and the page template
examples/hello/     a finished, trained example
```

## License

MIT: copy it, change it, and make your own living pages. See [LICENSE](LICENSE).

The mowalker.ai site itself, with its wordmark, icons and trained organisms, lives in its own repository and is not covered by this license.
