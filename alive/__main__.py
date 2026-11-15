"""alive: grow text and logos out of living cells that heal when you cut them.

  python -m alive make mysite "hello"                  all of it: draw, train, build
  python -m alive serve mysite                         open the page locally

Step by step:
  python -m alive new mysite "hello" [options]          start a project folder and draw the shapes
  python -m alive draw mysite                          draw again after editing mysite/alive.json
  python -m alive train mysite [--steps N] [--jobs N]  train every organism (minutes each)
  python -m alive test mysite                          how reliably each one grows back
  python -m alive build mysite                         write the page to mysite/site
  python -m alive runtime DIR                          copy the browser files into DIR, for a page of your own

Options for new and make:
  --font PATH.ttf         any TrueType or OpenType font (default: Pillow's built-in Aileron)
  --size N                height of a lowercase x in cells (default 20); bigger is finer but slower to train
  --weight N              cells to add to every stroke (default: thicken thin fonts automatically)
  --spacing N             extra cells between letters (default 1)
  --color '#rrggbb'       the tissue (default '#ff2a12')
  --paper '#rrggbb'       the dish (default '#f4f5f6')
  --mark IMAGE[=URL]      a logo grown under the text, optionally a link; repeat for more
  --title TEXT            the page title (default: the text)
"""
import argparse, os, shutil, sys

from . import project


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m alive', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, usage=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest='cmd', metavar='command')

    def shape_opts(p):
        p.add_argument('folder'); p.add_argument('text', nargs='?', default=None)
        p.add_argument('--font'); p.add_argument('--size', type=float); p.add_argument('--weight', type=float)
        p.add_argument('--spacing', type=float); p.add_argument('--color'); p.add_argument('--paper'); p.add_argument('--title')
        p.add_argument('--mark', action='append', default=[], metavar='IMAGE[=URL]')

    shape_opts(sub.add_parser('new', help='start a project folder and draw its shapes'))
    mk = sub.add_parser('make', help='draw, train and build in one go'); shape_opts(mk)
    sub.add_parser('draw', help='draw the shapes again from alive.json').add_argument('folder')
    for p in (sub.add_parser('train', help='train the organisms'), mk):
        p.add_argument('--steps', type=int, default=3000, help='training steps per shape before the first test (default 3000)')
        p.add_argument('--jobs', type=int, default=None, help='shapes to train at once (default: CPU threads / 8)')
    sub.choices['train'].add_argument('folder')
    sub.choices['train'].add_argument('--again', action='store_true', help='retrain shapes that are already trained')
    sub.add_parser('test', help='how reliably each organism grows back').add_argument('folder')
    b = sub.add_parser('build', help='write the page'); b.add_argument('folder'); b.add_argument('--out')
    s = sub.add_parser('serve', help='open the page locally'); s.add_argument('folder'); s.add_argument('--port', type=int, default=8000)
    sub.add_parser('runtime', help='copy the browser files into a folder').add_argument('dir')
    a = ap.parse_args(argv)

    if a.cmd in ('new', 'make'):
        path = os.path.join(a.folder, 'alive.json')
        cfg = project.config(a.folder) if os.path.exists(path) else dict(project.DEFAULTS)
        if a.text is not None: cfg['text'] = a.text
        for k in ('font', 'size', 'weight', 'spacing', 'color', 'paper', 'title'):
            if getattr(a, k) is not None: cfg[k] = getattr(a, k)
        if cfg['font']: cfg['font'] = os.path.abspath(cfg['font'])
        if a.mark:
            os.makedirs(os.path.join(a.folder, 'marks'), exist_ok=True)
            cfg['marks'] = []
            for m in a.mark:
                img, _, url = m.partition('=')
                dst = os.path.join('marks', os.path.basename(img))
                if os.path.abspath(img) != os.path.abspath(os.path.join(a.folder, dst)): shutil.copyfile(img, os.path.join(a.folder, dst))
                cfg['marks'].append({'image': dst.replace(os.sep, '/'), 'link': url, 'size': 36})
        project.save_config(a.folder, cfg)
        project.draw(a.folder)
        if a.cmd == 'new':
            print('next: python -m alive train %s' % a.folder)
            return 0
        ok = project.train(a.folder, steps=a.steps, jobs=a.jobs)
        project.build(a.folder)
        print('done. try it: python -m alive serve %s' % a.folder)
        return 0 if ok else 3
    if a.cmd == 'draw': project.draw(a.folder); return 0
    if a.cmd == 'train': return 0 if project.train(a.folder, steps=a.steps, jobs=a.jobs, again=a.again) else 3
    if a.cmd == 'test': project.test(a.folder); return 0
    if a.cmd == 'build': project.build(a.folder, a.out); return 0
    if a.cmd == 'serve': project.serve(a.folder, a.port); return 0
    if a.cmd == 'runtime': project.copy_runtime(a.dir); print('copied %s into %s' % (', '.join(project.RUNTIME), a.dir)); return 0
    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
