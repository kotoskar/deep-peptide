#!/usr/bin/env python3
"""Render main.tex as a Markdown document that reads like the built PDF.

Pandoc does the LaTeX parsing. This script does the two things pandoc cannot do
on its own here: it takes the real section/table/figure numbers out of main.aux
so cross-references read the way they do in the PDF, and it rewrites the pieces
pandoc leaves as raw HTML (the multi-header results table, figure blocks) into
Markdown a plain viewer will render.

    python3 texs/ai4dd/tex2md.py --pandoc /path/to/pandoc -o texs/ai4dd/main.md

Run it from the texs/ai4dd directory, or pass --dir: \input{figures/...} is
resolved relative to the working directory.
"""
from __future__ import annotations
import argparse, html, pathlib, re, subprocess, sys, tempfile

# --------------------------------------------------------------- preprocess ---

def preprocess(tex: str) -> str:
    """Make the source palatable to pandoc without changing what it says."""
    # \MaybeImage{path}{scale} is ours; pandoc knows \includegraphics.
    tex = re.sub(r'\\MaybeImage\{([^}]*)\}\{([^}]*)\}',
                 r'\\includegraphics[width=\2\\linewidth]{\1}', tex)
    # A \resizebox wrapper around a tabular hides the table from the reader.
    tex = tex.replace('\\resizebox{\\linewidth}{!}{%\n', '')
    tex = tex.replace('\\resizebox{\\linewidth}{!}{%', '')
    tex = re.sub(r'\\end\{tabular\}\}', r'\\end{tabular}', tex)
    tex = re.sub(r'(\\input\{figures/[^}]*\})\}', r'\1', tex)
    return tex


# ------------------------------------------------------------------- labels ---

def read_labels(aux_path: pathlib.Path) -> dict[str, str]:
    """label -> the number LaTeX printed for it, from main.aux."""
    out: dict[str, str] = {}
    for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}', aux_path.read_text()):
        name, num = m.group(1), m.group(2)
        if name.endswith('@cref'):
            continue
        num = re.sub(r'\\[a-zA-Z]+\s*', '', num).strip('{} ')
        if num:
            out[name] = num
    return out


KIND = {'sec': 'Section', 'tab': 'Table', 'fig': 'Figure', 'app': 'Appendix'}


def ref_text(label: str, labels: dict[str, str]) -> str:
    num = labels.get(label)
    if num is None:
        return f'[{label}]'
    prefix = label.split(':', 1)[0]
    kind = KIND.get(prefix, 'Section')
    if prefix == 'sec' and not num[0].isdigit():
        kind = 'Appendix'          # A-F are appendices, 1-6 are sections
    return f'{kind}&nbsp;{num}'


# ------------------------------------------------------------- post-process ---

MATH = [
    (r'\\pm\s*', '±'), (r'\\times', '×'), (r'\\approx', '≈'), (r'\\to', '→'),
    (r'\\ge', '≥'), (r'\\le', '≤'), (r'\\emph\{([^}]*)\}', r'*\1*'),
    (r'\\mathrm\{([^}]*)\}', r'\1'), (r'\\textsf\{([^}]*)\}', r'\1'),
    (r'\\dots', '…'), (r'\\,', ' '), (r'\{,\}', ','), (r'\\cdot', '·'),
    (r'\\tau', 'τ'), (r'\\cup', '∪'), (r'\\cap', '∩'), (r'\\alpha', 'α'), (r'\\in\b', ' ∈ '), (r'\\checkmark', '✓'),
]


def demath(expr: str) -> str:
    expr = expr.replace('\\{', '\x01').replace('\\}', '\x02')
    for pat, sub in MATH:
        expr = re.sub(pat, sub, expr)
    expr = expr.replace('{', '').replace('}', '').replace('\\', '')
    expr = expr.replace('\x01', '{').replace('\x02', '}')
    return re.sub(r'\s+', ' ', expr).strip()


def unwrap_math(md: str) -> str:
    md = re.sub(r'\$`(.+?)`\$', lambda m: demath(m.group(1)), md, flags=re.S)
    md = re.sub(r'<span class="math inline">(.+?)</span>',
                lambda m: demath(html.unescape(m.group(1))), md, flags=re.S)
    return md.replace('<!-- -->', '')


def fix_refs(md: str, labels: dict[str, str]) -> str:
    def one(m):
        return ref_text(m.group(1).lstrip('#'), labels)
    # A \Cref with several targets becomes one anchor carrying a comma list.
    def many(m):
        parts = [p for p in m.group(1).lstrip('#').split(',') if p]
        return ' and '.join(ref_text(p, labels) for p in parts)
    md = re.sub(r'<a href="#([^"]*,[^"]*)"[^>]*>.*?</a>', many, md, flags=re.S)
    md = re.sub(r'<a href="(#[^"]+)"[^>]*>.*?</a>', one, md, flags=re.S)
    return md


def fix_figures(md: str, labels: dict[str, str]) -> str:
    def one(m):
        block = m.group(0)
        label = re.search(r'id="([^"]+)"', block)
        src = re.search(r'src="([^"]+)"', block)
        cap = re.search(r'<figcaption>(.*?)</figcaption>', block, re.S)
        num = labels.get(label.group(1), '?') if label else '?'
        caption = cap.group(1).strip() if cap else ''
        path = src.group(1) if src else ''
        return f'![Figure {num}]({path})\n\n***Figure&nbsp;{num}.*** {caption}\n'
    return re.sub(r'<figure[^>]*>.*?</figure>', one, md, flags=re.S)


def number_headings(md: str, labels: dict[str, str], tex: str) -> str:
    """Prefix each heading with the number LaTeX gave that section."""
    order: list[tuple[str, str]] = []          # (title, label)
    for m in re.finditer(r'\\(sub)?section\{([^}]*)\}\s*(?:\\label\{([^}]*)\})?', tex):
        order.append((m.group(2), m.group(3) or ''))
    titles = {t: labels.get(l, '') for t, l in order if l}

    def one(m):
        hashes, title = m.group(1), m.group(2).strip()
        num = titles.get(title, '')
        return f'{hashes} {num} {title}' if num else f'{hashes} {title}'
    return re.sub(r'^(#{1,3}) (.+)$', one, md, flags=re.M)


RESULTS_TABLE = """
| Additions | P (±3) | R (±3) | F1 ±3 | F1 ±2 | F1 ±1 | F1 exact | Growth |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ESM-2** | | | | | | | |
| none | 0.621 ±.025 | 0.539 ±.045 | 0.576 ±.029 | 0.544 ±.027 | 0.467 ±.020 | 0.354 ±.010 | — |
| boundary head | 0.673 ±.029 | 0.538 ±.036 | 0.597 ±.026 | 0.566 ±.024 | 0.488 ±.015 | 0.384 ±.015 | +0.010 |
| adapter | 0.628 ±.019 | 0.579 ±.040 | 0.602 ±.026 | 0.572 ±.024 | 0.491 ±.011 | 0.396 ±.020 | +0.016 |
| both | 0.667 ±.032 | 0.598 ±.026 | **0.630 ±.021** | 0.599 ±.023 | 0.518 ±.016 | 0.432 ±.013 | +0.023 |
| **ESM-C 6B** | | | | | | | |
| none | 0.620 ±.017 | 0.560 ±.027 | 0.588 ±.016 | 0.558 ±.013 | 0.483 ±.007 | 0.371 ±.013 | +0.005 |
| boundary head | 0.731 ±.010 | 0.572 ±.030 | 0.641 ±.022 | 0.608 ±.026 | 0.532 ±.035 | 0.429 ±.030 | +0.010 |
| adapter | 0.598 ±.012 | 0.607 ±.031 | 0.602 ±.017 | 0.573 ±.019 | 0.497 ±.016 | 0.394 ±.031 | +0.014 |
| both | 0.690 ±.013 | 0.646 ±.033 | **0.666 ±.018** | 0.636 ±.015 | 0.554 ±.016 | 0.466 ±.021 | +0.021 |
"""



def html_table_to_pipe(block: str) -> str:
    """Fallback for tables pandoc could not express as a pipe table."""
    cap = re.search(r'<caption>(.*?)</caption>', block, re.S)
    rows: list[list[str]] = []
    for tr in re.findall(r'<tr>(.*?)</tr>', block, re.S):
        cells = [re.sub(r'<[^>]+>', '', c).strip()
                 for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, re.S)]
        if any(cells):
            rows.append(cells)
    if not rows:
        return block
    width = max(len(r) for r in rows)
    rows = [r + [''] * (width - len(r)) for r in rows]
    out = ['| ' + ' | '.join(rows[0]) + ' |',
           '|' + '---|' * width]
    out += ['| ' + ' | '.join(r) + ' |' for r in rows[1:]]
    body = '\n'.join(out)
    if cap:
        body = body + '\n\n***' + re.sub(r'<[^>]+>', '', cap.group(1)).strip() + '***'
    return body + '\n'


def fix_tables(md: str) -> str:
    def one(m):
        block = m.group(0)
        if 'nested cross-validation with the corrected matcher' in block:
            cap = re.search(r'<caption>(.*?)</caption>', block, re.S)
            text = re.sub(r'<[^>]+>', '', cap.group(1)).strip() if cap else ''
            return RESULTS_TABLE + '\n***Table&nbsp;1.*** ' + text + '\n'
        return html_table_to_pipe(block)
    return re.sub(r'<table>.*?</table>', one, md, flags=re.S)


def front_matter(tex: str) -> str:
    title = re.search(r'\\title\{(.+?)\}\s*\n', tex, re.S)
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S)
    parts = []
    if title:
        parts.append('# ' + re.sub(r'\s+', ' ', title.group(1)).strip())
    parts.append('*Anonymous submission — AI4DD @ NeurIPS 2026. '
                 'Rendered from `texs/ai4dd/main.tex`.*')
    if abstract:
        body = abstract.group(1)
        body = re.sub(r'%.*', '', body)
        body = re.sub(r'\\cite[a-z]*\{[^}]*\}', '', body)
        body = demath(re.sub(r'\$(.+?)\$', lambda m: demath(m.group(1)), body, flags=re.S)) \
            if False else re.sub(r'\$(.+?)\$', lambda m: demath(m.group(1)), body, flags=re.S)
        body = body.replace('--', '\u2013')
        parts.append('## Abstract\n\n' + re.sub(r'\s+', ' ', body).strip())
    return '\n\n'.join(parts) + '\n\n---\n'


# --------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.')
    ap.add_argument('--tex', default='main.tex')
    ap.add_argument('--aux', default='main.aux')
    ap.add_argument('--bib', default='refs.bib')
    ap.add_argument('--pandoc', default='pandoc')
    ap.add_argument('-o', '--out', default='main.md')
    args = ap.parse_args()

    root = pathlib.Path(args.dir).resolve()
    tex = (root / args.tex).read_text(encoding='utf-8')
    labels = read_labels(root / args.aux)

    with tempfile.NamedTemporaryFile('w', suffix='.tex', dir=root, delete=False,
                                     encoding='utf-8') as fh:
        fh.write(preprocess(tex))
        staged = pathlib.Path(fh.name)
    try:
        proc = subprocess.run(
            [args.pandoc, '-f', 'latex', '-t', 'gfm', '--wrap=none',
             f'--bibliography={args.bib}', '--citeproc', staged.name],
            cwd=root, capture_output=True, text=True)
    finally:
        staged.unlink(missing_ok=True)
    if proc.returncode:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    md = proc.stdout
    md = fix_tables(md)
    md = fix_figures(md, labels)
    md = fix_refs(md, labels)
    md = unwrap_math(md)
    md = number_headings(md, labels, tex)
    md = md.replace('<span class="nocase">', '').replace('</span>', '')
    md = re.sub(r'\n{3,}', '\n\n', md)
    md = front_matter(tex) + md

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.write_text(md, encoding='utf-8')
    print(f'wrote {out} ({len(md.splitlines())} lines)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
