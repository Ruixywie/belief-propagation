"""Convert BP applications report (Markdown) to standalone GitHub Pages HTML with KaTeX."""

import re
from pathlib import Path
from markdown_it import MarkdownIt

SRC = Path("bp_applications_report.md")
OUT = Path("bp_applications_report.html")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    # ── 1. Markdown → HTML ─────────────────────────────────────────────
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(text)

    # ── 2. Wrap in HTML template ───────────────────────────────────────
    html = HTML_TEMPLATE.replace("{{BODY}}", body)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Generated {OUT}  ({size_kb:.1f} KB)")


# ── HTML template (reuses existing report styling) ─────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BP Applications Report / 置信传播应用报告</title>

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false}
    ],
    throwOnError: false
  });"></script>

<style>
/* ── Base ─────────────────────────────────────────── */
:root {
  --bg: #ffffff;
  --fg: #1a1a2e;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --code-bg: #f3f4f6;
  --border: #e5e7eb;
  --caption: #4b5563;
}
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: "Noto Sans SC", "Segoe UI", system-ui, sans-serif;
  line-height: 1.8;
  color: var(--fg);
  background: var(--bg);
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}

/* ── Headings ─────────────────────────────────────── */
h1 { font-size: 2rem; border-bottom: 3px solid var(--accent); padding-bottom: .4em; }
h2 { font-size: 1.5rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: .3em; }
h3 { font-size: 1.25rem; margin-top: 2rem; }
h4 { font-size: 1.1rem; margin-top: 1.5rem; }

/* ── Links ────────────────────────────────────────── */
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Code ─────────────────────────────────────────── */
code {
  font-family: "Fira Code", "Cascadia Code", monospace;
  font-size: .9em;
  background: var(--code-bg);
  padding: .15em .35em;
  border-radius: 3px;
}
pre {
  background: var(--code-bg);
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
}
pre code { background: none; padding: 0; }

/* ── Tables ───────────────────────────────────────── */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  font-size: .95rem;
}
th, td {
  border: 1px solid var(--border);
  padding: .5rem .75rem;
  text-align: left;
}
th { background: var(--code-bg); font-weight: 600; }

/* ── Blockquotes ──────────────────────────────────── */
blockquote {
  border-left: 4px solid var(--accent);
  margin: 1rem 0;
  padding: .5rem 1rem;
  color: var(--caption);
  background: #f0f7ff;
  border-radius: 0 6px 6px 0;
}

/* ── Lists ────────────────────────────────────────── */
ul, ol { padding-left: 1.5rem; }
li { margin-bottom: .3rem; }

/* ── Horizontal rule ──────────────────────────────── */
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ── KaTeX overrides ──────────────────────────────── */
.katex-display { overflow-x: auto; overflow-y: hidden; padding: .5rem 0; }

/* ── Back link ────────────────────────────────────── */
.back-link {
  display: inline-block;
  margin-bottom: 1.5rem;
  padding: .35rem 1rem;
  font-size: .9rem;
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 4px;
  transition: background .2s, color .2s;
}
.back-link:hover {
  background: var(--accent);
  color: #fff;
  text-decoration: none;
}

/* ── Responsive ───────────────────────────────────── */
@media (max-width: 600px) {
  body { padding: 1rem; }
  h1 { font-size: 1.5rem; }
  h2 { font-size: 1.25rem; }
  table { font-size: .85rem; }
}
</style>
</head>
<body>
<a class="back-link" href="index.html">← BP 理论报告 / BP Theory Report</a>
{{BODY}}
</body>
</html>
"""

if __name__ == "__main__":
    main()
