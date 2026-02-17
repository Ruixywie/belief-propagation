"""Convert BP markdown report to GitHub Pages HTML with video, KaTeX, and downloads."""

import re
from pathlib import Path
from markdown_it import MarkdownIt

SRC = Path("belief_propagation_report-zh-CN-dual.md")
OUT = Path("index.html")

# ── 5 MP4 references ────────────────────────────────────────────────
VIDEOS = [
    ("Message Passing Mechanism", "消息传递机制", "MessagePassing.mp4"),
    ("Sum-Product Algorithm", "和积算法", "SumProductAlgorithm.mp4"),
    ("Exact Inference on Trees", "树上的精确推理", "TreeBP.mp4"),
    ("Loopy BP on a Cyclic Graph", "循环图上的循环 BP", "LoopyBP.mp4"),
    ("Belief Convergence", "信念收敛", "BeliefConvergence.mp4"),
]

VIDEO_DIR = "media/videos/bp_scenes/720p30"


def build_video_pattern(en_title: str, cn_title: str, filename: str) -> tuple[re.Pattern, str]:
    """Build regex pattern and replacement for a single video reference."""
    # Match the two-line blockquote format:
    # > *Animation: EN — see `media/videos/bp_scenes/720p30/FILE.mp4`
    # > 动画：CN — 参见 `media/videos/bp_scenes/720p30/FILE.mp4`*
    pattern = re.compile(
        r"> \*Animation: "
        + re.escape(en_title)
        + r" — see `"
        + re.escape(f"{VIDEO_DIR}/{filename}")
        + r"`\n> 动画："
        + re.escape(cn_title)
        + r" — 参见 `"
        + re.escape(f"{VIDEO_DIR}/{filename}")
        + r"`\*",
        re.MULTILINE,
    )
    src = f"{VIDEO_DIR}/{filename}"
    replacement = (
        f'<div class="video-container">\n'
        f'  <p class="video-caption">{en_title} / {cn_title}</p>\n'
        f'  <video controls preload="metadata">\n'
        f'    <source src="{src}" type="video/mp4">\n'
        f"  </video>\n"
        f'  <a class="download-link" href="{src}" download>⬇ 下载视频</a>\n'
        f"</div>"
    )
    return pattern, replacement


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    # ── 1. Remove GitHub placeholder link ────────────────────────────
    text = re.sub(
        r"\[https://github\.com/user-attachments/assets/placeholder-[^\]]*\]\([^)]*\)\n*",
        "",
        text,
    )

    # ── 2. Replace MP4 references with <video> tags ──────────────────
    for en, cn, fname in VIDEOS:
        pat, repl = build_video_pattern(en, cn, fname)
        text = pat.sub(repl, text)

    # ── 3. Markdown → HTML ───────────────────────────────────────────
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(text)

    # ── 4. Post-process: wrap images with download links ─────────────
    body = re.sub(
        r'<img\s+src="([^"]+)"\s+alt="([^"]*)"[^>]*/?>',
        r'<div class="image-container">'
        r'<img src="\1" alt="\2">'
        r'<a class="download-link" href="\1" download>⬇ 下载图片</a>'
        r"</div>",
        body,
    )

    # ── 5. Wrap in HTML template ─────────────────────────────────────
    html = HTML_TEMPLATE.replace("{{BODY}}", body)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Generated {OUT}  ({size_kb:.1f} KB)")


# ── HTML template ────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Belief Propagation Report / 信念传播报告</title>

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

/* ── Images ───────────────────────────────────────── */
.image-container {
  text-align: center;
  margin: 1.5rem 0;
}
.image-container img {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0,0,0,.1);
}

/* ── Videos ───────────────────────────────────────── */
.video-container {
  text-align: center;
  margin: 2rem 0;
  background: #f9fafb;
  border-radius: 8px;
  padding: 1rem;
}
.video-container video {
  max-width: 100%;
  border-radius: 6px;
}
.video-caption {
  font-weight: 600;
  color: var(--caption);
  margin-bottom: .5rem;
}

/* ── Download link ────────────────────────────────── */
.download-link {
  display: inline-block;
  margin-top: .5rem;
  padding: .35rem 1rem;
  font-size: .875rem;
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 4px;
  transition: background .2s, color .2s;
}
.download-link:hover {
  background: var(--accent);
  color: #fff;
  text-decoration: none;
}

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
{{BODY}}
</body>
</html>
"""

if __name__ == "__main__":
    main()
