"""Render paper/blog_post.md into paper/medium/blog_post.html: a page whose article can be copied as rich text
and pasted into Medium (which has no Markdown import). Images and the CodePen embed become highlighted
placeholders, because Medium does not upload pasted local images or embed pasted links."""

import argparse
import re
import subprocess
from pathlib import Path

from geomap.report import ROOT

SRC = ROOT / "paper" / "blog_post.md"
OUT = ROOT / "paper" / "medium" / "blog_post.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    s = src.read_text().replace("\n---\n", "\n")
    images = []

    def img(m):
        path = (src.parent / m.group(2)).resolve().relative_to(ROOT)
        images.append(str(path))
        return f"**[ IMAGE {len(images)}: upload {path} ]**\n\n*{m.group(1)}*"

    s = re.sub(r"^!\[(.*?)\]\((.*?)\)[ \t]*$", img, s, flags=re.M)
    s = re.sub(r"^(https://codepen\.io/\S+)[ \t]*$",
               r"**[ EMBED: paste \1 alone on an empty line, then press Enter ]**", s, flags=re.M)
    body = subprocess.run(["pandoc", "-f", "markdown-implicit_figures", "-t", "html5"], input=s,
                          capture_output=True, text=True, check=True).stdout
    files = "".join(f"<li><code>{p}</code></li>" for p in images)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Medium copy</title><style>
:root{{--bg:#fff;--ink:#222;--muted:#666;--accent:#2a78d6;--note:#fff3c4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--ink:#e8e8e8;--muted:#9aa;--accent:#3987e5;--note:#4a4020}}}}
body{{background:var(--bg);color:var(--ink);font:18px/1.6 Georgia,serif;max-width:700px;margin:0 auto;padding:16px}}
.bar{{font:15px/1.5 system-ui,sans-serif;border:1px solid var(--muted);border-radius:8px;padding:12px 16px;margin:16px 0 32px}}
.bar ol{{margin:6px 0 0;padding-left:20px}} .bar ul{{margin:4px 0;font-size:13px}}
button{{font:600 15px system-ui;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}}
#ok{{margin-left:10px;color:var(--muted)}} article p > strong:only-child{{background:var(--note)}}
</style></head><body>
<div class="bar"><button id="copy">Copy article</button><span id="ok"></span>
<ol><li>Click <b>Copy article</b>, open a new Medium story, click in the title, paste.</li>
<li>At each highlighted line: delete it, click the <b>+</b> on the empty line, upload the named file, and paste the
italic line under it as the caption. For the CodePen line: paste the link alone on the line and press Enter.</li>
<li>Check the headings: Medium has two sizes; select a heading and use the big or small <b>T</b> if needed.</li></ol>
Files to upload, in order (in the repo folder):<ul>{files}</ul></div>
<article id="post">{body}</article>
<script>
document.getElementById('copy').onclick=()=>{{const r=document.createRange();r.selectNodeContents(document.getElementById('post'));
const s=getSelection();s.removeAllRanges();s.addRange(r);const ok=document.execCommand('copy');s.removeAllRanges();
document.getElementById('ok').textContent=ok?'Copied. Paste into Medium.':'Select the article text and press Cmd+C.';}};
</script></body></html>""")
    print(out, f"{len(images)} images")


if __name__ == "__main__":
    main()
