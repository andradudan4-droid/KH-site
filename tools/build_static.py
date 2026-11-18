"""Render the Flask pages to plain HTML for Cloudflare.

Run from the repo root after changing any page in app.py:
    python3 tools/build_static.py
Settings for this site are in cloudflare/site.json. Output goes to public/.
"""
import html
import json
import re
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as site  # noqa: E402

OUT = ROOT / "public"
CFG = json.loads((ROOT / "cloudflare" / "site.json").read_text())


def swap_chat_script(html):
    """Replace the old inline chat script with the shared one, keeping the look."""
    boot = f"<script>window.CHAT={json.dumps(CFG['chat'])};</script><script src=\"/chat-frame.js?v=1\"></script>"
    start = html.rindex(CFG.get("chat_script_start", "<script>"))
    end = html.index("</script>", start) + len("</script>")
    return html[:start] + boot + html[end:]


def add_shim(html):
    """Load chat-shim.js first so the page's own chat code keeps working unchanged."""
    boot = f"<script>window.CHAT={json.dumps(CFG['chat'])};</script><script src=\"/chat-shim.js?v=1\"></script>"
    m = re.search(r"<head[^>]*>", html)
    return html[:m.end()] + boot + html[m.end():]


def extra_pages():
    """Service/area pages built from an existing page so they share its design.

    cloudflare/seo_pages.json: {"site", "template", "start", "end", "pages": [{path,title,desc,body}],
    "links": {"file", "before", "html"}}. The part of the template between `start` and `end`
    is replaced by each page's body; title, description, canonical and og tags are set.
    """
    f = ROOT / "cloudflare" / "seo_pages.json"
    if not f.exists():
        return
    cfg = json.loads(f.read_text())
    tpl = (OUT / cfg["template"]).read_text()
    a = tpl.index(cfg["start"]) + len(cfg["start"])
    b = tpl.index(cfg["end"], a)
    head, tail = tpl[:a], tpl[b:]
    sm = (OUT / "sitemap.xml").read_text()
    for pg in cfg["pages"]:
        url = cfg["site"] + pg["path"]
        t, d = html.escape(pg["title"]), html.escape(pg["desc"])
        h = re.sub(r"<title>.*?</title>", f"<title>{t}</title>", head, count=1, flags=re.S)
        h = re.sub(r'<meta name="description" content="[^"]*"\s*/?>', "", h)
        h = re.sub(r'<link rel="canonical" href="[^"]*"\s*/?>', "", h)
        h = re.sub(r'<meta property="og:(title|description|url)" content="[^"]*"\s*/?>', "", h)
        h = h.replace(f"<title>{t}</title>", f'<title>{t}</title><meta name="description" content="{d}">'
                      f'<link rel="canonical" href="{url}"><meta property="og:title" content="{t}">'
                      f'<meta property="og:description" content="{d}"><meta property="og:url" content="{url}">', 1)
        (OUT / (pg["path"].strip("/") + ".html")).write_text(h + pg["body"] + tail)
        if url not in sm:
            sm = sm.replace("</urlset>", f"<url><loc>{url}</loc></url></urlset>")
    (OUT / "sitemap.xml").write_text(sm)
    if cfg.get("links"):
        lf = OUT / cfg["links"]["file"]
        x = lf.read_text()
        i = x.index(cfg["links"]["before"])
        lf.write_text(x[:i] + cfg["links"]["html"] + x[i:])


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    client = site.app.test_client()

    for path, name in CFG["pages"].items():
        res = client.get(path, base_url=CFG.get("base_url", "http://localhost"))
        assert res.status_code == 200, (path, res.status_code)
        html = res.get_data(as_text=True)
        if name in CFG.get("chat_pages", []):
            html = swap_chat_script(html)
        if name in CFG.get("shim_pages", []):
            html = add_shim(html)
        (OUT / name).write_text(html)

    for path, name in CFG.get("files", {}).items():
        res = client.get(path, base_url=CFG.get("base_url", "http://localhost"))
        assert res.status_code == 200, (path, res.status_code)
        (OUT / name).write_bytes(res.get_data())

    tracked = subprocess.run(["git", "ls-files", "-z", "static"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\0")
    for rel in filter(None, tracked):
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)

    extra_pages()

    for f in (ROOT / "cloudflare").iterdir():
        if f.name not in ("site.json", "seo_pages.json"):
            shutil.copy(f, OUT / f.name)
    print("built", len(list(OUT.rglob("*"))), "files into public/")


if __name__ == "__main__":
    main()
