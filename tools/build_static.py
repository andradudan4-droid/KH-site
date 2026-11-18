"""Render the Flask pages to plain HTML for Cloudflare.

Run from the repo root after changing any page in app.py:
    python3 tools/build_static.py
Settings for this site are in cloudflare/site.json. Output goes to public/.
"""
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

    for f in (ROOT / "cloudflare").iterdir():
        if f.name != "site.json":
            shutil.copy(f, OUT / f.name)
    print("built", len(list(OUT.rglob("*"))), "files into public/")


if __name__ == "__main__":
    main()
