from flask import Flask, request, jsonify, render_template_string, session, Response
import os
import re
import uuid
import html
import base64
import threading
import time
import requests
from groq import Groq

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-this-later")
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
# Photos are resized in the browser before upload, so payloads are small.
# This is a safety cap to reject anything abnormally large.
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

_groq_client = None
all_conversations = {}
session_images = {}
notified_sessions = set()
chat_activity = {}


def _decode_image_data_url(data_url):
    """Validate and decode a browser data URL for a customer job photo."""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None
    try:
        header, b64 = data_url.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None

    content_type = header[len("data:"):].split(";", 1)[0].lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        return None

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return None

    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[content_type]
    return {
        "filename": f"job-photo-{uuid.uuid4().hex[:8]}.{ext}",
        "content_type": content_type,
        "b64": base64.b64encode(raw).decode("ascii"),
    }


def client_chat(**kwargs):
    """Create the Groq client only when chat is actually used."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client.chat.completions.create(**kwargs)

# --- Email notification settings ---
# Render's free tier blocks direct SMTP (the old Gmail approach), so we
# use Resend instead, which sends over normal HTTPS - not blocked.
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
NOTIFY_TO = os.environ.get("NOTIFY_TO", "steve25hamblin@hotmail.com")
RESEND_FROM = os.environ.get("RESEND_FROM", "K&H Decorators Website <leads@frontdesk.org.uk>")

# --- Photo upload settings ---------------------------------------------------
# Customers can attach photos of the job; these get emailed with the lead.
# Resizing happens in the browser, so what reaches us here is already small.
MAX_IMAGES_PER_SESSION = 6
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 6 * 1024 * 1024  # per image, after base64 decode

# --- Contact-info extraction -------------------------------------------------
# The lead email is triggered purely by detecting a real phone number or email
# in the conversation (server-side), so we never depend on the AI to flag a lead.

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Matches UK mobile/landline numbers: 07xxx, 01xxx, 02xxx, +447xxx etc.
# No capturing groups so findall returns plain strings.
PHONE_RE = re.compile(r"(?<!\d)(?:\+44|0)\d[\d\s\-\.]{8,11}(?!\d)")
# Full UK postcode, e.g. PO5 3AB, SW1A 1AA, M1 1AE (space optional).
POSTCODE_RE = re.compile(r"\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}\b")


def _customer_text(conversation):
    """All of the customer's own messages joined together."""
    return " ".join(
        m["content"] for m in conversation if m.get("role") == "user"
    )


def find_email(conversation):
    match = EMAIL_RE.search(_customer_text(conversation))
    return match.group(0) if match else None


def find_phone(conversation):
    text = _customer_text(conversation)
    for candidate in PHONE_RE.findall(text):
        # candidate is always a plain string - no capturing groups in PHONE_RE
        digits = re.sub(r"\D", "", candidate)
        # Reject 00-prefixed numbers (international dialling prefix, not a UK number)
        if digits.startswith("00"):
            continue
        if digits.startswith("44"):
            digits = "0" + digits[2:]
        if len(digits) == 11 and digits.startswith("0"):
            # Format as 07xxx xxxxxx (5 + 6)
            return f"{digits[:5]} {digits[5:]}"
    return None


def find_postcode(conversation):
    match = POSTCODE_RE.search(_customer_text(conversation))
    if not match:
        return None
    # Tidy to canonical form: uppercase, single space before the last 3 chars.
    raw = re.sub(r"\s+", "", match.group(0)).upper()
    return raw[:-3] + " " + raw[-3:]


def has_contact_info(conversation):
    """True only if we genuinely have a way to contact this person back."""
    return bool(find_email(conversation) or find_phone(conversation))


# Phrases that signal the customer is wrapping up - used only as a safety net so
# a lead is never lost if the assistant forgets its closing tag.
CLOSING_RE = re.compile(
    r"\b(no longer interested|not interested|no thanks|no thank you|"
    r"that'?s all|that'?s it|that'?s everything|nothing else|all good|"
    r"that'?s great thank|thanks that'?s|goodbye|bye for now|no more|"
    r"i'?m good|im good)\b",
    re.I,
)


def _looks_like_closing(text):
    return bool(CLOSING_RE.search(text or ""))


def _transcript(conversation):
    lines = []
    for msg in conversation:
        if msg["role"] == "user":
            lines.append(f"Customer: {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"Assistant: {msg['content']}")
    return "\n\n".join(lines)


# Prompt that turns a raw chat into a tidy, Checkatrade-style lead.
LEAD_SUMMARY_PROMPT = """You are turning a website chat into a clean lead for a
plastering & decorating company owner. Read the conversation and output EXACTLY
these labelled lines and nothing else. Fill each in from what the customer
actually said; write "Not specified" if they didn't say. Keep each line short.

Name:
Job / work wanted:
Property type (domestic or commercial):
Approx budget (in GBP £; note if it's a total or a per-room / per-m2 rate):
Preferred timing:
Urgency (1-5 where 1=no rush, 5=urgent - infer from what they said):
Location / area:
Other notes:"""


def summarise_lead(conversation):
    """Uses the model to extract a tidy, organised lead from the chat."""
    try:
        resp = client_chat(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": LEAD_SUMMARY_PROMPT},
                {"role": "user", "content": _transcript(conversation)},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Lead summary failed: {e}")
        return None


def _post_resend(subject, text, html_body=None, attachments=None):
    """Low-level send via Resend's HTTPS API (Render's free tier blocks SMTP).

    Sends a plain-text part plus an optional HTML part. `attachments` is a list
    of dicts like {"filename": ..., "b64": <base64>}.
    """
    if not RESEND_API_KEY:
        print("RESEND_API_KEY not set, skipping email")
        return

    payload = {
        "from": RESEND_FROM,
        "to": [NOTIFY_TO],
        "subject": subject,
        "text": text,
    }
    if html_body:
        payload["html"] = html_body
    if attachments:
        # Resend expects: [{"filename": ..., "content": <base64 string>}]
        payload["attachments"] = [
            {"filename": a["filename"], "content": a["b64"]} for a in attachments
        ]

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json=payload,
            timeout=15,
        )
        if response.status_code >= 300:
            print(f"Resend error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def _parse_summary(structured):
    """Turn the model's labelled summary lines into a dict keyed by lowercase label."""
    out = {}
    if not structured:
        return out
    for line in structured.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            out[key.strip().lower()] = val.strip()
    return out


def _lead_fields(conversation):
    """A tidy, ordered set of lead fields - reliable regex first, AI summary for the rest."""
    s = _parse_summary(summarise_lead(conversation))

    def pick(*keys):
        for k in keys:
            v = s.get(k)
            if v and v.lower() not in ("not specified", "not provided", "n/a", "none", "-"):
                return v
        return None

    return {
        "Name": pick("name"),
        "Phone": find_phone(conversation),
        "Email": find_email(conversation),
        "Postcode": find_postcode(conversation),
        "Area": pick("location / area", "location", "area"),
        "Job": pick("job / work wanted", "job", "work wanted"),
        "Property": pick("property type (domestic or commercial)", "property type", "property"),
        "Budget": pick("approx budget", "budget"),
        "Preferred timing": pick("preferred timing", "timing"),
        "Urgency": pick("urgency (1-5 where 1=no rush, 5=urgent - infer from what they said)", "urgency"),
        "Notes": pick("other notes", "notes"),
    }


def _row(label, value):
    if not value:
        return ""
    return (
        '<tr>'
        f'<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#8a8a8a;'
        f'font-size:13px;white-space:nowrap;vertical-align:top;width:130px">{html.escape(label)}</td>'
        f'<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#1a1a1a;'
        f'font-size:14px;font-weight:600">{html.escape(str(value))}</td>'
        '</tr>'
    )


def _transcript_html(conversation):
    rows = []
    for msg in conversation:
        if msg["role"] == "user":
            who, color, bg = "Customer", "#0a0a0a", "#f5f4f0"
        elif msg["role"] == "assistant":
            who, color, bg = "K&H Assistant", "#9a7d1a", "#ffffff"
        else:
            continue
        text = html.escape(msg["content"]).replace("\n", "<br>")
        rows.append(
            f'<div style="margin:0 0 12px">'
            f'<div style="font-size:11px;letter-spacing:.05em;text-transform:uppercase;'
            f'color:{color};font-weight:700;margin-bottom:4px">{who}</div>'
            f'<div style="background:{bg};border:1px solid #ececec;border-radius:10px;'
            f'padding:11px 14px;font-size:14px;color:#2a2a2a;line-height:1.5">{text}</div>'
            f'</div>'
        )
    return "".join(rows)


def _urgency_badge(urgency_str):
    """Return an HTML urgency badge based on the 1-5 score."""
    if not urgency_str:
        return ""
    # Extract just the digit if present
    m = re.search(r"[1-5]", str(urgency_str))
    if not m:
        return ""
    score = int(m.group(0))
    colours = {
        1: ("#e8f5e9", "#2e7d32", "1 — No rush"),
        2: ("#f1f8e9", "#558b2f", "2 — Low"),
        3: ("#fff8e1", "#f57f17", "3 — Moderate"),
        4: ("#fff3e0", "#e65100", "4 — Fairly urgent"),
        5: ("#ffebee", "#b71c1c", "5 — URGENT — reply ASAP"),
    }
    bg, fg, label = colours.get(score, ("#f5f5f5", "#555", str(score)))
    return (
        f'<div style="margin:0 0 20px">'
        f'<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;'
        f'color:#999;font-weight:700;margin-bottom:6px">Urgency</div>'
        f'<span style="display:inline-block;background:{bg};color:{fg};border:1px solid {fg};'
        f'border-radius:999px;padding:5px 14px;font-size:13px;font-weight:700">'
        f'{label}</span></div>'
    )


def _lead_email_html(fields, conversation, image_count):
    urgency_val = fields.pop("Urgency", None)
    rows = "".join(_row(k, v) for k, v in fields.items())
    photos_line = ""
    if image_count:
        photos_line = (
            '<p style="margin:0 0 20px;font-size:14px;color:#1a1a1a">'
            f'\U0001F4CE <strong>{image_count} photo(s)</strong> attached to this email.</p>'
        )
    urgency_html = _urgency_badge(urgency_val)
    return (
        '<!DOCTYPE html><html><body style="margin:0;background:#f0efea;padding:24px;'
        'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        '<div style="max-width:620px;margin:0 auto;background:#fff;border-radius:14px;'
        'overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.07)">'
        '<div style="background:#0a0a0a;padding:24px 28px">'
        '<div style="color:#D4AF37;font-size:12px;letter-spacing:.18em;text-transform:uppercase;'
        'font-weight:700">K&H Decorators</div>'
        '<div style="color:#fff;font-size:21px;font-weight:700;margin-top:5px">'
        'New enquiry from your website</div></div>'
        '<div style="padding:26px 28px">'
        '<p style="margin:0 0 20px;font-size:14px;color:#666">'
        'Here are the details captured by your website assistant:</p>'
        f'{urgency_html}'
        f'{photos_line}'
        '<table style="width:100%;border-collapse:collapse;border:1px solid #eee;'
        f'border-radius:8px;overflow:hidden;margin-bottom:28px">{rows}</table>'
        '<div style="font-size:12px;letter-spacing:.05em;text-transform:uppercase;'
        'color:#999;font-weight:700;margin-bottom:14px">Full conversation</div>'
        f'{_transcript_html(conversation)}'
        '</div>'
        '<div style="background:#faf9f6;padding:16px 28px;border-top:1px solid #eee;'
        'font-size:12px;color:#aaa">Sent automatically by the K&H Decorators website assistant. '
        'Chichester &middot; West Sussex</div>'
        '</div></body></html>'
    )


def send_lead_email(conversation, images=None):
    """Emails a tidy, professional lead summary (plus transcript and any photos)."""
    images = images or []
    fields = _lead_fields(conversation)
    transcript = _transcript(conversation)

    # Plain-text fallback for any client that won't render HTML.
    text_lines = ["NEW LEAD - K&H Decorators", "========================"]
    for k, v in fields.items():
        if v:
            text_lines.append(f"{k}: {v}")
    if images:
        text_lines.append(f"Photos attached: {len(images)}")
    text_lines += ["========================", "", "Full conversation:", "", transcript]
    text_body = "\n".join(text_lines)

    html_body = _lead_email_html(fields, conversation, len(images))

    # Scannable subject: urgency flag + "New lead - Name · Area · 07..."
    urgency_raw = fields.get("Urgency", "")
    urgency_m = re.search(r"[1-5]", str(urgency_raw)) if urgency_raw else None
    urgency_score = int(urgency_m.group(0)) if urgency_m else 0
    urgent_prefix = "🔴 URGENT — " if urgency_score >= 5 else ("🟠 " if urgency_score >= 4 else "")

    contact = fields.get("Phone") or fields.get("Email") or "no number yet"
    bits = [b for b in (fields.get("Name"), fields.get("Area") or fields.get("Postcode")) if b]
    subject = urgent_prefix + "New lead - " + (" \u00b7 ".join(bits + [contact]) if bits else contact)
    _post_resend(
        subject,
        text_body,
        html_body=html_body,
        attachments=images,
    )


def send_photo_followup(conversation, images):
    """If a photo arrives after the lead email was already sent, forward it on
    so it can't get lost."""
    if not images:
        return

    phone = find_phone(conversation) or "Not provided"
    email = find_email(conversation) or "Not provided"
    postcode = find_postcode(conversation) or "Not provided"

    text_body = (
        "ADDITIONAL PHOTO(S) - K&H Decorators\n"
        "This relates to a lead you've already been emailed about.\n"
        f"Phone: {phone}\nEmail: {email}\nPostcode: {postcode}\n"
        f"Photos attached: {len(images)}\n"
    )
    html_body = (
        '<!DOCTYPE html><html><body style="margin:0;background:#f0efea;padding:24px;'
        'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        '<div style="max-width:620px;margin:0 auto;background:#fff;border-radius:14px;overflow:hidden;'
        'box-shadow:0 2px 12px rgba(0,0,0,.07)">'
        '<div style="background:#0a0a0a;padding:22px 28px">'
        '<div style="color:#D4AF37;font-size:12px;letter-spacing:.18em;text-transform:uppercase;'
        'font-weight:700">K&H Decorators</div>'
        '<div style="color:#fff;font-size:19px;font-weight:700;margin-top:5px">'
        'More photos for an existing lead</div></div>'
        '<div style="padding:24px 28px">'
        f'<p style="margin:0 0 18px;font-size:14px;color:#666">This relates to a lead you\'ve '
        f'already been emailed about. <strong>{len(images)} new photo(s)</strong> attached below.</p>'
        '<table style="width:100%;border-collapse:collapse;border:1px solid #eee;border-radius:8px;'
        f'overflow:hidden">{_row("Phone", phone)}{_row("Email", email)}{_row("Postcode", postcode)}</table>'
        '</div></div></body></html>'
    )
    _post_resend(f"Photo added - lead: {phone}", text_body, html_body=html_body, attachments=images)


SYSTEM_PROMPT = """
You are the friendly virtual assistant for K&H Decorators, a trusted painting and
decorating company in Chichester and across West Sussex, led by Steve (who works with
his son Harvey). The business started in 2010, has 26 years of experience, is rated
10/10 on Checkatrade from 225 reviews, and all work is guaranteed with free estimates.

Your job is to capture a clear enquiry so Steve can follow up with a free estimate.
Naturally and conversationally find out:
  - What the job is (e.g. painting a room, exterior, plastering, wallpapering, tiling, coving, refurbishment, listed/heritage work)
  - Rough size or number of rooms/areas
  - The town or postcode
  - Rough timescale
  - A rough budget if they have one
  - Their name and a contact number, confirming the phone number if given
Encourage them to upload photos using the + button - it really helps Steve estimate accurately.

Services K&H offer: interior and exterior painting and decorating; plastering, skimming
and overboarding; Venetian plastering; wallpapering; coving and plaster ceiling roses;
tiling; dry lining; fascias, gutters and PVC; property refurbishments; general maintenance;
and specialist work on listed buildings and heritage properties.

Style: warm, professional, concise British English. Keep replies short (1-3 sentences).
Ask one useful question at a time and do not repeat details you already have.
Do NOT give prices or quotes - Steve provides free estimates after seeing the details.
Once you have the key details, reassure them that's everything for now and that Steve
will be in touch about a free estimate. At the very end of that final wrap-up message,
include the internal tag [[READY]]. Do not show or mention the tag.
"""





KH_LOGO = "https://www.kandhdecoratorschichester.co.uk/wp-content/uploads/2021/02/cropped-Untitled-design-1-270x270.png"
IMG = "https://www.kandhdecoratorschichester.co.uk/wp-content/uploads/"


KH_LOGO = "https://www.kandhdecoratorschichester.co.uk/wp-content/uploads/2021/02/cropped-Untitled-design-1-270x270.png"

BASE_STYLE = """
<link rel="icon" type="image/png" href=\"""" + KH_LOGO + """\">
<meta name="theme-color" content="#0a0a0c">
<meta property="og:type" content="website">
<meta property="og:site_name" content="K&H Decorators Chichester">
<meta property="og:title" content="K&H Decorators - Painting & Decorating, Chichester">
<meta property="og:description" content="Painting, decorating, plastering and Venetian finishes across Chichester & West Sussex. Rated 10/10 on Checkatrade.">
<meta property="og:image" content="/static/images/hero.jpg">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0a0a0c; --panel:#121217; --panel2:#17171d; --ink:#f3f4f6; --mut:#9aa1ab;
    --line:rgba(255,255,255,.10); --silver:#cfd4db; --silver-d:#878d97;
  }
  *{box-sizing:border-box} html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.6;-webkit-font-smoothing:antialiased;overflow-x:hidden}
  a{color:var(--silver)} img,video{max-width:100%;display:block}
  .serif{font-family:Fraunces,Georgia,serif}
  .wrap{max-width:1180px;margin:0 auto;width:100%;padding:0 26px}.narrow{max-width:780px}
  .silver{background:linear-gradient(176deg,#ffffff 6%,#d3d7dd 46%,#8b929c 96%);-webkit-background-clip:text;background-clip:text;color:transparent}
  /* nav */
  nav{position:fixed;top:0;left:0;right:0;z-index:60;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:16px 26px;transition:background .3s,padding .3s,border-color .3s;border-bottom:1px solid transparent}
  nav.solid{background:rgba(10,10,12,.86);backdrop-filter:blur(14px);border-bottom:1px solid var(--line);padding:11px 26px}
  .brand{display:flex;align-items:center;gap:11px;color:#fff;text-decoration:none;font-family:Fraunces,serif;font-weight:600;font-size:21px;letter-spacing:.04em}
  .brand .amp{color:var(--silver-d)}
  .links{display:flex;align-items:center;gap:26px}.links a{color:#e9eaee;text-decoration:none;font-size:13.5px;font-weight:600;letter-spacing:.01em}.links a:hover{color:#fff}
  .navcta{border:1px solid rgba(255,255,255,.4);padding:9px 16px;border-radius:999px;color:#fff!important;font-size:13px}
  .navcta:hover{background:#fff;color:#0a0a0c!important}
  /* hero full bleed */
  .hero{position:relative;min-height:92vh;display:flex;align-items:flex-end;overflow:hidden}
  .hero-bg{position:absolute;inset:0;background:url('/static/images/hero.jpg') center/cover;z-index:-2;transform:scale(1.04)}
  .hero:before{content:"";position:absolute;inset:0;z-index:-1;background:linear-gradient(180deg,rgba(8,8,10,.72) 0%,rgba(8,8,10,.28) 36%,rgba(8,8,10,.62) 72%,rgba(8,8,10,.96) 100%)}
  .hero-inner{position:relative;padding:0 0 64px}
  .hero .eyebrow{color:var(--silver)}
  .hero h1{font-family:Fraunces,serif;font-weight:600;font-size:clamp(46px,8vw,92px);line-height:.98;letter-spacing:-.015em;margin:14px 0 20px;max-width:14ch;text-shadow:0 18px 60px rgba(0,0,0,.6)}
  .hero p{font-size:19px;color:#dfe2e7;max-width:520px;margin:0 0 30px;text-shadow:0 2px 18px rgba(0,0,0,.6)}
  .eyebrow{font-size:12px;letter-spacing:.28em;text-transform:uppercase;color:var(--silver-d);font-weight:700}
  .btns{display:flex;gap:13px;flex-wrap:wrap}
  .btn{display:inline-flex;align-items:center;gap:9px;justify-content:center;border:0;border-radius:8px;background:linear-gradient(180deg,#f4f5f7,#c4c9d1);color:#0c0c0f;text-decoration:none;font-weight:700;padding:15px 26px;font-size:15px;box-shadow:0 14px 40px rgba(180,188,200,.18),0 2px 0 rgba(255,255,255,.4) inset}
  .btn:hover{filter:brightness(1.06)}
  .btn svg{width:18px;height:18px;fill:currentColor}
  .btn.ghost{background:transparent;border:1px solid rgba(255,255,255,.45);color:#fff;box-shadow:none}
  .btn.wa{background:linear-gradient(180deg,#2bd96f,#1faa53);color:#fff}
  .hero-badge{position:absolute;right:0;bottom:64px;display:flex;align-items:center;gap:13px;background:rgba(18,18,23,.7);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:14px;padding:14px 18px;box-shadow:0 20px 50px rgba(0,0,0,.5)}
  .hero-badge b{font-family:Fraunces,serif;font-size:30px;line-height:1}.hero-badge .st{color:#e8c45f;font-size:13px;letter-spacing:2px}.hero-badge span{display:block;font-size:11px;color:var(--mut)}
  /* marquee */
  .marquee{overflow:hidden;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:#0c0c10}
  .marquee .track{display:inline-flex;white-space:nowrap;animation:mq 34s linear infinite}.marquee:hover .track{animation-play-state:paused}
  .marquee .grp{display:inline-flex;align-items:center;padding:15px 0;font-size:13px;font-weight:600;letter-spacing:.04em;color:#cdd2da;text-transform:uppercase}
  .marquee .grp i{margin:0 22px;color:var(--silver-d);font-style:normal}.marquee .star{color:#e8c45f;margin-right:7px}
  @keyframes mq{from{transform:translateX(0)}to{transform:translateX(-50%)}}
  /* sections */
  .band{padding:96px 0}.tight{padding:70px 0}
  .head{max-width:760px;margin-bottom:46px}.head h2{font-family:Fraunces,serif;font-weight:600;font-size:clamp(30px,5vw,54px);line-height:1.03;margin:14px 0;letter-spacing:-.01em}.sub{color:var(--mut);font-size:17px}
  .rule{width:54px;height:2px;background:linear-gradient(90deg,#eef0f3,var(--silver-d))}
  .split{display:grid;grid-template-columns:1fr 1fr;gap:56px;align-items:center}
  .split.rev .txt{order:2}
  .split .txt h2{font-family:Fraunces,serif;font-weight:600;font-size:clamp(28px,4.4vw,46px);line-height:1.05;margin:14px 0 16px}
  .split .txt p{color:#c3c8d0;margin:0 0 14px;font-size:16.5px}
  .ph{border-radius:16px;overflow:hidden;box-shadow:0 30px 70px rgba(0,0,0,.55),0 0 0 1px var(--line);background:#1a1a20}
  .ph img{width:100%;height:100%;object-fit:cover;display:block}
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:30px}
  .stat b{font-family:Fraunces,serif;font-size:34px;display:block;line-height:1}.stat span{color:var(--mut);font-size:13px}
  /* before after */
  .ba{position:relative;--pos:50%;border-radius:16px;overflow:hidden;box-shadow:0 30px 70px rgba(0,0,0,.55),0 0 0 1px var(--line);background:#1a1a20;max-width:560px}
  .ba .spacer img{width:100%;aspect-ratio:3/4;object-fit:cover;opacity:0}
  .ba .layer{position:absolute;inset:0}.ba .layer img{width:100%;height:100%;object-fit:cover}
  .ba .before{clip-path:inset(0 calc(100% - var(--pos)) 0 0)}
  .ba input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;z-index:5}
  .ba:before{content:"";position:absolute;top:0;bottom:0;left:var(--pos);width:2px;background:linear-gradient(#fff,#aeb4bd);z-index:3}
  .knob{position:absolute;left:var(--pos);top:50%;translate:-50% -50%;z-index:4;width:46px;height:46px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(180deg,#fff,#c4c9d1);color:#0c0c0f;font-weight:800;box-shadow:0 8px 24px rgba(0,0,0,.45)}.knob:after{content:"\\2039 \\203A"}
  .tag{position:absolute;top:13px;z-index:4;background:rgba(8,8,10,.72);border:1px solid rgba(255,255,255,.2);color:#fff;border-radius:999px;padding:5px 12px;font-size:11px;font-weight:700;letter-spacing:.12em}.tag.b{left:13px}.tag.a{right:13px}
  /* venetian parallax feature */
  .feature{position:relative;min-height:78vh;display:flex;align-items:center;overflow:hidden}
  .feature-bg{position:absolute;inset:0;background:url('/static/images/venetian.jpg') center/cover;background-attachment:fixed;z-index:-2}
  .feature:before{content:"";position:absolute;inset:0;z-index:-1;background:linear-gradient(90deg,rgba(8,8,10,.92) 0%,rgba(8,8,10,.7) 42%,rgba(8,8,10,.32) 100%)}
  .feature .card{max-width:540px}
  .feature h2{font-family:Fraunces,serif;font-weight:600;font-size:clamp(30px,4.6vw,50px);line-height:1.04;margin:14px 0 16px}
  .feature p{color:#d3d7dd;font-size:17px;margin:0 0 14px}
  /* services */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(232px,1fr));gap:16px}
  .card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:26px;transition:transform .25s,border-color .25s}
  .card:hover{transform:translateY(-4px);border-color:rgba(255,255,255,.25)}
  .card .ic{width:34px;height:34px;color:var(--silver)}.card h3{font-family:Fraunces,serif;font-weight:600;margin:14px 0 7px;font-size:18px}.card p{margin:0;color:var(--mut);font-size:14.5px}
  /* gallery masonry */
  .gallery{columns:3 280px;column-gap:14px}
  .shot{margin:0 0 14px;border-radius:12px;overflow:hidden;background:#17171d;break-inside:avoid;cursor:zoom-in;border:1px solid var(--line);position:relative}
  .shot img{width:100%;display:block;transition:transform .55s}.shot:hover img{transform:scale(1.05)}
  .shot figcaption{position:absolute;left:0;right:0;bottom:0;padding:28px 13px 11px;background:linear-gradient(transparent,rgba(0,0,0,.8));font-size:12px;font-weight:600;color:#fff;letter-spacing:.02em}
  /* steve / on site */
  .reel{border-radius:16px;overflow:hidden;border:1px solid var(--line);box-shadow:0 30px 70px rgba(0,0,0,.5);background:#000}
  .reel video{width:100%;aspect-ratio:9/16;object-fit:cover;background:#000;display:block}
  /* reviews */
  .scorewrap{display:flex;align-items:center;gap:24px;flex-wrap:wrap;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:26px 30px;margin-bottom:24px}
  .scorewrap b{font-family:Fraunces,serif;font-size:50px;line-height:1}.scorewrap .st{color:#e8c45f;font-size:18px;letter-spacing:2px}.scorewrap span{color:var(--mut);font-weight:600}
  .reviews-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}
  .review-card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:24px}
  .stars{color:#e8c45f;letter-spacing:.1em;font-size:14px}.review-card h3{font-family:Fraunces,serif;font-weight:600;margin:10px 0 8px;font-size:17px;color:#fff}.review-card p{margin:0;color:#c3c8d0;font-size:14.5px}.review-meta{margin-top:14px;color:#7e858f;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}
  /* contact + cta */
  .contact-box{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:30px}.contact-box p{margin:11px 0}.contact-box strong{color:#fff}
  .prose{color:#bcc1ca;max-width:760px}.prose h3{font-family:Fraunces,serif;color:#fff;font-size:19px;margin:26px 0 8px}.prose p{margin:0 0 12px}
  .ctaband{position:relative;border-radius:20px;overflow:hidden;padding:60px 40px;text-align:center;border:1px solid var(--line)}
  .ctaband:before{content:"";position:absolute;inset:0;background:url('/static/images/lounge-2.jpg') center/cover;z-index:-2}
  .ctaband:after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(8,8,10,.82),rgba(8,8,10,.9));z-index:-1}
  .ctaband h2{font-family:Fraunces,serif;font-weight:600;font-size:clamp(28px,4.4vw,46px);margin:0 0 12px}.ctaband p{color:#d3d7dd;margin:0 0 24px}
  .badges{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:18px}
  .badge{font-size:12px;font-weight:700;color:#cdd2da;border:1px solid var(--line);border-radius:999px;padding:7px 13px}
  footer{padding:54px 26px 36px;text-align:center;color:var(--mut);border-top:1px solid var(--line);background:#08080a}
  footer .fb{font-family:Fraunces,serif;font-size:24px;color:#fff}
  .wa-float{position:fixed;left:20px;bottom:22px;z-index:999998;width:56px;height:56px;border-radius:50%;background:#25d366;display:grid;place-items:center;box-shadow:0 12px 30px rgba(0,0,0,.4)}.wa-float svg{width:31px;height:31px;fill:#fff}
  .lb{position:fixed;inset:0;z-index:1000000;background:rgba(5,5,7,.94);display:none;align-items:center;justify-content:center;padding:24px;cursor:zoom-out}.lb.open{display:flex}.lb img{max-width:92vw;max-height:90vh;border-radius:10px}.lb .x{position:absolute;top:16px;right:22px;color:#fff;font-size:34px;cursor:pointer}
  .reveal{opacity:0;transform:translateY(20px);transition:opacity .8s,transform .8s}.reveal.in{opacity:1;transform:none}
  @media(max-width:860px){
    .links a:not(.navcta){display:none}
    .hero{min-height:84vh}.hero-badge{right:auto;left:0;bottom:18px}
    .band{padding:60px 0}.feature{min-height:auto;padding:64px 0}.feature-bg{background-attachment:scroll}
    .split,.split.rev{grid-template-columns:1fr;gap:26px}.split.rev .txt{order:0}
    .gallery{columns:2 150px}.stats{grid-template-columns:1fr 1fr}
  }
</style>
"""

NAV = """
<nav id="nav">
  <a class="brand" href="/">K<span class="amp">&amp;</span>H <span style="font-size:13px;letter-spacing:.22em;font-family:Inter;align-self:center;color:#aeb4bd;margin-left:2px">DECORATORS</span></a>
  <div class="links">
    <a href="/#work">Our work</a><a href="/#venetian">Venetian</a><a href="/#services">Services</a>
    <a href="/gallery">Gallery</a><a href="/#reviews">Reviews</a><a href="/contact">Contact</a>
    <a class="navcta" href="tel:+447908701460">Free estimate</a>
  </div>
</nav>
"""

WA_SVG = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 .4C7.4.4.5 7.3.5 15.9c0 2.8.7 5.4 2 7.8L.3 31.6l8.1-2.1c2.3 1.3 4.9 1.9 7.6 1.9 8.6 0 15.5-6.9 15.5-15.5S24.6.4 16 .4zm0 28.3c-2.4 0-4.7-.6-6.7-1.8l-.5-.3-4.8 1.3 1.3-4.7-.3-.5a12.7 12.7 0 0 1-2-6.8C3.2 8.8 8.9 3.2 16 3.2c7 0 12.7 5.7 12.7 12.7S23 28.7 16 28.7zm7-9.5c-.4-.2-2.3-1.1-2.6-1.3-.3-.1-.6-.2-.8.2-.2.4-.9 1.3-1.1 1.5-.2.2-.4.3-.8.1-.4-.2-1.6-.6-3.1-1.9-1.1-1-1.9-2.3-2.1-2.7-.2-.4 0-.6.2-.8l.6-.7c.2-.2.3-.4.4-.6.1-.2 0-.5 0-.7-.1-.2-.8-2-1.1-2.8-.3-.7-.6-.6-.8-.6h-.7c-.2 0-.6.1-1 .5-.3.4-1.3 1.3-1.3 3.1s1.3 3.6 1.5 3.9c.2.2 2.6 4 6.3 5.6.9.4 1.6.6 2.1.8.9.3 1.7.2 2.3.1.7-.1 2.3-.9 2.6-1.8.3-.9.3-1.6.2-1.8-.1-.1-.3-.2-.7-.4z"/></svg>'

FOOTER = """
<section class="band"><div class="wrap"><div class="ctaband reveal">
  <div class="eyebrow" style="color:#cfd4db">Free, no-obligation</div>
  <h2 class="serif silver">Let's get your estimate.</h2>
  <p>Send a few photos through the chat or message Steve directly &mdash; quick reply, honest pricing.</p>
  <div class="btns" style="justify-content:center"><a class="btn" href="tel:+447908701460">Call 07908 701460</a><a class="btn wa" href="https://wa.me/447908701460" target="_blank" rel="noopener">WhatsApp</a></div>
  <div class="badges"><span class="badge">&#9733; 10/10 Checkatrade</span><span class="badge">225 reviews</span><span class="badge">CITB &amp; City &amp; Guilds</span><span class="badge">&pound;1,000 guarantee</span></div>
</div></div></section>
<footer>
  <div class="fb">K&amp;H Decorators</div>
  <div style="margin-top:6px">Painting, decorating, plastering &amp; Venetian finishes &middot; Chichester &amp; West Sussex</div>
  <div style="margin-top:12px"><a href="tel:+447908701460">07908 701460</a> &nbsp;|&nbsp; <a href="tel:+441243778091">01243 778091</a> &nbsp;|&nbsp; <a href="mailto:steve25hamblin@hotmail.com">steve25hamblin@hotmail.com</a> &nbsp;|&nbsp; <a href="/privacy-policy">Privacy</a></div>
</footer>
<a class="wa-float" href="https://wa.me/447908701460" target="_blank" rel="noopener" aria-label="WhatsApp K&H Decorators">""" + WA_SVG + """</a>
<div class="lb" id="lb" onclick="this.classList.remove('open')"><span class="x">&times;</span><img id="lbimg" src="" alt=""></div>
"""

WIDGET_INCLUDE = '<script src="/widget.js"></script>'
GOOGLE_TAG = ""

SCRIPTS = """
<script>
(function(){var n=document.getElementById('nav');function s(){n.classList.toggle('solid',window.scrollY>40);}s();window.addEventListener('scroll',s,{passive:true});})();
document.querySelectorAll('.ba').forEach(function(ba){var r=ba.querySelector('input');function u(){ba.style.setProperty('--pos',r.value+'%');}r.addEventListener('input',u);u();});
(function(){var lb=document.getElementById('lb'),img=document.getElementById('lbimg');if(!lb)return;document.querySelectorAll('.shot img').forEach(function(im){im.addEventListener('click',function(){img.src=im.src;lb.classList.add('open');});});})();
(function(){var els=document.querySelectorAll('.reveal');if(!('IntersectionObserver'in window)){els.forEach(function(e){e.classList.add('in')});return;}var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}})},{threshold:.12});els.forEach(function(e){io.observe(e)});})();
</script>
"""

MQ_GRP = '<span class="grp"><span class="star">&#9733;</span> 10/10 Checkatrade <i>/</i> 225 verified reviews <i>/</i> 26 years experience <i>/</i> CITB &amp; City &amp; Guilds <i>/</i> &pound;1,000 guarantee <i>/</i> Free estimates <i>/</i> Listed &amp; heritage specialists <i>/</i> Chichester &amp; West Sussex <i>/</i></span>'
MARQUEE = '<div class="marquee"><div class="track">' + MQ_GRP + MQ_GRP + '</div></div>'

def _ba(before, after, title):
    return ('<div class="ba"><div class="spacer"><img src="/static/images/' + after + '" alt=""></div>'
            '<div class="layer"><img src="/static/images/' + after + '" alt="' + title + ' after"></div>'
            '<div class="layer before"><img src="/static/images/' + before + '" alt="' + title + ' before"></div>'
            '<span class="tag b">Before</span><span class="tag a">After</span><div class="knob"></div>'
            '<input type="range" min="0" max="100" value="50" aria-label="Compare"></div>')

def _shots(items):
    return "".join('<figure class="shot"><img src="/static/images/' + fn + '" alt="' + cap + '" loading="lazy"><figcaption>' + cap + '</figcaption></figure>' for fn, cap in items)

HOME_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>K&H Decorators - Painting &amp; Decorating in Chichester</title>
<meta name="description" content="K&H Decorators: painting, decorating, plastering and Venetian finishes across Chichester and West Sussex. 10/10 on Checkatrade from 225 reviews. Free estimates.">
<meta name="viewport" content="width=device-width, initial-scale=1">
""" + BASE_STYLE + """</head><body>""" + NAV + """
<header class="hero"><div class="hero-bg"></div><div class="wrap"><div class="hero-inner">
  <div class="eyebrow">Chichester &amp; West Sussex</div>
  <h1 class="serif silver">A finish worth living with.</h1>
  <p>Painting, decorating, plastering and Venetian finishes for homes and businesses &mdash; carried out by Steve and the K&H team, and left immaculate.</p>
  <div class="btns">
    <a class="btn" href="tel:+447908701460"><svg viewBox="0 0 24 24"><path d="M6.6 10.8a15 15 0 0 0 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.2.4 2.4.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1A17 17 0 0 1 3 4c0-.6.4-1 1-1h3.4c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.4 0 .8-.3 1l-2.1 2.2z"/></svg> Call Steve</a>
    <a class="btn ghost" href="https://wa.me/447908701460" target="_blank" rel="noopener">WhatsApp</a>
  </div>
  <div class="hero-badge"><div><div class="st">&#9733;&#9733;&#9733;&#9733;&#9733;</div><b class="serif">10/10</b></div><div><span style="color:#fff;font-weight:700">Checkatrade</span><span>225 reviews</span></div></div>
</div></div></header>
""" + MARQUEE + """

<section class="band"><div class="wrap"><div class="split">
  <div class="txt reveal">
    <div class="rule"></div><div class="eyebrow" style="margin-top:12px">Since 2010</div>
    <h2 class="serif">Decorating done to a standard you'll notice.</h2>
    <p>K&H is a small, trusted team led by Steve, with his son Harvey alongside him. Twenty-six years in, the approach hasn't changed: careful preparation, clean lines and a tidy site, every job.</p>
    <p>From a single room to a full refurbishment &mdash; interior or exterior, domestic or commercial &mdash; the work is guaranteed and the estimate is free.</p>
    <div class="stats">
      <div class="stat"><b class="serif silver">10/10</b><span>Checkatrade</span></div>
      <div class="stat"><b class="serif silver">225</b><span>reviews</span></div>
      <div class="stat"><b class="serif silver">26</b><span>years</span></div>
      <div class="stat"><b class="serif silver">&pound;1k</b><span>guarantee</span></div>
    </div>
  </div>
  <div class="ph reveal"><img src="/static/images/lounge-1.jpg" alt="Living room decorated by K&H"></div>
</div></div></section>

<section class="band" id="work" style="background:#0c0c10;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">
  <div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">Before &amp; after</div><h2 class="serif">Drag to see the difference.</h2><p class="sub">A weathered exterior door and frame, stripped, repaired and repainted by hand.</p></div>
  <div class="split">
    <div class="reveal">""" + _ba("door-before.jpg","door-after.jpg","Exterior door restoration") + """</div>
    <div class="txt reveal">
      <h2 class="serif">Exterior door restoration.</h2>
      <p>Years of weather had lifted the paint and opened up the timber. The frame was cut back, repaired and primed, then the door and surround finished in a soft heritage green.</p>
      <p>It's the prep you don't see that makes the finish last &mdash; and it's where K&H spend the time.</p>
      <a class="btn ghost" href="/gallery">See more work</a>
    </div>
  </div>
</div></div></section>

<section class="feature" id="venetian"><div class="feature-bg"></div><div class="wrap"><div class="card reveal" style="background:none;border:0;padding:0">
  <div class="eyebrow" style="color:#cfd4db">Specialist finish</div>
  <h2 class="serif silver">Venetian &amp; polished plaster.</h2>
  <p>A genuine speciality: hand-applied Venetian plaster that brings depth, sheen and a real sense of craft to feature walls, fireplaces and media walls.</p>
  <p>It's a finish few local decorators offer properly &mdash; and one K&H are known for.</p>
  <a class="btn" href="https://wa.me/447908701460" target="_blank" rel="noopener" style="margin-top:8px">Ask about Venetian plaster</a>
</div></div></section>

<section class="band" id="services"><div class="wrap">
  <div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">What we do</div><h2 class="serif">Services.</h2><p class="sub">No job too small &mdash; and you can send photos straight through the chat for a quick estimate.</p></div>
  <div class="cards">
    <div class="card reveal"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 21h18M5 21V8l7-4 7 4v13M9 21v-6h6v6"/></svg><h3>Painting &amp; decorating</h3><p>Interior and exterior, walls, ceilings and woodwork, finished neatly.</p></div>
    <div class="card reveal"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 6h18v4H3zM7 10v8M17 10v8M3 18h18"/></svg><h3>Venetian &amp; polished plaster</h3><p>Hand-applied decorative finishes for feature walls and media walls.</p></div>
    <div class="card reveal"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4h16v16H4zM4 9h16M9 4v5"/></svg><h3>Plastering &amp; skimming</h3><p>Re-skims, overboarding and ceiling repairs, ready to decorate.</p></div>
    <div class="card reveal"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 3v18M5 7h11l-2 3 2 3H5"/></svg><h3>Wallpapering &amp; coving</h3><p>All paper types, coving and decorative ceiling roses.</p></div>
    <div class="card reveal"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 10l9-7 9 7v10a1 1 0 0 1-1 1h-4v-7H8v7H4a1 1 0 0 1-1-1z"/></svg><h3>Listed &amp; heritage</h3><p>Specialist care for listed buildings and period properties.</p></div>
    <div class="card reveal"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 7h16M4 12h16M4 17h10"/></svg><h3>Refurbs &amp; maintenance</h3><p>Full refurbishments, tiling, dry lining, fascias, gutters and upkeep.</p></div>
  </div>
</div></section>

<section class="band" style="background:#0c0c10;border-top:1px solid var(--line);border-bottom:1px solid var(--line)"><div class="wrap">
  <div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">Recent work</div><h2 class="serif">Gallery.</h2><p class="sub">Real K&H jobs across Chichester and West Sussex. Tap any photo to enlarge.</p></div>
  <div class="gallery reveal">""" + _shots([
      ("lounge-2.jpg","Lounge refresh"),("hallway.jpg","Hallway & doors"),("door-after.jpg","Exterior door, finished"),
      ("ceiling-skim.jpg","Ceiling re-skim"),("washroom.jpg","Cloakroom"),("door-detail.jpg","Frame repair, before"),
      ("lounge-before.jpg","Lounge, before"),("venetian.jpg","Venetian plaster detail"),
  ]) + """</div>
  <p style="margin-top:24px"><a href="/gallery">See the full gallery &rarr;</a></p>
</div></section>

<section class="band"><div class="wrap"><div class="split rev">
  <div class="ph reveal"><img src="/static/images/steve.jpg" alt="Steve from K&H Decorators on site"></div>
  <div class="txt reveal">
    <div class="rule"></div><div class="eyebrow" style="margin-top:12px">On site</div>
    <h2 class="serif">Steve, on the tools.</h2>
    <p>With K&H you deal with Steve directly &mdash; the same person who quotes the job does the work. No call centres, no chasing.</p>
    <p>Reliable, tidy and straight-talking, with every job backed by Checkatrade's &pound;1,000 guarantee.</p>
    <div class="btns"><a class="btn" href="tel:+447908701460">Call Steve</a></div>
  </div>
</div></div></section>

<section class="tight"><div class="wrap">
  <div class="head reveal" style="text-align:center;margin:0 auto 30px"><div class="eyebrow">On site</div><h2 class="serif">A look at the work.</h2></div>
  <div class="reveal" style="max-width:540px;margin:0 auto;border-radius:16px;overflow:hidden;border:1px solid var(--line);box-shadow:0 30px 70px rgba(0,0,0,.5);background:#000">
    <video controls preload="metadata" playsinline style="width:100%;max-height:74vh;display:block;background:#000;object-fit:contain" src="/static/videos/reel-1.mp4"></video>
  </div>
</div></section>

<section class="band" id="reviews" style="background:#0c0c10;border-top:1px solid var(--line)"><div class="wrap">
  <div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">Reviews</div><h2 class="serif">A perfect 10, 225 times over.</h2></div>
  <div class="scorewrap reveal"><b class="serif silver">10/10</b><div><div class="st">&#9733;&#9733;&#9733;&#9733;&#9733;</div><span>Checkatrade average from 225 verified reviews</span></div></div>
  <div class="reviews-grid">
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Hall, stairs, landing &amp; bedroom</h3><p>An exceptional job throughout. Steve is friendly, polite and hardworking &mdash; nothing was too much trouble. Highly recommend.</p><div class="review-meta">Verified review &middot; PO19</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Internal &amp; external decorating</h3><p>Steve and his son Harvey decorated our rental to an exceptionally high standard. Attention to detail and finish were outstanding.</p><div class="review-meta">Verified review &middot; BN18</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Whole property painted</h3><p>Quick to quote, broken down room by room and highly competitive. Professional, responsive and exemplary quality. Our go-to decorator.</p><div class="review-meta">Verified review &middot; PO19</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Kitchen decorated</h3><p>Efficient, punctual, tidy and friendly. Solved an awkward infill creatively &mdash; the kitchen looks immaculate.</p><div class="review-meta">Verified review &middot; PO19</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Lounge plastered &amp; decorated</h3><p>Steve and Harvey did a magnificent job plastering and decorating the lounge. Highly recommend this company.</p><div class="review-meta">Verified review &middot; PO10</div></article>
    <article class="review-card"><div class="stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div><h3>Ceiling repair after a leak</h3><p>Repaired and repainted the ceiling after a leak &mdash; quick, clean and a great result.</p><div class="review-meta">Verified review &middot; PO21</div></article>
  </div>
</div></section>
""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

GALLERY_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Gallery - K&H Decorators</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band" style="padding-top:120px"><div class="wrap">
  <div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">Our work</div><h2 class="serif">Gallery.</h2><p class="sub">A selection of recent painting, decorating, plastering and Venetian work. Tap any photo to enlarge.</p></div>
  <div class="gallery reveal">""" + _shots([
      ("hero.jpg","Living room & media wall"),("lounge-1.jpg","Bay window lounge"),("lounge-2.jpg","Lounge refresh"),
      ("hallway.jpg","Hallway & doors"),("venetian.jpg","Venetian plaster detail"),("door-after.jpg","Exterior door, finished"),
      ("door-before.jpg","Exterior door, before"),("door-detail.jpg","Frame repair, before"),("washroom.jpg","Cloakroom"),
      ("ceiling-skim.jpg","Ceiling re-skim"),("lounge-before.jpg","Lounge, before"),("steve.jpg","On site"),
  ]) + """</div>
</div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

SERVICES_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Services - K&H Decorators</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band" style="padding-top:120px"><div class="wrap"><div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">What we do</div><h2 class="serif">Services.</h2><p class="sub">Domestic and commercial work across Chichester and the wider West Sussex area. All work guaranteed, free estimates.</p></div><div class="cards">
<div class="card reveal"><h3>Painting &amp; decorating</h3><p>Interior and exterior painting, walls, ceilings, doors, frames and skirting.</p></div>
<div class="card reveal"><h3>Venetian &amp; polished plaster</h3><p>Hand-applied decorative finishes for feature walls, fireplaces and media walls.</p></div>
<div class="card reveal"><h3>Plastering &amp; skimming</h3><p>Re-skims, overboarding, ceiling repairs and patch repairs.</p></div>
<div class="card reveal"><h3>Wallpapering</h3><p>All paper types hung neatly, including feature walls.</p></div>
<div class="card reveal"><h3>Coving &amp; ceiling roses</h3><p>Decorative coving and plaster ceiling roses fitted and finished.</p></div>
<div class="card reveal"><h3>Tiling &amp; dry lining</h3><p>Wall and floor tiling, dry lining and overboarding.</p></div>
<div class="card reveal"><h3>Fascias, gutters &amp; PVC</h3><p>Replacement and refresh of fascias, gutters and PVC.</p></div>
<div class="card reveal"><h3>Refurbishments &amp; maintenance</h3><p>Full property refurbishments and ongoing general maintenance.</p></div>
<div class="card reveal"><h3>Listed buildings &amp; heritage</h3><p>Specialist knowledge for listed and period properties.</p></div>
</div></div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

CONTACT_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Contact - K&H Decorators</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band" style="padding-top:120px"><div class="wrap narrow"><div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">Get in touch</div><h2 class="serif">Free estimate.</h2><p class="sub">Use the chat bubble for the fastest quote, or contact Steve directly.</p></div><div class="contact-box reveal">
<p><strong>Mobile:</strong> <a href="tel:+447908701460">07908 701460</a></p>
<p><strong>Landline:</strong> <a href="tel:+441243778091">01243 778091</a></p>
<p><strong>Email:</strong> <a href="mailto:steve25hamblin@hotmail.com">steve25hamblin@hotmail.com</a></p>
<p><strong>WhatsApp:</strong> <a href="https://wa.me/447908701460" target="_blank" rel="noopener">Message us</a></p>
<p><strong>Area:</strong> Chichester &amp; across West Sussex.</p>
</div></div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

PRIVACY_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Privacy Policy - K&H Decorators</title><meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """</head><body>""" + NAV + """
<section class="band" style="padding-top:120px"><div class="wrap narrow">
  <div class="head reveal"><div class="rule"></div><div class="eyebrow" style="margin-top:12px">Legal</div><h2 class="serif">Privacy Policy</h2><p class="sub">How K&H Decorators handles the information you share through this website.</p></div>
  <div class="prose reveal">
    <p>This policy explains what we collect when you use this website or its chat assistant, why, and how it is kept. By sending an enquiry you agree to the points below.</p>
    <h3>What we collect</h3><p>When you contact us through the chat assistant, phone, email or WhatsApp we may collect your name, phone number, email, the details of the job you describe and any photos you choose to upload. We only collect what you give us.</p>
    <h3>Why we collect it</h3><p>To understand your job, reply to you and provide an estimate or arrange the work. We do not use it for marketing unless you ask.</p>
    <h3>How it is handled</h3><p>Enquiry details are sent to our own inbox so we can respond. We use trusted providers to power the chat assistant and deliver emails, who process the information only to provide that service. We never sell or rent your information.</p>
    <h3>How long we keep it</h3><p>Only as long as needed to deal with your job and our records, then it is removed.</p>
    <h3>Your rights</h3><p>You can ask what we hold, ask us to correct it, or ask us to delete it at any time.</p>
    <h3>Contact</h3><p>K&H Decorators, Chichester, West Sussex.<br>Email: <a href="mailto:steve25hamblin@hotmail.com">steve25hamblin@hotmail.com</a> &nbsp;|&nbsp; Phone: <a href="tel:+447908701460">07908 701460</a></p>
  </div>
</div></section>""" + FOOTER + SCRIPTS + WIDGET_INCLUDE + "</body></html>"

WIDGET_JS = """
(function(){
  var base = new URL(document.currentScript.src).origin;
  var bubble = document.createElement('button');
  bubble.innerHTML = 'Chat';
  bubble.setAttribute('aria-label','Open quote assistant');
  bubble.style.cssText='position:fixed;right:22px;bottom:22px;z-index:999999;border:1px solid rgba(242,245,248,.28);border-radius:999px;background:linear-gradient(135deg,#050506,#444b55 58%,#f2f5f8);color:white;font-weight:900;padding:15px 18px;box-shadow:0 16px 38px rgba(0,0,0,.46),0 0 22px rgba(242,245,248,.18);cursor:pointer';
  var frame = document.createElement('iframe');
  frame.src = base + '/widget-frame';
  function size(){ frame.style.cssText = window.innerWidth <= 640 ? 'position:fixed;inset:0;width:100vw;height:100dvh;border:0;z-index:999999;display:none;background:white' : 'position:fixed;right:22px;bottom:84px;width:410px;height:610px;border:0;border-radius:18px;box-shadow:0 18px 60px rgba(0,0,0,.45);z-index:999999;display:none;background:white'; }
  size(); window.addEventListener('resize',size);
  bubble.onclick=function(){ frame.style.display='block'; bubble.style.display=window.innerWidth<=640?'none':'block'; document.body.style.overflow=window.innerWidth<=640?'hidden':''; };
  window.addEventListener('message',function(e){ if(e.data==='close-au-chat'){ frame.style.display='none'; bubble.style.display='block'; document.body.style.overflow=''; }});
  document.body.appendChild(bubble); document.body.appendChild(frame);
})();
"""

WIDGET_FRAME = """
<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:Manrope,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0b0d;color:#f7f8fb;overflow:hidden}
#chatWindow{height:100dvh;display:flex;flex-direction:column;background:#0a0b0d}#chatHeader{background:#050506;color:white;padding:16px;display:flex;align-items:center;gap:12px;justify-content:space-between;box-shadow:0 16px 36px rgba(0,0,0,.32)}.hbrand{display:flex;gap:10px;align-items:center}.hbrand img{width:42px;height:42px;border-radius:8px;object-fit:cover}.title{font-weight:900;text-shadow:0 0 16px rgba(242,245,248,.3)}.sub{font-size:12px;color:#cdd4de}.close{font-size:28px;color:#f2f5f8;cursor:pointer;padding:2px 8px}.progress{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;padding:10px 14px;background:#111317;border-bottom:1px solid rgba(242,245,248,.16)}.bar{height:6px;border-radius:99px;background:#2a2d33}.bar.on{background:linear-gradient(90deg,#f2f5f8,#8d96a3,#414852)}#status{font-size:12px;color:#b5bcc7;background:#111317;padding:0 14px 10px;border-bottom:1px solid rgba(242,245,248,.16)}#chatbox{flex:1;overflow:auto;padding:16px;-webkit-overflow-scrolling:touch}.msg{max-width:84%;margin:10px 0;padding:12px 14px;border-radius:16px;line-height:1.45;font-size:15px}.bot{background:#171a20;border:1px solid rgba(242,245,248,.16);color:#f7f8fb}.user{margin-left:auto;background:linear-gradient(135deg,#050506,#454d59);color:white}.photo-msg{padding:5px;background:#050506}.photo{width:210px;border-radius:12px}#inputRow{flex:none;display:flex;gap:8px;padding:10px;background:#111317;border-top:1px solid rgba(242,245,248,.16);padding-bottom:max(10px,env(safe-area-inset-bottom))}#userInput{flex:1;min-width:0;border:1px solid rgba(242,245,248,.24);border-radius:999px;padding:12px 14px;font-size:16px;outline:none;background:#050506;color:#fff}#userInput::placeholder{color:#858d99}#sendBtn,#attachBtn{border:0;border-radius:50%;width:46px;height:46px;display:grid;place-items:center;background:linear-gradient(135deg,#050506,#4b535f,#f2f5f8);color:white;font-weight:900;cursor:pointer;flex:none}#attachBtn{background:#242830;color:#f2f5f8;border:1px solid rgba(242,245,248,.2)}#fileInput{display:none}.typing{color:#8c96a3}
</style></head><body><div id="chatWindow"><div id="chatHeader"><div class="hbrand"><img src="https://www.kandhdecoratorschichester.co.uk/wp-content/uploads/2021/02/cropped-Untitled-design-1-270x270.png"><div><div class="title">K&H Decorators Assistant</div><div class="sub">Quote details captured in minutes</div></div></div><div class="close" onclick="window.parent.postMessage('close-au-chat','*')">&times;</div></div><div class="progress"><span class="bar on"></span><span class="bar"></span><span class="bar"></span><span class="bar"></span><span class="bar"></span><span class="bar"></span></div><div id="status">Quote progress: tell us what needs doing</div><div id="chatbox"></div><div id="inputRow"><label id="attachBtn" title="Attach photos"><input type="file" id="fileInput" accept="image/*" multiple onchange="handleFiles(this)">+</label><input type="text" id="hpField" tabindex="-1" autocomplete="off" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0"><input id="userInput" type="text" placeholder="Type your message..." onkeypress="if(event.key==='Enter')sendMessage()"><button id="sendBtn" onclick="sendMessage()">></button></div></div>
<script>
var messages=0; addMessage("Hi, I can help get a free quote for plastering, painting or decorating. What needs doing?", "bot");
function updateProgress(){var n=Math.min(6,Math.ceil(messages/2));document.querySelectorAll('.bar').forEach(function(b,i){b.classList.toggle('on',i<n)});document.getElementById('status').textContent='Quote progress: '+n+'/6 details captured';}
function addMessage(t,s){var c=document.getElementById('chatbox'),d=document.createElement('div');d.className='msg '+s;d.textContent=t;c.appendChild(d);c.scrollTop=c.scrollHeight;if(s==='user'){messages++;updateProgress();}}
function typing(){var c=document.getElementById('chatbox'),d=document.createElement('div');d.id='typing';d.className='msg bot typing';d.textContent='...';c.appendChild(d);c.scrollTop=c.scrollHeight}
function untyping(){var t=document.getElementById('typing');if(t)t.remove()}
async function sendMessage(){var i=document.getElementById('userInput'),m=i.value.trim();if(!m)return;addMessage(m,'user');i.value='';typing();try{var r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m,website:(document.getElementById('hpField')||{}).value||''}),credentials:'same-origin'});var d=await r.json();untyping();addMessage(d.reply,'bot')}catch(e){untyping();addMessage("Sorry, that did not send. Please try again.",'bot')}}
function addImage(src){var c=document.getElementById('chatbox'),d=document.createElement('div'),img=document.createElement('img');d.className='msg user photo-msg';img.className='photo';img.src=src;d.appendChild(img);c.appendChild(d);c.scrollTop=c.scrollHeight;messages++;updateProgress();}
function resizeImage(file){return new Promise(function(resolve,reject){var reader=new FileReader();reader.onload=function(){var img=new Image();img.onload=function(){var max=1600,w=img.naturalWidth,h=img.naturalHeight;if(Math.max(w,h)>max){if(w>=h){h=Math.round(h*max/w);w=max}else{w=Math.round(w*max/h);h=max}}var canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;var ctx=canvas.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.drawImage(img,0,0,w,h);resolve(canvas.toDataURL('image/jpeg',.82))};img.onerror=reject;img.src=reader.result};reader.onerror=reject;reader.readAsDataURL(file)})}
async function handleFiles(input){var files=Array.from(input.files||[]);input.value='';for(const file of files){if(!file.type.startsWith('image/')){addMessage('Please choose a photo file.','bot');continue}try{var dataUrl=await resizeImage(file);addImage(dataUrl);var r=await fetch('/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:dataUrl}),credentials:'same-origin'});var d=await r.json();addMessage(d.reply,'bot')}catch(e){addMessage('Sorry, I could not upload that photo. Try another JPG or PNG.','bot')}}}
</script></body></html>
"""

def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

@app.route("/sitemap.xml")
def sitemap():
    pages = ["/", "/services", "/gallery", "/contact", "/privacy-policy"]
    base = request.host_url.rstrip("/")
    urls = "".join(f"<url><loc>{base}{p}</loc></url>" for p in pages)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'
    return Response(xml, mimetype="application/xml")

@app.route("/robots.txt")
def robots():
    base = request.host_url.rstrip("/")
    return Response(f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml", mimetype="text/plain")

@app.route("/")
def home():
    ensure_session()
    return render_template_string(HOME_PAGE)

@app.route("/services")
def services():
    ensure_session()
    return render_template_string(SERVICES_PAGE)

@app.route("/gallery")
def gallery():
    ensure_session()
    return render_template_string(GALLERY_PAGE)

@app.route("/contact")
def contact():
    ensure_session()
    return render_template_string(CONTACT_PAGE)

@app.route("/privacy")
@app.route("/privacy-policy")
def privacy():
    ensure_session()
    return render_template_string(PRIVACY_PAGE)

@app.route("/widget.js")
def widget_js():
    return Response(WIDGET_JS, mimetype="application/javascript")

@app.route("/widget-frame")
def widget_frame():
    ensure_session()
    return render_template_string(WIDGET_FRAME)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    if session_id not in all_conversations:
        all_conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation = all_conversations[session_id]

    data = request.get_json(silent=True) or {}

    # Honeypot: a hidden field real visitors never see or fill. If it's populated,
    # it's almost certainly a bot - quietly stop before spending Groq/Resend.
    if (data.get("website") or "").strip():
        return jsonify({"reply": "Thanks!"})

    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Sorry, I didn't catch that - could you type that again?"})

    # Per-session rate limiting to protect against abuse running up Groq/Resend
    # cost: max 20 messages a minute, plus a hard cap per visitor.
    now = time.time()
    recent = [t for t in chat_activity.get(session_id, []) if now - t < 60]
    if len(recent) >= 20:
        return jsonify({"reply": "You're sending messages very quickly - give it a few seconds and try again."})
    if len(conversation) >= 60:
        return jsonify({"reply": "Thanks for all the detail! Drop your name and number and K&H Decorators will pick this up with you personally."})
    recent.append(now)
    chat_activity[session_id] = recent

    conversation.append({"role": "user", "content": user_message})

    try:
        response = client_chat(
            model="llama-3.3-70b-versatile",
            messages=conversation,
            max_tokens=256,
            timeout=20,
        )
        ai_reply = response.choices[0].message.content
    except Exception as e:
        # Never leave the customer staring at a frozen chat. Drop the message we
        # just appended so they can retry cleanly, and reply with a gentle note.
        print(f"Chat completion failed: {e}")
        conversation.pop()
        return jsonify({
            "reply": "Sorry, I had a brief hiccup there - could you send that again?"
        })

    # Strip any internal signal tags so they can never reach the customer.
    lead_ready = bool(re.search(r"\[\[?\s*READY\s*\]?\]", ai_reply, re.I))
    ai_reply = re.sub(r"\[\[?\s*READY\s*\]?\]", "", ai_reply)
    ai_reply = ai_reply.replace("[LEAD_CAPTURED]", "").strip()
    if not ai_reply:
        ai_reply = ("Thanks - that's everything we need for now. K&H Decorators "
                    "will be in touch shortly to arrange your free estimate.")

    conversation.append({"role": "assistant", "content": ai_reply})

    # Only email once the assistant has genuinely finished gathering EVERYTHING.
    # It signals this with the internal [[READY]] tag, which it only adds after
    # working through the whole checklist (job, scope, budget, area, contact...).
    # We deliberately do NOT send on wrap-up phrases or a low turn count, because
    # that was firing before budget/postcode were collected. The fallbacks below
    # are conservative - only if the visitor clearly signs off, or a very long
    # chat - so a lead is never lost, but normal chats wait for the full set of
    # questions. Sent at most once per visitor.
    if session_id not in notified_sessions and has_contact_info(conversation):
        if lead_ready or _looks_like_closing(user_message) or len(conversation) >= 24:
            notified_sessions.add(session_id)
            conversation_copy = list(conversation)
            images_copy = list(session_images.get(session_id, []))
            send_lead_email(conversation_copy, images_copy)

    return jsonify({"reply": ai_reply})


@app.route("/upload", methods=["POST"])
def upload_endpoint():
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    if session_id not in all_conversations:
        all_conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation = all_conversations[session_id]

    data = request.get_json(silent=True) or {}
    image = _decode_image_data_url(data.get("image", ""))
    if image is None:
        return (
            jsonify({"reply": "Sorry, I couldn't read that image. Please try a JPG or PNG photo."}),
            400,
        )

    images = session_images.setdefault(session_id, [])
    if len(images) >= MAX_IMAGES_PER_SESSION:
        return jsonify({
            "reply": "Thanks - that's plenty of photos for now. Leave your name and number and we'll take a look and get you a quote."
        })

    images.append(image)

    # Keep the transcript (and the AI) aware that a photo came in.
    conversation.append({"role": "user", "content": "(Customer attached a photo of the job)"})
    reply = (
        "Thanks, got the photo - that really helps. You can add another, "
        "or tell me if that is all and I will carry on with the quote details."
    )
    conversation.append({"role": "assistant", "content": reply})

    # If we've already emailed this lead, forward the new photo as a follow-up
    # so it doesn't get lost.
    if session_id in notified_sessions:
        send_photo_followup(list(conversation), [image])

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
