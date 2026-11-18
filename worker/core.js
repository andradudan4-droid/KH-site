// Shared chat-assistant engine for the FRONTDESK sites on Cloudflare.
// Each site has worker/site.js with its prompts, wording and email layout.
// Copy this file unchanged into each site's worker/ folder.

const MODEL = "openai/gpt-oss-120b";
const MAX_MESSAGES = 60;
const MAX_TEXT = 2000;
const MAX_IMAGES = 6;
const MAX_IMAGE_BYTES = 6 * 1024 * 1024;
const IMAGE_EXT = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp" };

const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/;
const PHONE_RE = /(?<!\d)(?:\+44|0)\d[\d\s\-.]{8,11}(?!\d)/g;
const POSTCODE_RE = /\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}\b/;
const READY_RE = /\[\[?\s*READY\s*\]?\]/gi;
export const CLOSING_RE = new RegExp(
  "\\b(no longer interested|not interested|no thanks|no thank you|" +
  "that'?s all|that'?s it|that'?s everything|nothing else|all good|" +
  "that'?s great thank|thanks that'?s|goodbye|bye for now|no more|" +
  "i'?m good|im good)\\b", "i");

export const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;" }[c]));

// ---------- contact details ----------
const customerText = (conv) => conv.filter((m) => m.role === "user").map((m) => m.content).join(" ");
export function findEmail(conv) { const m = customerText(conv).match(EMAIL_RE); return m ? m[0] : null; }
export function findPhone(conv) {
  for (const cand of customerText(conv).match(PHONE_RE) || []) {
    let d = cand.replace(/\D/g, "");
    if (d.startsWith("00")) continue;
    if (d.startsWith("44")) d = "0" + d.slice(2);
    if (d.length === 11 && d.startsWith("0")) return `${d.slice(0, 5)} ${d.slice(5)}`;
  }
  return null;
}
export function findPostcode(conv) {
  const m = customerText(conv).match(POSTCODE_RE);
  if (!m) return null;
  const raw = m[0].replace(/\s+/g, "").toUpperCase();
  return raw.slice(0, -3) + " " + raw.slice(-3);
}
export const hasContact = (conv) => !!(findEmail(conv) || findPhone(conv));

// ---------- email helpers ----------
export const transcript = (conv) =>
  conv.map((m) => `${m.role === "user" ? "Customer" : "Assistant"}: ${m.content}`).join("\n\n");

export function row(label, value, width = 130) {
  if (!value) return "";
  return '<tr>' +
    `<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#8a8a8a;font-size:13px;white-space:nowrap;vertical-align:top;width:${width}px">${esc(label)}</td>` +
    `<td style="padding:10px 16px;border-bottom:1px solid #eee;color:#1a1a1a;font-size:14px;font-weight:600">${esc(value)}</td>` +
    '</tr>';
}

export function transcriptHtml(conv, o) {
  return conv.map((m) => {
    const [who, color, bg] = m.role === "user" ? ["Customer", "#0a0a0a", o.userBg] : [o.botLabel, o.botColor, "#ffffff"];
    return '<div style="margin:0 0 12px">' +
      `<div style="font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:${color};font-weight:700;margin-bottom:4px">${who}</div>` +
      `<div style="background:${bg};border:1px solid ${o.border || "#ececec"};border-radius:10px;padding:11px 14px;font-size:14px;color:#2a2a2a;line-height:1.5">${esc(m.content).replace(/\n/g, "<br>")}</div>` +
      '</div>';
  }).join("");
}

export function urgencyScore(v) { const m = v && String(v).match(/[1-5]/); return m ? Number(m[0]) : 0; }
export function urgencyBadge(v) {
  const score = urgencyScore(v);
  if (!score) return "";
  const [bg, fg, label] = {
    1: ["#e8f5e9", "#2e7d32", "1 — No rush"], 2: ["#f1f8e9", "#558b2f", "2 — Low"],
    3: ["#fff8e1", "#f57f17", "3 — Moderate"], 4: ["#fff3e0", "#e65100", "4 — Fairly urgent"],
    5: ["#ffebee", "#b71c1c", "5 — URGENT — reply ASAP"],
  }[score];
  return '<div style="margin:0 0 20px">' +
    '<div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#999;font-weight:700;margin-bottom:6px">Urgency</div>' +
    `<span style="display:inline-block;background:${bg};color:${fg};border:1px solid ${fg};border-radius:999px;padding:5px 14px;font-size:13px;font-weight:700">${label}</span></div>`;
}
export const urgentPrefix = (v) => { const s = urgencyScore(v); return s >= 5 ? "🔴 URGENT — " : s >= 4 ? "🟠 " : ""; };

export const photosLine = (n) => n
  ? `<p style="margin:0 0 20px;font-size:14px;color:#1a1a1a">\u{1F4CE} <strong>${n} photo(s)</strong> attached to this email.</p>` : "";
export const missedLine = (n) => n
  ? `<p style="margin:0 0 20px;font-size:14px;color:#1a1a1a">The customer sent ${n} photo(s) but left the page before they could be attached. Ask them to send them by WhatsApp.</p>` : "";

// ---------- outside services ----------
export async function groq(env, body) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { authorization: `Bearer ${env.GROQ_API_KEY}`, "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, ...body }),
    signal: AbortSignal.timeout(20000),
  });
  if (!res.ok) {
    const err = new Error(`Groq ${res.status}: ${(await res.text()).slice(0, 300)}`);
    err.status = res.status;
    err.retryAfter = Number(res.headers.get("retry-after")) || 0;
    throw err;
  }
  return (await res.json()).choices?.[0]?.message?.content || "";
}

export async function sendEmail(env, { subject, text, html, attachments = [], replyTo = null }) {
  const payload = { from: env.RESEND_FROM, to: [env.NOTIFY_TO], subject, text, html };
  if (replyTo) payload.reply_to = replyTo;
  if (attachments.length) payload.attachments = attachments;
  if (env.DRY_RUN === "1" || !env.RESEND_API_KEY) {
    console.log("DRY RUN email to", env.NOTIFY_TO, "|", subject, "|", attachments.length, "photo(s)");
    return { dryRun: true, subject, to: env.NOTIFY_TO, text };
  }
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { authorization: `Bearer ${env.RESEND_API_KEY}`, "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Resend ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return { sent: true };
}

export async function limited(limiter, request) {
  if (!limiter) return false;
  const { success } = await limiter.limit({ key: request.headers.get("cf-connecting-ip") || "unknown" });
  return !success;
}

// ---------- input checks ----------
function cleanHistory(raw) {
  if (!Array.isArray(raw)) return null;
  const out = [];
  for (const m of raw.slice(-MAX_MESSAGES)) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") return null;
    const content = m.content.slice(0, MAX_TEXT).trim();
    if (content) out.push({ role: m.role, content });
  }
  return out;
}

function decodeImage(dataUrl, i) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) return null;
  const comma = dataUrl.indexOf(",");
  if (comma < 0) return null;
  const header = dataUrl.slice(5, comma);
  const ext = IMAGE_EXT[header.split(";")[0].toLowerCase()];
  if (!ext || !header.includes(";base64")) return null;
  const b64 = dataUrl.slice(comma + 1);
  if (!/^[A-Za-z0-9+/]+=*$/.test(b64)) return null;
  const bytes = Math.floor((b64.length * 3) / 4);
  if (!bytes || bytes > MAX_IMAGE_BYTES) return null;
  return { filename: `job-photo-${crypto.randomUUID().slice(0, 8)}-${i + 1}.${ext}`, content: b64 };
}

// The model's "Label: value" summary, keyed by lower-case label.
export async function summary(env, site, conv) {
  let text = "";
  try {
    text = await groq(env, {
      messages: [{ role: "system", content: site.LEAD_SUMMARY_PROMPT }, { role: "user", content: transcript(conv) }],
      // gpt-oss thinks before it writes; 250 tokens cut the summary off.
      max_tokens: 1200, temperature: 0.2, reasoning_effort: "low",
    });
  } catch (e) { console.log("Lead summary failed:", e.message); }
  const out = {};
  for (const line of text.split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) out[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
  }
  const pick = (...keys) => {
    for (const k of keys) {
      const v = out[k];
      if (v && !["not specified", "not provided", "n/a", "none", "-"].includes(v.toLowerCase())) return v;
    }
    return null;
  };
  return pick;
}

// ---------- routes ----------
async function chat(request, env, site) {
  if (await limited(env.CHAT_LIMIT, request)) {
    return json({ reply: "You're sending messages very quickly - give it a few seconds and try again." });
  }
  const body = await request.json().catch(() => null);
  if (!body) return json({ reply: "Sorry, I didn't catch that - could you type that again?" });
  if ((body.website || "").trim()) return json({ reply: "Thanks!" });
  const conv = cleanHistory(body.history);
  if (!conv || conv.at(-1)?.role !== "user") return json({ reply: "Sorry, I didn't catch that - could you type that again?" });
  if (conv.length >= MAX_MESSAGES - 1) return json({ reply: site.MAXED_REPLY });

  let reply;
  try {
    reply = await groq(env, { messages: [{ role: "system", content: site.SYSTEM_PROMPT }, ...conv], max_tokens: site.MAX_TOKENS || 256,
      ...(site.TEMPERATURE !== undefined ? { temperature: site.TEMPERATURE } : {}) });
  } catch (e) {
    console.log("Chat completion failed:", e.message);
    if (e.status === 429) return json({ error: "busy", retryAfter: Math.min(Math.max(e.retryAfter, 3), 20) }, 429);
    return json({ error: "failed", reply: "Sorry, I had a brief hiccup there - could you send that again?" }, 503);
  }
  const ready = new RegExp(READY_RE.source, "i").test(reply);
  reply = reply.replace(READY_RE, "").replace("[LEAD_CAPTURED]", "").trim() || site.EMPTY_REPLY;

  const full = [...conv, { role: "assistant", content: reply }];
  const contact = hasContact(full);
  // As in the Flask apps: send at once when finished, otherwise the browser sends after 90s quiet.
  // full.length + 1 counts the system prompt, as the Flask list did.
  const finished = ready || CLOSING_RE.test(conv.at(-1).content) ||
    (site.sendNowAfter ? site.sendNowAfter(full, conv) : full.length + 1 >= 24);
  const lead = !contact ? null : finished ? "now" : "later";
  return json({ reply, lead });
}

async function lead(request, env, site) {
  if (await limited(env.LEAD_LIMIT, request)) return json({ error: "slow down" }, 429);
  const body = await request.json().catch(() => null);
  if (!body || (body.website || "").trim()) return json({ ok: true });
  const conv = cleanHistory(body.history);
  if (!conv || !hasContact(conv)) return json({ error: "no contact details" }, 400);
  const images = (Array.isArray(body.images) ? body.images.slice(0, MAX_IMAGES) : []).map(decodeImage).filter(Boolean);
  const missed = Math.max(0, Math.min(MAX_IMAGES, Number(body.missedPhotos) || 0));
  try {
    let result = { skipped: true };
    if (!body.followUp) {
      const pick = await summary(env, site, conv);
      const mail = site.leadEmail({ pick, conv, images, missed });
      result = await sendEmail(env, { ...mail, attachments: images, replyTo: findEmail(conv) });
    } else if (images.length) {
      result = await sendEmail(env, { ...site.followUpEmail({ conv, images }), attachments: images });
    }
    return json({ ok: true, ...(env.DRY_RUN === "1" ? { result } : {}) });
  } catch (e) {
    console.log("Lead email failed:", e.message);
    return json({ error: "email failed" }, 502);
  }
}

export function makeWorker(site) {
  return {
    async fetch(request, env) {
      const url = new URL(request.url);
      const extra = site.routes?.[url.pathname];
      if (url.pathname.startsWith("/api/") || extra) {
        if (request.method !== "POST") return extra ? env.ASSETS.fetch(request) : json({ error: "method" }, 405);
        const origin = request.headers.get("origin");
        if (origin && new URL(origin).host !== url.host) return json({ error: "origin" }, 403);
        if (url.pathname === "/api/chat") return chat(request, env, site);
        if (url.pathname === "/api/lead") return lead(request, env, site);
        if (extra) return extra(request, env);
        return json({ error: "not found" }, 404);
      }
      return env.ASSETS.fetch(request);
    },
  };
}
