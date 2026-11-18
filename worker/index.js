// Cloudflare Worker for K&H Decorators. The shared chat engine is in core.js;
// this file holds the K&H wording and email layout (same as app.py).
import { makeWorker, row, transcript, transcriptHtml, urgencyBadge, urgentPrefix, photosLine, missedLine,
  findPhone, findEmail, findPostcode } from "./core.js";
import { SYSTEM_PROMPT, LEAD_SUMMARY_PROMPT } from "./prompts.js";

const PAGE_TOP = '<!DOCTYPE html><html><body style="margin:0;background:#f0efea;padding:24px;' +
  'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">' +
  '<div style="max-width:620px;margin:0 auto;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.07)">';
const header = (title, pad, size) => `<div style="background:#0a0a0a;padding:${pad}">` +
  '<div style="color:#c9ced6;font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:700">K&H Decorators</div>' +
  `<div style="color:#fff;font-size:${size}px;font-weight:700;margin-top:5px">${title}</div></div>`;

function leadEmail({ pick, conv, images, missed }) {
  const fields = {
    Name: pick("name"),
    Phone: findPhone(conv),
    Email: findEmail(conv),
    Postcode: findPostcode(conv),
    Area: pick("location / area", "location", "area"),
    Job: pick("job / work wanted", "job", "work wanted"),
    Property: pick("property type (domestic or commercial)", "property type", "property"),
    Budget: pick("approx budget (in gbp £; note if it's a total or a per-room / per-m2 rate)", "approx budget", "budget"),
    "Preferred timing": pick("preferred timing", "timing"),
    Urgency: pick("urgency (1-5 where 1=no rush, 5=urgent - infer from what they said)", "urgency"),
    Notes: pick("other notes", "notes"),
  };
  const text = ["NEW LEAD - K&H Decorators", "========================"];
  for (const [k, v] of Object.entries(fields)) if (v) text.push(`${k}: ${v}`);
  if (images.length) text.push(`Photos attached: ${images.length}`);
  if (missed) text.push(`Photos not attached (customer left the page): ${missed}`);
  text.push("========================", "", "Full conversation:", "", transcript(conv));

  const { Urgency, ...rest } = fields;
  const html = PAGE_TOP + header("New enquiry from your website", "24px 28px", 21) +
    '<div style="padding:26px 28px">' +
    '<p style="margin:0 0 20px;font-size:14px;color:#666">Here are the details captured by your website assistant:</p>' +
    urgencyBadge(Urgency) + photosLine(images.length) + missedLine(missed) +
    '<table style="width:100%;border-collapse:collapse;border:1px solid #eee;border-radius:8px;overflow:hidden;margin-bottom:28px">' +
    Object.entries(rest).map(([k, v]) => row(k, v)).join("") + '</table>' +
    '<div style="font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:#999;font-weight:700;margin-bottom:14px">Full conversation</div>' +
    transcriptHtml(conv, { userBg: "#f5f4f0", botLabel: "K&H Assistant", botColor: "#6a7480" }) +
    '</div>' +
    '<div style="background:#faf9f6;padding:16px 28px;border-top:1px solid #eee;font-size:12px;color:#aaa">Sent automatically by the K&H Decorators website assistant. Chichester &middot; West Sussex</div>' +
    '</div></body></html>';

  const contact = fields.Phone || fields.Email || "no number yet";
  const bits = [fields.Name, fields.Area || fields.Postcode].filter(Boolean);
  return {
    subject: urgentPrefix(Urgency) + "New lead - " + (bits.length ? [...bits, contact].join(" · ") : contact),
    text: text.join("\n"),
    html,
  };
}

function followUpEmail({ conv, images }) {
  const phone = findPhone(conv) || "Not provided";
  const email = findEmail(conv) || "Not provided";
  const postcode = findPostcode(conv) || "Not provided";
  const html = PAGE_TOP + header("More photos for an existing lead", "22px 28px", 19) +
    '<div style="padding:24px 28px">' +
    `<p style="margin:0 0 18px;font-size:14px;color:#666">This relates to a lead you've already been emailed about. <strong>${images.length} new photo(s)</strong> attached below.</p>` +
    '<table style="width:100%;border-collapse:collapse;border:1px solid #eee;border-radius:8px;overflow:hidden">' +
    row("Phone", phone) + row("Email", email) + row("Postcode", postcode) + '</table>' +
    '</div></div></body></html>';
  return {
    subject: `Photo added - lead: ${phone}`,
    text: "ADDITIONAL PHOTO(S) - K&H Decorators\nThis relates to a lead you've already been emailed about.\n" +
      `Phone: ${phone}\nEmail: ${email}\nPostcode: ${postcode}\nPhotos attached: ${images.length}\n`,
    html,
  };
}

export default makeWorker({
  SYSTEM_PROMPT,
  LEAD_SUMMARY_PROMPT,
  MAXED_REPLY: "Thanks for all the detail! Drop your name and number and K&H Decorators will pick this up with you personally.",
  EMPTY_REPLY: "Thanks - that's everything we need for now. K&H Decorators will be in touch shortly to arrange your free estimate.",
  // app.py sends straight away after 5 customer messages.
  sendNowAfter: (full) => full.filter((m) => m.role === "user").length >= 5,
  leadEmail,
  followUpEmail,
});
