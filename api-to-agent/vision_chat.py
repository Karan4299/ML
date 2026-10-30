"""
Vision chat UI for LarpRouter.

Attach images (file picker, drag & drop, or paste), ask questions about them,
and see the images inline in the conversation. Responses are streamed.

Run:
    .venv/bin/python vision_chat.py
Then open http://127.0.0.1:8000
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------- env / client
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("'\"")

client = OpenAI(
    api_key=os.environ["LARPROUTER_API_KEY"],
    base_url="https://larprouter.com/v1",
    timeout=120,  # don't hang forever if the upstream provider stalls
)

DEFAULT_MODEL = "deepseek-v4.1-flash"
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 8000))


def vision_models() -> list[str]:
    """Return chat models available to this key that accept image input."""
    try:
        raw = client.models.list().model_dump()["data"]
    except Exception:
        return [DEFAULT_MODEL]
    ids = [
        m["id"]
        for m in raw
        if m.get("kind", "chat") == "chat"
        and "image" in ((m.get("modalities") or {}).get("input") or [])
    ]
    return ids or [DEFAULT_MODEL]


def to_api_messages(messages: list[dict]) -> list[dict]:
    """
    Convert UI messages -> OpenAI/LarpRouter format.

    UI message: {"role": "user"|"assistant", "text": str, "images": [dataURL, ...]}
    API message (with images):
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    """
    out = []
    for m in messages:
        images = m.get("images") or []
        text = m.get("text") or ""
        if m["role"] == "user" and images:
            parts = [{"type": "text", "text": text or "Describe this image."}]
            parts += [{"type": "image_url", "image_url": {"url": u}} for u in images]
            out.append({"role": "user", "content": parts})
        else:
            out.append({"role": m["role"], "content": text})
    return out


# ---------------------------------------------------------------- HTTP server
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/models":
            body = json.dumps({"models": vision_models(), "default": DEFAULT_MODEL})
            self._send(200, body.encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, b"not found", "text/plain")

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))

        # Stream plain-text chunks back to the browser.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            stream = client.chat.completions.create(
                model=payload.get("model") or DEFAULT_MODEL,
                messages=to_api_messages(payload["messages"]),
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:  # reasoning-only / final chunks have content=None
                    self.wfile.write(piece.encode())
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser stopped / closed the tab
        except Exception as e:
            try:
                self.wfile.write(f"\n\n[error] {e}".encode())
            except Exception:
                pass

    def log_message(self, fmt, *args):  # quieter logs
        if "/api/chat" in (args[0] if args else ""):
            super().log_message(fmt, *args)


# ---------------------------------------------------------------- UI
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LarpRouter Vision Chat</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --muted:#8a90a0; --text:#e6e8ee; --accent:#6c8cff; --user:#243049; --border:#2a2f3a; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui, -apple-system, sans-serif; background:var(--bg); color:var(--text);
         display:flex; flex-direction:column; height:100vh; }
  header { padding:10px 16px; border-bottom:1px solid var(--border); display:flex; gap:12px; align-items:center; }
  header h1 { font-size:16px; margin:0; flex:1; }
  select, button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:6px 10px; font:inherit; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  #log { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:14px; }
  .msg { max-width:780px; padding:10px 14px; border-radius:12px; white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:var(--user); }
  .assistant { align-self:flex-start; background:var(--panel); border:1px solid var(--border); }
  .assistant.pending::after { content:"▍"; animation:blink 1s steps(1) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  .imgs { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:6px; }
  .imgs img { max-width:240px; max-height:240px; border-radius:8px; cursor:zoom-in; display:block; }
  .empty { margin:auto; color:var(--muted); text-align:center; }
  footer { border-top:1px solid var(--border); padding:12px 16px; }
  #previews { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
  .thumb { position:relative; }
  .thumb img { width:72px; height:72px; object-fit:cover; border-radius:8px; border:1px solid var(--border); display:block; }
  .thumb button { position:absolute; top:-6px; right:-6px; width:20px; height:20px; padding:0; border-radius:50%; font-size:12px; line-height:1; }
  .row { display:flex; gap:8px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:10px; padding:10px; font:inherit; min-height:44px; max-height:200px; }
  body.drag footer { outline:2px dashed var(--accent); outline-offset:-6px; }
  #lightbox { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center; justify-content:center; cursor:zoom-out; }
  #lightbox img { max-width:95vw; max-height:95vh; }
  .hint { color:var(--muted); font-size:12px; margin-top:6px; }
</style>
</head>
<body>
<header>
  <h1>LarpRouter Vision Chat</h1>
  <select id="model" title="Image-capable models"></select>
  <button id="clear">New chat</button>
</header>

<div id="log"><div class="empty" id="empty">Attach an image and ask something about it.</div></div>

<footer>
  <div id="previews"></div>
  <div class="row">
    <button id="attach" title="Attach images">📎</button>
    <input type="file" id="file" accept="image/*" multiple hidden>
    <textarea id="input" rows="1" placeholder="Ask about your images…  (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="send" class="primary">Send</button>
  </div>
  <div class="hint">Attach with 📎, drag &amp; drop, or paste (⌘V). Click any image to enlarge.</div>
</footer>

<div id="lightbox"><img alt=""></div>

<script>
const $ = s => document.querySelector(s);
const log = $("#log"), input = $("#input"), sendBtn = $("#send"), modelSel = $("#model");
let history = [];   // [{role, text, images:[dataURL]}]
let pending = [];   // images attached but not yet sent
let busy = false;

// ---- models
fetch("/api/models").then(r => r.json()).then(({models, default: def}) => {
  for (const m of models) modelSel.add(new Option(m, m, false, m === def));
});

// ---- image handling (downscale big images so requests stay small)
const MAX_SIDE = 2048;
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = reject;
    reader.onload = () => {
      const img = new Image();
      img.onerror = reject;
      img.onload = () => {
        const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
        if (scale === 1 && file.size < 4e6) return resolve(reader.result);
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL("image/jpeg", 0.9));
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

async function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    try { pending.push(await fileToDataURL(f)); } catch (e) { alert("Could not read " + f.name); }
  }
  renderPreviews();
}

function renderPreviews() {
  const box = $("#previews"); box.innerHTML = "";
  pending.forEach((url, i) => {
    const d = document.createElement("div"); d.className = "thumb";
    const img = document.createElement("img"); img.src = url; img.onclick = () => openLightbox(url);
    const x = document.createElement("button"); x.textContent = "×"; x.title = "Remove";
    x.onclick = () => { pending.splice(i, 1); renderPreviews(); };
    d.append(img, x); box.append(d);
  });
}

$("#attach").onclick = () => $("#file").click();
$("#file").onchange = e => { addFiles(e.target.files); e.target.value = ""; };
document.addEventListener("paste", e => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) { e.preventDefault(); addFiles(files); }
});
["dragenter", "dragover"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); document.body.classList.add("drag"); }));
["dragleave", "drop"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); if (t === "drop" || e.target === document.documentElement) document.body.classList.remove("drag"); }));
document.addEventListener("drop", e => addFiles(e.dataTransfer.files));

// ---- lightbox
const lb = $("#lightbox");
function openLightbox(url) { lb.querySelector("img").src = url; lb.style.display = "flex"; }
lb.onclick = () => lb.style.display = "none";
document.addEventListener("keydown", e => { if (e.key === "Escape") lb.style.display = "none"; });

// ---- rendering
function addBubble(role, text, images = []) {
  $("#empty")?.remove();
  const div = document.createElement("div"); div.className = "msg " + role;
  if (images.length) {
    const wrap = document.createElement("div"); wrap.className = "imgs";
    for (const url of images) {
      const img = document.createElement("img"); img.src = url; img.onclick = () => openLightbox(url);
      wrap.append(img);
    }
    div.append(wrap);
  }
  const span = document.createElement("span"); span.textContent = text;
  div.append(span); log.append(div);
  log.scrollTop = log.scrollHeight;
  return {div, span};
}

// ---- send
async function send() {
  const text = input.value.trim();
  if (busy || (!text && !pending.length)) return;
  busy = true; sendBtn.disabled = true;

  const userMsg = {role: "user", text, images: pending};
  history.push(userMsg);
  addBubble("user", text, pending);
  input.value = ""; pending = []; renderPreviews(); autosize();

  const {div, span} = addBubble("assistant", "");
  div.classList.add("pending");
  let reply = "";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model: modelSel.value, messages: history}),
    });
    const reader = res.body.getReader(), dec = new TextDecoder();
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      reply += dec.decode(value, {stream: true});
      span.textContent = reply;
      log.scrollTop = log.scrollHeight;
    }
  } catch (e) {
    reply += "\n[error] " + e.message; span.textContent = reply;
  }
  div.classList.remove("pending");
  history.push({role: "assistant", text: reply});
  busy = false; sendBtn.disabled = false; input.focus();
}

sendBtn.onclick = send;
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; }
input.addEventListener("input", autosize);

$("#clear").onclick = () => {
  history = []; pending = []; renderPreviews();
  log.innerHTML = '<div class="empty" id="empty">Attach an image and ask something about it.</div>';
};
</script>
</body>
</html>
"""


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Vision chat running at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
