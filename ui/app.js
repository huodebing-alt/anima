/* Anima UI — vanilla JS, no build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const api = {
  get: (p) => fetch(p).then((r) => r.json()),
  post: (p, body) => fetch(p, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }).then((r) => r.json()),
};

const THEMES = {
  aurora: ["#8b5cf6", "#22d3ee"],
  ocean: ["#3b82f6", "#14b8a6"],
  ember: ["#f97316", "#f43f5e"],
  forest: ["#10b981", "#a3e635"],
  rose: ["#ec4899", "#a78bfa"],
  mono: ["#94a3b8", "#e2e8f0"],
};
const EMOJI_AVATARS = ["🌙", "✨", "🫧", "🦊", "🐈", "🤖", "🌸", "🔮"];
let settings = null;
const loadTs = Date.now() / 1000;
const sentRecently = new Map(); // text -> ts, to dedupe our own SSE echoes

/* ---------------- boot ---------------- */
async function boot() {
  settings = await api.get("/api/settings");
  applyAppearance();
  buildPickers();
  fillSettingsForm();
  initVoices();
  const hist = await api.get("/api/history?limit=100");
  (hist.entries || []).forEach((e) => routeEntry(e, false));
  scrollBottom($("chatLog"));
  openStream();
  loadDreams();
  loadMemories();
}

/* ---------------- appearance ---------------- */
function applyAppearance() {
  document.body.dataset.theme = settings.theme || "aurora";
  document.body.dataset.mode = settings.mode || "dark";
  $("agentName").textContent = settings.agent_name || "Anima";
  document.title = settings.agent_name || "Anima";
  const emoji = (settings.avatar || "").startsWith("emoji:")
    ? settings.avatar.slice(6) : "";
  $("avatarEmoji").classList.toggle("hidden", !emoji);
  $("avatarEmoji").textContent = emoji;
}

function buildPickers() {
  const sw = $("swatches");
  sw.innerHTML = "";
  Object.entries(THEMES).forEach(([t, [c1, c2]]) => {
    const d = document.createElement("div");
    d.className = "swatch" + (settings.theme === t ? " sel" : "");
    d.title = t;
    d.style.background = `linear-gradient(135deg, ${c1}, ${c2})`;
    d.onclick = () => { settings.theme = t; applyAppearance(); buildPickers(); };
    sw.appendChild(d);
  });
  const av = $("avatars");
  av.innerHTML = "";
  const orb = document.createElement("div");
  orb.className = "av-opt orb" + (!String(settings.avatar || "").startsWith("emoji:") ? " sel" : "");
  orb.title = "gradient orb";
  orb.onclick = () => { settings.avatar = "orb"; applyAppearance(); buildPickers(); };
  av.appendChild(orb);
  EMOJI_AVATARS.forEach((e) => {
    const d = document.createElement("div");
    d.className = "av-opt" + (settings.avatar === "emoji:" + e ? " sel" : "");
    d.textContent = e;
    d.onclick = () => { settings.avatar = "emoji:" + e; applyAppearance(); buildPickers(); };
    av.appendChild(d);
  });
  document.querySelectorAll(".seg-btn").forEach((b) => {
    b.classList.toggle("sel", b.dataset.mode === settings.mode);
    b.onclick = () => { settings.mode = b.dataset.mode; applyAppearance(); buildPickers(); };
  });
}
/* ---------------- settings drawer ---------------- */
function fillSettingsForm() {
  $("setAgentName").value = settings.agent_name || "";
  $("setUserName").value = settings.user_name || "";
  $("setRate").value = settings.voice_rate ?? 1;
  $("setPitch").value = settings.voice_pitch ?? 1;
  $("rateVal").textContent = Number($("setRate").value).toFixed(2);
  $("pitchVal").textContent = Number($("setPitch").value).toFixed(2);
  $("setAutoSpeak").checked = !!settings.auto_speak;
  $("setWorkspace").value = settings.workspace_dir || "";
  $("setBrowse").checked = !!settings.allow_browse;
  $("setFiles").checked = !!settings.allow_files;
  $("setVisionModel").value = settings.vision_model || "";
}

async function saveSettings() {
  settings.agent_name = $("setAgentName").value.trim() || "Anima";
  settings.user_name = $("setUserName").value.trim() || "you";
  settings.voice = $("setVoice").value;
  settings.voice_rate = Number($("setRate").value);
  settings.voice_pitch = Number($("setPitch").value);
  settings.auto_speak = $("setAutoSpeak").checked;
  settings.workspace_dir = $("setWorkspace").value.trim();
  settings.allow_browse = $("setBrowse").checked;
  settings.allow_files = $("setFiles").checked;
  settings.vision_model = $("setVisionModel").value.trim();
  settings = await api.post("/api/settings", settings);
  applyAppearance();
  $("saveNote").textContent = "saved — the mind picks this up on its next thought";
  setTimeout(() => { $("saveNote").textContent = ""; }, 3500);
}

$("settingsBtn").onclick = async () => {
  // re-fetch so the form never shows stale grants (another client or an
  // earlier session may have changed them since this page loaded)
  const fresh = await api.get("/api/settings");
  settings = Object.assign({}, fresh,
    { theme: settings.theme, mode: settings.mode, avatar: settings.avatar });
  fillSettingsForm();
  toggleDrawer(true);
};
$("scrim").onclick = () => toggleDrawer(false);
$("saveBtn").onclick = saveSettings;
function toggleDrawer(open) {
  $("drawer").classList.toggle("hidden", !open);
  $("scrim").classList.toggle("hidden", !open);
}
$("modeBtn").onclick = () => {
  settings.mode = settings.mode === "dark" ? "light" : "dark";
  applyAppearance(); buildPickers();
  api.post("/api/settings", { mode: settings.mode });
};
$("sleepBtn").onclick = () => api.post("/api/control", { command: "sleep" });
$("stopBtn").onclick = () => {
  if (confirm("Stop the mind process? Its memory is safe; it can be restarted."))
    api.post("/api/control", { command: "stop" });
};

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("view-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "dreams") loadDreams();
    if (b.dataset.tab === "memories") loadMemories();
  };
});

/* ---------------- chat ---------------- */
function scrollBottom(el) { el.scrollTop = el.scrollHeight + 999; }
const fmtT = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function addChat(who, text, ts) {
  const row = document.createElement("div");
  row.className = "msg " + who;
  const bubble = `<div class="bubble"></div>`;
  row.innerHTML = who === "agent"
    ? `<div class="mini-avatar">${(settings.avatar || "").startsWith("emoji:") ? settings.avatar.slice(6) : ""}</div>${bubble}<span class="t">${fmtT(ts)}</span>`
    : `<span class="t">${fmtT(ts)}</span>${bubble}`;
  row.querySelector(".bubble").textContent = text;
  $("chatLog").appendChild(row);
  scrollBottom($("chatLog"));
}

function addMind(kind, text, ts) {
  const card = document.createElement("div");
  const k = kind.startsWith("sleep") ? "sleep" : kind;
  card.className = "mind-card k-" + k;
  card.innerHTML = `<span class="k"></span><span class="body"></span><span class="when">${fmtT(ts)}</span>`;
  card.querySelector(".k").textContent = kind;
  card.querySelector(".body").textContent = text;
  const log = $("mindLog");
  log.appendChild(card);
  while (log.children.length > 250) log.firstChild.remove();
  scrollBottom(log);
}

async function sendMessage() {
  const text = $("input").value.trim();
  if (!text) return;
  $("input").value = "";
  autosize();
  sentRecently.set(text, Date.now());
  addChat("user", text, Date.now() / 1000);
  await api.post("/api/message", { text });
}
$("sendBtn").onclick = sendMessage;
$("input").addEventListener("keydown", (e) => {
  const isEnter = e.key === "Enter" || e.code === "Enter" || e.keyCode === 13;
  if (isEnter && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
function autosize() {
  const el = $("input");
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 130) + "px";
}
$("input").addEventListener("input", autosize);

/* ---------------- event routing ---------------- */
function routeEntry(e, live) {
  const kind = e.kind, ts = e.ts || Date.now() / 1000;
  if (kind === "say") {
    addChat("agent", e.text, ts);
    if (live && settings.auto_speak && ts > loadTs) speak(e.text);
    flashAvatar("speaking");
  } else if (kind === "percept" || kind === "message") {
    if ((e.source || "user") === "user") {
      const sent = sentRecently.get(e.text);
      if (sent && Date.now() - sent < 90000) { sentRecently.delete(e.text); return; }
      addChat("user", e.text, ts);
    } else {
      addMind(e.source || "sense", e.text, ts);
    }
  } else if (kind === "thought") {
    addMind("thought", e.text, ts);
    if (live) flashAvatar("thinking");
  } else if (kind === "sleep.dream") {
    addMind("dream", e.dream || e.text || "", ts);
  } else if (kind === "browse") {
    addMind("browse", e.url + (e.error ? " — failed: " + e.error : ""), ts);
  } else if (kind === "file") {
    addMind("file", e.path || e.error || "", ts);
  } else if (kind === "remember" || kind === "goal" || kind === "grant" ||
             kind === "control") {
    addMind(kind, e.text || "", ts);
  } else if (kind === "sleep.begin") {
    addMind("sleep", "falling asleep…", ts);
    dayNote("💤 " + (settings.agent_name || "Anima") + " fell asleep");
  } else if (kind === "sleep.end") {
    const bits = [];
    if ((e.gists || []).length) bits.push(`${e.gists.length} new gists`);
    if ((e.dreams || []).length) bits.push(`${e.dreams.length} dream(s)`);
    if ((e.decay || {}).archived) bits.push(`forgot ${e.decay.archived}`);
    addMind("sleep", "woke up — " + (bits.join(", ") || "light sleep"), ts);
    dayNote("🌅 woke up" + ((e.dreams || []).length ? " after dreaming" : ""));
    loadDreams();
  } else if (kind === "birth") {
    dayNote("🐣 first boot");
  } else if (kind === "reboot") {
    dayNote("⚡ back — memory intact");
  }
}
function dayNote(text) {
  const d = document.createElement("div");
  d.className = "day-note";
  d.textContent = text;
  $("chatLog").appendChild(d);
  scrollBottom($("chatLog"));
}

/* ---------------- SSE ---------------- */
let es;
function openStream() {
  es = new EventSource("/api/stream");
  es.addEventListener("entry", (ev) => routeEntry(JSON.parse(ev.data), true));
  es.addEventListener("state", (ev) => renderState(JSON.parse(ev.data)));
  es.onerror = () => {
    es.close();
    setTimeout(openStream, 3000);
  };
}

function renderState(st) {
  const dot = $("statusDot"), txt = $("statusText"), av = $("avatar");
  av.classList.remove("sleeping");
  $("moon").classList.add("hidden");
  if (!st.daemon_alive) {
    dot.className = "dot dead";
    txt.textContent = "mind not running — `python3 -m anima run`";
  } else if (st.phase === "sleeping") {
    dot.className = "dot sleeping";
    txt.textContent = "sleeping · consolidating memories";
    av.classList.add("sleeping");
    $("moon").classList.remove("hidden");
  } else {
    dot.className = "dot awake";
    const mem = st.memory && st.memory.total ? ` · ${st.memory.total} memories` : "";
    txt.textContent = `awake · tick ${st.tick || 0}${mem}` +
      (st.focus ? ` · ${String(st.focus).slice(0, 42)}` : "");
  }
  $("fatigueBar").style.width = Math.min(100, (st.sleep_pressure || 0) * 100) + "%";
}

let avatarTimer;
function flashAvatar(cls) {
  const av = $("avatar");
  av.classList.add(cls);
  clearTimeout(avatarTimer);
  avatarTimer = setTimeout(() => av.classList.remove("thinking", "speaking"), 4000);
}

/* ---------------- voice out (TTS) ---------------- */
let voices = [];
function initVoices() {
  const fill = () => {
    voices = speechSynthesis.getVoices();
    const sel = $("setVoice");
    sel.innerHTML = "<option value=''>system default</option>";
    voices.forEach((v) => {
      const o = document.createElement("option");
      o.value = v.name;
      o.textContent = `${v.name} (${v.lang})`;
      if (settings.voice === v.name) o.selected = true;
      sel.appendChild(o);
    });
  };
  fill();
  speechSynthesis.onvoiceschanged = fill;
  $("setRate").oninput = () => { $("rateVal").textContent = Number($("setRate").value).toFixed(2); };
  $("setPitch").oninput = () => { $("pitchVal").textContent = Number($("setPitch").value).toFixed(2); };
  $("testVoice").onclick = () => speak(
    `Hello, I'm ${$("setAgentName").value || "Anima"}. This is how I sound.`, true);
}
function speak(text, force) {
  if (!("speechSynthesis" in window)) return;
  if (!force && !settings.auto_speak) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.slice(0, 600));
  const v = voices.find((x) => x.name === ($("setVoice").value || settings.voice));
  if (v) u.voice = v;
  u.rate = Number($("setRate").value || settings.voice_rate || 1);
  u.pitch = Number($("setPitch").value || settings.voice_pitch || 1);
  u.onstart = () => $("avatar").classList.add("speaking");
  u.onend = () => $("avatar").classList.remove("speaking");
  speechSynthesis.speak(u);
}

/* ---------------- voice in (dictation + ambient) ---------------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let dictation = null, ambient = null, ambientWanted = false;

$("micBtn").onclick = () => {
  if (!SR) { alert("Speech recognition needs Chrome/Edge."); return; }
  if (dictation) { stopDictation(); return; }
  pauseAmbient();
  dictation = new SR();
  dictation.interimResults = true;
  dictation.continuous = false;
  $("micBtn").classList.add("listening");
  let finalText = "";
  dictation.onresult = (ev) => {
    let interim = "";
    for (const r of ev.results) (r.isFinal ? (finalText += r[0].transcript) : (interim += r[0].transcript));
    $("input").value = (finalText + " " + interim).trim();
    autosize();
  };
  dictation.onend = () => {
    stopDictation();
    if ($("input").value.trim()) sendMessage();
  };
  dictation.onerror = stopDictation;
  dictation.start();
};
function stopDictation() {
  if (dictation) { try { dictation.stop(); } catch (e) {} }
  dictation = null;
  $("micBtn").classList.remove("listening");
  if (ambientWanted) startAmbient();
}

$("earBtn").onclick = () => setAmbient(!ambientWanted);
$("setEar").onchange = () => setAmbient($("setEar").checked);
function setAmbient(on) {
  ambientWanted = on;
  $("earBtn").classList.toggle("on", on);
  $("setEar").checked = on;
  if (on) startAmbient(); else pauseAmbient();
}
let lastHeardTs = 0;
function startAmbient() {
  if (!SR || ambient || dictation) return;
  ambient = new SR();
  ambient.continuous = true;
  ambient.interimResults = false;
  ambient.onresult = (ev) => {
    const r = ev.results[ev.results.length - 1];
    const text = r[0].transcript.trim();
    if (!text || text.split(" ").length < 3) return;
    if (Date.now() - lastHeardTs < 15000) return; // throttle
    lastHeardTs = Date.now();
    api.post("/api/percept", {
      source: "hearing",
      text: `I overheard someone say: "${text}"`,
      importance: 0.5,
    });
    addMind("hearing", `overheard: "${text}"`, Date.now() / 1000);
  };
  ambient.onend = () => { ambient = null; if (ambientWanted) setTimeout(startAmbient, 800); };
  ambient.onerror = () => {};
  try { ambient.start(); } catch (e) { ambient = null; }
}
function pauseAmbient() {
  if (ambient) { const a = ambient; ambient = null; try { a.onend = null; a.stop(); } catch (e) {} }
}

/* ---------------- camera watching ---------------- */
let camStream = null, camTimer = null, lastFrame = null;
let lastMotionTs = 0, lastVisionTs = 0;

$("camBtn").onclick = () => setCamera(!camStream);
$("setCam").onchange = () => setCamera($("setCam").checked);

async function setCamera(on) {
  $("setCam").checked = on;
  if (!on) {
    if (camStream) camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
    clearInterval(camTimer);
    $("camBtn").classList.remove("on");
    $("camStrip").classList.add("hidden");
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { width: 480 } });
  } catch (e) {
    alert("Camera permission denied.");
    $("setCam").checked = false;
    return;
  }
  $("camVideo").srcObject = camStream;
  $("camBtn").classList.add("on");
  $("camStrip").classList.remove("hidden");
  api.post("/api/percept", {
    source: "vision", importance: 0.4,
    text: "My camera just turned on. I can see now.",
  });
  const canvas = document.createElement("canvas");
  canvas.width = 64; canvas.height = 48;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  camTimer = setInterval(() => watchFrame(ctx, canvas), 1600);
}

function watchFrame(ctx, canvas) {
  const video = $("camVideo");
  if (!camStream || video.readyState < 2) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const frame = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  if (lastFrame) {
    let diff = 0;
    for (let i = 0; i < frame.length; i += 16) diff += Math.abs(frame[i] - lastFrame[i]);
    const motion = diff / (frame.length / 16) / 255;
    if (motion > 0.06 && Date.now() - lastMotionTs > 45000) {
      lastMotionTs = Date.now();
      if (settings.vision_model) {
        snapshotAndDescribe();
      } else {
        api.post("/api/percept", {
          source: "vision", importance: 0.35,
          text: "Through my camera I notice movement in the room.",
        });
        addMind("vision", "movement noticed", Date.now() / 1000);
      }
    }
  }
  lastFrame = frame.slice(0);
  // periodic scene description when a vision model is configured
  if (settings.vision_model && Date.now() - lastVisionTs > 120000) {
    lastVisionTs = Date.now();
    snapshotAndDescribe();
  }
}

function snapshotAndDescribe() {
  lastVisionTs = Date.now();
  const video = $("camVideo");
  const c = document.createElement("canvas");
  c.width = 480; c.height = Math.round(480 * video.videoHeight / (video.videoWidth || 640));
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  const b64 = c.toDataURL("image/jpeg", 0.7);
  api.post("/api/vision", { image: b64 }).then((r) => {
    if (r.description) addMind("vision", r.description, Date.now() / 1000);
  }).catch(() => {});
}

/* ---------------- dreams + memories ---------------- */
async function loadDreams() {
  const d = await api.get("/api/dreams");
  const log = $("dreamLog");
  log.innerHTML = "";
  if (!(d.dreams || []).length) {
    log.innerHTML = "<div class='day-note'>no dreams yet — dreams come with sleep</div>";
    return;
  }
  d.dreams.forEach((x) => {
    const card = document.createElement("div");
    card.className = "dream-card";
    card.innerHTML = `<div class="when">${new Date(x.ts * 1000).toLocaleString()}</div>`;
    const p = document.createElement("div");
    p.textContent = x.dream;
    card.appendChild(p);
    log.appendChild(card);
  });
}

async function loadMemories() {
  const kind = $("memKind").value;
  const m = await api.get("/api/memories?limit=80" + (kind ? "&kind=" + kind : ""));
  const log = $("memLog");
  log.innerHTML = "";
  (m.memories || []).forEach((x) => {
    const card = document.createElement("div");
    card.className = "mem-card";
    card.innerHTML = `<span class="k"></span><span class="body"></span>
      <span class="bars"></span>`;
    card.querySelector(".k").textContent = x.kind;
    card.querySelector(".body").textContent = x.text;
    card.querySelector(".bars").textContent =
      `imp ${x.importance.toFixed(2)} · str ${x.strength.toFixed(2)} · ×${x.access_count}`;
    log.appendChild(card);
  });
  if (!(m.memories || []).length)
    log.innerHTML = "<div class='day-note'>nothing here yet</div>";
}
$("memRefresh").onclick = loadMemories;
$("memKind").onchange = loadMemories;
$("selfBtn").onclick = async () => {
  const s = await api.get("/api/self");
  alert(`self-model v${s.version}\n\n${s.text || "(none yet)"}`);
};

boot();
