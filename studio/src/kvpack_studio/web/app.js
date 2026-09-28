// kvpack Studio dashboard. Plain JavaScript, no build step.
// Everything user-provided is inserted with textContent, never as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const KEY_STORAGE = "kvpack.apiKey";

const state = {
  key: null,
  config: null,
  me: null,
  cartridges: [],
  selectedId: null,
  pollTimer: null,
  files: [],
  sourceType: "git",
  chats: {}, // cartridge id -> [{role, content}]
  snippetLang: "python",
};

// ---------------------------------------------------------------- helpers

function readKey() {
  try { return localStorage.getItem(KEY_STORAGE); } catch { return null; }
}
function writeKey(value) {
  try { value ? localStorage.setItem(KEY_STORAGE, value) : localStorage.removeItem(KEY_STORAGE); } catch { /* private mode */ }
}

async function api(path, { method = "GET", json, form, raw = false } = {}) {
  const headers = {};
  if (state.key) headers.Authorization = `Bearer ${state.key}`;
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  if (form) body = form;
  const response = await fetch(path, { method, headers, body });
  if (raw && response.ok) return response;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data?.error?.message || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

function showError(id, message) {
  const node = $(id);
  node.textContent = message || "";
  node.hidden = !message;
}

let toastTimer;
function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 2200);
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch { toast("Couldn't copy. Select the text instead."); }
}

const fmt = {
  int: (n) => (n ?? 0).toLocaleString("en-US"),
  pct: (x) => `${(x * 100).toFixed(1)}%`,
  bytes: (n) => (n > 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`),
  date: (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "never"),
  duration: (s) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`),
};

// ---------------------------------------------------------------- views

const VIEWS = ["auth", "newkey", "cartridges", "keys", "usage"];
function showView(name) {
  for (const view of VIEWS) $(`view-${view}`).hidden = view !== name;
  const signedIn = !!state.me;
  $("tabs").hidden = !signedIn || name === "newkey";
  $("account").hidden = !signedIn;
  for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("active", b.dataset.tab === name);
  if (name === "keys") loadKeys();
  if (name === "usage") loadUsage();
}

for (const b of document.querySelectorAll("#tabs button")) b.addEventListener("click", () => showView(b.dataset.tab));

// ---------------------------------------------------------------- auth

async function signIn(key) {
  state.key = key;
  state.me = await api("/v1/me");
  writeKey(key);
  $("account-email").textContent = state.me.email;
}

$("signin-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("signin-error");
  try {
    await signIn($("signin-key").value.trim());
    await enterApp();
  } catch (err) {
    state.key = null;
    showError("signin-error", err.status === 401 ? "That API key isn't valid." : err.message);
  }
});

$("signup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("signup-error");
  try {
    const result = await api("/v1/signup", {
      method: "POST",
      json: { email: $("signup-email").value.trim(), invite_code: $("signup-invite").value.trim() },
    });
    await signIn(result.api_key);
    showNewKey(result.api_key, () => enterApp());
  } catch (err) {
    showError("signup-error", err.message);
  }
});

$("sign-out").addEventListener("click", () => {
  writeKey(null);
  Object.assign(state, { key: null, me: null, cartridges: [], selectedId: null, chats: {} });
  clearTimeout(state.pollTimer);
  $("signin-key").value = "";
  showView("auth");
});

function showNewKey(key, onDone) {
  $("newkey-value").textContent = key;
  $("newkey-copy").onclick = () => copy(key);
  $("newkey-done").onclick = () => { $("newkey-value").textContent = ""; onDone(); };
  showView("newkey");
}

// ---------------------------------------------------------------- cartridges

async function enterApp() {
  showView("cartridges");
  await refreshCartridges();
}

async function refreshCartridges() {
  clearTimeout(state.pollTimer);
  try {
    state.cartridges = (await api("/v1/cartridges")).data;
  } catch (err) {
    if (err.status === 401) return $("sign-out").click();
    toast(err.message);
  }
  renderCartridges();
  if (state.selectedId) renderDetail();
  if (state.cartridges.some((c) => c.status === "queued" || c.status === "building")) {
    state.pollTimer = setTimeout(refreshCartridges, 2000);
  }
}

const SYNCABLE = new Set(["git", "web", "folder", "upload"]);
const building = (c) => c.status === "queued" || c.status === "building";

function statusLabel(c) {
  if (building(c)) return c.servable ? "updating" : c.status === "queued" ? "queued" : c.stage || "building";
  if (c.status === "failed" && c.servable) return "ready";
  return c.status;
}

function pillClass(c) {
  if (building(c) && c.servable) return "updating";
  if (c.status === "failed" && c.servable) return "ready";
  return c.status;
}

function listMeta(c) {
  if (c.servable) {
    const note = c.status === "failed" ? " · last update failed" : "";
    return `${fmt.int(c.num_tokens)} tokens · v${c.version}${note}`;
  }
  if (c.status === "failed") return "Build failed";
  return `${Math.round(c.progress * 100)}% · ${c.base_model}`;
}

function renderCartridges() {
  const list = $("cartridges");
  list.replaceChildren();
  $("cartridges-empty").hidden = state.cartridges.length > 0;
  $("cartridge-count").textContent = state.cartridges.length ? `${state.cartridges.length} of ${state.config.limits.max_cartridges}` : "";
  for (const c of state.cartridges) {
    const button = el("button", { type: "button", class: c.id === state.selectedId ? "selected" : "", onclick: () => select(c.id) },
      el("span", { class: "c-name", text: c.name }),
      el("span", { class: `pill ${pillClass(c)}`, text: statusLabel(c) }),
      el("span", { class: "c-meta", text: listMeta(c) }),
    );
    if (building(c)) {
      const bar = el("div", { class: "mini-progress" }, el("div"));
      bar.firstChild.style.width = `${Math.round(c.progress * 100)}%`;
      button.append(bar);
    }
    list.append(el("li", {}, button));
  }
}

function select(id) {
  state.selectedId = id;
  renderCartridges();
  renderDetail();
  $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderDetail() {
  const c = state.cartridges.find((x) => x.id === state.selectedId);
  $("detail").hidden = !c;
  if (!c) return;
  $("detail-id").textContent = c.id;
  $("detail-name").textContent = c.name;
  $("detail-download").hidden = !c.servable;
  $("detail-delete").hidden = false;
  $("detail-delete-confirm").hidden = true;

  const syncable = (c.sources || []).some((src) => SYNCABLE.has(src.type));
  const autoSyncable = (c.sources || []).some((src) => src.type !== "upload" && SYNCABLE.has(src.type));
  $("detail-sync").hidden = !syncable;
  $("detail-sync").disabled = building(c);
  $("detail-sync-every").closest("label").hidden = !autoSyncable;
  $("detail-sync-every").value = String(c.sync_every_hours || 0);
  $("detail-sources").replaceChildren(...(c.sources || []).map((src) =>
    el("li", {}, el("span", { class: "source-type", text: src.type }), el("span", { text: describeSource(src) }))));
  $("detail-version").textContent = c.servable
    ? [`Version ${c.version}`, c.last_synced_at && `last synced ${fmt.date(c.last_synced_at)}`,
       c.sync_every_hours && `checks every ${hoursLabel(c.sync_every_hours)}`].filter(Boolean).join(" · ")
    : "";

  $("detail-building").hidden = !building(c);
  $("detail-progress").style.width = `${Math.round(c.progress * 100)}%`;
  const stages = { fetching: "Fetching the sources", synthesizing: "Generating practice conversations", training: "Training the cartridge" };
  const what = c.servable ? "Updating. The current version keeps answering" : "Building";
  $("detail-stage").textContent = c.status === "queued"
    ? `${c.servable ? "Update" : "Build"} queued…`
    : `${what} · ${stages[c.stage] || "Starting"} · ${Math.round(c.progress * 100)}%`;
  showError("detail-error", c.status === "failed" ? (c.servable ? `The last update failed: ${c.error} Version ${c.version} is still serving.` : c.error) : "");

  $("detail-ready").hidden = !c.servable;
  if (!c.servable) return;

  const facts = [
    ["Base model", c.base_model],
    ["Cartridge", `${fmt.int(c.num_tokens)} tokens`],
    ["Documents", c.corpus_tokens ? `${fmt.int(c.corpus_tokens)} tokens` : `${c.file_count} files`],
    ["Compression", c.corpus_tokens ? `${(c.corpus_tokens / c.num_tokens).toFixed(1)}×` : "–"],
  ];
  $("detail-facts").replaceChildren(...facts.map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: v }))));

  const rows = [
    ["No context", c.metrics?.no_context],
    ["Cartridge before training", c.metrics?.before_training],
    ["Cartridge after training", c.metrics?.after_training],
  ].filter(([, m]) => m);
  $("detail-metrics").replaceChildren(...rows.map(([label, m], i) =>
    el("tr", { class: i === rows.length - 1 ? "hl" : "" }, el("td", { text: label }), el("td", { class: "num", text: fmt.pct(m.agreement) }))));

  const n = c.metrics?.held_out_conversations;
  $("detail-metrics-note").textContent = !rows.length
    ? "No quality check for this cartridge."
    : n
      ? `Share of answer tokens where the model agrees with its answer when the documents are in the prompt, measured on ${n} held-out practice conversation${n === 1 ? "" : "s"}.${n < 10 ? " That's too few for a reliable number; build with more practice conversations." : ""}`
      : "Share of answer tokens where the model agrees with its answer when the documents are in the prompt.";

  renderChat(c);
  renderSnippet(c);
}

function describeSource(src) {
  if (src.type === "upload") return `${(src.files || []).length} uploaded file${(src.files || []).length === 1 ? "" : "s"}: ${(src.files || []).join(", ")}`;
  if (src.type === "import") return `Imported from ${src.file}`;
  if (src.type === "git") return [src.url, src.branch && `branch ${src.branch}`, src.subdir && `folder ${src.subdir}`].filter(Boolean).join(" · ");
  if (src.type === "web") return `${src.url} (up to ${src.max_pages} pages)`;
  return src.path || src.url || "";
}

function hoursLabel(h) {
  return { 1: "hour", 6: "6 hours", 24: "day", 168: "week" }[h] || `${h} hours`;
}

$("detail-sync").addEventListener("click", async () => {
  try {
    await api(`/v1/cartridges/${state.selectedId}/sync`, { method: "POST" });
    toast("Sync started");
    await refreshCartridges();
  } catch (err) { toast(err.message); }
});

$("detail-sync-every").addEventListener("change", async (e) => {
  try {
    await api(`/v1/cartridges/${state.selectedId}`, { method: "PATCH", json: { sync_every_hours: Number(e.target.value) } });
    toast("Saved");
    await refreshCartridges();
  } catch (err) { toast(err.message); }
});

// Create -------------------------------------------------------------

for (const tab of document.querySelectorAll("#source-tabs button")) {
  tab.addEventListener("click", () => setSourceType(tab.dataset.source));
}

function setSourceType(type) {
  state.sourceType = type;
  for (const tab of document.querySelectorAll("#source-tabs button")) tab.classList.toggle("active", tab.dataset.source === type);
  for (const panel of document.querySelectorAll(".source-panel")) panel.hidden = panel.dataset.panel !== type;
  $("sync-row").hidden = type === "upload";
  updateSubmit();
}

function updateSubmit() {
  const button = $("create-submit");
  if (state.sourceType === "upload") {
    const total = state.files.reduce((sum, f) => sum + f.size, 0);
    button.disabled = state.files.length === 0;
    button.textContent = state.files.length ? `Build cartridge from ${state.files.length} file${state.files.length > 1 ? "s" : ""} (${fmt.bytes(total)})` : "Build cartridge";
  } else {
    button.disabled = false;
    button.textContent = "Connect and build";
  }
}

function sourceFromForm() {
  if (state.sourceType === "git") {
    const src = { type: "git", url: $("git-url").value.trim() };
    if ($("git-branch").value.trim()) src.branch = $("git-branch").value.trim();
    if ($("git-subdir").value.trim()) src.subdir = $("git-subdir").value.trim();
    if ($("git-token").value) src.token_env = $("git-token").value;
    return src;
  }
  if (state.sourceType === "web") return { type: "web", url: $("web-url").value.trim(), max_pages: Number($("web-pages").value) || 200 };
  return { type: "folder", path: $("folder-path").value.trim() };
}

function nameFromSource(src) {
  const raw = src.url || src.path || "";
  const parts = raw.replace(/\/+$/, "").split("/").filter(Boolean);
  return (parts.pop() || "cartridge").replace(/\.git$/, "").replace(/\.html?$/, "");
}


const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("files").click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("files").click(); } });
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
dropzone.addEventListener("drop", (e) => { e.preventDefault(); dropzone.classList.remove("over"); addFiles(e.dataTransfer.files); });
$("files").addEventListener("change", (e) => addFiles(e.target.files));

function addFiles(fileList) {
  const accepted = new Set(state.config.accepted_file_types);
  const rejected = [];
  for (const file of fileList) {
    const dot = file.name.lastIndexOf(".");
    const suffix = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
    if (accepted.has(suffix)) state.files.push(file);
    else rejected.push(file.name);
  }
  if (rejected.length) toast(`Skipped unsupported files: ${rejected.join(", ")}`);
  if (!$("create-name").value && state.files[0]) $("create-name").value = state.files[0].name.replace(/\.[^.]+$/, "");
  renderFiles();
}

function renderFiles() {
  $("file-list").replaceChildren(...state.files.map((f) => el("li", {}, el("span", { text: f.name }), el("span", { text: fmt.bytes(f.size) }))));
  updateSubmit();
}

$("create-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("create-error");
  const settings = {
    base_model: $("create-model").value,
    tokens: Number($("create-tokens").value),
    samples: Number($("create-samples").value),
  };
  $("create-submit").disabled = true;
  try {
    let record;
    if (state.sourceType === "upload") {
      const form = new FormData();
      for (const f of state.files) form.append("files", f, f.name);
      form.append("name", $("create-name").value.trim());
      for (const [k, v] of Object.entries(settings)) form.append(k, v);
      record = await api("/v1/cartridges/upload", { method: "POST", form });
      state.files = [];
      renderFiles();
    } else {
      const source = sourceFromForm();
      if (!source.url && !source.path) throw new Error("Enter where the source lives first.");
      record = await api("/v1/cartridges", {
        method: "POST",
        json: { name: $("create-name").value.trim() || nameFromSource(source), sources: [source],
                sync_every_hours: Number($("create-sync").value), ...settings },
      });
    }
    $("create-name").value = "";
    state.selectedId = record.id;
    await refreshCartridges();
    toast("Build started");
  } catch (err) {
    showError("create-error", err.message);
  } finally {
    updateSubmit();
  }
});

$("import-button").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  const form = new FormData();
  form.append("file", file, file.name);
  showError("create-error");
  try {
    const record = await api("/v1/cartridges/import", { method: "POST", form });
    state.selectedId = record.id;
    await refreshCartridges();
    toast(`Imported ${record.name}`);
  } catch (err) { showError("create-error", err.message); }
});

// Download and delete ----------------------------------------------

$("detail-download").addEventListener("click", async () => {
  const c = state.cartridges.find((x) => x.id === state.selectedId);
  try {
    const response = await api(`/v1/cartridges/${c.id}/download`, { raw: true });
    const url = URL.createObjectURL(await response.blob());
    el("a", { href: url, download: `${c.name}.safetensors` }).click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  } catch (err) { toast(err.message); }
});

$("detail-delete").addEventListener("click", () => {
  $("detail-delete").hidden = true;
  $("detail-delete-confirm").hidden = false;
});
$("detail-delete-confirm").addEventListener("click", async () => {
  const id = state.selectedId;
  try {
    await api(`/v1/cartridges/${id}`, { method: "DELETE" });
    state.selectedId = null;
    delete state.chats[id];
    await refreshCartridges();
    toast("Deleted");
  } catch (err) { toast(err.message); }
});

// Playground -------------------------------------------------------

// Minimal, safe Markdown: **bold** and `code`. Builds DOM nodes; never parses HTML.
function richText(text) {
  const fragment = document.createDocumentFragment();
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) fragment.append(el("strong", { text: part.slice(2, -2) }));
    else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) fragment.append(el("code", { text: part.slice(1, -1) }));
    else fragment.append(part);
  }
  return fragment;
}

function messageNode(m) {
  const node = el("div", { class: `msg ${m.role}` });
  node.append(m.role === "assistant" ? richText(m.content) : m.content);
  return node;
}

function renderChat(c) {
  const history = state.chats[c.id] || [];
  const chat = $("chat");
  chat.replaceChildren(...history.map(messageNode));
  if (!history.length) chat.append(el("p", { class: "muted", text: `Ask ${c.name} something. Answers come from the cartridge, with no documents in the prompt.` }));
  chat.scrollTop = chat.scrollHeight;
}

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const c = state.cartridges.find((x) => x.id === state.selectedId);
  const input = $("chat-input");
  const text = input.value.trim();
  if (!c || !text) return;
  const history = (state.chats[c.id] ||= []);
  history.push({ role: "user", content: text });
  const reply = { role: "assistant", content: "" };
  history.push(reply);
  input.value = "";
  $("chat-send").disabled = true;
  renderChat(c);
  const showReply = () => {
    // Re-find the bubble each time: a status refresh may have re-rendered the chat.
    const last = $("chat").lastElementChild;
    if (state.selectedId === c.id && last) last.replaceChildren(richText(reply.content));
    $("chat").scrollTop = $("chat").scrollHeight;
  };

  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.key}` },
      body: JSON.stringify({ model: c.id, stream: true, messages: history.slice(0, -1) }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data?.error?.message || `Request failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const event of events) {
        const line = event.trim();
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        const delta = JSON.parse(line.slice(6)).choices?.[0]?.delta?.content;
        if (delta) { reply.content += delta; showReply(); }
      }
    }
  } catch (err) {
    history.pop();
    history.push({ role: "error", content: err.message });
    renderChat(c);
    history.pop();
  } finally {
    $("chat-send").disabled = false;
    input.focus();
  }
});

// Snippets ---------------------------------------------------------

function snippetFor(c, lang) {
  const base = `${location.origin}/v1`;
  if (lang === "curl") {
    return `curl ${base}/chat/completions \\
  -H "Authorization: Bearer $KVPACK_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${c.id}", "messages": [{"role": "user", "content": "Summarize the key points."}]}'`;
  }
  return `import os
from openai import OpenAI

client = OpenAI(base_url="${base}", api_key=os.environ["KVPACK_API_KEY"])
reply = client.chat.completions.create(
    model="${c.id}",  # ${c.name}
    messages=[{"role": "user", "content": "Summarize the key points."}],
)
print(reply.choices[0].message.content)`;
}

function renderSnippet(c) {
  $("snippet").textContent = snippetFor(c, state.snippetLang);
  for (const b of document.querySelectorAll(".snippet-tabs button")) b.classList.toggle("active", b.dataset.snippet === state.snippetLang);
}
for (const b of document.querySelectorAll(".snippet-tabs button")) {
  b.addEventListener("click", () => {
    state.snippetLang = b.dataset.snippet;
    const c = state.cartridges.find((x) => x.id === state.selectedId);
    if (c) renderSnippet(c);
  });
}
$("snippet-copy").addEventListener("click", () => copy($("snippet").textContent));

// ---------------------------------------------------------------- keys and usage

async function loadKeys() {
  try {
    const keys = (await api("/v1/keys")).data;
    $("keys").replaceChildren(...keys.map((k) => el("tr", {},
      el("td", { text: k.name }),
      el("td", { class: "num", text: `${k.prefix}…` }),
      el("td", { text: fmt.date(k.created_at) }),
      el("td", { text: fmt.date(k.last_used_at) }),
      el("td", {}, el("button", { type: "button", class: "danger", text: "Revoke", onclick: (e) => revokeKey(k.id, e.target) })),
    )));
  } catch (err) { toast(err.message); }
}

async function revokeKey(id, button) {
  if (button.dataset.confirm !== "1") { button.dataset.confirm = "1"; button.textContent = "Revoke permanently"; return; }
  try {
    await api(`/v1/keys/${id}`, { method: "DELETE" });
    toast("Revoked");
    await loadKeys();
    await api("/v1/me").catch(() => $("sign-out").click()); // signed out if it was this session's key
  } catch (err) { toast(err.message); }
}

$("key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const key = await api("/v1/keys", { method: "POST", json: { name: $("key-name").value.trim() || "default" } });
    $("key-name").value = "";
    showNewKey(key.api_key, () => showView("keys"));
  } catch (err) { toast(err.message); }
});

async function loadUsage() {
  try {
    const u = await api("/v1/usage?days=30");
    const stats = [
      [fmt.int(u.builds), "cartridges built"],
      [fmt.duration(u.build_gpu_seconds), "of GPU time for builds"],
      [fmt.int(u.chat_requests), "chat requests"],
      [fmt.int(u.prompt_tokens + u.completion_tokens), `tokens (${fmt.int(u.completion_tokens)} generated)`],
    ];
    $("usage").replaceChildren(...stats.map(([value, label]) => el("div", { class: "stat" }, el("b", { text: value }), el("span", { text: label }))));
  } catch (err) { toast(err.message); }
}

// ---------------------------------------------------------------- boot

(async function boot() {
  try {
    state.config = await api("/v1/config");
  } catch {
    document.body.textContent = "kvpack Studio is unavailable right now. Try again in a minute.";
    return;
  }
  $("signup-form").hidden = !state.config.signup_open;
  $("invite-row").hidden = state.config.first_account;
  $("signup-invite").required = !state.config.first_account;
  if (state.config.first_account) $("signup-title").textContent = "Set up your Studio";
  const src = state.config.sources || {};
  document.querySelector('#source-tabs [data-source="folder"]').hidden = !src.folder;
  $("folder-hint").textContent = src.folder ? `Must be inside ${src.folder_roots.join(" or ")}.` : "";
  $("folder-path").placeholder = src.folder_roots?.[0] ? `${src.folder_roots[0]}/handbook` : "";
  $("git-token-row").hidden = !(src.git_token_envs || []).length;
  for (const name of src.git_token_envs || []) $("git-token").append(el("option", { value: name, text: name }));
  setSourceType("git");
  $("create-model").replaceChildren(...state.config.base_models.map((m) => el("option", { value: m, text: m })));
  $("create-tokens").value = state.config.defaults.tokens;
  $("create-tokens").max = state.config.limits.max_cartridge_tokens;
  $("create-samples").value = state.config.defaults.samples;
  $("create-samples").max = state.config.limits.max_samples;
  $("files").accept = state.config.accepted_file_types.join(",");

  const saved = readKey();
  if (saved) {
    try { await signIn(saved); return enterApp(); } catch { writeKey(null); state.key = null; }
  }
  showView("auth");
})();
