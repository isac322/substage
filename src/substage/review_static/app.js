"use strict";
// substage review UI: video list + video page (video, synced EN/공식/substage subtitles,
// cue table with filters, refeval scene jump list, notes). No dependencies.

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

function mmss(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const p = (x) => String(x).padStart(2, "0");
  return h ? `${h}:${p(m)}:${p(r)}` : `${p(m)}:${p(r)}`;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
  return body;
}

// ------------------------------------------------------------------ search helpers

// Index of the last item with item.s <= t (items sorted by s), or -1.
function lastStartingBefore(items, t, key = "s") {
  let lo = 0, hi = items.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid][key] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

// Text of every cue active at t (overlapping cues joined).
function activeCues(cues, t) {
  const i = lastStartingBefore(cues, t);
  const out = [];
  for (let j = i; j >= 0 && j > i - 8; j--) {
    if (cues[j].e > t) out.unshift(cues[j]);
  }
  return out;
}

function isQuotedOstText(txt) {
  const s = (txt || "").trim();
  return s.length >= 2 && /^["'“”‘’\[].*["'“”‘’\]]$/s.test(s);
}

function activeSplitText(cues, t) {
  const act = activeCues(cues, t);
  if (!act.length) return { top: "", bottom: "", all: "" };
  const hasMulti = act.length >= 2;
  const top = [], bottom = [];
  for (const c of act) {
    if (c.pos === "top" || (hasMulti && isQuotedOstText(c.t))) top.push(c.t);
    else bottom.push(c.t);
  }
  const allParts = (top.length && bottom.length)
    ? [`[상단] ${top.join(" / ")}`, `[하단] ${bottom.join(" / ")}`]
    : act.map((c) => c.t);
  return { top: top.join("\n"), bottom: bottom.join("\n"), all: allParts.join("\n") };
}

function activeText(cues, t) {
  return activeCues(cues, t).map((c) => c.t).join("\n");
}
// Character-bigram Dice similarity (0..100) ignoring spaces/punctuation.
function bigrams(s) {
  const t = s.replace(/[\s\p{P}\p{S}]/gu, "");
  const m = new Map();
  for (let i = 0; i < t.length - 1; i++) {
    const g = t.slice(i, i + 2);
    m.set(g, (m.get(g) || 0) + 1);
  }
  if (t.length === 1) m.set(t, 1);
  return m;
}
function dice(a, b) {
  const A = bigrams(a), B = bigrams(b);
  let na = 0, nb = 0, inter = 0;
  for (const v of A.values()) na += v;
  for (const v of B.values()) nb += v;
  if (!na && !nb) return 100;
  for (const [g, v] of A) inter += Math.min(v, B.get(g) || 0);
  return Math.round((200 * inter) / (na + nb));
}

// 어절(whitespace token) LCS: returns [flagsForA, flagsForB] where flag=true marks
// tokens absent from the other text (insertions/deletions). Flag slots cover every
// split part incl. whitespace runs (never rendered as marks).
function diffFlags(a, b) {
  const A = a.split(/(\s+)/), B = b.split(/(\s+)/);
  const fa = new Array(A.length).fill(false), fb = new Array(B.length).fill(false);
  if (a === b) return [fa, fb];
  if (A.length * B.length > 1e6) { fa.fill(true); fb.fill(true); return [fa, fb]; }
  const W = B.length + 1;
  const dp = new Uint32Array((A.length + 1) * W);
  for (let i = A.length - 1; i >= 0; i--) {
    for (let j = B.length - 1; j >= 0; j--) {
      dp[i * W + j] = A[i] === B[j] ? dp[(i + 1) * W + j + 1] + 1
        : Math.max(dp[i * W + j + 1], dp[(i + 1) * W + j]);
    }
  }
  let i = 0, j = 0;
  while (i < A.length && j < B.length) {
    if (A[i] === B[j]) { i++; j++; }
    else if (dp[i * W + j + 1] >= dp[(i + 1) * W + j]) { fb[j] = true; j++; }
    else { fa[i] = true; i++; }
  }
  for (; i < A.length; i++) fa[i] = true;
  for (; j < B.length; j++) fb[j] = true;
  return [fa, fb];
}


// Appends text to node, wrapping flagged non-whitespace runs in <span class="dmark">.
function markTokens(node, text, flags) {
  const parts = text.split(/(\s+)/);
  for (let i = 0; i < parts.length; i++) {
    const t = parts[i];
    if (!t) continue;
    if (!/^\s+$/.test(t) && flags[i]) {
      const s = document.createElement("span");
      s.className = "dmark";
      s.textContent = t;
      node.append(s);
    } else node.append(document.createTextNode(t));
  }
}

// ------------------------------------------------------------------ state

const S = {
  video: null,          // detail from /api/videos/<id>
  arm: null,
  rows: [],            // table rows for the current arm
  rowEls: [],
  groups: [],          // refeval groups sorted by start_ms
  metric: "dice",      // "chrf" when refeval groups carry per-arm chrF
  curRow: -1,
  clock: null,
  raf: 0,
  last: { en: null, off: null, ko: null },
  ov: { en: null, pair: null, topPair: null },  // last texts rendered into the video overlay
  prevT: 0,            // playback time at the previous tick (autopause boundary detect)
  pausedRow: -1,       // row index already auto-paused once
  hasVideo: false,
  dialogRow: null,
  opts: {},            // persisted UI options (see bindOpt)
};

// ------------------------------------------------------------------ routing

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", route);

function route() {
  const m = location.hash.match(/^#\/video\/([^/]+)$/);
  stopLoop();
  if (m) showVideo(decodeURIComponent(m[1]));
  else showList();
}

async function showList() {
  $("video-view").hidden = true;
  $("list-view").hidden = false;
  $("video-title").textContent = "";
  document.title = "substage review";
  const v = $("video");
  v.pause(); v.removeAttribute("src"); v.load();
  const ul = $("video-list");
  ul.replaceChildren();
  let videos = [];
  try { videos = await api("/api/videos"); } catch (e) { ul.append(el("li", "muted", `오류: ${e.message}`)); return; }
  $("list-empty").hidden = videos.length > 0;
  for (const f of videos) {
    const a = el("a", null, `${f.title}${f.year ? ` (${f.year})` : ""}`);
    a.href = `#/video/${encodeURIComponent(f.id)}`;
    const bits = [f.id, f.has_video ? "영상" : "텍스트만", `arm: ${f.arms.join(", ") || "-"}`];
    if (f.has_refeval) bits.push("refeval");
    if (f.n_notes) bits.push(`메모 ${f.n_notes}`);
    a.append(el("span", "meta", bits.join(" · ")));
    const li = el("li");
    li.append(a);
    ul.append(li);
  }
}

// ------------------------------------------------------------------ video page

async function showVideo(id) {
  $("list-view").hidden = true;
  $("video-view").hidden = false;
  let video;
  try { video = await api(`/api/videos/${encodeURIComponent(id)}`); } catch (e) {
    $("video-title").textContent = `오류: ${e.message}`;
    return;
  }
  S.video = video;
  $("video-title").textContent = `${video.title}${video.year ? ` (${video.year})` : ""}`;
  document.title = `${video.title} · substage review`;

  // refeval groups and metric
  const ref = video.refeval || {};
  S.groups = Array.isArray(ref.groups) ? ref.groups.filter((g) => Number.isFinite(g.start_ms)).slice()
    .sort((a, b) => a.start_ms - b.start_ms) : [];

  // arm selector
  const sel = $("arm-select");
  sel.replaceChildren(...video.arms.map((a) => { const o = el("option", null, a); o.value = a; return o; }));
  const saved = localStorage.getItem(`substage.arm.${id}`);
  S.arm = video.arms.includes(saved) ? saved : video.arms.includes("B") ? "B" : video.arms[0] || null;
  if (S.arm) sel.value = S.arm;
  sel.disabled = video.arms.length < 2;
  sel.onchange = () => {
    S.arm = sel.value;
    localStorage.setItem(`substage.arm.${id}`, S.arm);
    rebuild();
  };

  setupClock(video);
  renderMetrics(ref);
  rebuild();
  startLoop();
}

function rebuild() {
  const f = S.video;
  S.metric = S.groups.some((g) => g.chrf && S.arm in g.chrf) ? "chrf" : "dice";
  $("score-head").textContent = S.metric === "chrf" ? "chrF" : "유사도";
  $("thr-label").textContent = S.metric === "chrf" ? "chrF <" : "유사도 <";
  S.rows = buildRows(f.en, f.official, f.candidates[S.arm] || []);
  attachScores(S.rows);
  attachNotes();
  renderTable();
  renderScenes();
  renderNotesList();
  S.last = { en: null, off: null, ko: null };
  S.ov = { en: null, pair: null, topPair: null };
  S.curRow = -1;
  S.pausedRow = -1;
  tick(true);
}

// One row per English cue; official / substage cues attach to the English cue they
// overlap most. Cues overlapping no English cue become their own rows.
function buildRows(en, off, ko) {
  const base = en.map((c) => ({ s: c.s, e: c.e, en: [c.t], off: [], ko: [] }));
  const extra = [];
  const place = (cues, key) => {
    for (const c of cues) {
      let best = null, bestOv = 0;
      const hasSimultaneousSpoken = isQuotedOstText(c.t) && cues.some(
        (other) => other !== c && !isQuotedOstText(other.t) && Math.min(other.e, c.e) - Math.max(other.s, c.s) > 100
      );
      const isTopCue = c.pos === "top" || hasSimultaneousSpoken;
      if (!isTopCue) {
        const i = lastStartingBefore(base, c.e - 1);
        for (let j = i; j >= 0 && base[j].s > c.s - 60000; j--) {
          const ov = Math.min(c.e, base[j].e) - Math.max(c.s, base[j].s);
          const minDur = Math.min(c.e - c.s, base[j].e - base[j].s);
          if (ov >= Math.min(200, minDur * 0.3) && ov > bestOv) {
            bestOv = ov;
            best = base[j];
          }
        }
      }
      if (!best) {
        for (const r of extra) {
          if (Boolean(r.isTop) !== isTopCue) continue;
          const ov = Math.min(c.e, r.e) - Math.max(c.s, r.s);
          if (ov >= 150 && ov > bestOv) { bestOv = ov; best = r; }
        }
      }
      if (best) best[key].push(c.t);
      else {
        const r = { s: c.s, e: c.e, en: [], off: [], ko: [], isTop: isTopCue };
        r[key].push(c.t);
        extra.push(r);
      }
    }
  };
  place(off, "off");
  place(ko, "ko");
  return base.concat(extra).sort((a, b) => a.s - b.s || a.e - b.e);
}

function groupAt(t) {
  const i = lastStartingBefore(S.groups, t, "start_ms");
  if (i >= 0 && t < S.groups[i].end_ms) return S.groups[i];
  return null;
}

function attachScores(rows) {
  for (const r of rows) {
    r.group = S.groups.length ? groupAt((r.s + r.e) / 2) : null;
    const off = r.off.join("\n"), ko = r.ko.join("\n");
    if (S.metric === "chrf") {
      const v = r.group && r.group.chrf ? r.group.chrf[S.arm] : null;
      r.score = v == null ? null : Math.round(v <= 1 ? v * 100 : v);
    } else if (off || ko) {
      r.score = off && ko ? dice(off, ko) : 0;
    } else r.score = null;
  }
}

function attachNotes() {
  for (const r of S.rows) r.notes = [];
  for (const n of S.video.notes) {
    const i = Math.max(0, lastStartingBefore(S.rows, n.time_ms));
    if (S.rows[i]) S.rows[i].notes.push(n);
  }
}

function renderTable() {
  const tbody = $("cue-body");
  const frag = document.createDocumentFragment();
  S.rowEls = S.rows.map((r, i) => {
    const tr = el("tr");
    if (r.group && (i === 0 || S.rows[i - 1].group !== r.group)) tr.classList.add("group-start");
    const offText = r.off.join("\n"), koText = r.ko.join("\n");
    const [offFlags, koFlags] = diffFlags(offText, koText);
    const offTd = el("td", "off"), koTd = el("td", "ko");
    markTokens(offTd, offText, offFlags);
    markTokens(koTd, koText, koFlags);
    tr.append(el("td", "time", mmss(r.s)), el("td", "en", r.en.join("\n")), offTd, koTd);
    const sc = el("td", "score", r.score == null ? "" : String(r.score));
    if (r.group) sc.title = `${r.group.id || ""}${r.group.scene_id ? ` · ${r.group.scene_id}` : ""}`;
    tr.append(sc);
    const nt = el("td", "note");
    const btn = el("button", null, "＋");
    btn.type = "button";
    btn.title = "이 행에 메모";
    btn.onclick = (ev) => { ev.stopPropagation(); openNoteDialog(r); };
    nt.append(btn);
    for (const n of r.notes) nt.append(noteChip(n));
    tr.append(nt);
    tr.onclick = () => S.clock.seek(r.s + 1);
    r.search = [r.en, r.off, r.ko].flat().join("\n").toLowerCase()
      + "\n" + r.notes.map((n) => n.comment).join("\n").toLowerCase();
    frag.append(tr);
    return tr;
  });
  tbody.replaceChildren(frag);
  applyFilters();
}

function noteChip(n) {
  const d = el("div", "n");
  d.append(delButton(n), document.createTextNode(n.comment));
  d.title = `${n.created_at || ""}${n.arm ? ` · arm ${n.arm}` : ""}`;
  return d;
}

function delButton(n) {
  const b = el("button", "del", "삭제");
  b.type = "button";
  b.title = "메모 삭제";
  b.onclick = (ev) => { ev.stopPropagation(); deleteNote(n); };
  return b;
}

async function deleteNote(n) {
  if (!confirm("이 메모를 삭제할까요?")) return;
  try {
    await api(`/api/notes/${encodeURIComponent(S.video.id)}/${n.id}`, { method: "DELETE" });
  } catch (e) { alert(`메모 삭제 실패: ${e.message}`); return; }
  S.video.notes = S.video.notes.filter((x) => x !== n);
  const keepScroll = $("table-scroll").scrollTop;
  attachNotes();
  renderTable();
  renderNotesList();
  $("table-scroll").scrollTop = keepScroll;
}

function applyFilters() {
  const q = $("search").value.trim().toLowerCase();
  const diff = $("f-diff").checked, onlyNotes = $("f-notes").checked;
  const thr = Number($("f-thr").value);
  $("thr-val").textContent = String(thr);
  let shown = 0;
  S.rows.forEach((r, i) => {
    const low = r.score != null && r.score < thr;
    const tr = S.rowEls[i];
    tr.children[4].classList.toggle("low", low);
    const ok = (!q || r.search.includes(q)) && (!diff || low) && (!onlyNotes || r.notes.length);
    tr.hidden = !ok;
    if (ok) shown++;
  });
  $("row-count").textContent = `${shown} / ${S.rows.length}행`;
}
for (const id of ["search", "f-diff", "f-thr", "f-notes"]) $(id).addEventListener("input", applyFilters);

// ------------------------------------------------------------------ refeval panels

function renderScenes() {
  const ref = S.video.refeval || {};
  const scenes = Array.isArray(ref.scenes) ? ref.scenes : [];
  const box = $("scenes-box"), ol = $("scene-list");
  box.hidden = scenes.length === 0;
  $("scenes-count").textContent = scenes.length ? `(${scenes.length})` : "";
  const groupsById = new Map(S.groups.map((g) => [g.id, g]));
  ol.replaceChildren(...scenes.map((sc) => {
    const li = el("li");
    const head = el("div", null, `${sc.id || ""} ${mmss(sc.start_ms || 0)}–${mmss(sc.end_ms || 0)}`);
    if (sc.reason) head.append(el("span", "scores", ` · ${sc.reason}`));
    const v = sc.verdicts || {};
    const scores = Object.keys(v).map((a) => `${a}${a === S.arm ? "*" : ""} ${v[a] && v[a].score != null ? v[a].score : "-"}`);
    if (scores.length) head.append(el("div", "scores", scores.join(" · ")));
    li.append(head);
    const mine = v[S.arm];
    for (const line of (mine && mine.lines) || []) {
      if (!line || line.category === "none") continue;
      const g = groupsById.get(line.group_id);
      const d = el("div", `verdict sev${line.severity || 0}`,
        `${g ? mmss(g.start_ms) : ""} [${line.category}/${line.severity}] ${line.note || ""}`);
      if (g) d.onclick = (ev) => { ev.stopPropagation(); S.clock.seek(g.start_ms + 1); };
      li.append(d);
    }
    li.onclick = () => S.clock.seek((sc.start_ms || 0) + 1);
    return li;
  }));
}

function flatNumbers(obj, prefix = "", out = {}) {
  for (const [k, v] of Object.entries(obj || {})) {
    if (typeof v === "number" || v === null) out[prefix + k] = v;
    else if (v && typeof v === "object" && !Array.isArray(v) && !prefix) flatNumbers(v, `${k}.`, out);
  }
  return out;
}

function renderMetrics(ref) {
  const m = ref.metrics;
  const box = $("metrics-box"), div = $("metrics");
  div.replaceChildren();
  if (!m || typeof m !== "object" || !Object.keys(m).length) { box.hidden = true; return; }
  box.hidden = false;
  const arms = Object.keys(m);
  const flat = Object.fromEntries(arms.map((a) => [a, flatNumbers(m[a])]));
  const keys = [...new Set(arms.flatMap((a) => Object.keys(flat[a])))];
  const table = el("table"), hr = el("tr");
  hr.append(el("th", null, ""), ...arms.map((a) => el("th", null, a)));
  table.append(hr);
  for (const k of keys) {
    const tr = el("tr");
    tr.append(el("td", null, k), ...arms.map((a) => {
      const v = flat[a][k];
      return el("td", null, v == null ? "-" : Number.isInteger(v) ? String(v) : v.toFixed(3));
    }));
    table.append(tr);
  }
  div.append(table);
}

// ------------------------------------------------------------------ notes

function renderNotesList() {
  const notes = S.video.notes.slice().sort((a, b) => a.time_ms - b.time_ms);
  $("notes-count").textContent = notes.length ? `(${notes.length})` : "";
  $("notes-list").replaceChildren(...notes.map((n) => {
    const li = el("li");
    const who = el("div", "who", `${mmss(n.time_ms)}${n.arm ? ` · ${n.arm}` : ""} · ${(n.created_at || "").replace("T", " ").slice(0, 16)}`);
    who.append(delButton(n));
    li.append(who);
    li.append(el("div", null, n.comment));
    li.style.whiteSpace = "pre-line";
    li.onclick = () => S.clock.seek(n.time_ms + 1);
    return li;
  }));
}

async function saveNote(time_ms, cue, comment) {
  const note = await api(`/api/notes/${encodeURIComponent(S.video.id)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ time_ms: Math.round(time_ms), cue, arm: S.arm, comment }),
  });
  S.video.notes.push(note);
  const keepScroll = $("table-scroll").scrollTop;
  attachNotes();
  renderTable();
  renderNotesList();
  $("table-scroll").scrollTop = keepScroll;
  S.curRow = -1;
  tick(true);
}

function openNoteDialog(r) {
  S.dialogRow = r;
  $("note-dialog-time").textContent = mmss(r.s);
  const box = $("note-dialog-cue");
  box.replaceChildren();
  for (const [label, lines] of [["EN", r.en], ["공식", r.off], [`substage ${S.arm || ""}`, r.ko]]) {
    box.append(el("b", null, label), el("div", null, lines.join("\n") || "—"));
  }
  $("note-dialog-text").value = "";
  $("note-dialog").returnValue = "";
  $("note-dialog").showModal();
  $("note-dialog-text").focus();
}

$("note-dialog").addEventListener("close", async () => {
  const r = S.dialogRow;
  S.dialogRow = null;
  const text = $("note-dialog-text").value.trim();
  if ($("note-dialog").returnValue !== "save" || !r || !text) return;
  try {
    await saveNote(r.s, { en: r.en.join("\n"), official: r.off.join("\n"), substage: r.ko.join("\n") }, text);
  } catch (e) { alert(`메모 저장 실패: ${e.message}`); }
});
$("note-dialog-text").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) { ev.preventDefault(); $("note-dialog").close("save"); }
});

async function saveFreeNote() {
  const text = $("free-note").value.trim();
  if (!text || !S.video) return;
  const t = S.clock.now();
  try {
    await saveNote(t, { en: S.last.en || "", official: S.last.off || "", substage: S.last.ko || "" }, text);
    $("free-note").value = "";
  } catch (e) { alert(`메모 저장 실패: ${e.message}`); }
}
$("free-note-save").onclick = saveFreeNote;
$("free-note").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) { ev.preventDefault(); saveFreeNote(); }
});

// ------------------------------------------------------------------ clock / sync

function setupClock(video) {
  const v = $("video");
  const hasVideo = !!video.video;
  S.hasVideo = hasVideo;
  $("video-wrap").hidden = !hasVideo;
  $("textonly-bar").hidden = hasVideo;
  if (hasVideo) {
    if (v.getAttribute("src") !== video.video) v.src = video.video;
    S.clock = {
      now: () => v.currentTime * 1000,
      seek: (ms) => { v.currentTime = ms / 1000; tick(true); },
      paused: () => v.paused,
      pause: () => v.pause(),
      toggle: () => { if (v.paused) v.play().catch(() => {}); else v.pause(); },
    };
    return;
  }
  v.pause(); v.removeAttribute("src"); v.load();
  let base = 0, startedAt = null;
  const btn = $("clock-toggle");
  btn.textContent = "▶ 재생";
  btn.onclick = () => {
    if (startedAt == null) { startedAt = performance.now(); btn.textContent = "❚❚ 정지"; }
    else { base += performance.now() - startedAt; startedAt = null; btn.textContent = "▶ 재생"; }
  };
  S.clock = {
    now: () => (startedAt == null ? base : base + performance.now() - startedAt),
    seek: (ms) => { base = ms; if (startedAt != null) startedAt = performance.now(); tick(true); },
    paused: () => startedAt == null,
    pause: () => { if (startedAt != null) btn.onclick(); },
    toggle: () => btn.onclick(),
  };
}

function setText(id, key, text) {
  if (S.last[key] === text) return;
  S.last[key] = text;
  $(id).textContent = text;
}

function tick(force) {
  if (!S.video || !S.clock) return;
  const t = S.clock.now();
  const en = activeText(S.video.en, t);
  const offSplit = activeSplitText(S.video.official, t);
  const koSplit = activeSplitText(S.video.candidates[S.arm] || [], t);
  setText("sub-en", "en", en);
  setText("sub-official", "off", offSplit.all);
  setText("sub-substage", "ko", koSplit.all);

  // top overlay for {\an8} / simultaneous on-screen text cards
  const topOff = offSplit.top, topKo = koSplit.top;
  const topPair = topOff + "\0" + topKo;
  if (S.ov.topPair !== topPair) {
    S.ov.topPair = topPair;
    const [tOffFlags, tKoFlags] = diffFlags(topOff, topKo);
    const tOffEl = $("ov-top-off"), tKoEl = $("ov-top-ko");
    tOffEl.replaceChildren();
    tKoEl.replaceChildren();
    markTokens(tOffEl, topOff, tOffFlags);
    markTokens(tKoEl, topKo, tKoFlags);
  }
  $("ov-top").hidden = !(S.hasVideo && S.opts.overlay && (topOff || topKo));

  // bottom video overlay (spoken dialogue: 공식 | substage side by side, diff-marked; EN optional)
  const botOff = offSplit.bottom, botKo = koSplit.bottom;
  const showEn = S.opts.ovEn && !!en;
  $("ov-en").hidden = !showEn;
  if (showEn && S.ov.en !== en) { S.ov.en = en; $("ov-en").textContent = en; }
  const pair = botOff + "\0" + botKo;
  if (S.ov.pair !== pair) {
    S.ov.pair = pair;
    const [offFlags, koFlags] = diffFlags(botOff, botKo);
    const offEl = $("ov-off"), koEl = $("ov-ko");
    offEl.replaceChildren();
    koEl.replaceChildren();
    markTokens(offEl, botOff, offFlags);
    markTokens(koEl, botKo, koFlags);
  }
  $("ov").hidden = !(S.hasVideo && S.opts.overlay && (botOff || botKo || showEn));
  $("free-note-time").textContent = mmss(t);
  if (!$("textonly-bar").hidden) $("clock-time").textContent = mmss(t);
  const i = lastStartingBefore(S.rows, t);
  if (i !== S.curRow || force === true) {
    if (S.curRow >= 0 && S.rowEls[S.curRow]) S.rowEls[S.curRow].classList.remove("current");
    S.curRow = i;
    const tr = S.rowEls[i];
    if (tr) {
      tr.classList.add("current");
      if ($("f-follow").checked && !tr.hidden) scrollRowIntoView(tr);
    }
  }
  // optional auto-pause once, when playback crosses a cue's end
  const r = S.rows[i];
  if (r && S.opts.autopause && i !== S.pausedRow && !S.clock.paused()
      && S.prevT < r.e && t >= r.e) {
    S.clock.pause();
    S.pausedRow = i;
  }
  S.prevT = t;
}

function scrollRowIntoView(tr) {
  const box = $("table-scroll");
  const top = tr.offsetTop, bottom = top + tr.offsetHeight;
  if (top < box.scrollTop + 40 || bottom > box.scrollTop + box.clientHeight - 20) {
    box.scrollTop = Math.max(0, top - box.clientHeight / 3);
  }
}

function startLoop() {
  stopLoop();
  const loop = () => { tick(false); S.raf = requestAnimationFrame(loop); };
  S.raf = requestAnimationFrame(loop);
}

function stopLoop() {
  if (S.raf) cancelAnimationFrame(S.raf);
  S.raf = 0;
}

// ------------------------------------------------------------------ options (persisted)

function bindOpt(id, key, def, onChange) {
  const box = $(id);
  const v = localStorage.getItem(key);
  box.checked = v === null ? def : v === "1";
  S.opts[key] = box.checked;
  box.addEventListener("change", () => {
    S.opts[key] = box.checked;
    localStorage.setItem(key, box.checked ? "1" : "0");
    if (onChange) onChange();
  });
}
bindOpt("opt-overlay", "overlay", true);
bindOpt("opt-ov-en", "ovEn", false);
bindOpt("opt-panel", "panel", true, () => { $("subs-panel").hidden = !S.opts.panel; });
bindOpt("opt-autopause", "autopause", false);
$("subs-panel").hidden = !S.opts.panel;

// ------------------------------------------------------------------ keyboard shortcuts

function stepRow(d) {
  const t = S.clock.now();
  const i = lastStartingBefore(S.rows, t);
  const n = i < 0 ? 0 : i + d;
  if (S.rows[n]) S.clock.seek(S.rows[n].s + 1);
}

document.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (document.querySelector("dialog[open]")) return;
  const tag = (ev.target.tagName || "").toLowerCase();
  if (["input", "textarea", "select", "button"].includes(tag) || ev.target.isContentEditable) return;
  if (!S.video || !S.clock) return;
  switch (ev.key) {
    case " ":
      ev.preventDefault();
      S.clock.toggle();
      break;
    case "j": case "J": stepRow(-1); break;
    case "l": case "L": stepRow(1); break;
    case "k": case "K":
      if (S.rows[S.curRow]) S.clock.seek(S.rows[S.curRow].s + 1);
      break;
    case "n": case "N":
      ev.preventDefault();
      $("free-note").scrollIntoView({ block: "nearest" });
      $("free-note").focus();
      break;
    case "?":
      ev.preventDefault();
      $("help-dialog").showModal();
      break;
    default:
      if (/^[1-9]$/.test(ev.key) && S.video.arms[+ev.key - 1]) {
        S.arm = S.video.arms[+ev.key - 1];
        $("arm-select").value = S.arm;
        localStorage.setItem(`substage.arm.${S.video.id}`, S.arm);
        rebuild();
      }
  }
});
$("help-btn").onclick = () => $("help-dialog").showModal();

