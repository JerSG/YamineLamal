"""A self-contained HTML replay of tracking plus possession labels, for looking at the output.

build_html() writes one .html file with the tracking (subsampled, quantised) and the labels
embedded, and a small canvas player: pitch, players coloured by team, ball, the current label,
and a full-match timeline coloured by label that can be clicked to seek. It needs no server and
no network.

The file embeds tracking. For the competition match, write it only inside a gitignored folder
(submission/ by default) and never publish it: the match data is licensed for the competition
only. It shows the model's output next to the data the model read. It is not a labelling tool:
nothing seen in it is fed back into the model.
"""

import base64
import json

import numpy as np

NAN_CODE = 65535

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0f1720; --panel:#17212c; --text:#e6edf3; --muted:#8b98a5; --line:rgba(255,255,255,.55);
          --grass:#1f6b3a; --grass2:#226f3d; --home:#3b82f6; --away:#f43f5e; --dead:#9aa4ae; --ball:#ffd43b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  main { max-width:1100px; margin:0 auto; padding:16px; }
  header { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 16px; margin-bottom:10px; }
  h1 { font-size:18px; margin:0; font-weight:600; }
  .muted { color:var(--muted); }
  .stage { position:relative; background:var(--panel); border-radius:10px; padding:10px; }
  canvas#pitch { width:100%; height:auto; display:block; border-radius:6px; }
  .badge { position:absolute; top:18px; left:18px; padding:4px 10px; border-radius:999px;
           font-weight:700; letter-spacing:.02em; background:rgba(0,0,0,.55); }
  .bar { margin-top:10px; }
  .bar .name { font-size:12px; color:var(--muted); margin:6px 0 3px; }
  canvas.timeline { width:100%; height:18px; display:block; border-radius:3px; cursor:pointer; }
  canvas.zoom { height:14px; margin-top:3px; cursor:default; }
  .controls { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin-top:12px; }
  button, select { background:#243241; color:var(--text); border:1px solid #33475b; border-radius:6px;
                   padding:6px 12px; font:inherit; cursor:pointer; }
  button:hover, select:hover { background:#2c3e51; }
  input[type=range] { flex:1 1 240px; accent-color:#9ec5ff; }
  .clock { font-variant-numeric:tabular-nums; min-width:170px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:10px; font-size:12px; color:var(--muted); }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
  footer { margin-top:12px; font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="title"></h1>
    <span class="muted" id="subtitle"></span>
  </header>
  <div class="stage">
    <canvas id="pitch" width="1060" height="686"></canvas>
    <div class="badge" id="badge">-</div>
    <div id="bars"></div>
    <div class="controls">
      <button id="play" title="space">Play</button>
      <select id="speed" title="playback speed">
        <option value="1">1x</option><option value="2">2x</option>
        <option value="4" selected>4x</option><option value="8">8x</option><option value="16">16x</option>
      </select>
      <input type="range" id="seek" min="0" value="0" step="1" aria-label="seek">
      <span class="clock" id="clock"></span>
    </div>
    <div class="legend">
      <span><span class="sw" style="background:var(--home)"></span><span id="lh">H</span></span>
      <span><span class="sw" style="background:var(--away)"></span><span id="la">A</span></span>
      <span><span class="sw" style="background:var(--dead)"></span>D dead ball</span>
      <span><span class="sw" style="background:#2a3440"></span>X not graded</span>
      <span><span class="sw" style="background:var(--ball)"></span>ball (hollow = filled from broadcast)</span>
      <span>thin bars: two minutes around the cursor</span>
      <span>keys: space play/pause, left/right 1 s, shift+left/right 10 s</span>
    </div>
  </div>
  <footer id="footer"></footer>
</main>
<script>
const D = __DATA__;
function decode(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Uint16Array(u8.buffer);
}
const P = decode(D.players), B = decode(D.ball), BF = D.ball_filled ? decode(D.ball_filled) : null;
const N = D.n_samples, S = D.n_slots, NAN = 65535;
const css = getComputedStyle(document.documentElement);
const col = k => css.getPropertyValue(k).trim();
const COL = { H: col('--home'), A: col('--away'), D: col('--dead'), X: '#2a3440' };
document.getElementById('title').textContent = D.title;
document.getElementById('subtitle').textContent = D.subtitle;
document.getElementById('footer').textContent = D.footer;
document.getElementById('lh').textContent = 'H ' + D.team_names[0];
document.getElementById('la').textContent = 'A ' + D.team_names[1];

// labels: run-length encoded at full frame rate, expanded per sample for lookup
function expand(rle) {
  const out = new Array(D.n_frames); let i = 0;
  for (const [lab, n] of rle) { for (let k = 0; k < n; k++) out[i++] = lab; }
  return out;
}
const tracks = D.label_tracks.map(t => ({ name: t.name, rle: t.rle, frames: expand(t.rle) }));

const cv = document.getElementById('pitch'), cx = cv.getContext('2d');
const W = cv.width, H = cv.height, M = 30, L = D.pitch[0], Wd = D.pitch[1];
const sx = (W - 2 * M) / L, sy = (H - 2 * M) / Wd;
const X = x => M + x * (W - 2 * M), Y = y => M + y * (H - 2 * M);
function pitch() {
  for (let i = 0; i < 12; i++) {
    cx.fillStyle = i % 2 ? col('--grass') : col('--grass2');
    cx.fillRect(i * W / 12, 0, W / 12 + 1, H);
  }
  cx.strokeStyle = col('--line'); cx.lineWidth = 2;
  cx.strokeRect(M, M, W - 2 * M, H - 2 * M);
  cx.beginPath(); cx.moveTo(W / 2, M); cx.lineTo(W / 2, H - M); cx.stroke();
  cx.beginPath(); cx.arc(W / 2, H / 2, 9.15 * sx, 0, 2 * Math.PI); cx.stroke();
  for (const side of [0, 1]) {
    const x0 = side ? W - M : M, d = side ? -1 : 1;
    cx.strokeRect(Math.min(x0, x0 + d * 16.5 * sx), H / 2 - 20.16 * sy, 16.5 * sx, 40.32 * sy);
    cx.strokeRect(Math.min(x0, x0 + d * 5.5 * sx), H / 2 - 9.16 * sy, 5.5 * sx, 18.32 * sy);
  }
}
function draw(i) {
  pitch();
  const base = i * S * 2;
  for (let s = 0; s < S; s++) {
    const qx = P[base + 2 * s], qy = P[base + 2 * s + 1];
    if (qx === NAN) continue;
    cx.beginPath(); cx.arc(X(qx / 65534), Y(qy / 65534), 7, 0, 2 * Math.PI);
    cx.fillStyle = D.slot_team[s] === 0 ? COL.H : COL.A; cx.fill();
    cx.lineWidth = 1.5; cx.strokeStyle = 'rgba(0,0,0,.6)'; cx.stroke();
  }
  const bx = B[2 * i], by = B[2 * i + 1];
  if (bx !== NAN) {
    cx.beginPath(); cx.arc(X(bx / 65534), Y(by / 65534), 5, 0, 2 * Math.PI);
    const filled = BF && BF[i] === 1;
    cx.lineWidth = 2.5; cx.strokeStyle = col('--ball');
    if (filled) cx.stroke(); else { cx.fillStyle = col('--ball'); cx.fill(); cx.strokeStyle = '#000'; cx.lineWidth = 1; cx.stroke(); }
  }
  const f = Math.min(i * D.step, D.n_frames - 1), lab = tracks[0].frames[f];
  const badge = document.getElementById('badge');
  const names = { H: D.team_names[0] + ' in possession', A: D.team_names[1] + ' in possession', D: 'dead ball', X: 'not graded' };
  badge.textContent = tracks[0].name + ': ' + lab + ' - ' + names[lab];
  badge.style.color = lab === 'X' ? col('--muted') : COL[lab];
  const t = D.frame_id[i];
  const secs = f / D.fps, mm = Math.floor(secs / 60), ss = Math.floor(secs % 60);
  document.getElementById('clock').textContent =
    'video ' + String(mm).padStart(2, '0') + ':' + String(ss).padStart(2, '0') + '  frame ' + t;
  for (const b of bars) b.cursor(f);
}
// timelines
const bars = [];
const host = document.getElementById('bars');
for (const t of tracks) {
  const wrap = document.createElement('div'); wrap.className = 'bar';
  const name = document.createElement('div'); name.className = 'name'; name.textContent = t.name;
  const c = document.createElement('canvas'); c.className = 'timeline'; c.width = 1060; c.height = 18;
  const z = document.createElement('canvas'); z.className = 'timeline zoom'; z.width = 1060; z.height = 14;
  z.title = 'two minutes around the cursor';
  wrap.append(name, c, z); host.append(wrap);
  const gz = z.getContext('2d');
  const g = c.getContext('2d');
  const img = document.createElement('canvas'); img.width = c.width; img.height = c.height;
  const gi = img.getContext('2d'); let pos = 0;
  for (const [lab, n] of t.rle) {
    gi.fillStyle = COL[lab];
    gi.fillRect(pos / D.n_frames * c.width, 0, Math.max(n / D.n_frames * c.width, 0.6), c.height);
    pos += n;
  }
  bars.push({ cursor(f) {
    g.drawImage(img, 0, 0); const x = f / D.n_frames * c.width;
    g.fillStyle = '#fff'; g.fillRect(x - 1, 0, 2, c.height);
    const half = 60 * D.fps, a = f - half, px = z.width / (2 * half);
    gz.fillStyle = '#121a23'; gz.fillRect(0, 0, z.width, z.height);
    let k = Math.max(0, a);
    while (k < Math.min(D.n_frames, f + half)) {
      let e = k; const lab = t.frames[k];
      while (e < Math.min(D.n_frames, f + half) && t.frames[e] === lab) e++;
      gz.fillStyle = COL[lab]; gz.fillRect((k - a) * px, 0, (e - k) * px + 0.5, z.height); k = e;
    }
    gz.fillStyle = '#fff'; gz.fillRect(z.width / 2 - 1, 0, 2, z.height);
  }});
  c.addEventListener('click', e => {
    const r = c.getBoundingClientRect();
    go(Math.round((e.clientX - r.left) / r.width * D.n_frames / D.step));
  });
}
const seek = document.getElementById('seek'); seek.max = N - 1;
let cur = D.start || 0, playing = false, last = 0, acc = 0;
function go(i) { cur = Math.max(0, Math.min(N - 1, i)); seek.value = cur; draw(cur); }
seek.addEventListener('input', () => go(+seek.value));
const btn = document.getElementById('play');
function toggle() { playing = !playing; btn.textContent = playing ? 'Pause' : 'Play'; last = performance.now(); acc = 0; if (playing) requestAnimationFrame(tick); }
btn.addEventListener('click', toggle);
function tick(now) {
  if (!playing) return;
  acc += (now - last) / 1000 * (+document.getElementById('speed').value) * D.fps / D.step; last = now;
  if (acc >= 1) { const k = Math.floor(acc); acc -= k; go(cur + k); if (cur >= N - 1) toggle(); }
  requestAnimationFrame(tick);
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  const step1 = Math.round(D.fps / D.step);
  if (e.code === 'Space') { e.preventDefault(); toggle(); }
  else if (e.code === 'ArrowRight') go(cur + (e.shiftKey ? 10 : 1) * step1);
  else if (e.code === 'ArrowLeft') go(cur - (e.shiftKey ? 10 : 1) * step1);
});
go(cur);
</script>
</body>
</html>
"""


def _quantise(xy):
    """Normalised pitch coordinates (0..1) -> uint16, NaN -> NAN_CODE; base64 text."""
    q = np.full(xy.shape, NAN_CODE, dtype="<u2")
    ok = np.isfinite(xy)
    q[ok] = np.round(np.clip(xy[ok], 0.0, 1.0) * 65534).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def _rle(labels):
    labels = np.asarray(labels)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [[str(labels[a]), int(b - a)] for a, b in zip(starts, ends)]


def build_html(out_path, players, ball, slot_team, label_tracks, pitch, title, subtitle="",
               footer="", team_names=("home", "away"), frame_ids=None, ball_filled=None,
               step=5, fps=25, start_frame=0):
    """Write the replay page.

    players (T, slots, >=2) and ball (T, >=2) in normalised pitch coordinates (0..1, NaN where not
    tracked); label_tracks: [(name, labels as 'H'/'A'/'D'/'X' per frame)], the first one drives the
    badge; ball_filled: optional bool per frame, ball positions that came from the broadcast.
    """
    T = len(ball)
    idx = np.arange(0, T, step)
    data = {
        "title": title, "subtitle": subtitle, "footer": footer,
        "team_names": list(team_names), "pitch": [float(pitch[0]), float(pitch[1])],
        "fps": fps, "step": step, "n_frames": T, "n_samples": len(idx),
        "n_slots": int(players.shape[1]), "slot_team": [int(t) for t in slot_team],
        "players": _quantise(players[idx, :, :2]), "ball": _quantise(ball[idx, :2]),
        "ball_filled": (base64.b64encode(np.asarray(ball_filled)[idx].astype("<u2").tobytes())
                        .decode("ascii") if ball_filled is not None else None),
        "frame_id": [int(f) for f in (np.arange(T) if frame_ids is None
                                      else np.asarray(frame_ids))[idx]],
        "label_tracks": [{"name": n, "rle": _rle(lab)} for n, lab in label_tracks],
        "start": int(start_frame // step),
    }
    for _, lab in label_tracks:
        if len(lab) != T:
            raise ValueError("label track has {} frames, tracking has {}".format(len(lab), T))
    page = PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data,
                                                                           separators=(",", ":")))
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return out_path


def write_match_page(out, players, ball, ball_filled, meta, labels, team, step=5):
    """The competition replay page; also called by run.py with the tracking it already read."""
    import games
    from . import config as C
    first = games.half_video_blocks(games.GAME)[0][1]
    build_html(out, players, ball, meta["slot_team"], [("our labels", labels)],
               games.GAMES[games.GAME]["pitch"],
               title="{} vs {}".format(*meta["names"]),
               subtitle="team {}: possession from automatic tracking".format(team),
               footer="Tracking as provided (automatic, uncorrected); labels from {}.csv. Local "
                      "file: the match data is licensed for the competition only, do not "
                      "publish it.".format(team),
               team_names=meta["names"], ball_filled=ball_filled, step=step, fps=C.FPS,
               start_frame=first)
    return out
