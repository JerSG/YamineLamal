"""Draw docs/assets/pitch.svg: the animated README banner.

    python scripts/make_banner.py

A simulated, scripted 16-second passage of play (no match data): 22 players and the ball, with the
possession label switching H -> A -> D -> H as the model would label it, and a timeline of those
labels underneath. Pure SVG animation (SMIL), no JavaScript, so it plays inside GitHub's README.
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets", "pitch.svg")

LOOP_S = 16.0
DT = 0.25
S = 8.0                     # SVG units per metre
MX, MY = 24.0, 108.0        # pitch origin in the SVG
L, W = 105.0, 68.0
HOME, AWAY, DEAD, LINE = "#3b82f6", "#e3262b", "#9aa4ae", "rgba(255,255,255,0.6)"

# (t, x, y) of the ball, in metres; linear in between. Home attacks left to right.
BALL = [(0.0, 42, 34), (1.2, 43, 36), (1.9, 45, 50), (3.0, 52, 52), (3.4, 53, 51),
        (4.2, 64, 37), (5.0, 69, 36), (5.6, 70, 35), (6.4, 61, 23), (7.4, 53, 19),
        (8.2, 51, 18), (9.0, 40, -2), (11.4, 39, -1.5), (11.8, 39, -0.5), (12.4, 40, 11),
        (13.2, 40, 12), (14.0, 31, 22), (14.8, 32, 24), (16.0, 42, 34)]
# possession label over time: (start, end, label)
SPELLS = [(0.0, 5.0, "H"), (5.0, 9.0, "A"), (9.0, 11.8, "D"), (11.8, 16.0, "H")]
TEXT = {"H": "H · home in possession", "A": "A · away in possession",
        "D": "D · dead ball"}
COLOUR = {"H": HOME, "A": AWAY, "D": DEAD}

HOME_BASE = [(5, 34), (25, 10), (22, 27), (22, 41), (25, 58), (45, 18), (40, 34), (45, 50),
             (62, 14), (66, 34), (62, 54)]
AWAY_BASE = [(100, 34), (82, 12), (85, 28), (85, 40), (82, 56), (68, 20), (70, 34), (68, 48),
             (55, 22), (52, 36), (56, 50)]


def label_at(t):
    for a, b, lab in SPELLS:
        if a <= t < b:
            return lab
    return SPELLS[-1][2]


def ball_at(t):
    ts = [k[0] for k in BALL]
    return np.interp(t, ts, [k[1] for k in BALL]), np.interp(t, ts, [k[2] for k in BALL])


def taker(t):
    """How strongly the throw-in taker is drawn to the ball: in from 9.4 s, out by 13.4 s."""
    return float(np.interp(t, [9.4, 11.0, 12.0, 13.4], [0, 1, 1, 0]))


def positions(t):
    """Players drift with the play: the team in possession pushes up, everyone is drawn towards
    the ball, the nearest most strongly; during a dead ball they relax."""
    bx, by = ball_at(t)
    lab = label_at(t)
    out = []
    for team, base in ((0, HOME_BASE), (1, AWAY_BASE)):
        sign = 1 if team == 0 else -1
        push = {"H": 7 if team == 0 else 3, "A": 3 if team == 0 else 7, "D": 0}[lab] * sign
        pull = 0.25 if lab == "D" else 0.95
        for i, (x0, y0) in enumerate(base):
            phase = 1.7 * i + 3.1 * team
            x = x0 + (push if i else 0) + 1.2 * np.sin(2 * np.pi * t / LOOP_S * 2 + phase)
            y = y0 + 1.5 * np.sin(2 * np.pi * t / LOOP_S + phase) + 0.18 * (by - y0)
            if team == 0 and i == 5 and taker(t) > 0:     # walks over to take the throw-in
                w = taker(t)
                x, y = x + w * (bx - 0.6 - x), y + w * (max(by, -0.8) - y)
            elif i:
                d = np.hypot(bx - x, by - y)
                w = pull * np.exp(-d / 7.0)
                gap = 0.8 if (lab == "H") == (team == 0) and lab != "D" else 2.5
                tx = bx - gap * sign
                x, y = x + w * (tx - x), y + w * (np.clip(by, 0.5, W - 0.5) - y)
            out.append((x, y))
    return out


def svg():
    ts = np.arange(0.0, LOOP_S + 1e-9, DT)
    kt = ";".join("{:.4f}".format(t / LOOP_S) for t in ts)
    frames = [positions(t) for t in ts]
    frames[-1] = frames[0]                                   # seamless loop
    dur = "{:.0f}s".format(LOOP_S)
    px = lambda x: MX + x * S                                # noqa: E731
    py = lambda y: MY + y * S                                # noqa: E731
    w_svg, h_svg = 2 * MX + L * S, MY + W * S + 76
    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {:.0f} {:.0f}" width="100%" '
         'role="img" aria-label="Simulated passage of play: the possession label switches from '
         'home to away to dead ball and back to home">'.format(w_svg, h_svg),
         '<title>H, A, D frame by frame: a simulated passage of play with its possession labels</title>',
         '<rect width="100%" height="100%" rx="14" fill="#0f1720"/>',
         '<text x="{:.0f}" y="40" fill="#e6edf3" font-family="system-ui,-apple-system,Segoe UI,'
         'Roboto,sans-serif" font-size="26" font-weight="700">H · A · D, frame by frame</text>'.format(MX),
         '<text x="{:.0f}" y="62" fill="#8b98a5" font-family="system-ui,-apple-system,Segoe UI,'
         'Roboto,sans-serif" font-size="14">team YamineLamal · possession from automatic tracking · '
         'simulated players, not match data</text>'.format(MX)]
    # pitch
    for i in range(12):
        o.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="{}"/>'.format(
            px(i * L / 12), py(-4), L / 12 * S + 0.5, (W + 8) * S,
            "#1f6b3a" if i % 2 else "#226f3d"))
    g = '<g fill="none" stroke="{}" stroke-width="2">'.format(LINE)
    g += '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}"/>'.format(px(0), py(0), L * S,
                                                                            W * S)
    g += '<line x1="{0:.1f}" y1="{1:.1f}" x2="{0:.1f}" y2="{2:.1f}"/>'.format(px(L / 2), py(0),
                                                                             py(W))
    g += '<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}"/>'.format(px(L / 2), py(W / 2), 9.15 * S)
    for x0, d in ((0, 1), (L, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            g += '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}"/>'.format(
                px(min(x0, x0 + d * depth)), py(W / 2 - half), depth * S, 2 * half * S)
    o.append(g + '</g>')
    # players
    for k in range(len(frames[0])):
        colour = HOME if k < 11 else AWAY
        xs = ";".join("{:.1f}".format(px(f[k][0])) for f in frames)
        ys = ";".join("{:.1f}".format(py(f[k][1])) for f in frames)
        o.append('<circle r="8" fill="{}" stroke="rgba(0,0,0,.55)" stroke-width="1.5">'
                 '<animate attributeName="cx" values="{}" keyTimes="{}" dur="{}" '
                 'repeatCount="indefinite"/><animate attributeName="cy" values="{}" '
                 'keyTimes="{}" dur="{}" repeatCount="indefinite"/></circle>'.format(
                     colour, xs, kt, dur, ys, kt, dur))
    # ball
    bxs = ";".join("{:.1f}".format(px(ball_at(t)[0])) for t in ts)
    bys = ";".join("{:.1f}".format(py(ball_at(t)[1])) for t in ts)
    o.append('<circle r="9" fill="rgba(0,0,0,.45)">'
             '<animate attributeName="cx" values="{}" keyTimes="{}" dur="{}" '
             'repeatCount="indefinite"/><animate attributeName="cy" values="{}" keyTimes="{}" '
             'dur="{}" repeatCount="indefinite"/></circle>'.format(bxs, kt, dur, bys, kt, dur))
    o.append('<circle r="6" fill="#ffffff">'
             '<animate attributeName="cx" values="{}" keyTimes="{}" dur="{}" '
             'repeatCount="indefinite"/><animate attributeName="cy" values="{}" keyTimes="{}" '
             'dur="{}" repeatCount="indefinite"/></circle>'.format(bxs, kt, dur, bys, kt, dur))
    # label pill, one per label, shown only during its spells
    for lab in ("H", "A", "D"):
        vals, keys = [], []
        for a, b, l in SPELLS:
            vals.append("1" if l == lab else "0")
            keys.append("{:.4f}".format(a / LOOP_S))
        o.append('<g opacity="0"><animate attributeName="opacity" calcMode="discrete" '
                 'values="{}" keyTimes="{}" dur="{}" repeatCount="indefinite"/>'
                 '<rect x="{:.0f}" y="{:.0f}" width="270" height="34" rx="17" '
                 'fill="rgba(0,0,0,.6)"/><circle cx="{:.0f}" cy="{:.0f}" r="7" fill="{}"/>'
                 '<text x="{:.0f}" y="{:.0f}" fill="{}" font-family="system-ui,-apple-system,'
                 'Segoe UI,Roboto,sans-serif" font-size="16" font-weight="700">{}</text></g>'
                 .format(";".join(vals), ";".join(keys), dur, px(1.5), py(1.5), px(1.5) + 18,
                         py(1.5) + 17, COLOUR[lab], px(1.5) + 32, py(1.5) + 23, COLOUR[lab],
                         TEXT[lab]))
    # timeline of labels with a moving cursor
    ty = py(W) + 44
    for a, b, lab in SPELLS:
        o.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="14" fill="{}"/>'.format(
            px(a / LOOP_S * L), ty, (b - a) / LOOP_S * L * S, COLOUR[lab]))
    o.append('<rect y="{:.1f}" width="3" height="22" fill="#fff"><animate attributeName="x" '
             'values="{:.1f};{:.1f}" dur="{}" repeatCount="indefinite"/></rect>'.format(
                 ty - 4, px(0), px(L) - 3, dur))
    o.append('</svg>')
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="\n") as h:
        h.write(svg())
    print("wrote {} ({:.0f} kB)".format(os.path.relpath(OUT, ROOT), os.path.getsize(OUT) / 1e3))
