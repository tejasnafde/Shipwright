"""The self-healing loop as an SVG: static light and dark versions (README, blog) and an animated one (HF Space).

    python submission/make_loop_diagram.py     # writes submission/img/loop-{light,dark}.svg and submission/space/img/loop-animated.svg

Numbers on the steps are round 16's (the first unattended promotion, 2026-11-15).
"""

from pathlib import Path

DOCS = Path(__file__).resolve().parent

LIGHT = {"surface": "#fcfcfb", "box": "#ffffff", "line": "#d6d5ce", "ink": "#0b0b0b", "ink2": "#52514e",
         "ink3": "#6f6e69", "arrow": "#8a8984", "accent": "#2a78d6", "good": "#0f7b3f",
         "at_bg": "#e8f1fb", "at_fg": "#1d5fae", "ak_bg": "#fdeee7", "ak_fg": "#b4461c",
         "ours_bg": "#efeee9", "ours_fg": "#52514e", "glow": "#2a78d6"}
DARK = {"surface": "#1a1a19", "box": "#232321", "line": "#3a3a37", "ink": "#f0efec", "ink2": "#c3c2b7",
        "ink3": "#9a998f", "arrow": "#7a7973", "accent": "#3987e5", "good": "#4cc27f",
        "at_bg": "#1d2a3a", "at_fg": "#8fbcf2", "ak_bg": "#3a2419", "ak_fg": "#f3a27f",
        "ours_bg": "#2c2c2a", "ours_fg": "#c3c2b7", "glow": "#3987e5"}

RULES = """
.bg{{fill:{surface}}} .box{{fill:{box};stroke:{line};stroke-width:1.2}}
.weight{{stroke:{accent};stroke-width:1.6}}
.t{{fill:{ink};font-weight:700;font-size:15px}} .s{{fill:{ink2};font-size:12.5px}} .n{{fill:{ink3};font-size:12px}}
.arrow{{stroke:{arrow};stroke-width:1.6;fill:none}} .head{{fill:{arrow}}} .dash{{stroke-dasharray:5 4}}
.at{{fill:{at_bg}}} .at-t{{fill:{at_fg}}} .ak{{fill:{ak_bg}}} .ak-t{{fill:{ak_fg}}}
.ours{{fill:{ours_bg}}} .ours-t{{fill:{ours_fg}}} .pt{{font-size:11px;font-weight:600}}
.ok{{fill:{good}}} .bad{{fill:{ink3}}} .glow{{fill:none;stroke:{glow};stroke-width:3}} .dot{{fill:{accent}}}
.foot{{fill:{ink2};font-size:12.5px}} .foot b{{fill:{ink}}}
"""


def style_block(mode: str) -> str:
    if mode == "light":
        return RULES.format(**LIGHT)
    if mode == "dark":
        return RULES.format(**DARK)
    return RULES.format(**LIGHT) + "@media (prefers-color-scheme: dark){" + RULES.format(**DARK) + "}"


PILLS = {"AgentTune": "at", "AuditKIT": "ak", "ours": "ours", "tau2-bench": "ours"}


def pill(x, y, label):
    kind = PILLS[label]
    width = 13 + 6.6 * len(label)
    return (f'<rect class="{kind}" x="{x - width:.0f}" y="{y}" width="{width:.0f}" height="19" rx="9.5"/>'
            f'<text class="pt {kind}-t" x="{x - width / 2:.0f}" y="{y + 13.5}" text-anchor="middle">{label}</text>'), width


def box(x, y, w, h, title, sub, note, pills=(), extra_class=""):
    parts = [f'<rect class="box {extra_class}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>',
             f'<text class="t" x="{x + 14}" y="{y + 26}">{title}</text>',
             f'<text class="s" x="{x + 14}" y="{y + 48}">{sub}</text>']
    if note:
        parts.append(f'<text class="n" x="{x + 14}" y="{y + 68}">{note}</text>')
    right = x + w - 10
    for label in pills:
        svg, width = pill(right, y + 11, label)
        parts.append(svg)
        right -= width + 6
    return "".join(parts)


def chip(x, title, line1, line2, extra_class=""):
    return (f'<rect class="box {extra_class}" x="{x}" y="180" width="170" height="70" rx="10"/>'
            f'<text class="t" x="{x + 14}" y="204">{title}</text>'
            f'<text class="s" x="{x + 14}" y="223">{line1}</text>'
            f'<text class="n" x="{x + 14}" y="240">{line2}</text>')


def arrow(d, dashed=False):
    return f'<path class="arrow{" dash" if dashed else ""}" d="{d}" marker-end="url(#h)"/>'


# Step centres the animated dot passes through, with the fraction of the path at which it arrives.
PATH = "M135,82 H915 V150 H745 V342 H225 V420 H145 V492 H12 V82 H135"
GLOWS = [  # (x, y, w, h, arrival fraction)
    (290, 40, 210, 84, 0.099), (550, 40, 210, 84, 0.198), (810, 40, 210, 84, 0.297),
    (660, 180, 170, 70, 0.411), (640, 300, 210, 84, 0.461), (390, 300, 210, 84, 0.556),
    (110, 300, 230, 84, 0.659), (30, 450, 230, 84, 0.746), (30, 40, 210, 84, 0.985),
]
SECONDS = 10


def diagram(mode: str, animated: bool) -> str:
    body = [f'<rect class="bg" width="1040" height="610"/>']
    body.append(box(30, 40, 210, 84, "1  Run", "Tiny Aya 3.35B student", "+ guards · 74 tasks · $0.53", ["tau2-bench"]))
    body.append(box(290, 40, 210, 84, "2  Detect", "tau2 grader + FailureDetector", "every failed conversation", ["AgentTune"]))
    body.append(box(550, 40, 210, 84, "3  Classify", "rules, then FailureClassifier", "a root cause each · $0.05", ["AgentTune"]))
    body.append(box(810, 40, 210, 84, "4  Route", "by root cause", "tool · model · environment", ["ours"]))
    for d in ("M240,82 H286", "M500,82 H546", "M760,82 H806"):
        body.append(arrow(d))
    # The router's three branches.
    body.append('<path class="arrow" d="M915,124 V150 H555 M745,150 V176 M935,150"/>')
    body.append(arrow("M555,150 V176"))
    body.append(arrow("M745,150 V176"))
    body.append(arrow("M915,150 H935 V176"))
    body.append(chip(470, "Tool fault", "→ guards, tool text", "a person approves"))
    body.append(chip(660, "Model fault", "→ weight fix", "continues below", extra_class="weight"))
    body.append(chip(850, "Environment", "→ no fix", "never trained on"))
    body.append(arrow("M745,250 V296"))
    body.append(box(640, 300, 210, 84, "5  Correct", "teacher fixes each wrong turn", "59 new (235 in all) · $0.12", ["ours"]))
    body.append(box(390, 300, 210, 84, "6  Train", "distill trainer, fresh LoRA", "789 examples · $1.17", ["AgentTune"]))
    body.append(box(110, 300, 230, 84, "7  Gate", "8 checks, all required", "8 of 8 passed · $2.02", ["AgentTune", "AuditKIT"]))
    body.append(arrow("M640,342 H604"))
    body.append(arrow("M390,342 H344"))
    # Gate outcomes.
    body.append('<path class="arrow" d="M225,384 V420"/>')
    body.append(arrow("M225,420 H145 V446"))
    body.append(arrow("M225,420 H405 V446"))
    body.append(box(30, 450, 230, 84, 'Promote <tspan class="ok">✓</tspan>', "ledger on the Hub, rollback", None))
    body.append('<text class="n" x="44" y="518"><tspan class="ok" font-weight="700">1 time:</tspan> round 16</text>')
    body.append(box(290, 450, 230, 84, 'Refuse <tspan class="bad">✗</tspan>', "the old model keeps serving", None))
    body.append('<text class="n" x="304" y="518"><tspan font-weight="700">11 times,</tspan> reasons traced</text>')
    # Back to Run: promotion solid, refusal dashed.
    body.append('<path class="arrow" d="M30,492 H12 V82"/>')
    body.append(arrow("M12,82 H26"))
    body.append('<path class="arrow dash" d="M405,534 V560 H12 V492"/>')
    # Footer: legend and the round's totals.
    legend_x = 30
    for label in ("AgentTune", "AuditKIT", "ours"):
        svg, width = pill(legend_x + 13 + 6.6 * len(label), 576, label)
        body.append(svg)
        legend_x += width + 8
    body.append(f'<text class="foot" x="{legend_x + 8}" y="590">which kit runs each step</text>')
    body.append('<text class="foot" x="1020" y="590" text-anchor="end">Round 16: <tspan font-weight="700">'
                '2 h 24 min · $3.89 · no person in the loop</tspan></text>')
    if animated:
        for x, y, w, h, at in GLOWS:
            start, peak, end = max(at - 0.015, 0), at, min(at + 0.07, 1)
            body.append(f'<rect class="glow" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" opacity="0">'
                        f'<animate attributeName="opacity" dur="{SECONDS}s" repeatCount="indefinite" '
                        f'values="0;0;1;0;0" keyTimes="0;{start:.3f};{peak:.3f};{end:.3f};1"/></rect>')
        body.append(f'<circle class="dot" r="7"><animateMotion dur="{SECONDS}s" repeatCount="indefinite" '
                    f'path="{PATH}"/></circle>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 610" '
            'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Inter, Roboto, sans-serif" role="img" '
            'aria-label="The self-healing loop: run, detect, classify, route by root cause, correct, train, gate, '
            'then promote or refuse and run again.">'
            f'<style>{style_block(mode)}</style>'
            '<defs><marker id="h" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
            + "".join(body) + "</svg>\n")


if __name__ == "__main__":
    (DOCS / "img").mkdir(exist_ok=True)
    (DOCS / "space" / "img").mkdir(parents=True, exist_ok=True)
    (DOCS / "img" / "loop-light.svg").write_text(diagram("light", animated=False))
    (DOCS / "img" / "loop-dark.svg").write_text(diagram("dark", animated=False))
    (DOCS / "space" / "img" / "loop-animated.svg").write_text(diagram("auto", animated=True))
    print("wrote loop-light.svg, loop-dark.svg, space/img/loop-animated.svg")
