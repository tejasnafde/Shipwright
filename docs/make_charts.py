"""The write-up's charts, light and dark, from numbers measured in the gate runs (sources in comments).

    pip install matplotlib && python docs/make_charts.py      # writes docs/img/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "img"

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9",
              "accent": "#2a78d6", "context": "#a3a29b"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#2c2c2a",
             "accent": "#3987e5", "context": "#6f6e69"},
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, t, labels, values, highlight, fmt):
    colors = [t["accent"] if i in highlight else t["context"] for i in range(len(values))]
    rects = ax.bar(labels, values, color=colors, width=0.62, edgecolor=t["surface"], linewidth=2)
    for rect, value in zip(rects, values):
        ax.annotate(fmt(value), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, color=t["text"], fontweight="bold")


def title(fig, t, text, sub):
    fig.text(0.02, 0.965, text, fontsize=14, fontweight="bold", color=t["text"], va="top")
    fig.text(0.02, 0.895, sub, fontsize=10, color=t["muted"], va="top")


def cost_vs_success(t):
    # Gate 16 (base, round 16) and the teacher's held-out ceiling run, 20 tasks x 3 trials.
    # Cost: measured model calls per conversation (16.8 student, 9.8 teacher) x $0.13 / $1.19 per 1,000 replies.
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=t["surface"],
                                      gridspec_kw={"width_ratios": [3, 2]})
    style(left, t)
    style(right, t)
    bars(left, t, ["Base\nTiny Aya 3.35B", "Self-healed\nTiny Aya 3.35B", "Teacher API\nDeepSeek V4.1 Flash"],
         [10, 15, 31.7], {1}, lambda v: f"{v:.0f}%")
    left.set_ylim(0, 40)
    left.set_yticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40%"])
    left.set_title("Tickets resolved end to end", loc="left", fontsize=11, color=t["text"])
    bars(right, t, ["Self-healed\nTiny Aya", "Teacher API"], [0.0146, 0.0368], {0}, lambda v: f"${v:.3f}")
    right.set_ylim(0, 0.045)
    right.set_yticks([0, 0.01, 0.02, 0.03, 0.04], ["$0", "0.01", "0.02", "0.03", "$0.04"])
    right.set_title("Serving cost per resolved ticket", loc="left", fontsize=11, color=t["text"])
    title(fig, t, "A 3B model that heals itself, at 2.5x less per solved ticket",
          "tau2-bench retail, 20 held-out tasks x 3 trials. Student on one A10G ($1.50/h); teacher at API list price, uncached.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.17, wspace=0.25)
    return fig


def wasted_calls(t):
    # scripts/trajectory_efficiency.py on gates 16, 12, 13, 14, 16 and the teacher's held-out run.
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=t["surface"])
    style(ax, t)
    bars(ax, t, ["Base", "Round 7\n+ guards", "Round 13", "Round 13\n+ guards", "Round 16\n(promoted)", "Teacher"],
         [69, 46, 26, 29, 21, 5], {4}, lambda v: f"{v}%")
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80], ["0", "20", "40", "60", "80%"])
    title(fig, t, "Healing rounds cut wasted tool calls from 69% to 21%",
          "Share of tool calls that were repeated, failed, or blocked by a guard. Same 20 held-out tasks x 3 trials.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.17)
    return fig


def gates(t):
    # Gate decisions (configs/gate.yaml): candidate pass^1 per gate; the base scored 10% in all but gate 4 (12%).
    numbers = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16]
    candidate = [15, 7, 13, 13, 13, 23.3, 12, 13.3, 15, 15, 13.3, 15]
    why = {3: "forgot tools", 4: "worse", 8: "overflow", 12: "overflow", 13: "overflow", 14: "overflow"}
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=t["surface"])
    style(ax, t)
    x = list(range(len(numbers)))
    ax.axhline(10, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(len(x) - 0.55, 10.6, "base 10%", color=t["muted"], fontsize=9, ha="right")
    for i, (gate, value) in enumerate(zip(numbers, candidate)):
        promoted = gate == 16
        ax.plot([i], [value], marker="o", markersize=11 if promoted else 9,
                color=t["accent"] if promoted else t["surface"],
                markeredgecolor=t["accent"] if promoted else t["context"], markeredgewidth=2, linestyle="none")
        if promoted:
            ax.annotate("promoted: 8 of 8 checks,\nno person in the loop", (i, value), xytext=(i - 0.1, 26.5),
                        textcoords="data", ha="right", va="center", fontsize=10, color=t["text"], fontweight="bold",
                        arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 1})
        elif gate in why:
            ax.annotate(why[gate], (i, value), xytext=(0, 10), textcoords="offset points", ha="center",
                        fontsize=9, color=t["muted"])
    ax.set_xticks(x, [f"gate {g}" for g in numbers], fontsize=9)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30], ["0", "10", "20", "30%"])
    title(fig, t, "Eleven refusals, then an unattended promotion",
          "Candidate task success per gate. Hollow: refused by our 8-check gate (AgentTune's own gate approved 4 of them).")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.12)
    return fig


def round16_jump(t):
    # scripts/trajectory_efficiency.py, gate 14 (round 13 + guards, the round before) vs gate 16, same 20 tasks x 3.
    # 95% task-bootstrap intervals of the change: wasted -14 to -3 pts, attempts -4.1 to -0.8, blocks -1.9 to -0.7.
    panels = [("Wasted tool calls", [29.3, 20.8], lambda v: f"{v:.0f}%", 35, "−8 pts"),
              ("Tool calls tried per conversation", [10.4, 8.0], lambda v: f"{v:.1f}", 12.5, "−23%"),
              ("Calls a guard had to block", [2.58, 1.27], lambda v: f"{v:.1f}", 3.2, "−51%")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), facecolor=t["surface"])
    for ax, (label, values, fmt, top, change) in zip(axes, panels):
        style(ax, t)
        bars(ax, t, ["Round before\n(gate 14)", "Round 16"], values, {1}, fmt)
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.yaxis.grid(False)
        ax.set_title(label, loc="left", fontsize=11, color=t["text"])
        ax.text(0.98, 0.97, change, transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold", color=t["accent"])
    title(fig, t, "Round 16 used its tools more cleanly than the round before",
          "Same 20 held-out tasks x 3 trials. All three drops hold under a task bootstrap (95% intervals exclude zero).")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.74, bottom=0.17, wspace=0.18)
    return fig


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, make in {"cost-vs-success": cost_vs_success, "wasted-calls": wasted_calls, "gates": gates,
                       "round16-jump": round16_jump}.items():
        for mode, theme in THEMES.items():
            figure = make(theme)
            figure.savefig(OUT / f"{name}-{mode}.png", dpi=160, facecolor=theme["surface"])
            plt.close(figure)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
