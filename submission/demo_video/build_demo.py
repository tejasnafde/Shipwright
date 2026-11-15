# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright==1.65.2", "imageio-ffmpeg==0.8.1"]
# ///
"""Build the demo video from the real control room: narration, title cards, recorded scenes, one 1080p MP4.

    uv run submission/demo_video/build_demo.py                     # control room already on :8501, voice "Samantha"
    uv run submission/demo_video/build_demo.py --start-app         # start the control room itself
    uv run submission/demo_video/build_demo.py --voice "Ava (Premium)" --music bed.mp3
    uv run submission/demo_video/build_demo.py --voiceover-dir my_takes/   # your own <scene>.wav or .mp3 per scene

Before the first run: `uv run --with playwright==1.65.2 playwright install chromium`, and the round 16 run dirs in
runs/ (scripts/pull_runs.py --owner mitushi; the control room's sidebar pulls gate results). Nothing is staged: every
screen is the control room reading the service's own run files.

Steps:
  1. narration.md -> one audio file per scene (macOS `say`, or your own takes); each scene lasts as long as its words.
  2. Title and closing cards are rendered from HTML at 1920x1080 and given a slow push-in.
  3. Each control-room scene is recorded by Playwright at 1920x1080 (one browser context per scene), with captions and
     highlights drawn into the page, then trimmed to the narration.
  4. ffmpeg encodes every segment with its audio and fades, concatenates them, and writes final_video.mp4 plus
     final_video.srt (subtitles from the same narration).
Output: submission/demo_video/out/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Page, sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
WIDTH, HEIGHT, FPS = 1920, 1080, 30
APP_WIDTH, APP_HEIGHT = 1536, 864  # the app is recorded at 80% and scaled to 1080p: text reads at 125%
LEAD_IN, TAIL = 0.6, 0.9          # seconds of picture before the voice starts and after it ends
ACCENT = "#ff6b35"


# ---------------------------------------------------------------- narration and audio

def read_narration() -> dict[str, str]:
    """{scene id: text} from narration.md."""
    sections, current = {}, None
    for line in (HERE / "narration.md").read_text().splitlines():
        heading = re.match(r"^## (\w+)\s*$", line)
        if heading:
            current = heading.group(1)
            sections[current] = ""
        elif current and line.strip():
            sections[current] = (sections[current] + " " + line.strip()).strip()
    return sections


def run_ffmpeg(*arguments: str) -> str:
    completed = subprocess.run([FFMPEG, "-hide_banner", "-y", *arguments], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:])
    return completed.stderr


def media_seconds(path: Path) -> float:
    probe = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    hours, minutes, seconds = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe).groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def make_voice(scene: str, text: str, options: argparse.Namespace, work: Path) -> Path:
    """The scene's narration as 48 kHz stereo WAV: a take from --voiceover-dir, else macOS `say`."""
    wav = work / f"{scene}.wav"
    if options.voiceover_dir:
        for suffix in (".wav", ".mp3", ".m4a", ".aiff"):
            take = Path(options.voiceover_dir) / f"{scene}{suffix}"
            if take.exists():
                run_ffmpeg("-i", str(take), "-ar", "48000", "-ac", "2", "-af", "loudnorm=I=-16:TP=-1.5", str(wav))
                return wav
    aiff = work / f"{scene}.aiff"
    subprocess.run(["say", "-v", options.voice, "-r", str(options.rate), "-o", str(aiff), text], check=True)
    run_ffmpeg("-i", str(aiff), "-ar", "48000", "-ac", "2", "-af", "loudnorm=I=-16:TP=-1.5", str(wav))
    return wav


# ---------------------------------------------------------------- page helpers (drawn into the control room)

OVERLAY_SCRIPT = """
window.__demo = {
  caption(text) {
    let bar = document.getElementById('demo-caption');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'demo-caption';
      Object.assign(bar.style, {position: 'fixed', left: '50%', bottom: '34px', transform: 'translateX(-50%)',
        maxWidth: '1100px', padding: '13px 26px', borderRadius: '14px', background: 'rgba(17,24,39,0.88)',
        color: '#fff', font: '600 23px/1.35 -apple-system, Segoe UI, Inter, sans-serif', zIndex: 999999,
        boxShadow: '0 10px 30px rgba(0,0,0,0.25)', transition: 'opacity 350ms', opacity: 0, textAlign: 'center'});
      document.body.appendChild(bar);
    }
    bar.style.opacity = 0;
    setTimeout(() => { bar.innerHTML = text; bar.style.opacity = text ? 1 : 0; }, text ? 200 : 0);
  },
  highlight(el) {
    if (!el) return;
    el.style.transition = 'outline-color 300ms, box-shadow 300ms';
    el.style.outline = '4px solid ACCENT';
    el.style.outlineOffset = '6px';
    el.style.borderRadius = '8px';
    el.style.boxShadow = '0 0 0 12px rgba(255,107,53,0.12)';
    el.dataset.demoHighlight = '1';
  },
  clear() {
    document.querySelectorAll('[data-demo-highlight]').forEach(el => {
      el.style.outline = ''; el.style.boxShadow = ''; delete el.dataset.demoHighlight; });
  },
};
const style = document.createElement('style');
style.textContent = '[data-testid=stToolbar],[data-testid=stDecoration],[data-testid=stStatusWidget]{display:none!important}';
document.addEventListener('DOMContentLoaded', () => document.head.appendChild(style));
""".replace("ACCENT", ACCENT)


@dataclass
class Director:
    """Moves the camera inside the control room: captions, smooth scrolls, highlights."""
    page: Page

    def caption(self, text: str) -> None:
        self.page.evaluate("text => window.__demo.caption(text)", text)

    def scroll_to(self, text: str, block: str = "start", nth: int = 0) -> None:
        target = self.page.get_by_text(text, exact=False).nth(nth)
        target.evaluate(f"el => el.scrollIntoView({{behavior: 'smooth', block: '{block}'}})")
        time.sleep(1.2)

    def highlight(self, text: str, nth: int = 0, container: str | None = None) -> None:
        target = self.page.get_by_text(text, exact=False).nth(nth)
        selector = f"el => window.__demo.highlight(el.closest('{container}') || el)" if container else \
            "el => window.__demo.highlight(el)"
        target.evaluate(selector)

    def clear(self) -> None:
        self.page.evaluate("() => window.__demo.clear()")

    def tab(self, name: str) -> None:
        self.page.get_by_role("tab", name=name).click()
        time.sleep(2.5)

    def scroll_top(self) -> None:
        self.page.evaluate("() => { const m = document.querySelector('[data-testid=stMain]'); "
                           "if (m) m.scrollTo({top: 0}); window.scrollTo({top: 0}); }")
        time.sleep(0.6)

    def choose(self, label: str, typed: str) -> None:
        """Pick an option in a Streamlit selectbox by typing part of it."""
        box = self.page.locator("[data-testid=stSelectbox]").filter(has_text=label).last
        box.click()
        time.sleep(0.4)
        self.page.keyboard.type(typed, delay=40)
        time.sleep(0.5)
        self.page.keyboard.press("Enter")
        time.sleep(2.5)


_clock = {"cursor": 0.0}


def start_clock() -> None:
    _clock["cursor"] = time.monotonic()


def pace(seconds: float) -> None:
    """Hold until this beat's slot ends. Beats run on a fixed clock from the scene's start, so a slow scroll or click
    eats into its own beat instead of pushing every later caption behind the narration."""
    _clock["cursor"] += seconds
    time.sleep(max(_clock["cursor"] - time.monotonic(), 0.3))


# ---------------------------------------------------------------- the scenes in the control room

def scene_control_room(director: Director, duration: float) -> None:
    director.caption("The control room: a replay of a real round, read from the service's own files")
    pace(duration * 0.30)
    director.highlight("GATE SAID YES")
    director.caption("Round 16 ran unattended for $3.89, and the gate said yes: 8 of 8 checks")
    pace(duration * 0.30)
    director.clear()
    director.highlight("Fix / Generate", container=".cr-pipeline")
    director.caption("Run, detect, classify, route, fix, train, gate, promote")
    pace(duration * 0.40)
    director.clear()


def scene_what_changed(director: Director, duration: float) -> None:
    director.scroll_to("Round 16: the gate said yes")
    director.highlight("Tasks passed on a single try")
    director.caption("7\u00d7 fewer failing tool calls (44% to 6%); tickets resolved 10% to 15%")
    pace(duration * 0.22)
    director.clear()
    director.highlight("Runs that failed by skipping a step")
    director.caption("And what got worse, said just as plainly")
    pace(duration * 0.14)
    director.clear()
    director.scroll_to("Read one task side by side")
    director.caption("One held-out task, base model against the healed student")
    pace(duration * 0.14)
    director.scroll_to("Error: Order not found", block="center")
    director.highlight("Error: Order not found")
    director.caption("The base model invents order #W123456: “Order not found”")
    pace(duration * 0.22)
    director.clear()
    director.highlight("but first I need to verify your identity")
    director.caption("The healed student verifies the customer, looks up the real orders, and passes")
    pace(duration * 0.28)
    director.clear()


def scene_corrections(director: Director, duration: float) -> None:
    director.scroll_to("On-policy corrections")
    director.caption("Every call a policy guard blocked became a lesson")
    pace(duration * 0.22)
    director.scroll_to("Read one correction")
    director.highlight("The student tried")
    director.caption("Here the student made up an order id: #W1234567")
    pace(duration * 0.24)
    director.clear()
    director.highlight("The teacher's turn, trained into the next adapter")
    director.caption("The teacher, seeing exactly what the student saw, looks up the real customer")
    pace(duration * 0.27)
    director.clear()
    director.scroll_to("On-policy corrections")
    director.highlight("Corrections kept")
    director.caption("59 new corrections this round ($0.12), 235 across all three rounds; only the corrected turn is trained")
    pace(duration * 0.27)
    director.clear()


def scene_gate(director: Director, duration: float) -> None:
    director.scroll_to("PROMOTED")
    director.caption("Both systems, the same 20 held-out tasks, 3 times each")
    pace(duration * 0.35)
    director.scroll_to("Measured deltas")
    director.highlight("BFCL tool-calling score")
    director.caption("Task success up, targeted failures down, general tool calling 45 to 73")
    pace(duration * 0.40)
    director.clear()
    director.scroll_to("The gate's tests, in plain words")
    director.caption("All eight checks must pass")
    pace(duration * 0.25)


def scene_history(director: Director, duration: float) -> None:
    director.scroll_to("gates,")
    director.caption("12 gates: 11 refused, then one promotion")
    pace(duration * 0.40)
    director.highlight("AgentTune's DeploymentGate on its own")
    director.caption("AgentTune's own check alone would have approved 10 of the 11")
    pace(duration * 0.30)
    director.clear()
    director.scroll_to("Every decision, and why")
    director.caption("Every refusal has a reason the traces explain")
    pace(duration * 0.30)


APP_SCENES = {"control_room": scene_control_room, "what_changed": scene_what_changed,
              "corrections": scene_corrections, "gate": scene_gate, "history": scene_history}
CARD_SCENES = ("title", "loop", "outro")
ORDER = ("title", "loop", "control_room", "what_changed", "corrections", "gate", "history", "outro")


def record_app_scene(browser, scene: str, duration: float, url: str, work: Path) -> Path:
    """Record one scene; return a clip trimmed to the moment the scene was ready, LEAD_IN + duration + TAIL long."""
    video_dir = work / f"rec-{scene}"
    context = browser.new_context(viewport={"width": APP_WIDTH, "height": APP_HEIGHT}, device_scale_factor=1,
                                  record_video_dir=str(video_dir),
                                  record_video_size={"width": APP_WIDTH, "height": APP_HEIGHT})
    context.add_init_script(OVERLAY_SCRIPT)
    created = time.monotonic()
    page = context.new_page()
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("text=Which round do you want to look at?", timeout=90000)
    page.wait_for_selector("text=GATE SAID YES", timeout=90000)
    guide = page.get_by_text("Start here: what you are looking at")
    if guide.count():
        guide.first.click()  # collapse the reading guide: the narration explains the page
    time.sleep(2.5)
    director = Director(page)
    if scene != "control_room":  # the sidebar only matters in the first scene; then land on the scene's tab
        collapse = page.locator("[data-testid=stSidebarCollapseButton] button")
        if collapse.count():
            page.locator("[data-testid=stSidebar]").hover()
            collapse.first.click(force=True)
            time.sleep(1.0)
        APP_SCENES_SETUP.get(scene, lambda d: None)(director)
        time.sleep(2.0)  # let Streamlit finish the rerun, so no fading stale element is recorded
    ready = time.monotonic() - created
    time.sleep(LEAD_IN)
    start_clock()
    APP_SCENES[scene](director, duration)
    time.sleep(TAIL)
    director.caption("")
    page.close()
    context.close()
    raw = next(video_dir.glob("*.webm"))
    clip = work / f"{scene}-clip.mp4"
    run_ffmpeg("-ss", f"{ready:.2f}", "-i", str(raw), "-t", f"{LEAD_IN + duration + TAIL:.2f}",
               "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", str(clip))
    return clip


APP_SCENES_SETUP = {
    "what_changed": lambda director: director.tab("What changed?"),
    "corrections": lambda director: director.tab("Fix & Train"),
    "gate": lambda director: director.tab("Gate & Outcome"),
    "history": lambda director: director.tab("History"),
}


# ---------------------------------------------------------------- title and closing cards

CARD_STYLE = """
<style>
  html,body{margin:0;width:1920px;height:1080px;overflow:hidden;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,sans-serif;color:#0f172a;
    background:radial-gradient(1200px 700px at 20% 10%,#eef4ff 0%,#f7f5ef 55%,#f1efe9 100%)}
  .wrap{position:absolute;inset:0;padding:110px 140px;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center}
  .eyebrow{font:700 26px/1 'JetBrains Mono',ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;color:#ff6b35}
  h1{font-size:132px;line-height:1;margin:24px 0 18px;letter-spacing:-.03em}
  h2{font-size:56px;line-height:1.1;margin:0 0 24px;letter-spacing:-.02em}
  p{font-size:38px;line-height:1.35;color:#334155;max-width:1500px;margin:0}
  .stats{display:flex;gap:28px;margin-top:56px}
  .stat{background:#fff;border:1px solid #e2ddd0;border-radius:18px;padding:26px 34px;min-width:250px;box-shadow:0 6px 24px rgba(15,23,42,.06)}
  .num{font:800 64px/1 -apple-system,Inter,sans-serif;color:#1f7a4d}
  .cap{font-size:24px;color:#475569;margin-top:10px}
  .foot{position:absolute;left:140px;right:140px;bottom:70px;display:flex;justify-content:space-between;font-size:24px;color:#64748b}
  img.loop{height:760px;margin-top:6px;align-self:center}
</style>
"""


def card_html(scene: str) -> str:
    if scene == "title":
        body = """<div class="wrap"><div class="eyebrow">Lexsi Labs hackathon &middot; Track 3 &middot; self-healing agents</div>
<h1>Shipwright</h1><p>A support agent that finds its own mistakes, fixes them,<br>and refuses to ship until the fix is measured.</p>
<div class="stats"><div class="stat"><div class="num">0 people</div><div class="cap">in the loop, one command, $3.89</div></div>
<div class="stat"><div class="num">7&times; fewer</div><div class="cap">tool calls that fail</div></div>
<div class="stat"><div class="num">2.5&times; cheaper</div><div class="cap">per resolved ticket than the teacher</div></div></div></div>
<div class="foot"><span>Tiny Aya Global 3.35B + LoRA &middot; &tau;&sup2;-bench retail</span><span>AgentTune &middot; AuditKIT</span></div>"""
    elif scene == "loop":
        diagram = (ROOT / "submission" / "img" / "loop-light.svg").resolve().as_uri()
        body = f"""<div class="wrap" style="padding-top:80px"><div class="eyebrow">How a round works</div>
<h2>Run &rarr; detect &rarr; classify &rarr; route by cause &rarr; fix &rarr; gate &rarr; promote or refuse</h2>
<img class="loop" src="{diagram}"></div>"""
    else:
        body = """<div class="wrap"><div class="eyebrow">Shipwright</div>
<h2>One command. An unattended round.<br>A gate that can say no.</h2>
<div class="stats"><div class="stat"><div class="num">$3.89</div><div class="cap">the round that shipped</div></div>
<div class="stat"><div class="num">11</div><div class="cap">candidates refused first</div></div>
<div class="stat"><div class="num">0% &rarr; 10%</div><div class="cap">on hidden tasks, opened once</div></div></div>
<p style="margin-top:56px;font-size:32px">Every number is recomputed from public run files: <b>python scripts/reproduce.py</b></p></div>
<div class="foot"><span>huggingface.co/spaces/mitushigoyal5/shipwright</span><span>Tejas Nafde &middot; Ishank Giri &middot; Mitushi Goyal</span></div>"""
    return f"<!doctype html><html><head><meta charset='utf-8'>{CARD_STYLE}</head><body>{body}</body></html>"


def render_card(browser, scene: str, duration: float, work: Path) -> Path:
    page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
    html = work / f"{scene}.html"
    html.write_text(card_html(scene))
    page.goto(html.as_uri(), wait_until="networkidle")
    time.sleep(0.5)
    still = work / f"{scene}.png"
    page.screenshot(path=str(still))
    page.close()
    length = LEAD_IN + duration + TAIL
    frames = int(length * FPS)
    clip = work / f"{scene}-clip.mp4"
    # a slow push-in (3% over the card) so a still does not look frozen
    run_ffmpeg("-loop", "1", "-i", str(still), "-t", f"{length:.2f}", "-vf",
               f"scale=3840:-1,zoompan=z='1+0.03*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
               f":d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", str(clip))
    return clip


# ---------------------------------------------------------------- assembly

def finish_segment(scene: str, clip: Path, voice: Path, duration: float, work: Path) -> Path:
    length = LEAD_IN + duration + TAIL
    segment = work / f"{scene}-segment.mp4"
    delay_ms = int(LEAD_IN * 1000)
    run_ffmpeg("-i", str(clip), "-i", str(voice),
               "-filter_complex",
               f"[0:v]fade=t=in:st=0:d=0.35,fade=t=out:st={length - 0.35:.2f}:d=0.35[v];"
               f"[1:a]adelay={delay_ms}|{delay_ms},apad,atrim=0:{length:.2f}[a]",
               "-map", "[v]", "-map", "[a]", "-t", f"{length:.2f}", "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               str(segment))
    return segment


def srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_subtitles(narration: dict[str, str], durations: dict[str, float], path: Path) -> None:
    """One cue per sentence, spread over the scene's spoken time in proportion to its length."""
    cues, start_of_scene, index = [], 0.0, 1
    for scene in ORDER:
        sentences = [part.strip() for part in re.split(r"(?<=[.?!])\s+", narration[scene]) if part.strip()]
        total_characters = sum(len(sentence) for sentence in sentences)
        cursor = start_of_scene + LEAD_IN
        for sentence in sentences:
            length = durations[scene] * len(sentence) / total_characters
            cues.append(f"{index}\n{srt_time(cursor)} --> {srt_time(cursor + length)}\n{sentence}\n")
            cursor += length
            index += 1
        start_of_scene += LEAD_IN + durations[scene] + TAIL
    path.write_text("\n".join(cues))


def assemble(segments: list[Path], music: Path | None, output: Path, work: Path) -> None:
    listing = work / "segments.txt"
    listing.write_text("".join(f"file '{segment}'\n" for segment in segments))
    joined = work / "joined.mp4"
    run_ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined))
    if music is None:
        shutil.copy(joined, output)
        return
    total = media_seconds(joined)
    run_ffmpeg("-i", str(joined), "-stream_loop", "-1", "-i", str(music), "-filter_complex",
               f"[1:a]volume=0.07,afade=t=in:st=0:d=2,afade=t=out:st={total - 3:.2f}:d=3[m];"
               "[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}",
               str(output))


def wait_for_app(url: str, seconds: int = 90) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except OSError:
            time.sleep(1)
    raise SystemExit(f"the control room did not answer at {url}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--start-app", action="store_true", help="start the control room on --url's port")
    parser.add_argument("--voice", default="Samantha", help="a macOS voice (say -v '?')")
    parser.add_argument("--rate", type=int, default=172, help="words per minute for `say`")
    parser.add_argument("--voiceover-dir", type=Path, default=None, help="<scene>.wav/.mp3 takes that replace `say`")
    parser.add_argument("--music", type=Path, default=None, help="a background track, mixed quietly under the voice")
    parser.add_argument("--only", nargs="*", default=None, help="build only these scenes (for a quick check)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    narration = read_narration()
    missing = [scene for scene in ORDER if scene not in narration]
    if missing:
        raise SystemExit(f"narration.md has no section for {missing}")
    scenes = [scene for scene in ORDER if not options.only or scene in options.only]
    OUT.mkdir(exist_ok=True)

    app = None
    if options.start_app:
        port = options.url.rsplit(":", 1)[-1].strip("/")
        app = subprocess.Popen([str(ROOT / ".venv" / "bin" / "streamlit"), "run",
                                str(ROOT / "src" / "shipwright" / "control_room" / "app.py"), "--server.headless",
                                "true", "--server.port", port, "--browser.gatherUsageStats", "false"],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if any(scene in APP_SCENES for scene in scenes):
            wait_for_app(options.url)
        with tempfile.TemporaryDirectory() as scratch, sync_playwright() as playwright:
            work = Path(scratch)
            browser = playwright.chromium.launch()
            durations, segments = {}, []
            for scene in scenes:
                voice = make_voice(scene, narration[scene], options, work)
                durations[scene] = media_seconds(voice)
                print(f"{scene:13s} {durations[scene]:5.1f} s of narration", flush=True)
                if scene in CARD_SCENES:
                    clip = render_card(browser, scene, durations[scene], work)
                else:
                    clip = record_app_scene(browser, scene, durations[scene], options.url, work)
                segments.append(finish_segment(scene, clip, voice, durations[scene], work))
            browser.close()
            name = "final_video" if not options.only else "preview_" + "_".join(scenes)
            output = OUT / f"{name}.mp4"
            assemble(segments, options.music, output, work)
            if not options.only:
                write_subtitles(narration, durations, OUT / "final_video.srt")
            (OUT / f"{name}.json").write_text(json.dumps(
                {"scenes": scenes, "narration_seconds": durations, "total_seconds": round(media_seconds(output), 1),
                 "voice": str(options.voiceover_dir or options.voice)}, indent=2))
            print(f"wrote {output} ({media_seconds(output):.0f} s)")
    finally:
        if app is not None:
            app.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
