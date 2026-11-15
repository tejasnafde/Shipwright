# Demo video storyboard

About 2 minutes 40 seconds, 1920x1080. A short framing, then the real control room replaying round 16, the round the
loop promoted with no person involved. Every screen after the opening cards is the control room reading the service's
own run files; nothing is staged. The words are in `narration.md`; `build_demo.py` records and assembles everything.

| # | Scene | Picture | What the viewer should take away |
|---|---|---|---|
| 1 | `title` | Title card: Shipwright, the one-line claim, three numbers (no person in the loop for $3.89, 7x fewer failing tool calls, 2.5x cheaper per resolved ticket) | the problem (agents fail quietly) and the promise (find, fix, prove) |
| 2 | `loop` | The loop diagram (`submission/img/loop-light.svg`) | run, detect, classify, route by cause, fix in the tool layer or the weights, gate |
| 3 | `control_room` | Control room on round 16: header tiles, GATE SAID YES, the eight-step pipeline | this is a replay of a real, unattended round: $3.89, 8 of 8 |
| 4 | `what_changed` | What changed? tab: the Better and Worse lists, then held-out task 9 side by side | tickets resolved 10% to 15%, 7x fewer failing tool calls, made-up ids gone, the trade-offs shown too; the base invents `#W123456` and gets "Order not found", the healed student verifies the customer and passes |
| 5 | `corrections` | Fix & Train tab: on-policy corrections, the guard table, task 28's correction side by side | every call a guard blocked became a lesson: the student's made-up `#W1234567` against the teacher's lookup of the real customer; 59 corrections for $0.12 |
| 6 | `gate` | Gate & Outcome tab: PROMOTED, the eight tests in plain words, the measured deltas | same 20 held-out tasks, 3 trials each, all eight checks must pass; BFCL 45 to 73 |
| 7 | `history` | History tab: every gate, base against candidate, why each refusal happened | 12 gates, 11 refused; AgentTune's own check alone would have approved 10 of those 11 |
| 8 | `outro` | Closing card: $3.89, 11 refusals first, hidden split 0% to 10%, `python scripts/reproduce.py`, team | one command, a gate that can say no, numbers anyone can recompute |

## Production notes

- Captions and orange highlights are drawn into the page during recording (`OVERLAY_SCRIPT`); the app's own toolbar is
  hidden. The app is recorded at 1536x864 and scaled to 1080p, so its text reads at 125%.
- Each scene lasts exactly as long as its narration, plus a short lead-in and tail, with fades between scenes.
- Subtitles (`out/final_video.srt`) come from the same narration, one cue per sentence.
- The voice is Google Cloud Text-to-Speech, Chirp 3 HD (`make_voiceover.py`, default `en-US-Chirp3-HD-Charon`;
  `--samples Charon Aoede Kore` writes the title line in several voices to compare). Build with
  `--voiceover-dir submission/demo_video/out/voice`. Without takes, `build_demo.py` falls back to macOS `say`; a human
  recording per scene (`<scene>.wav`) works the same way. `--music <file>` mixes a background track quietly under the
  voice.
- Any claim added to the narration must be visible in the control room or in `submission/WRITEUP.md`, and checkable
  against the run files (`python scripts/reproduce.py`).
