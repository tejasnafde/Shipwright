# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Narration takes from Google Cloud Text-to-Speech (Chirp 3 HD voices), one WAV per sentence of narration.md.

    uv run submission/demo_video/make_voiceover.py                                  # voice en-US-Chirp3-HD-Charon
    uv run submission/demo_video/make_voiceover.py --voice en-US-Chirp3-HD-Aoede
    uv run submission/demo_video/make_voiceover.py --samples Charon Aoede Kore Puck  # the title line in each voice
    uv run submission/demo_video/build_demo.py --voiceover-dir submission/demo_video/out/voice

Needs the Text-to-Speech API on the project and a gcloud account config with access (`--gcloud-config`, default
`lexsi`). The whole narration is about 3,000 characters, inside the API's monthly free tier. The token is never
written to disk or printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"


def read_narration() -> dict[str, str]:
    sections, current = {}, None
    for line in (HERE / "narration.md").read_text().splitlines():
        heading = re.match(r"^## (\w+)\s*$", line)
        if heading:
            current = heading.group(1)
            sections[current] = ""
        elif current and line.strip():
            sections[current] = (sections[current] + " " + line.strip()).strip()
    return sections


def split_sentences(text: str) -> list[str]:
    """The same split as build_demo.split_sentences (kept here so this script runs on its own)."""
    return [part.strip() for part in re.split(r"(?<=[.?!])\s+", text) if part.strip()]


def access_token(gcloud_config: str) -> str:
    completed = subprocess.run(["gcloud", f"--configuration={gcloud_config}", "auth", "print-access-token"],
                               capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def synthesize(text: str, voice: str, token: str, project: str, speaking_rate: float) -> bytes:
    language = "-".join(voice.split("-")[:2])
    body = {"input": {"text": text}, "voice": {"languageCode": language, "name": voice},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 48000, "speakingRate": speaking_rate}}
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {token}", "x-goog-user-project": project, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return base64.b64decode(json.loads(response.read())["audioContent"])


def full_voice_name(voice: str) -> str:
    return voice if voice.startswith("en-") else f"en-US-Chirp3-HD-{voice}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", default="en-US-Chirp3-HD-Charon", help="a Chirp 3 HD voice, full name or short")
    parser.add_argument("--speaking-rate", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "out" / "voice")
    parser.add_argument("--samples", nargs="*", default=None, help="only the title line, once per voice listed")
    parser.add_argument("--gcloud-config", default="lexsi")
    parser.add_argument("--project", default="lexsi-hackathon")
    options = parser.parse_args(argv)

    narration = read_narration()
    token = access_token(options.gcloud_config)
    if options.samples:
        samples_dir = options.output_dir.parent / "voice-samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for voice in options.samples:
            audio = synthesize(narration["title"], full_voice_name(voice), token, options.project,
                               options.speaking_rate)
            path = samples_dir / f"{voice}.wav"
            path.write_bytes(audio)
            print(f"wrote {path}")
        return 0

    options.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in options.output_dir.glob("*.wav"):  # takes of an older narration must not be picked up
        stale.unlink()
    voice = full_voice_name(options.voice)
    characters = takes = 0
    for scene, text in narration.items():
        # One take per sentence (<scene>-<n>.wav, n from 0), split exactly as build_demo.py splits: the build joins
        # them with a fixed gap, so it knows when every sentence is spoken and syncs captions and subtitles to it.
        for index, sentence in enumerate(split_sentences(text)):
            (options.output_dir / f"{scene}-{index}.wav").write_bytes(
                synthesize(sentence, voice, token, options.project, options.speaking_rate))
            characters += len(sentence)
            takes += 1
        print(f"{scene:13s} {len(text):4d} characters, {len(split_sentences(text))} sentences", flush=True)
    print(f"wrote {takes} sentence takes in {voice} to {options.output_dir} ({characters} characters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
