"""build_sft.py: which teacher run dirs may join the training data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_sft  # noqa: E402


def test_only_model_skips_runs_written_by_another_teacher(tmp_path):
    run_directory = tmp_path / "glm-run"
    run_directory.mkdir()
    (run_directory / "run_config.json").write_text(json.dumps({"agent_model": "vertex/zai-org/glm-5.2-maas", "task_split": "train"}))
    build_options = build_sft.parse_args(["--only-model", "vertex/deepseek-ai/deepseek-v4.1-flash-maas"])
    notes: list[str] = []
    records = build_sft.collect([run_directory], "teacher", build_options, notes)
    assert records == []
    assert "skipped, written by vertex/zai-org/glm-5.2-maas" in notes[0]


def test_general_replay_rows_get_token_counts_so_the_statistics_work(tmp_path):
    replay_file = tmp_path / "replay.jsonl"
    row = {"messages": [{"role": "system", "content": "tools"}, {"role": "user", "content": "area of a triangle?"},
                        {"role": "assistant", "content": "```json\n{\"name\": \"area\", \"arguments\": {}}\n```"}],
           "source": "general_replay", "task_id": "simple_100"}
    replay_file.write_text(json.dumps(row) + "\n")
    records, dropped = build_sft.load_general_replay(replay_file, max_tokens=8192)
    assert records[0]["tokens"] > 0 and records[0]["reply_tokens_max"] > 0 and dropped == {
        "over_max_tokens": 0, "reply_over_max_tokens": 0}
    statistics = build_sft.sft_data.stats(records, dropped)  # the call that crashed without the counts
    assert statistics["examples"] == 1
