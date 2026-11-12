"""Correction rows: only the teacher's final reply is trained, never the student's wrong earlier turns."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_lora  # noqa: E402


def test_only_the_last_assistant_turn_of_a_correction_row_is_trained():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "wrong call 1"}, {"role": "user", "content": "result"},
                {"role": "assistant", "content": "wrong call 2"}, {"role": "user", "content": "result"},
                {"role": "assistant", "content": "the teacher's reply"}]
    all_assistant_trained = [True, True, False, True, False, True, False]  # True = masked
    masked = train_lora.mask_all_but_last_assistant(messages, all_assistant_trained)
    assert masked == [True, True, True, True, True, True, False]


def test_a_row_without_assistant_turns_is_left_alone():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    assert train_lora.mask_all_but_last_assistant(messages, [True, True]) == [True, True]
