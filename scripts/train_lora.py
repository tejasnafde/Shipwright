# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "agenttune==1.1.3",
#     "huggingface_hub>=1.0",
# ]
# ///
"""Train a LoRA adapter on Tiny Aya Global with AgentTune, from the SFT dataset on the Hub.

Runs on HF Jobs (see scripts/train_job.py), not locally. Steps:

1. Download sft.jsonl (scripts/build_sft.py) from the dataset repo.
2. Turn each example into a full-tier AgentTune ``EventLog``: user turns become
   OBSERVATION events (loss-masked), assistant turns become TEXT events (trained).
   The system prompt (policy + compact tools + JSON block protocol) is kept as the
   first message, loss-masked, so the rows are exactly the student's prompts.
3. ``agenttune.create_distill_trainer(student, trajectories=..., trainer_factory=...)``
   runs ``Project.distill``. Our factory tokenizes each row with Tiny Aya's own chat
   template, sets labels to -100 everywhere except the assistant turns (from
   AgentTune's ``loss_mask``), and returns a TRL ``SFTTrainer`` with a LoRA config.
4. Save the adapter and push it to the model repo on branch --revision (e.g. round-1).

Example (on a GPU box with HF_TOKEN set):
  uv run scripts/train_lora.py --dataset mitushigoyal5/shipwright-sft --revision round-1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

STUDENT = "CohereLabs/tiny-aya-global"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", default=STUDENT)
    p.add_argument("--dataset", default="mitushigoyal5/shipwright-sft")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--output-repo", default="mitushigoyal5/shipwright-lora")
    p.add_argument("--revision", default="round-1", help="branch of the output repo")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--output-dir", default="/tmp/shipwright-lora")
    p.add_argument("--no-push", action="store_true")
    return p.parse_args(argv)


# ---- pure helpers (no heavy imports; tested offline in tests/test_train_lora.py) ----


def split_example(messages: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    """(system prompt, [(event kind, text)]) for one SFT example.

    user -> "observation" (masked), assistant -> "text" (trained).
    """
    system = ""
    events: list[tuple[str, str]] = []
    for i, m in enumerate(messages):
        if m["role"] == "system":
            if i != 0:
                raise ValueError("system message must come first")
            system = m["content"]
        elif m["role"] == "user":
            events.append(("observation", m["content"]))
        elif m["role"] == "assistant":
            events.append(("text", m["content"]))
        else:
            raise ValueError(f"unexpected role {m['role']!r}")
    return system, events


def target_char_spans(render, messages: list[dict], loss_mask: list[bool]) -> list[tuple[int, int]]:
    """Char spans of the trained (unmasked) messages in render(messages).

    ``render(msgs, add_generation_prompt)`` returns the chat-template text. A trained
    span starts after the generation prompt of its turn and runs to the end of the
    turn (so the end-of-turn token is trained: the model learns to stop).
    """
    full = render(messages, False)
    spans = []
    for i, masked in enumerate(loss_mask):
        if masked:
            continue
        start_text = render(messages[:i], True)
        end_text = render(messages[: i + 1], False)
        if not (full.startswith(start_text) and full.startswith(end_text) and len(end_text) > len(start_text)):
            raise ValueError(f"chat template is not prefix-stable at message {i}")
        spans.append((len(start_text), len(end_text)))
    return spans


def labels_from_offsets(input_ids: list[int], offsets: list[tuple[int, int]],
                        spans: list[tuple[int, int]]) -> list[int]:
    """input_ids inside a trained span, -100 elsewhere."""
    labels = []
    for tid, (a, b) in zip(input_ids, offsets, strict=True):
        inside = any(s <= a < e for s, e in spans) and b > a
        labels.append(tid if inside else -100)
    return labels


# ---- AgentTune glue ----


def make_eventlog_class():
    from dataclasses import dataclass

    from agenttune.agentic import EventLog

    @dataclass
    class SystemPromptEventLog(EventLog):
        """An EventLog whose SFT rows start with the student's system prompt (masked).

        AgentTune's EventLog has no system event kind; the system prompt carries the
        policy and the tool list, so the student must be trained with it in place.
        """

        system: str = ""

        def as_dataset_rows(self, fmt: str) -> list[dict]:
            rows = super().as_dataset_rows(fmt)
            for r in rows:
                r["messages"].insert(0, {"role": "system", "content": self.system})
                r["segment_weights"].insert(0, 0.0)
                r["loss_mask"].insert(0, True)
            return rows

    return SystemPromptEventLog


def to_eventlog(cls, messages: list[dict]):
    from agenttune.agentic import Event, EventKind

    system, events = split_example(messages)
    log = cls(tier="full", system=system)
    for kind, text in events:
        log.append(Event(EventKind.OBSERVATION if kind == "observation" else EventKind.TEXT, {"text": text}))
    return log


class MaskedSFTFactory:
    """AgentTune trainer_factory: Tiny Aya + LoRA + TRL SFTTrainer, loss on assistant turns only."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.trainer = None
        self.tokenizer = None
        self.dropped_too_long = 0
        self.dropped_template = 0
        self.n_rows = 0
        self.max_len_seen = 0

    def tokenize(self, row: dict) -> dict | None:
        tok = self.tokenizer

        def render(msgs: list[dict], gen: bool) -> str:
            if not msgs:
                return ""
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

        try:
            spans = target_char_spans(render, row["messages"], row["loss_mask"])
        except ValueError:
            self.dropped_template += 1
            return None
        text = render(row["messages"], False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if len(ids) > self.args.max_seq_len:
            self.dropped_too_long += 1
            return None
        self.max_len_seen = max(self.max_len_seen, len(ids))
        return {"input_ids": ids, "labels": labels_from_offsets(ids, enc["offset_mapping"], spans)}

    def __call__(self, *, model, train_dataset, **kwargs):
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        a = self.args
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        rows = [r for r in (self.tokenize(row) for row in train_dataset) if r is not None]
        self.n_rows = len(rows)
        print(f"tokenized {len(rows)} rows; dropped {self.dropped_too_long} over {a.max_seq_len} tokens, "
              f"{self.dropped_template} template errors; longest {self.max_len_seen}", flush=True)
        if not rows:
            raise SystemExit("No training rows left after tokenization.")

        student = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto")
        cfg = SFTConfig(
            output_dir=kwargs.get("output_dir", a.output_dir),
            num_train_epochs=a.epochs,
            learning_rate=a.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            per_device_train_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accum,
            gradient_checkpointing=True,
            bf16=True,
            max_length=a.max_seq_len,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=a.seed,
        )
        lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                          bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
        self.trainer = SFTTrainer(model=student, args=cfg, train_dataset=Dataset.from_list(rows),
                                  processing_class=self.tokenizer, peft_config=lora)
        return self.trainer


def load_examples(repo: str, revision: str) -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, "sft.jsonl", repo_type="dataset", revision=revision)
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    try:
        stats = json.loads(Path(hf_hub_download(repo, "sft_stats.json", repo_type="dataset",
                                                revision=revision)).read_text())
    except Exception:  # noqa: BLE001 - stats are optional
        stats = {}
    return records, stats


MODEL_CARD = """---
base_model: {student}
library_name: peft
license: cc-by-nc-4.0
tags: [lora, agenttune, tau2-bench, shipwright]
---

# Shipwright LoRA ({revision})

LoRA adapter for `{student}`, trained with Lexsi AgentTune (`create_distill_trainer`,
`Project.distill`) on successful tau2-bench retail trajectories rendered in the
prompted JSON tool-call format of the Shipwright agent.

- Data: `{dataset}` (revision `{dataset_revision}`), {n_rows} rows
- LoRA r={r}, alpha={alpha}, lr={lr}, epochs={epochs}, max_seq_len={max_seq_len}
- Final train loss: {loss}

Research artifact for the Lexsi AI Safety and Alignment Hackathon. Non-commercial
(the base model is CC-BY-NC).
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from agenttune import create_distill_trainer

    t0 = time.time()
    records, data_stats = load_examples(args.dataset, args.dataset_revision)
    cls = make_eventlog_class()
    logs = [to_eventlog(cls, r["messages"]) for r in records]
    factory = MaskedSFTFactory(args)
    trainer = create_distill_trainer(args.student, trajectories=logs, trainer_factory=factory,
                                     output_dir=args.output_dir)

    # The rows AgentTune will train on must be the student's prompts, unchanged.
    rows = trainer.sft_dataset()
    assert len(rows) == len(records)
    for row, rec in zip(rows, records, strict=True):
        assert row["messages"] == [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
    print(f"{len(rows)} AgentTune rows match sft.jsonl", flush=True)

    result = trainer.train()
    loss = getattr(result, "training_loss", None)
    out = Path(args.output_dir) / "adapter"
    factory.trainer.model.save_pretrained(out)
    factory.tokenizer.save_pretrained(out)
    meta = {
        "student": args.student,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "data_stats": data_stats,
        "rows_trained": factory.n_rows,
        "dropped_too_long": factory.dropped_too_long,
        "dropped_template": factory.dropped_template,
        "longest_tokens": factory.max_len_seen,
        "hyperparams": {k: v for k, v in vars(args).items() if k not in ("output_dir",)},
        "train_loss": loss,
        "agenttune_events": [[e.stage, e.kind] for e in trainer.events()],
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "training_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "README.md").write_text(MODEL_CARD.format(
        student=args.student, revision=args.revision, dataset=args.dataset,
        dataset_revision=args.dataset_revision, n_rows=factory.n_rows, r=args.lora_r,
        alpha=args.lora_alpha, lr=args.lr, epochs=args.epochs, max_seq_len=args.max_seq_len, loss=loss))
    print(json.dumps(meta, indent=2, default=str), flush=True)

    if not args.no_push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.output_repo, exist_ok=True, private=False)
        api.create_branch(args.output_repo, branch=args.revision, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.output_repo, revision=args.revision,
                          commit_message=f"LoRA {args.revision}: loss {loss}")
        print(f"Pushed https://huggingface.co/{args.output_repo}/tree/{args.revision}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
