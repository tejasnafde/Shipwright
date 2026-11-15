# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "agenttune==1.1.3",
#     "huggingface_hub>=1.0",
#     "transformers==5.22.2",
#     "trl==1.7.1",
#     "peft==0.21.2",
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # The model and the data.
    parser.add_argument("--student", default=STUDENT)
    parser.add_argument("--dataset", default="mitushigoyal5/shipwright-sft")
    parser.add_argument("--dataset-revision", default="main")

    # Training settings.
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=300)

    # Output.
    parser.add_argument("--output-repo", default="mitushigoyal5/shipwright-lora")
    parser.add_argument("--revision", default="round-1", help="branch of the output repo")
    parser.add_argument("--output-dir", default="/tmp/shipwright-lora")
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args(argv)


# ---- pure helpers (no heavy imports; tested offline in tests/heal/test_sft_data.py) ----


def split_example(messages: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    """(system prompt, [(event kind, text)]) for one SFT example.

    user -> "observation" (masked), assistant -> "text" (trained).
    """
    system_prompt = ""
    events: list[tuple[str, str]] = []
    for message_index, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            if message_index != 0:
                raise ValueError("system message must come first")
            system_prompt = message["content"]
        elif role == "user":
            events.append(("observation", message["content"]))
        elif role == "assistant":
            events.append(("text", message["content"]))
        else:
            raise ValueError(f"unexpected role {role!r}")
    return system_prompt, events


def mask_all_but_last_assistant(messages: list[dict], loss_mask: list[bool]) -> list[bool]:
    """A correction row holds the student's own (wrong) earlier turns as context: only the teacher's final reply is trained.

    ``loss_mask[i]`` is True when message i is masked (not trained)."""
    last_assistant = max((index for index, message in enumerate(messages) if message["role"] == "assistant"), default=-1)
    return [is_masked or (message["role"] == "assistant" and index != last_assistant)
            for index, (message, is_masked) in enumerate(zip(messages, loss_mask))]


def target_char_spans(render, messages: list[dict], loss_mask: list[bool]) -> list[tuple[int, int]]:
    """Char spans of the trained (unmasked) messages in render(messages).

    ``render(messages, add_generation_prompt)`` returns the chat-template text. A trained
    span starts after the generation prompt of its turn and runs to the end of the
    turn (so the end-of-turn token is trained: the model learns to stop).
    """
    full_text = render(messages, False)
    spans = []
    for message_index, is_masked in enumerate(loss_mask):
        if is_masked:
            continue
        text_before_turn = render(messages[:message_index], True)
        text_after_turn = render(messages[: message_index + 1], False)
        template_is_prefix_stable = (
            full_text.startswith(text_before_turn)
            and full_text.startswith(text_after_turn)
            and len(text_after_turn) > len(text_before_turn)
        )
        if not template_is_prefix_stable:
            raise ValueError(f"chat template is not prefix-stable at message {message_index}")
        spans.append((len(text_before_turn), len(text_after_turn)))
    return spans


def labels_from_offsets(
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> list[int]:
    """input_ids inside a trained span, -100 elsewhere."""
    labels = []
    for token_id, (token_start, token_end) in zip(input_ids, offsets, strict=True):
        is_inside_trained_span = any(
            span_start <= token_start < span_end for span_start, span_end in spans
        ) and token_end > token_start
        if is_inside_trained_span:
            labels.append(token_id)
        else:
            labels.append(-100)
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
            for row in rows:
                if self.system:
                    row["messages"].insert(0, {"role": "system", "content": self.system})
                    row["segment_weights"].insert(0, 0.0)
                    row["loss_mask"].insert(0, True)
            return rows

    return SystemPromptEventLog


def to_eventlog(eventlog_class, messages: list[dict]):
    from agenttune.agentic import Event, EventKind

    system_prompt, events = split_example(messages)
    event_log = eventlog_class(tier="full", system=system_prompt)
    for event_kind, text in events:
        if event_kind == "observation":
            agenttune_kind = EventKind.OBSERVATION
        else:
            agenttune_kind = EventKind.TEXT
        event_log.append(Event(agenttune_kind, {"text": text}))
    return event_log


def check_loss_against_the_models_own(model, inputs: dict, windowed_loss) -> None:
    """Stop before any training step if the hand-computed loss is not the model's own labels= loss.

    The first step only, in eval mode (no LoRA dropout) and without gradients, on the full logits once. Round 3
    attempt 3 trained for minutes with an unshifted loss (155 instead of about 2): this makes that failure a
    one-second error instead of a wasted GPU hour."""
    import torch

    was_training = model.training
    model.eval()
    with torch.no_grad():
        models_own_loss = model(**inputs).loss
        recomputed_windowed_loss = windowed_loss.detach()
    if was_training:
        model.train()
    relative_gap = abs(float(recomputed_windowed_loss) - float(models_own_loss)) / max(float(models_own_loss), 1e-6)
    print(f"loss check: windowed {float(recomputed_windowed_loss):.4f} vs the model's own {float(models_own_loss):.4f}", flush=True)
    if relative_gap > 0.05:
        raise RuntimeError(
            f"the windowed loss ({float(recomputed_windowed_loss):.4f}) differs from the model's own loss "
            f"({float(models_own_loss):.4f}): fix compute_loss before training"
        )


def windowed_sft_trainer():
    """A TRL ``SFTTrainer`` subclass that skips computing logits for the loss-masked
    prefix before the first trained token.

    Tiny Aya's 262,144-token vocabulary makes the full ``[seq_len, vocab]`` logits
    tensor the memory bottleneck, not the base model's own weights or LoRA's ~30M
    trainable params - at our own 8192-token ``--max-seq-len``, one bf16 logits
    tensor alone is already seq_len x vocab x 2 bytes ~= 4.3 GB before gradients or
    activations. ``logits_to_keep`` plus a windowed loss skips the masked prefix. It is
    applied to our own ``SFTTrainer`` subclass, since ``create_distill_trainer``'s
    default factory builds a plain TRL ``SFTTrainer``.

    Safe to apply here specifically because ``per_device_train_batch_size=1`` (no
    per-example window to reconcile across a batch), and ``logits_to_keep`` as a
    plain Python int ("only project the last N positions through the lm_head") is a
    long-standing, architecture-independent `transformers` generation API, not
    something that needs verifying per model family. The loss itself is computed by
    hand on the returned (smaller) logits tensor, not via the model's own ``labels=``
    path, specifically so correctness here does not depend on how any particular
    model class threads ``labels`` through ``logits_to_keep`` internally - that
    interaction is genuinely model-specific and was NOT verified against Tiny Aya's
    real modeling code (this repo has no GPU to check on). If the loaded model
    rejects a plain int ``logits_to_keep`` kwarg, this falls back to the unwindowed
    loss rather than guess.
    """
    from trl import SFTTrainer

    class WindowedSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch.nn.functional as functional

            labels = inputs.get("labels")
            if labels is None or labels.shape[0] != 1:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

            label_row = labels[0]
            trained_positions = (label_row != -100).nonzero(as_tuple=True)[0]
            if trained_positions.numel() == 0:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

            first_trained_position = int(trained_positions[0].item())
            # The logits at position p predict the token at p + 1, so keep one position before the first trained
            # token, and compare logits[:-1] with the labels from the first trained position on. Without this
            # shift the loss asks every position to repeat its own token (round 3, attempt 3: loss 154).
            window_start = max(first_trained_position - 1, 0)
            positions_to_keep = label_row.shape[0] - window_start
            model_inputs = {name: value for name, value in inputs.items() if name != "labels"}
            try:
                outputs = model(**model_inputs, logits_to_keep=positions_to_keep)
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

            predicting_logits = outputs.logits[:, :-1]
            target_labels = label_row[window_start + 1:].unsqueeze(0)
            loss = functional.cross_entropy(
                predicting_logits.reshape(-1, predicting_logits.size(-1)).float(),
                target_labels.reshape(-1),
                ignore_index=-100,
            )
            if not getattr(self, "_loss_was_checked", False):
                self._loss_was_checked = True
                check_loss_against_the_models_own(model, inputs, loss)
            if num_items_in_batch is not None:
                # transformers does not divide by the accumulation steps when it passes num_items_in_batch (it expects
                # a loss already normalised by it): without this the logged loss is steps times too high.
                loss = loss / self.args.gradient_accumulation_steps
            if return_outputs:
                return loss, outputs
            return loss

    return WindowedSFTTrainer


class MaskedSFTFactory:
    """AgentTune trainer_factory: Tiny Aya + LoRA + TRL SFTTrainer, loss on assistant turns only."""

    def __init__(self, training_options: argparse.Namespace):
        self.training_options = training_options
        self.trainer = None
        self.tokenizer = None
        self.dropped_too_long = 0
        self.dropped_template = 0
        self.rows_trained = 0
        self.longest_row_tokens = 0
        self.last_only: list[bool] = []  # per row of the dataset: train only its final assistant turn (corrections)

    def tokenize(self, row: dict, last_only: bool = False) -> dict | None:
        """Token ids and labels of one SFT row, or None if it is dropped (template error, too long)."""
        tokenizer = self.tokenizer
        if last_only:
            row = {**row, "loss_mask": mask_all_but_last_assistant(row["messages"], row["loss_mask"])}

        def render(messages: list[dict], add_generation_prompt: bool) -> str:
            if not messages:
                return ""
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

        try:
            trained_spans = target_char_spans(render, row["messages"], row["loss_mask"])
        except ValueError:
            self.dropped_template += 1
            return None

        full_text = render(row["messages"], False)
        encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = encoding["input_ids"]
        if len(input_ids) > self.training_options.max_seq_len:
            self.dropped_too_long += 1
            return None
        self.longest_row_tokens = max(self.longest_row_tokens, len(input_ids))
        labels = labels_from_offsets(input_ids, encoding["offset_mapping"], trained_spans)
        return {"input_ids": input_ids, "labels": labels}

    def __call__(self, *, model, train_dataset, **kwargs):
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        options = self.training_options
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenized_rows = []
        for row_index, row in enumerate(train_dataset):
            tokenized_row = self.tokenize(row, last_only=bool(self.last_only and self.last_only[row_index]))
            if tokenized_row is not None:
                tokenized_rows.append(tokenized_row)
        self.rows_trained = len(tokenized_rows)
        print(
            f"tokenized {len(tokenized_rows)} rows; dropped {self.dropped_too_long} over"
            f" {options.max_seq_len} tokens, {self.dropped_template} template errors;"
            f" longest {self.longest_row_tokens}",
            flush=True,
        )
        if not tokenized_rows:
            raise SystemExit("No training rows left after tokenization.")

        student_model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto")
        sft_config = SFTConfig(
            output_dir=kwargs.get("output_dir", options.output_dir),
            num_train_epochs=options.epochs,
            learning_rate=options.lr,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,  # a float below 1 is a share of the training steps (transformers 5 dropped warmup_ratio)
            per_device_train_batch_size=options.batch_size,
            gradient_accumulation_steps=options.grad_accum,
            gradient_checkpointing=True,
            bf16=True,
            max_length=options.max_seq_len,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=options.seed,
        )
        lora_config = LoraConfig(
            r=options.lora_r,
            lora_alpha=options.lora_alpha,
            lora_dropout=options.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        )
        if options.batch_size == 1:
            trainer_class = windowed_sft_trainer()
        else:
            trainer_class = SFTTrainer
        self.trainer = trainer_class(
            model=student_model,
            args=sft_config,
            train_dataset=Dataset.from_list(tokenized_rows),
            processing_class=self.tokenizer,
            peft_config=lora_config,
        )
        return self.trainer


def load_examples(repo: str, revision: str) -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download

    sft_path = hf_hub_download(repo, "sft.jsonl", repo_type="dataset", revision=revision)
    with open(sft_path) as sft_file:
        records = [json.loads(line) for line in sft_file if line.strip()]
    try:
        stats_path = hf_hub_download(repo, "sft_stats.json", repo_type="dataset", revision=revision)
        data_stats = json.loads(Path(stats_path).read_text())
    except Exception:  # noqa: BLE001 - stats are optional
        data_stats = {}
    return records, data_stats


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


def check_rows_match_records(agenttune_rows: list[dict], records: list[dict]) -> None:
    """The rows AgentTune will train on must be the student's prompts, unchanged."""
    assert len(agenttune_rows) == len(records)
    for row, record in zip(agenttune_rows, records, strict=True):
        expected_messages = [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
        assert row["messages"] == expected_messages


def push_adapter(training_options: argparse.Namespace, adapter_directory: Path, final_loss) -> None:
    from huggingface_hub import HfApi

    hf_api = HfApi(token=os.environ.get("HF_TOKEN"))
    hf_api.create_repo(training_options.output_repo, exist_ok=True, private=False)
    hf_api.create_branch(training_options.output_repo, branch=training_options.revision, exist_ok=True)
    hf_api.upload_folder(
        folder_path=str(adapter_directory),
        repo_id=training_options.output_repo,
        revision=training_options.revision,
        commit_message=f"LoRA {training_options.revision}: loss {final_loss}",
    )
    print(f"Pushed https://huggingface.co/{training_options.output_repo}/tree/{training_options.revision}", flush=True)


def write_lineage_record(adapter_directory: Path, training_options: argparse.Namespace, final_loss) -> None:
    """``lexsi_provenance.json`` (AgentTune's ``write_provenance``): the base model, the SFT dataset and its revision,
    and the training settings, the lineage record every Lexsi library writes. AgentTune's own trainers write it; our
    distill factory did not (2026-11-15). A failure here only warns: it must never lose a trained adapter."""
    try:
        from agenttune.utils.provenance import write_provenance

        write_provenance(
            adapter_directory,
            method="agentic.distill.sft_lora",
            base_model=training_options.student,
            dataset=training_options.dataset,
            dataset_config=training_options.dataset_revision,
            params={
                "output_repo": training_options.output_repo,
                "revision": training_options.revision,
                "lora_r": training_options.lora_r,
                "lora_alpha": training_options.lora_alpha,
                "lr": training_options.lr,
                "epochs": training_options.epochs,
                "max_seq_len": training_options.max_seq_len,
                "train_loss": final_loss,
            },
        )
    except Exception as error:  # noqa: BLE001
        print(f"warning: lexsi_provenance.json not written: {error}", flush=True)


def main(argv: list[str] | None = None) -> int:
    training_options = parse_args(argv)
    from agenttune import create_distill_trainer

    started_at = time.time()
    records, data_stats = load_examples(training_options.dataset, training_options.dataset_revision)
    eventlog_class = make_eventlog_class()
    event_logs = [to_eventlog(eventlog_class, record["messages"]) for record in records]
    trainer_factory = MaskedSFTFactory(training_options)
    trainer_factory.last_only = [bool(record.get("train_last_only")) for record in records]
    trainer = create_distill_trainer(
        training_options.student,
        trajectories=event_logs,
        trainer_factory=trainer_factory,
        output_dir=training_options.output_dir,
    )

    agenttune_rows = trainer.sft_dataset()
    check_rows_match_records(agenttune_rows, records)
    print(f"{len(agenttune_rows)} AgentTune rows match sft.jsonl", flush=True)

    training_result = trainer.train()
    final_loss = getattr(training_result, "training_loss", None)
    adapter_directory = Path(training_options.output_dir) / "adapter"
    trainer_factory.trainer.model.save_pretrained(adapter_directory)
    trainer_factory.tokenizer.save_pretrained(adapter_directory)

    training_meta = {
        "student": training_options.student,
        "dataset": training_options.dataset,
        "dataset_revision": training_options.dataset_revision,
        "data_stats": data_stats,
        "rows_trained": trainer_factory.rows_trained,
        "dropped_too_long": trainer_factory.dropped_too_long,
        "dropped_template": trainer_factory.dropped_template,
        "longest_tokens": trainer_factory.longest_row_tokens,
        "hyperparams": {
            name: value for name, value in vars(training_options).items() if name not in ("output_dir",)
        },
        "train_loss": final_loss,
        "agenttune_events": [[event.stage, event.kind] for event in trainer.events()],
        "minutes": round((time.time() - started_at) / 60, 1),
    }
    (adapter_directory / "training_meta.json").write_text(json.dumps(training_meta, indent=2, default=str))
    write_lineage_record(adapter_directory, training_options, final_loss)
    (adapter_directory / "README.md").write_text(MODEL_CARD.format(
        student=training_options.student,
        revision=training_options.revision,
        dataset=training_options.dataset,
        dataset_revision=training_options.dataset_revision,
        n_rows=trainer_factory.rows_trained,
        r=training_options.lora_r,
        alpha=training_options.lora_alpha,
        lr=training_options.lr,
        epochs=training_options.epochs,
        max_seq_len=training_options.max_seq_len,
        loss=final_loss,
    ))
    print(json.dumps(training_meta, indent=2, default=str), flush=True)

    if not training_options.no_push:
        push_adapter(training_options, adapter_directory, final_loss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
