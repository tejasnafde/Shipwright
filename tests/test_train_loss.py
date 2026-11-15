"""The windowed loss of scripts/train_lora.py must equal the model's own labels= loss (HF shifts the labels by one).

Round 3, attempt 3 trained with no shift (loss 154, an adapter that copies the current token): a random-weight smoke
run cannot see that, this comparison can. Needs torch and trl (the HF Job and CI-less laptops with them); skipped otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_lora  # noqa: E402


def tiny_model():
    config = transformers.Cohere2Config(
        vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    return transformers.Cohere2ForCausalLM(config).float().eval()


def test_windowed_loss_equals_the_models_own_shifted_loss():
    model = tiny_model()
    input_ids = torch.randint(0, 200, (1, 40))
    labels = input_ids.clone()
    labels[0, :25] = -100  # the system prompt and early turns are masked
    labels[0, 31:34] = -100  # a masked user turn in the middle

    trainer_class = train_lora.windowed_sft_trainer()
    trainer = object.__new__(trainer_class)  # compute_loss needs no trainer state here
    windowed_loss = trainer_class.compute_loss(trainer, model, {"input_ids": input_ids, "labels": labels})
    models_own_loss = model(input_ids=input_ids, labels=labels).loss
    assert torch.allclose(windowed_loss, models_own_loss, atol=1e-5), (windowed_loss.item(), models_own_loss.item())


def test_the_first_step_check_refuses_a_loss_that_is_not_the_models_own():
    model = tiny_model()
    input_ids = torch.randint(0, 200, (1, 40))
    labels = input_ids.clone()
    labels[0, :25] = -100
    inputs = {"input_ids": input_ids, "labels": labels}
    models_own_loss = model(**inputs).loss
    train_lora.check_loss_against_the_models_own(model, inputs, models_own_loss.detach())  # the right loss passes
    with pytest.raises(RuntimeError, match="differs from the model's own loss"):
        train_lora.check_loss_against_the_models_own(model, inputs, models_own_loss.detach() * 20)
