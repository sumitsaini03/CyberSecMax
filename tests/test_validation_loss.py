import math
from types import SimpleNamespace

import pytest

from cyber_lora_utils import manual_validation_loss, token_weighted_loss_from_log_probs
from tests.fake_tokenizer import FakeTokenizer


def test_hand_computed_token_weighted_loss():
    # Three supervised targets have probabilities 1/2, 1/4 and 1/8.
    result = token_weighted_loss_from_log_probs([math.log(0.5), math.log(0.25), math.log(0.125)], 3)
    assert result == pytest.approx(math.log(4.0))
    with pytest.raises(ValueError):
        token_weighted_loss_from_log_probs([math.log(0.5)], 2)


def test_manual_validation_loss_matches_hand_computed_shift_and_mask():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.device = torch.device("cpu")
        def forward(self, input_ids, attention_mask):
            logits = torch.zeros((*input_ids.shape, 2), dtype=torch.float32, device=input_ids.device)
            logits[..., 1] = torch.where(input_ids.eq(0), 0.0, math.log(3.0))
            return SimpleNamespace(logits=logits)

    examples = [
        {"input_ids": [0, 0, 0], "labels": [-100, 1, -100], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 1], "labels": [-100, 1], "attention_mask": [1, 1]},
    ]
    result = manual_validation_loss(TinyModel(), FakeTokenizer(), examples, batch_size=2)
    expected = (math.log(2.0) + math.log(4.0 / 3.0)) / 2
    assert result["loss"] == pytest.approx(expected, rel=1e-6)
    assert result["valid_target_tokens"] == 2
