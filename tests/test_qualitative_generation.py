from contextlib import nullcontext
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import train_cyber_lora


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, key):
        return ("slice", key)


class FakeBatchEncoding(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return FakeBatchEncoding(
            input_ids=FakeTensor((1, 2)),
            attention_mask=FakeTensor((1, 2)),
        )

    def decode(self, tokens, skip_special_tokens=True):
        assert tokens[0] == "slice"
        return "response"


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_calls = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        assert "input_ids" in kwargs
        assert "attention_mask" in kwargs
        return FakeTensor((1, 5))


def test_qualitative_generation_unpacks_batch_encoding():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    fake_torch = SimpleNamespace(no_grad=nullcontext)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        outputs = train_cyber_lora.generate_qualitative(model, tokenizer, max_new_tokens=3)

    assert len(outputs) == len(train_cyber_lora.QUALITATIVE_PROMPTS)
    assert len(model.generate_calls) == len(outputs)
    assert outputs[0]["response"] == "response"
    assert outputs[0]["truncated_by_token_cap"] is True


def test_full_training_receives_tokenizer_explicitly():
    parameters = inspect.signature(train_cyber_lora.train_model).parameters
    assert "tokenizer" in parameters
