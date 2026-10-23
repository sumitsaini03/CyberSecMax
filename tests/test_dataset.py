import pytest

from cyber_lora_utils import (
    SYSTEM_PROMPT, deduplicate_records, normalize_benchmark_row,
    normalize_training_record, split_records, validate_messages,
    tokenize_supervised_example, assert_supervised_decode,
)
from tests.fake_tokenizer import FakeTokenizer


def test_schema_validation_and_fixed_system_prompt():
    row = {"messages": [{"role": "user", "content": "Explain a firewall."},
                        {"role": "assistant", "content": "It filters traffic."}], "id": "doc-1"}
    normalized = normalize_training_record(row, "synthetic")
    assert normalized["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert normalized["source"] == "synthetic"
    assert normalized["source_id"] == "doc-1"
    assert normalized["source_group_id"].startswith("synthetic:")


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "user", "content": "question"}],
    [{"role": "user", "content": "question"}, {"role": "assistant", "content": "  "}],
    [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}],
    [{"role": "tool", "content": "not allowed"}],
])
def test_rejects_malformed_conversations(messages):
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_exact_deduplication_and_group_aware_split():
    rows = [
        normalize_training_record({"id": "shared", "messages": [
            {"role": "user", "content": "Question one"}, {"role": "assistant", "content": "Answer one"}]}, "s"),
        normalize_training_record({"id": "shared", "messages": [
            {"role": "user", "content": "Question two"}, {"role": "assistant", "content": "Answer two"}]}, "s"),
        normalize_training_record({"id": "other", "messages": [
            {"role": "user", "content": "Question three"}, {"role": "assistant", "content": "Answer three"}]}, "s"),
    ]
    kept, removed = deduplicate_records(rows + [rows[0]])
    assert len(kept) == 3 and removed == 1
    train, validation, strategy = split_records(kept, seed=42)
    assert strategy == "source_id"
    shared_splits = {"validation" if row["source_id"] in {x["source_id"] for x in validation} else "train"
                     for row in kept if row["source_id"] == "shared"}
    assert len(shared_splits) == 1


def test_benchmark_answer_mapping_and_invalid_rows():
    good = {"question": "Pick one", "answers": {"A": "a", "B": "b", "C": "c", "D": "d"}, "solution": "b"}
    assert normalize_benchmark_row(good)["solution"] == "B"
    with pytest.raises(ValueError, match="exactly A, B, C and D"):
        normalize_benchmark_row({**good, "answers": {"A": "a", "B": "b"}})


def test_chat_format_masks_non_assistant_and_includes_assistant_end_token():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What does MFA protect against?"},
        {"role": "assistant", "content": "It reduces account takeover risk."},
    ]
    feature = tokenize_supervised_example(tokenizer, messages, max_length=2048)
    assert_supervised_decode(tokenizer, messages, feature)
    assert [label for label in feature["labels"] if label != -100][-1] == 2
    assert feature["labels"].count(-100) > 0


def test_chat_loss_mask_uses_offsets_from_full_rendered_text():
    class OffsetOnlyTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            raise AssertionError("loss masking must not tokenize prefixes independently")

    tokenizer = OffsetOnlyTokenizer()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Explain a firewall."},
        {"role": "assistant", "content": "It filters traffic."},
    ]
    feature = tokenize_supervised_example(tokenizer, messages, max_length=2048)
    assert_supervised_decode(tokenizer, messages, feature)
    assert [label for label in feature["labels"] if label != -100][-1] == 2


def test_chat_loss_mask_supervises_indivisible_token_crossing_answer_start():
    class BoundaryMergeTokenizer(FakeTokenizer):
        def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
            ids = []
            offsets = []
            i = 0
            while i < len(text):
                matched = next((token for token in (self.im_start_token, self.im_end_token)
                                if text.startswith(token, i)), None)
                if matched:
                    ids.append(self._special_to_id[matched])
                    offsets.append((i, i + len(matched)))
                    i += len(matched)
                elif text[i] == "\n" and i + 1 < len(text):
                    ids.append(50000 + ord(text[i + 1]))
                    offsets.append((i, i + 2))
                    i += 2
                else:
                    ids.append(ord(text[i]) + 1000)
                    offsets.append((i, i + 1))
                    i += 1
            result = {"input_ids": ids}
            if return_offsets_mapping:
                result["offset_mapping"] = offsets
            return result

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            expanded = []
            for token_id in ids:
                token_id = int(token_id)
                if token_id >= 50000:
                    expanded.extend([1000 + ord("\n"), 1000 + token_id - 50000])
                else:
                    expanded.append(token_id)
            return super().decode(expanded, skip_special_tokens, clean_up_tokenization_spaces)

    tokenizer = BoundaryMergeTokenizer()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Explain a firewall."},
        {"role": "assistant", "content": "It filters traffic."},
    ]
    feature = tokenize_supervised_example(tokenizer, messages, max_length=2048)
    assert_supervised_decode(tokenizer, messages, feature)
    supervised = [token for token, label in zip(feature["input_ids"], feature["labels"]) if label != -100]
    assert tokenizer.decode(supervised, skip_special_tokens=False).startswith("\nI")
