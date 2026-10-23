import pytest

from cyber_lora_utils import (
    ANSWER_LABELS, benchmark_prompt, compare_predictions, normalize_benchmark_row,
    validate_answer_tokens,
)
from tests.fake_tokenizer import FakeTokenizer


def sample_row():
    return {"question": "Which option?", "answers": {"A": "first", "B": "second", "C": "third", "D": "fourth"}, "solution": "B"}


def test_answer_boundary_ids_are_validated():
    tokenizer = FakeTokenizer()
    prompt, _ = benchmark_prompt(sample_row())
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": "system"}, {"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
    result = validate_answer_tokens(tokenizer, rendered)
    assert result["strategy"] == "single_token"
    assert set(result["candidate_ids"]) == set(ANSWER_LABELS)
    assert len(set(result["candidate_ids"].values())) == 4


def test_cyclic_option_permutation_remaps_correct_label():
    row = normalize_benchmark_row(sample_row())
    prompt0, solution0 = benchmark_prompt(row, rotation=0)
    prompt1, solution1 = benchmark_prompt(row, rotation=1)
    assert solution0 == "B"
    assert solution1 == "A"
    assert "A) second" in prompt1


def test_paired_metrics_and_invalid_answer_handling():
    before = [
        {"predicted": "A", "expected": "A", "solution": "A", "correct": True},
        {"predicted": "B", "expected": "A", "solution": "A", "correct": False},
        {"predicted": "C", "expected": "C", "solution": "C", "correct": True},
    ]
    after = [
        {"predicted": "B", "expected": "A", "solution": "A", "correct": False},
        {"predicted": "A", "expected": "A", "solution": "A", "correct": True},
        {"predicted": "C", "expected": "C", "solution": "C", "correct": True},
    ]
    metrics = compare_predictions(before, after)
    assert metrics["n"] == 3
    assert metrics["wrong_to_right"] == 1 and metrics["right_to_wrong"] == 1
    assert metrics["mcnemar_exact_two_sided_p"] == 1.0
    assert metrics["letter_distributions"]["answer_key"] == {"A": 2, "B": 0, "C": 1, "D": 0}
    bad = {"question": "Q", "answers": {"A": "a", "B": "b", "C": "c", "D": "d"}, "solution": "E"}
    with pytest.raises(ValueError, match="solution"):
        normalize_benchmark_row(bad)
