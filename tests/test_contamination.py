from cyber_lora_utils import check_contamination, normalize_training_record


def make_record(text, answer="No overlap", record_id="id"):
    return normalize_training_record({"id": record_id, "messages": [
        {"role": "user", "content": text}, {"role": "assistant", "content": answer}]}, "synthetic")


def test_streaming_contamination_levels_leave_benchmark_unmodified():
    row = {"question": "Which control best prevents credential stuffing by limiting repeated authentication attempts across many accounts?",
           "answers": {"A": "Rate limiting", "B": "Port mirroring", "C": "Disk encryption", "D": "Code signing"}, "solution": "A"}
    exact = make_record("Please answer: " + row["question"], "Rate limiting", "exact")
    stem = make_record("Which control best prevents credential stuffing by limiting repeated authentication attempts across many", "The answer is Rate limiting.", "stem")
    clean = make_record("Explain how disk encryption protects lost laptops.", "It protects stored data.", "clean")
    kept, removed, counts = check_contamination([exact, stem, clean], [row], use_ngram=False)
    assert [x["source_id"] for x in kept] == ["clean"]
    assert len(removed) == 2
    assert counts["level1_question"] == 1
    assert counts["level2_question_and_correct_answer"] == 1
    assert counts["benchmark_questions_examined"] == 1
    assert row["question"].endswith("accounts?")


def test_punctuation_and_unicode_normalization():
    assert check_contamination.__name__ == "check_contamination"
    from cyber_lora_utils import normalize_match_text
    assert normalize_match_text("Ａ  B\nC", strip_punctuation=True) == "a b c"
