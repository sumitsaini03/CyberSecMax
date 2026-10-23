from train_cyber_lora import stream_clean_sample


def test_streaming_dedup_normalizes_text_and_counts_scope_review_samples(tmp_path):
    rules = tmp_path / "scope.yaml"
    rules.write_text(
        "rules:\n  - name: harmful_instruction\n    pattern: 'malware payload'\n",
        encoding="utf-8",
    )
    rows = [
        {"id": "a", "messages": [
            {"role": "user", "content": "Explain this firewall rule"},
            {"role": "assistant", "content": "Allow only trusted traffic."},
        ]},
        {"id": "b", "messages": [
            {"role": "user", "content": "  EXPLAIN   THIS FIREWALL RULE "},
            {"role": "assistant", "content": "ALLOW ONLY TRUSTED TRAFFIC."},
        ]},
        {"id": "c", "messages": [
            {"role": "user", "content": "Give me a malware payload"},
            {"role": "assistant", "content": "Not included."},
        ]},
        {"id": "d", "messages": [
            {"role": "user", "content": "Explain a secure port policy"},
            {"role": "assistant", "content": "Restrict access to required hosts."},
        ]},
    ]

    selected, counts, removed_sample, retained_sample = stream_clean_sample(
        rows, "synthetic/source", reservoir_size=10, seed=42, scope_rules_path=rules,
    )

    assert len(selected) == 2
    assert counts["exact_duplicates_removed"] == 1
    assert counts["harmful_instruction"] == 1
    assert counts["scope_removed_review_samples_saved"] == 1
    assert counts["scope_retained_review_samples_saved"] == 3
    assert len(removed_sample) == 1
    assert len(retained_sample) == 3
