import json

from cyber_lora_utils import atomic_write_json, latest_valid_checkpoint, mark_checkpoint, validate_baseline_artifact


def test_baseline_reuse_requires_matching_fingerprint_and_complete_flag(tmp_path):
    path = tmp_path / "baseline.json"
    atomic_write_json(path, {"complete": True, "fingerprint": "abc", "value": 7})
    assert validate_baseline_artifact(path, "abc")["value"] == 7
    assert validate_baseline_artifact(path, "different") is None
    atomic_write_json(path, {"complete": False, "fingerprint": "abc"})
    assert validate_baseline_artifact(path, "abc") is None


def test_checkpoint_recovery_ignores_unmarked_checkpoints(tmp_path):
    check_dir = tmp_path / "checkpoints"
    invalid = check_dir / "checkpoint-10"
    invalid.mkdir(parents=True)
    assert latest_valid_checkpoint(check_dir, "run-fp") is None
    valid = check_dir / "checkpoint-20"
    valid.mkdir(parents=True)
    (valid / "adapter_config.json").write_text("{}", encoding="utf-8")
    (valid / "adapter_model.safetensors").write_bytes(b"weights")
    (valid / "trainer_state.json").write_text("{}", encoding="utf-8")
    mark_checkpoint(valid, "run-fp")
    assert latest_valid_checkpoint(check_dir, "run-fp") == valid


def test_checkpoint_with_marker_but_missing_weights_is_rejected(tmp_path):
    check_dir = tmp_path / "checkpoints"
    incomplete = check_dir / "checkpoint-30"
    incomplete.mkdir(parents=True)
    (incomplete / "adapter_config.json").write_text("{}", encoding="utf-8")
    (incomplete / "trainer_state.json").write_text("{}", encoding="utf-8")
    (incomplete / "cyber_lora_checkpoint.json").write_text(
        '{"fingerprint":"run-fp","complete":true}', encoding="utf-8")
    assert latest_valid_checkpoint(check_dir, "run-fp") is None
