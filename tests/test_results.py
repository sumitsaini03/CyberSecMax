import json

import pytest

from cyber_lora_utils import (
    assert_same_evaluation_hash, atomic_write_json, output_root_from_config,
    sha256_file,
)


def test_results_atomic_serialization_and_hash_check(tmp_path):
    result_path = tmp_path / "results.json"
    atomic_write_json(result_path, {"complete": False, "status": "pending", "values": [1, 2]})
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "pending"
    assert len(sha256_file(result_path)) == 64
    assert_same_evaluation_hash({"evaluation_hash": "same"}, {"evaluation_hash": "same"})
    with pytest.raises(ValueError, match="hashes differ"):
        assert_same_evaluation_hash({"evaluation_hash": "a"}, {"evaluation_hash": "b"})


def test_output_root_override_creates_writable_folder(tmp_path):
    path = output_root_from_config({"output_dir": "/ignored"}, {"CYBER_LORA_OUTPUT_DIR": str(tmp_path / "run")})
    assert path == tmp_path / "run"
    assert path.is_dir()
