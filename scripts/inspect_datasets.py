#!/usr/bin/env python3
"""Inspect configured Hugging Face sources without copying dataset rows to Git."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _serializable_splits(info: Any) -> dict[str, int | None]:
    output: dict[str, int | None] = {}
    splits = getattr(info, "splits", None) or {}
    for name, split in splits.items():
        value = getattr(split, "num_examples", None)
        output[str(name)] = int(value) if value is not None else None
    return output


def _features(info: Any) -> Any:
    features = getattr(info, "features", None)
    if features is None:
        return "UNVERIFIED"
    try:
        return json.loads(json.dumps(features, default=lambda item: item.to_dict()))
    except Exception:
        return str(features)


def _inspect_dataset(repo_id: str, configs: list[str], token: str | None, *, load_for_exact_counts: bool = False) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from datasets import get_dataset_config_info, get_dataset_config_names, load_dataset

    api = HfApi(token=token)
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "access_status": "UNVERIFIED",
        "revision": "UNVERIFIED",
        "license": "UNVERIFIED",
        "gated": "UNVERIFIED",
        "configs": {},
        "error": None,
    }
    try:
        info = api.dataset_info(repo_id, files_metadata=False)
        result["revision"] = getattr(info, "sha", None) or "UNVERIFIED"
        result["gated"] = getattr(info, "gated", None)
        result["private"] = bool(getattr(info, "private", False))
        card = getattr(info, "cardData", None)
        result["license"] = getattr(card, "license", None) or "UNVERIFIED"
        result["access_status"] = "repository_metadata_accessible"
    except Exception as exc:
        result["access_status"] = "repository_metadata_unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        discovered = get_dataset_config_names(repo_id, revision=result["revision"], token=token)
        selected = configs if configs != ["*"] else list(discovered)
        result["available_configs"] = list(discovered)
    except Exception as exc:
        selected = configs
        result["config_discovery_error"] = f"{type(exc).__name__}: {exc}"

    for config_name in selected:
        config_item: dict[str, Any] = {"config": config_name}
        try:
            info = get_dataset_config_info(repo_id, config_name, revision=result["revision"], token=token)
            config_item["features"] = _features(info)
            config_item["splits"] = _serializable_splits(info)
            if load_for_exact_counts:
                loaded = load_dataset(repo_id, config_name, revision=result["revision"], token=token)
                config_item["exact_split_counts"] = {name: len(split) for name, split in loaded.items()}
                config_item["loaded_splits"] = list(loaded)
                config_item["data_access"] = "accessible"
            else:
                if token:
                    split_names = list(_serializable_splits(info))
                    sample_split = "train" if "train" in split_names else (split_names[0] if split_names else "train")
                    stream = load_dataset(repo_id, config_name, split=sample_split, revision=result["revision"],
                                          token=token, streaming=True)
                    next(iter(stream.take(1)), None)
                    config_item["data_access"] = "accessible_sampled_one_row"
                else:
                    config_item["data_access"] = "metadata_only"
            config_item["metadata_status"] = "verified"
        except Exception as exc:
            config_item["metadata_status"] = "UNVERIFIED"
            config_item["data_access"] = "not_loaded"
            config_item["error"] = f"{type(exc).__name__}: {exc}"
        result["configs"][config_name] = config_item
    if load_for_exact_counts and not any(item.get("data_access") == "accessible" for item in result["configs"].values()):
        result["access_status"] = "data_access_failed"
    elif result["gated"] and not token:
        result["access_status"] = "gated_no_token"
    elif any(item.get("data_access") == "accessible" for item in result["configs"].values()):
        result["access_status"] = "data_accessible"
    elif any(item.get("data_access") == "accessible_sampled_one_row" for item in result["configs"].values()):
        result["access_status"] = "data_accessible"
    elif result["gated"] and token:
        result["access_status"] = "gated_access_not_verified"
    elif any(item.get("data_access") == "not_loaded" for item in result["configs"].values()):
        result["access_status"] = "data_access_failed"
    return result


def inspect_all(config: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    dataset_cfg = config["datasets"]
    source_map = {
        "redsage_conv": (dataset_cfg["tier1_id"], [dataset_cfg["tier1_config"]], False),
        "primus_instruct": (dataset_cfg["tier2_id"], ["*"] , False),
        "primus_reasoning": (dataset_cfg["tier3_id"], ["*"] , False),
        "ultrachat_candidate": ("HuggingFaceH4/ultrachat_200k", ["*"] , False),
        "cybermetric_500": (dataset_cfg["benchmark_id"], [dataset_cfg["benchmark_config"]], True),
    }
    sources: dict[str, Any] = {}
    for key, (repo_id, configs, exact_counts) in source_map.items():
        sources[key] = _inspect_dataset(repo_id, configs, token, load_for_exact_counts=exact_counts)
    model_id = config["model"]["id"]
    model: dict[str, Any]
    try:
        model_info = HfApi(token=token).model_info(model_id, files_metadata=False)
        card = getattr(model_info, "cardData", None)
        model = {
            "repo_id": model_id,
            "revision": getattr(model_info, "sha", None) or "UNVERIFIED",
            "license": getattr(card, "license", None) or "UNVERIFIED",
            "access_status": "accessible",
        }
    except Exception as exc:
        model = {"repo_id": model_id, "revision": "UNVERIFIED", "license": "UNVERIFIED",
                 "access_status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "token_present": bool(token),
        "model": model,
        "sources": sources,
        "profile_decision": "deferred_to_training_preflight_after_access_and_terms_checks",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "dataset_metadata.json"))
    parser.add_argument("--require-benchmark", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token
        token = token or get_token()
    except Exception:
        pass
    try:
        inventory = inspect_all(config, token=token)
    except ImportError as exc:
        print(f"Missing inspection dependency: {exc}. Install requirements-colab.txt first.", file=sys.stderr)
        return 2
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, item in inventory["sources"].items():
        print(f"{name}: {item['access_status']} (revision={item['revision']})")
    if args.require_benchmark and inventory["sources"]["cybermetric_500"]["access_status"] not in {"data_accessible", "repository_metadata_accessible"}:
        print("CyberMetric-500 could not be verified or loaded.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
