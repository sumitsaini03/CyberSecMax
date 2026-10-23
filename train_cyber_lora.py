#!/usr/bin/env python3
"""Reproducible cybersecurity QLoRA experiment orchestration for Google Colab.

The base-model benchmark, validation-loss, and qualitative baselines are saved
before any PEFT/LoRA configuration or Trainer is constructed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from cyber_lora_utils import (
    ANSWER_LABELS, QUALITATIVE_PROMPTS, SYSTEM_PROMPT, assert_same_evaluation_hash,
    assert_supervised_decode, atomic_write_json, benchmark_prompt, canonical_json,
    check_contamination, compare_predictions, deduplicate_records, fingerprint,
    latest_valid_checkpoint, manual_validation_loss, normalize_benchmark_row,
    normalize_training_record, output_root_from_config, paired_bootstrap_ci,
    read_jsonl, score_benchmark, sha256_file, sha256_text, split_records,
    tokenize_supervised_example, validate_answer_tokens, validate_baseline_artifact,
    normalized_conversation_key, valid_checkpoint, write_jsonl,
)


class ExperimentError(RuntimeError):
    pass


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate", "trl")
    output = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("experiment YAML must contain a mapping")
    return config


def verify_gpu() -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise ExperimentError("CUDA is unavailable. Select a GPU runtime in Colab before running the experiment.")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    info = {
        "cuda_available": True,
        "gpu_name": props.name,
        "total_vram_bytes": int(total),
        "free_vram_bytes_at_start": int(free),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    print(f"GPU: {info['gpu_name']} | free VRAM: {free / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB")
    return info


def load_tokenizer(model_id: str, revision: str, token: str | None) -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ExperimentError("A fast tokenizer is required for deterministic loss-mask alignment.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_base(model_id: str, revision: str, token: str | None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "quantization_config": quant,
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    # Transformers 5 renamed torch_dtype to dtype; keep the loader explicit.
    try:
        major = int(importlib.metadata.version("transformers").split(".", 1)[0])
    except Exception:
        major = 4
    load_kwargs["dtype" if major >= 5 else "torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model.config.use_cache = False
    return model


def model_runtime_metadata(model: Any) -> dict[str, Any]:
    try:
        footprint = int(model.get_memory_footprint())
    except Exception:
        footprint = None
    return {"parameter_memory_footprint_bytes": footprint, "device_map": getattr(model, "hf_device_map", None)}


def load_benchmark_rows(config: Mapping[str, Any], dataset_meta: Mapping[str, Any], token: str | None) -> tuple[list[dict[str, Any]], str, str]:
    from datasets import load_dataset
    source = dataset_meta["sources"]["cybermetric_500"]
    revision = source.get("revision")
    if not revision or revision == "UNVERIFIED":
        raise ExperimentError("CyberMetric dataset revision is UNVERIFIED; refusing to score an unpinned test set.")
    rows = load_dataset(
        config["datasets"]["benchmark_id"], config["datasets"]["benchmark_config"],
        split=config["datasets"]["benchmark_split"], revision=revision, token=token,
    )
    normalized = [normalize_benchmark_row(row, i) for i, row in enumerate(rows)]
    if len(normalized) != 500:
        raise ExperimentError(f"Expected 500 CyberMetric rows, received {len(normalized)}.")
    evaluation_hash = sha256_text(canonical_json(normalized))
    return normalized, evaluation_hash, revision


def load_source_rows(repo_id: str, config_name: str | None, revision: str, token: str | None) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset
    dataset = load_dataset(repo_id, config_name, revision=revision, token=token)
    if "train" in dataset:
        split_name = "train"
    elif len(dataset) == 1:
        split_name = next(iter(dataset))
    else:
        raise ExperimentError(f"{repo_id}: expected a train split or one unambiguous split; found {list(dataset)}")
    return [dict(row) for row in dataset[split_name]], split_name


def stream_source(repo_id: str, config_name: str | None, split_name: str, revision: str, token: str | None) -> Any:
    from datasets import load_dataset
    return load_dataset(repo_id, config_name, split=split_name, revision=revision, token=token, streaming=True)


def reservoir_add(reservoir: list[dict[str, Any]], item: Mapping[str, Any], seen: int,
                  capacity: int, rng: random.Random) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(dict(item))
    else:
        index = rng.randrange(seen + 1)
        if index < capacity:
            reservoir[index] = dict(item)


def stream_clean_sample(
    rows: Any, repo_id: str, *, reservoir_size: int, seed: int,
    scope_rules_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize one row at a time, count all filters, and retain a seeded reservoir."""
    compiled = []
    if scope_rules_path:
        with open(scope_rules_path, encoding="utf-8") as handle:
            rules = yaml.safe_load(handle)["rules"]
        compiled = [(rule["name"], re.compile(rule["pattern"])) for rule in rules]
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    removed_sample: list[dict[str, Any]] = []
    retained_sample: list[dict[str, Any]] = []
    counts: dict[str, Any] = {name: 0 for name, _ in compiled}
    counts.update({"raw_rows_examined": 0, "malformed_records_removed": 0,
                   "scope_records_removed": 0, "exact_duplicates_removed": 0,
                   "valid_unique_records": 0})
    seen_keys: set[str] = set()
    retained_for_review = 0
    removed_for_review = 0
    for raw in rows:
        counts["raw_rows_examined"] += 1
        try:
            record = normalize_training_record(raw, repo_id)
        except (ValueError, TypeError):
            counts["malformed_records_removed"] += 1
            continue
        if compiled:
            content = "\n".join(message["content"] for message in record["messages"])
            rule_name = next((name for name, pattern in compiled if pattern.search(content)), None)
            if rule_name:
                counts[rule_name] += 1
                counts["scope_records_removed"] += 1
                record["scope_filter_rule"] = rule_name
                reservoir_add(removed_sample, record, removed_for_review, 100, rng)
                removed_for_review += 1
                continue
            reservoir_add(retained_sample, record, retained_for_review, 100, rng)
            retained_for_review += 1
        key = sha256_text(normalized_conversation_key(record))
        if key in seen_keys:
            counts["exact_duplicates_removed"] += 1
            continue
        seen_keys.add(key)
        counts["valid_unique_records"] += 1
        reservoir_add(selected, record, counts["valid_unique_records"] - 1, reservoir_size, rng)
    counts["scope_records_retained"] = counts["raw_rows_examined"] - counts["malformed_records_removed"] - counts["scope_records_removed"]
    counts["reservoir_records"] = len(selected)
    counts["scope_removed_review_samples_saved"] = len(removed_sample)
    counts["scope_retained_review_samples_saved"] = len(retained_sample)
    return selected, counts, removed_sample, retained_sample


def scope_filter(records: Sequence[Mapping[str, Any]], rules_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    with open(rules_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    compiled = [(rule["name"], re.compile(rule["pattern"])) for rule in config["rules"]]
    counts = {name: 0 for name, _ in compiled}
    retained, removed = [], []
    for raw in records:
        record = dict(raw)
        content = "\n".join(message["content"] for message in record["messages"])
        matched = next((name for name, pattern in compiled if pattern.search(content)), None)
        if matched:
            counts[matched] += 1
            record["scope_filter_rule"] = matched
            removed.append(record)
        else:
            retained.append(record)
    counts["records_examined"] = len(records)
    counts["records_removed"] = len(removed)
    counts["records_retained"] = len(retained)
    return retained, removed, counts


def write_scope_review_samples(root: Path, removed: Sequence[Mapping[str, Any]], retained: Sequence[Mapping[str, Any]], seed: int = 42) -> None:
    review_dir = root / "manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sample_removed = rng.sample(list(removed), min(100, len(removed))) if removed else []
    sample_retained = rng.sample(list(retained), min(100, len(retained))) if retained else []
    atomic_write_json(review_dir / "scope_removed_sample.json", sample_removed)
    atomic_write_json(review_dir / "scope_retained_sample.json", sample_retained)


def select_profile(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str, str]:
    dcfg = config["datasets"]
    sources = metadata["sources"]
    def data_accessible(key: str) -> bool:
        return sources[key].get("access_status") == "data_accessible"
    t1 = sources["redsage_conv"]
    t1_license = str(t1.get("license", "UNVERIFIED")).casefold()
    allowed_t1_licenses = {"apache-2.0", "mit", "odc-by", "cc-by-4.0", "cc-by-sa-4.0"}
    if (dcfg.get("allow_tier1_training") and dcfg.get("accepted_gated_dataset_conditions")
            and data_accessible("redsage_conv") and t1_license in allowed_t1_licenses):
        return "T1", "RedSage-Conv is accessible and its metadata lists an accepted training license; user flags confirm terms review."
    primus = sources["primus_instruct"]
    if (dcfg.get("allow_primus_training") and dcfg.get("accepted_gated_dataset_conditions")
            and dcfg.get("accepted_teacher_output_terms") and data_accessible("primus_instruct")):
        return "T2", "Primus-Instruct is accessible and the user confirmed review of gated conditions and teacher-output terms."
    raise ExperimentError(
        "No training profile is eligible. RedSage-Conv remains gated and its license is not published in the inspected card; "
        "Primus-Instruct requires accepted HF conditions and a separate user decision about GPT-4o-generated outputs. "
        "Review configs/experiment.yaml, set only the applicable explicit acceptance flags after reading the terms, "
        "then rerun. No substitute dataset will be selected."
    )


def data_preparation_context(
    config: Mapping[str, Any], profile: str, primary_revision: str,
    evaluation_hash: str, general_revision: str | None,
) -> dict[str, Any]:
    dcfg = config["datasets"]
    return {
        "profile": profile,
        "primary_repo": dcfg["tier1_id"] if profile == "T1" else dcfg["tier2_id"],
        "primary_config": dcfg["tier1_config"] if profile == "T1" else "default",
        "primary_revision": primary_revision,
        "general_repo": "HuggingFaceH4/ultrachat_200k" if profile == "T1" and general_revision else None,
        "general_split": "train_sft" if profile == "T1" and general_revision else None,
        "general_revision": general_revision if profile == "T1" else None,
        "general_replay_enabled": bool(dcfg.get("allow_general_replay")) if profile == "T1" else False,
        "tier1_max_records": int(dcfg["max_tier1_records"]) if profile == "T1" else None,
        "seed": int(config["seed"]),
        "evaluation_hash": evaluation_hash,
        "scope_filter_sha256": sha256_file(Path(__file__).parent / "configs" / "scope_filter.yaml") if profile == "T1" else None,
        "acceptance_flags": {
            "tier1": bool(dcfg.get("allow_tier1_training")),
            "primus": bool(dcfg.get("allow_primus_training")),
            "gated_conditions": bool(dcfg.get("accepted_gated_dataset_conditions")),
            "teacher_output_terms": bool(dcfg.get("accepted_teacher_output_terms")),
        },
    }


def prepare_frozen_data(
    root: Path, config: Mapping[str, Any], metadata: Mapping[str, Any], benchmark: Sequence[Mapping[str, Any]],
    evaluation_hash: str, token: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path, manifest_path = data_dir / "frozen_train.jsonl", data_dir / "frozen_validation.jsonl", data_dir / "dataset_manifest.json"
    has_any_frozen = manifest_path.exists() or train_path.exists() or val_path.exists()
    if has_any_frozen:
        if not (manifest_path.exists() and train_path.exists() and val_path.exists()):
            raise ExperimentError("An incomplete frozen-data set exists. Inspect the output directory; use a fresh run directory to refreeze.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        profile = previous.get("profile")
        if profile not in {"T1", "T2"}:
            raise ExperimentError("Frozen data has no valid profile marker.")
        dcfg = config["datasets"]
        if profile == "T1" and not (dcfg.get("allow_tier1_training") and dcfg.get("accepted_gated_dataset_conditions")):
            raise ExperimentError("The frozen T1 data requires the same explicit access/license acceptance flags used when it was prepared.")
        if profile == "T2" and not (dcfg.get("allow_primus_training") and dcfg.get("accepted_gated_dataset_conditions") and dcfg.get("accepted_teacher_output_terms")):
            raise ExperimentError("The frozen T2 data requires the same gated-access and teacher-output terms acceptance flags used when it was prepared.")
        primary_key = "redsage_conv" if profile == "T1" else "primus_instruct"
        current_primary_revision = metadata["sources"][primary_key].get("revision")
        old_context = previous.get("preparation_context", {})
        frozen_primary_revision = previous.get("source_revision")
        if current_primary_revision not in (None, "UNVERIFIED", frozen_primary_revision):
            raise ExperimentError("The training source revision changed since data was frozen; use a new output directory.")
        frozen_general_revision = old_context.get("general_revision")
        if profile == "T1" and frozen_general_revision:
            general_info = metadata["sources"].get("ultrachat_candidate", {})
            current_general_revision = general_info.get("revision")
            if current_general_revision not in (None, "UNVERIFIED", frozen_general_revision):
                raise ExperimentError("The general replay source revision changed since data was frozen; use a new output directory.")
        else:
            current_general_revision = frozen_general_revision
        context = data_preparation_context(config, profile, frozen_primary_revision, evaluation_hash, current_general_revision)
        context_fp = fingerprint(context)
        if previous.get("preparation_context_fingerprint") != context_fp:
            raise ExperimentError("The preparation config, evaluation hash, or source revision changed; use a new output directory.")
        if sha256_file(train_path) != previous.get("training_file_sha256") or sha256_file(val_path) != previous.get("validation_file_sha256"):
            raise ExperimentError("Frozen train/validation hash mismatch; refusing to overwrite baseline data.")
        train, valid = read_jsonl(train_path), read_jsonl(val_path)
        atomic_write_json(root / "dataset_metadata.json", {**dict(metadata), "preparation": previous})
        return train, valid, previous

    profile, decision = select_profile(config, metadata)
    dcfg = config["datasets"]
    if profile == "T1":
        info = metadata["sources"]["redsage_conv"]
        repo_id, config_name = dcfg["tier1_id"], dcfg["tier1_config"]
    else:
        info = metadata["sources"]["primus_instruct"]
        repo_id, config_name = dcfg["tier2_id"], None
    revision = info.get("revision")
    if not revision or revision == "UNVERIFIED":
        raise ExperimentError(f"Training source revision for {repo_id} is UNVERIFIED.")
    general_info = metadata["sources"].get("ultrachat_candidate", {})
    general_revision_context = general_info.get("revision") if (
        profile == "T1" and dcfg.get("allow_general_replay")
        and str(general_info.get("license", "")).casefold() == "mit"
        and general_info.get("revision") not in (None, "UNVERIFIED")
    ) else None
    preparation_context = data_preparation_context(config, profile, revision, evaluation_hash, general_revision_context)
    preparation_context_fp = fingerprint(preparation_context)
    seed = int(config["seed"])
    scope_counts: dict[str, Any]
    source_counts: dict[str, Any] = {}
    general_status: str
    if profile == "T1":
        cap = int(dcfg["max_tier1_records"])
        general_revision = general_info.get("revision")
        general_license = str(general_info.get("license", "UNVERIFIED")).casefold()
        use_general = bool(dcfg.get("allow_general_replay")) and general_license == "mit" and general_revision not in (None, "UNVERIFIED")
        general_target = int(round(cap * 0.15)) if use_general else 0
        cyber_target = cap - general_target
        cyber_stream = stream_source(repo_id, config_name, "train", revision, token)
        cyber_sample, scope_counts, scope_removed_sample, scope_retained_sample = stream_clean_sample(
            cyber_stream, repo_id, reservoir_size=cyber_target, seed=seed,
            scope_rules_path=Path(__file__).parent / "configs" / "scope_filter.yaml")
        if not cyber_sample:
            raise ExperimentError(f"No valid RedSage-Conv records remain after the scope filter: {scope_counts}")
        source_counts[repo_id] = {**scope_counts, "revision": revision, "split": "train"}
        review_dir = root / "manual_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(review_dir / "scope_removed_sample.json", scope_removed_sample)
        atomic_write_json(review_dir / "scope_retained_sample.json", scope_retained_sample)
        general_sample: list[dict[str, Any]] = []
        if use_general:
            try:
                general_stream = stream_source("HuggingFaceH4/ultrachat_200k", None, "train_sft", general_revision, token)
                general_sample, general_counts, _, _ = stream_clean_sample(
                    general_stream, "HuggingFaceH4/ultrachat_200k", reservoir_size=general_target, seed=seed + 1)
                source_counts["HuggingFaceH4/ultrachat_200k"] = {**general_counts, "revision": general_revision, "split": "train_sft"}
                general_status = "included; MIT card license verified; 15% target of the T1 total"
            except Exception as exc:
                use_general = False
                general_sample = []
                general_target = 0
                source_counts["HuggingFaceH4/ultrachat_200k"] = {"status": "not_loaded", "error": f"{type(exc).__name__}: {exc}"}
                general_status = "omitted because the public candidate could not be loaded at the inspected revision"
        else:
            general_status = "omitted by config or because the MIT license/revision was not verified"
        # Rebalance to approximately 15% of the selected mix if the cyber reservoir is smaller than requested.
        if use_general:
            actual_general_target = min(len(general_sample), round(len(cyber_sample) * 0.15 / 0.85))
            rng = random.Random(seed + 2)
            rng.shuffle(general_sample)
            general_sample = general_sample[:actual_general_target]
        combined = cyber_sample + general_sample
        per_source_duplicates = sum(
            int(item.get("exact_duplicates_removed", 0))
            for item in source_counts.values() if isinstance(item, Mapping)
        )
        selected, cross_source_duplicates = deduplicate_records(combined)
        exact_duplicates = per_source_duplicates + cross_source_duplicates
        clean, contaminated, contamination_counts = check_contamination(selected, benchmark, use_ngram=True)
        split_name = "train"
        malformed = sum(item.get("malformed_records_removed", 0) for item in source_counts.values() if isinstance(item, Mapping))
        raw_count = sum(item.get("raw_rows_examined", 0) for item in source_counts.values() if isinstance(item, Mapping))
        general_source_counts = source_counts.get("HuggingFaceH4/ultrachat_200k", {})
    else:
        raw_rows, split_name = load_source_rows(repo_id, config_name, revision, token)
        normalized: list[dict[str, Any]] = []
        malformed = 0
        for row in raw_rows:
            try:
                normalized.append(normalize_training_record(row, repo_id))
            except (ValueError, TypeError):
                malformed += 1
        if not normalized:
            raise ExperimentError(f"No valid training conversations were found in {repo_id}.")
        scope_counts = {"status": "not_required_for_T2", "records_examined": len(normalized)}
        deduplicated, exact_duplicates = deduplicate_records(normalized)
        clean, contaminated, contamination_counts = check_contamination(deduplicated, benchmark, use_ngram=True)
        selected = clean
        raw_count = len(raw_rows)
        source_counts[repo_id] = {"raw_rows_examined": raw_count, "valid_records": len(normalized),
                                  "revision": revision, "split": split_name}
        general_status = "not applicable to T2"
        general_source_counts = {}
    train, valid, split_strategy = split_records(selected, seed=int(config["seed"]), validation_fraction=0.1)
    if not train or not valid:
        raise ExperimentError(f"Insufficient cleaned records after filtering: train={len(train)}, validation={len(valid)}")
    data_manifest = {
        "profile": profile, "profile_decision": decision, "source": repo_id, "source_revision": revision,
        "source_config": config_name or "default", "source_split": split_name,
        "raw_rows_loaded": raw_count, "source_counts": source_counts,
        "malformed_records_removed": malformed,
        "scope_filter": scope_counts, "scope_removed_count": scope_counts.get("scope_records_removed", 0),
        "exact_duplicates_removed": exact_duplicates, "contamination": contamination_counts,
        "candidate_records_before_contamination": len(selected), "contamination_records_removed": len(contaminated),
        "records_after_contamination": len(clean), "records_selected": len(clean),
        "training_count": len(train), "validation_count": len(valid),
        "split_strategy": split_strategy, "seed": int(config["seed"]),
        "general_instruction_replay": general_status,
        "general_instruction_source_counts": general_source_counts,
        "preparation_context": preparation_context,
        "preparation_context_fingerprint": preparation_context_fp,
        "benchmark_hash": evaluation_hash,
        "limitations": [
            "The scope filter is keyword/regex triage; it can miss offensive instructions and can flag benign defensive discussion.",
            "Exact and 5-gram matching cannot detect paraphrases or semantic overlap.",
            "CyberMetric questions derive from public sources that can also occur in training corpora.",
            "The base model may have encountered CyberMetric during pretraining; this cannot be checked here.",
        ],
    }
    atomic_write_json(root / "dataset_metadata.json", {**dict(metadata), "preparation": data_manifest})
    write_jsonl(train_path, train)
    write_jsonl(val_path, valid)
    data_manifest["training_file_sha256"] = sha256_file(train_path)
    data_manifest["validation_file_sha256"] = sha256_file(val_path)
    data_manifest["preparation_fingerprint"] = preparation_context_fp
    atomic_write_json(manifest_path, data_manifest)
    return train, valid, data_manifest


def resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def inspect_sources(root: Path, config_path: Path) -> dict[str, Any]:
    output = root / "dataset_metadata.json"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "scripts" / "inspect_datasets.py"),
         "--config", str(config_path), "--output", str(output)],
        text=True,
    )
    if result.returncode != 0 or not output.exists():
        raise ExperimentError("Dataset inspection failed. See the command output and resolve the access or dependency issue before training.")
    return json.loads(output.read_text(encoding="utf-8"))


def chat_template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    return sha256_text(str(template))


def model_fingerprints(config: Mapping[str, Any], tokenizer: Any, model_revision: str,
                       eval_hash: str, validation_hash: str, train_hash: str,
                       max_length: int, profile: str, versions: Mapping[str, str]) -> dict[str, str]:
    common = {
        "model_id": config["model"]["id"], "model_revision": model_revision,
        "tokenizer_id": config["model"]["id"], "tokenizer_revision": model_revision,
        "quantization": {"4bit": "NF4", "double_quant": True, "compute_dtype": "float16", "attention": "sdpa"},
        "system_prompt": SYSTEM_PROMPT, "chat_template_sha256": chat_template_hash(tokenizer),
        "seed": int(config["seed"]), "versions": dict(versions),
    }
    benchmark = fingerprint({**common, "evaluation_hash": eval_hash,
        "prompt_template": "CyberMetric fixed prompt v1", "scoring": "answer boundary token validation; logits or full-sequence LL",
        "decoding": {"method": "greedy_next_token"}})
    validation = fingerprint({**common, "validation_hash": validation_hash, "max_seq_length": max_length,
        "masking": "assistant bodies plus each im_end; causal shift; fp32 token-weighted CE", "truncation": "right"})
    qualitative = fingerprint({**common, "prompts": list(QUALITATIVE_PROMPTS), "max_new_tokens": 300, "do_sample": False})
    experiment = fingerprint({**common, "evaluation_hash": eval_hash, "validation_hash": validation_hash,
        "training_hash": train_hash, "max_seq_length": max_length, "profile": profile,
        "scoring": "answer boundary token validation; logits or full-sequence LL", "decoding": "greedy, 300 tokens"})
    return {"benchmark": benchmark, "validation": validation, "qualitative": qualitative, "experiment": experiment}


def generate_qualitative(model: Any, tokenizer: Any, max_new_tokens: int = 300) -> list[dict[str, Any]]:
    import torch
    device = getattr(model, "device", None) or next(model.parameters()).device
    outputs = []
    model.eval()
    for index, prompt in enumerate(QUALITATIVE_PROMPTS, start=1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(device)
        input_ids = encoded["input_ids"]
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        text = tokenizer.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True)
        outputs.append({"prompt_id": index, "prompt": prompt, "response": text,
                        "truncated_by_token_cap": generated.shape[-1] - input_ids.shape[-1] >= max_new_tokens})
    return outputs


def pad_batch(features: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch
    width = max(len(row["input_ids"]) for row in features)
    inputs, masks, labels = [], [], []
    for row in features:
        n = len(row["input_ids"])
        inputs.append(list(row["input_ids"]) + [pad_token_id] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        labels.append(list(row["labels"]) + [-100] * (width - n))
    return {"input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)}


class TokenizedDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = list(rows)
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {key: row[key] for key in ("input_ids", "labels", "attention_mask")}


class FingerprintCheckpointCallback:
    def __init__(self, experiment_fingerprint: str):
        self.experiment_fingerprint = experiment_fingerprint
    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        from cyber_lora_utils import mark_checkpoint
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        has_weights = ((checkpoint / "adapter_model.safetensors").is_file()
                       or (checkpoint / "adapter_model.bin").is_file())
        if (checkpoint.exists() and (checkpoint / "adapter_config.json").is_file()
                and has_weights and (checkpoint / "trainer_state.json").is_file()):
            mark_checkpoint(checkpoint, self.experiment_fingerprint, complete=True)
        return control


def validate_target_modules(model: Any, target_modules: Sequence[str]) -> None:
    names = [name.split(".")[-1] for name, _ in model.named_modules()]
    missing = sorted(set(target_modules) - set(names))
    if missing:
        raise ExperimentError(f"Requested LoRA target modules are absent from the loaded model: {missing}")


def make_lora_model(base: Any, config: Mapping[str, Any]) -> Any:
    import inspect
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    tcfg = config["training"]
    prepare_sig = inspect.signature(prepare_model_for_kbit_training)
    prep_kwargs: dict[str, Any] = {"use_gradient_checkpointing": True}
    if "gradient_checkpointing_kwargs" in prepare_sig.parameters:
        prep_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    base = prepare_model_for_kbit_training(base, **{k: v for k, v in prep_kwargs.items() if k in prepare_sig.parameters})
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    validate_target_modules(base, tcfg["target_modules"])
    lora_config = LoraConfig(
        r=int(tcfg["lora_rank"]), lora_alpha=int(tcfg["lora_alpha"]),
        lora_dropout=float(tcfg["lora_dropout"]), bias="none",
        target_modules=list(tcfg["target_modules"]), task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ExperimentError("Only LoRA adapter parameters must be trainable; inspect the PEFT model before proceeding.")
    print(f"Trainable adapter parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def choose_optimizer(model: Any) -> tuple[Any, str]:
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(params, lr=1e-4, weight_decay=0.01)
        return optimizer, "bitsandbytes.PagedAdamW8bit"
    except Exception as exc:
        print(f"Paged AdamW 8-bit unavailable ({type(exc).__name__}); using torch.optim.AdamW.")
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01), "torch.optim.AdamW fallback"


def run_preflight(root: Path, config: Mapping[str, Any], tokenizer: Any, revision: str,
                  train_features: Sequence[Mapping[str, Any]], model_id: str, token: str | None,
                  experiment_fingerprint: str) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    preflight_path = root / "preflight_adapter"
    marker = root / "preflight.json"
    if marker.exists():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if (previous.get("fingerprint") == experiment_fingerprint and previous.get("passed") is True
                and (preflight_path / "adapter_config.json").exists()):
            return previous
    base = load_quantized_base(model_id, revision, token)
    model = make_lora_model(base, config)
    model.train()
    batch = pad_batch(train_features[:1], tokenizer.pad_token_id)
    device = getattr(model, "device", None) or next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer, optimizer_name = choose_optimizer(model)
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    output = model(**batch)
    loss = output.loss
    if not torch.isfinite(loss):
        raise ExperimentError("Preflight training loss is not finite.")
    loss.backward()
    trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not trainable_grads or any(not torch.isfinite(grad).all() for grad in trainable_grads):
        raise ExperimentError("Preflight adapter gradients are missing or non-finite.")
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(config["training"]["max_grad_norm"]))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-6)
    tokens = int(batch["attention_mask"].sum().item())
    tokens_per_second = tokens / elapsed
    if preflight_path.exists():
        shutil.rmtree(preflight_path)
    model.save_pretrained(preflight_path)
    base_for_reload = model.unload()
    del model
    reloaded = PeftModel.from_pretrained(base_for_reload, preflight_path, is_trainable=False)
    reloaded.eval()
    with torch.no_grad():
        reload_check = reloaded(**batch).loss
    if not torch.isfinite(reload_check):
        raise ExperimentError("Saved/reloaded preflight adapter produced a non-finite loss.")
    result = {"fingerprint": experiment_fingerprint, "passed": True,
              "pilot_loss": float(loss.detach().float().item()), "reloaded_loss": float(reload_check.float().item()),
              "pilot_tokens": tokens, "elapsed_seconds": elapsed, "tokens_per_second": tokens_per_second,
              "optimizer": optimizer_name, "adapter_reload": "passed"}
    atomic_write_json(marker, result)
    del reloaded, base_for_reload, base, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_model(root: Path, config: Mapping[str, Any], tokenizer: Any,
                train_features: Sequence[Mapping[str, Any]], monitor_features: Sequence[Mapping[str, Any]],
                model_id: str, revision: str,
                token: str | None, profile: str, run_fingerprint: str) -> tuple[Path, dict[str, Any]]:
    import inspect
    import torch
    from transformers import Trainer, TrainingArguments
    base = load_quantized_base(model_id, revision, token)
    model = make_lora_model(base, config)
    tcfg = config["training"]
    if profile == "T1":
        epochs = int(tcfg["epochs_t1"]); learning_rate = float(tcfg["learning_rate_t1"])
    else:
        epochs = int(tcfg["epochs_t2"]); learning_rate = float(tcfg["learning_rate_t2"])
    optimizer, optimizer_name = choose_optimizer(model)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["weight_decay"] = float(tcfg["weight_decay"])
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    args_values: dict[str, Any] = {
        "output_dir": str(root / "checkpoints"),
        "per_device_train_batch_size": int(tcfg["per_device_batch_size"]),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(tcfg["gradient_accumulation_steps"]),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": float(tcfg["warmup_ratio"]),
        "lr_scheduler_type": "cosine",
        "weight_decay": float(tcfg["weight_decay"]),
        "max_grad_norm": float(tcfg["max_grad_norm"]),
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "save_strategy": "epoch",
        eval_key: "epoch",
        "save_total_limit": 2,
        "logging_steps": 5,
        "logging_nan_inf_filter": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": True,
        "load_best_model_at_end": profile == "T2",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
    }
    if "gradient_checkpointing_kwargs" in sig:
        args_values["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    args = TrainingArguments(**args_values)
    pad_id = tokenizer.pad_token_id
    def collator(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return pad_batch(items, pad_id)
    from transformers import TrainerCallback
    class Callback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            has_weights = ((checkpoint / "adapter_model.safetensors").is_file()
                           or (checkpoint / "adapter_model.bin").is_file())
            if ((checkpoint / "adapter_config.json").is_file() and has_weights
                    and (checkpoint / "trainer_state.json").is_file()):
                from cyber_lora_utils import mark_checkpoint
                mark_checkpoint(checkpoint, run_fingerprint, complete=True)
            return control
    trainer = Trainer(
        model=model, args=args,
        train_dataset=TokenizedDataset(train_features),
        eval_dataset=TokenizedDataset(monitor_features),
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=[Callback()],
    )
    latest = latest_valid_checkpoint(root / "checkpoints", run_fingerprint)
    train_result = trainer.train(resume_from_checkpoint=str(latest) if latest else None)
    adapter_stage = root / f"final_staging_{run_fingerprint[:12]}"
    if adapter_stage.exists():
        shutil.rmtree(adapter_stage)
    loss_events: list[tuple[int, str, float]] = []
    for event in trainer.state.log_history:
        if "loss" in event:
            loss_events.append((int(event.get("step", 0)), "train_loss", float(event["loss"])))
        if "eval_loss" in event:
            loss_events.append((int(event.get("step", 0)), "eval_loss", float(event["eval_loss"])))
    divergences = [(step, name, value) for step, name, value in loss_events if not math.isfinite(value)]
    selected_checkpoint = None
    if divergences and profile == "T2":
        step, name, _ = divergences[0]
        raise ExperimentError(f"T2 training produced a non-finite {name} at step {step}; refusing to publish an adapter.")
    if divergences and profile == "T1":
        first_bad_step = min(step for step, _, _ in divergences)
        finite_eval_steps = {step for step, name, value in loss_events
                             if name == "eval_loss" and math.isfinite(value) and step < first_bad_step}
        checkpoint_candidates = sorted(
            (root / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
            reverse=True,
        )
        chosen = next((path for path in checkpoint_candidates
                       if path.name.rsplit("-", 1)[-1].isdigit()
                       and int(path.name.rsplit("-", 1)[-1]) in finite_eval_steps
                       and valid_checkpoint(path, run_fingerprint)), None)
        if chosen is None:
            raise ExperimentError("T1 loss diverged and no complete fingerprint-matched checkpoint has a finite in-training validation loss.")
        adapter_stage.mkdir(parents=True)
        copied = []
        for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            source = chosen / filename
            if source.is_file():
                shutil.copy2(source, adapter_stage / filename)
                copied.append(filename)
        if not {"adapter_config.json"}.issubset(copied) or not ({"adapter_model.safetensors", "adapter_model.bin"} & set(copied)):
            raise ExperimentError(f"Finite checkpoint {chosen.name} is missing adapter artifacts.")
        selected_checkpoint = chosen.name
    else:
        trainer.model.save_pretrained(adapter_stage)
    trainer.model.config.use_cache = True
    divergence_info = ({"step": divergences[0][0], "metric": divergences[0][1]} if divergences else None)
    atomic_write_json(adapter_stage / "run_fingerprint.json", {
        "fingerprint": run_fingerprint, "complete": True, "loss_diverged": bool(divergences),
        "selected_checkpoint": selected_checkpoint, "divergence": divergence_info,
    })
    final_dir = root / "final"
    if final_dir.exists():
        existing = final_dir / "run_fingerprint.json"
        if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("fingerprint") == run_fingerprint:
            shutil.rmtree(adapter_stage)
        else:
            raise ExperimentError("A final adapter already exists for a different fingerprint; use a new output root.")
    else:
        os.replace(adapter_stage, final_dir)
    metrics = {"train_result": train_result.metrics, "optimizer": optimizer_name,
               "epochs": epochs, "learning_rate": learning_rate, "global_step": int(trainer.state.global_step),
               "loss_diverged": bool(divergences), "selected_checkpoint": selected_checkpoint,
               "divergence": divergence_info}
    del trainer, model, base, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return final_dir, metrics


def get_model_info(model_id: str, pinned_revision: str) -> tuple[str, str | None]:
    from huggingface_hub import HfApi
    try:
        info = HfApi(token=resolve_hf_token()).model_info(model_id, revision=pinned_revision, files_metadata=False)
    except Exception as exc:
        raise ExperimentError(f"Could not load pinned model revision {pinned_revision}: {type(exc).__name__}: {exc}") from exc
    return str(info.sha), getattr(getattr(info, "cardData", None), "license", None)


def ensure_baselines(
    root: Path, config: Mapping[str, Any], tokenizer: Any, revision: str, token: str | None,
    benchmark: Sequence[Mapping[str, Any]], eval_hash: str, eval_revision: str,
    validation_records: Sequence[Mapping[str, Any]], validation_features: Sequence[Mapping[str, Any]],
    max_length: int, profile: str, train_hash: str, hardware: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_dir = root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    versions = package_versions()
    validation_hash = sha256_file(root / "data" / "frozen_validation.jsonl")
    fps = model_fingerprints(config, tokenizer, revision, eval_hash, validation_hash, train_hash, max_length, profile, versions)
    benchmark_path = baseline_dir / "cybermetric_baseline.json"
    loss_path = baseline_dir / f"validation_loss_{max_length}.json"
    qualitative_path = baseline_dir / "qualitative_baseline.json"
    benchmark_artifact = validate_baseline_artifact(benchmark_path, fps["benchmark"])
    loss_artifact = validate_baseline_artifact(loss_path, fps["validation"])
    qualitative_artifact = validate_baseline_artifact(qualitative_path, fps["qualitative"])
    need_model = not (benchmark_artifact and loss_artifact and qualitative_artifact)
    if need_model:
        model = load_quantized_base(config["model"]["id"], revision, token)
        tokenizer_validation = validate_answer_tokens(tokenizer, tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": benchmark_prompt(benchmark[0])[0]}],
            tokenize=False, add_generation_prompt=True,
        ))
        # Verify the requested supervised-token decode contract before training.
        checked = False
        for record in validation_records:
            feature = tokenize_supervised_example(tokenizer, record["messages"], max_length=max_length)
            if not feature["truncated"]:
                assert_supervised_decode(tokenizer, record["messages"], feature)
                checked = True
                break
        if not checked:
            raise ExperimentError("No untruncated validation conversation is available for the required supervised-token decode assertion.")
        if not benchmark_artifact:
            predictions, metrics = score_benchmark(model, tokenizer, benchmark, include_rotations=True, batch_size=2)
            artifact = {"complete": True, "fingerprint": fps["benchmark"], "evaluation_hash": eval_hash,
                        "evaluation_revision": eval_revision, "metrics": metrics, "predictions": predictions,
                        "tokenizer_validation": tokenizer_validation}
            atomic_write_json(benchmark_path, artifact)
            benchmark_artifact = artifact
            atomic_write_json(root / "baseline_predictions.json", artifact)
        if not loss_artifact:
            loss = manual_validation_loss(model, tokenizer, validation_features, batch_size=int(config["evaluation"]["validation_batch_size"]))
            loss.update({"complete": True, "fingerprint": fps["validation"],
                         "validation_file_sha256": validation_hash, "max_sequence_length": max_length})
            atomic_write_json(loss_path, loss)
            loss_artifact = loss
        if not qualitative_artifact:
            qualitative = generate_qualitative(model, tokenizer, int(config["evaluation"]["qualitative_max_new_tokens"]))
            artifact = {"complete": True, "fingerprint": fps["qualitative"], "samples": qualitative}
            atomic_write_json(qualitative_path, artifact)
            qualitative_artifact = artifact
        baseline_model_meta = {
            "complete": True, "model_id": config["model"]["id"], "model_revision": revision,
            "quantization": "4-bit NF4 with double quantization; FP16 compute; SDPA",
            "tokenizer_revision": revision, "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
            "evaluation_hash": eval_hash, "evaluation_revision": eval_revision,
            "validation_file_sha256": validation_hash, "hardware": dict(hardware),
            "library_versions": versions, "sub_fingerprints": fps,
        }
        atomic_write_json(baseline_dir / "model_metadata.json", baseline_model_meta)
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    # Keep the baseline artifact name required by the handoff.
    if benchmark_artifact and not (root / "baseline_predictions.json").exists():
        atomic_write_json(root / "baseline_predictions.json", benchmark_artifact)
    if qualitative_artifact:
        sample_path = root / "samples_before_after.json"
        if not sample_path.exists():
            atomic_write_json(sample_path, {"before": qualitative_artifact["samples"], "after": None,
                                           "manual_blind_rubric_review": "pending"})
    return {"fingerprints": fps, "benchmark": benchmark_artifact, "validation": loss_artifact,
            "qualitative": qualitative_artifact}


def tokenize_records(tokenizer: Any, records: Sequence[Mapping[str, Any]], max_length: int) -> list[dict[str, Any]]:
    return [tokenize_supervised_example(tokenizer, row["messages"], max_length=max_length) for row in records]


def select_training_size(root: Path, config: Mapping[str, Any], profile: str,
                         train_records: Sequence[Mapping[str, Any]], train_features: Sequence[Mapping[str, Any]],
                         pilot: Mapping[str, Any], max_length: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data_dir = root / "data"
    pool_hash = sha256_text(canonical_json(train_records))
    path = data_dir / f"final_train_{max_length}.jsonl"
    manifest_path = data_dir / f"selection_{max_length}.json"
    selection_fp = fingerprint({"pool_hash": pool_hash, "profile": profile, "seq": max_length,
                                "seed": int(config["seed"]), "epochs": config["training"]["epochs_t1" if profile == "T1" else "epochs_t2"],
                                "versions": package_versions()})
    if path.exists() or manifest_path.exists():
        if not (path.exists() and manifest_path.exists()):
            raise ExperimentError("Incomplete final training selection found; use a fresh output directory.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("selection_fingerprint") != selection_fp or sha256_file(path) != manifest.get("training_file_sha256"):
            raise ExperimentError("Final training selection differs from its frozen fingerprint; use a new output directory.")
        selected_records = read_jsonl(path)
        selected_ids = {row["source_id"] for row in selected_records}
        selected_features = [feature for row, feature in zip(train_records, train_features) if row["source_id"] in selected_ids]
        return selected_records, selected_features, manifest
    hours = float(config["training"]["training_budget_hours"])
    safe_seconds = hours * 3600 * (1 - float(config["training"]["safety_margin"]))
    tps = max(float(pilot["tokens_per_second"]), 0.1)
    epochs = int(config["training"]["epochs_t1"] if profile == "T1" else config["training"]["epochs_t2"])
    budget_tokens = tps * safe_seconds
    order = list(range(len(train_records)))
    random.Random(int(config["seed"])).shuffle(order)
    chosen_indices: list[int] = []
    projected = 0
    for index in order:
        cost = max(1, sum(train_features[index]["attention_mask"])) * epochs
        if projected + cost > budget_tokens and chosen_indices:
            continue
        if projected + cost > budget_tokens:
            continue
        chosen_indices.append(index)
        projected += cost
    minimum = int(config["training"]["min_training_records"])
    if len(chosen_indices) < min(minimum, len(train_records)):
        raise ExperimentError(
            f"Pilot throughput projects only {len(chosen_indices)} records within the configured budget. "
            "Increase training_budget_hours or use a new run directory with a justified smaller target."
        )
    # If everything fits, preserve the full deterministic candidate set rather than randomizing order.
    chosen_indices.sort()
    selected_records = [dict(train_records[i]) for i in chosen_indices]
    selected_features = [train_features[i] for i in chosen_indices]
    write_jsonl(path, selected_records)
    manifest = {
        "complete": True, "selection_fingerprint": selection_fp, "training_file_sha256": sha256_file(path),
        "pool_records": len(train_records), "selected_records": len(selected_records),
        "pilot_tokens_per_second": tps, "estimated_training_tokens": projected,
        "estimated_seconds": projected / tps, "configured_budget_seconds": hours * 3600,
        "safety_margin": float(config["training"]["safety_margin"]),
        "reason": "deterministic throughput-based estimate made after the pilot and before the full training run",
    }
    atomic_write_json(manifest_path, manifest)
    return selected_records, selected_features, manifest


def post_training_evaluation(
    root: Path, config: Mapping[str, Any], tokenizer: Any, revision: str, token: str | None,
    benchmark: Sequence[Mapping[str, Any]], eval_hash: str, eval_revision: str,
    validation_features: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any],
    final_dir: Path, profile: str, run_fingerprint: str,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    base = load_quantized_base(config["model"]["id"], revision, token)
    model = PeftModel.from_pretrained(base, final_dir, is_trainable=False)
    model.eval()
    with model.disable_adapter():
        adapter_off = manual_validation_loss(model, tokenizer, validation_features, batch_size=1)
    baseline_loss = float(baseline["validation"]["loss"])
    tolerance = float(config["training"]["baseline_adapter_loss_tolerance"])
    drift = abs(adapter_off["loss"] - baseline_loss)
    if drift > tolerance:
        raise ExperimentError(f"Adapter-disabled validation loss drift {drift:.6g} exceeds tolerance {tolerance}.")
    after_loss = manual_validation_loss(model, tokenizer, validation_features, batch_size=1)
    before_count = int(baseline["validation"]["valid_target_tokens"])
    if after_loss["valid_target_tokens"] != before_count:
        raise ExperimentError("Valid target-token count changed between baseline and fine-tuned evaluation.")
    after_predictions, after_metrics = score_benchmark(model, tokenizer, benchmark, include_rotations=True, batch_size=2)
    if after_metrics.get("scoring_strategy") != baseline["benchmark"]["metrics"].get("scoring_strategy"):
        raise ExperimentError("CyberMetric answer-token scoring strategy changed after training.")
    after_artifact = {"complete": True, "fingerprint": run_fingerprint, "evaluation_hash": eval_hash,
                      "evaluation_revision": eval_revision, "metrics": after_metrics, "predictions": after_predictions}
    atomic_write_json(root / "finetuned_predictions.json", after_artifact)
    after_qualitative = generate_qualitative(model, tokenizer, int(config["evaluation"]["qualitative_max_new_tokens"]))
    samples_path = root / "samples_before_after.json"
    prior_samples = json.loads(samples_path.read_text(encoding="utf-8"))
    atomic_write_json(samples_path, {"before": prior_samples["before"], "after": after_qualitative,
                                    "manual_blind_rubric_review": "pending; responses must be shuffled and rated using BRIEF.md §10"})
    before_predictions = baseline["benchmark"]["predictions"]
    assert_same_evaluation_hash({"evaluation_hash": eval_hash}, {"evaluation_hash": after_artifact["evaluation_hash"]})
    comparison = compare_predictions(before_predictions, after_predictions)
    baseline_full_loss = float(baseline["validation"]["loss"])
    results = {
        "complete": False,
        "experiment_fingerprint": run_fingerprint,
        "profile": profile,
        "model": {"id": config["model"]["id"], "revision": revision, "quantization": "4-bit NF4; double quantization; FP16 compute; SDPA"},
        "evaluation": {"dataset_id": config["datasets"]["benchmark_id"], "config": config["datasets"]["benchmark_config"],
                       "split": config["datasets"]["benchmark_split"], "revision": eval_revision, "sha256": eval_hash},
        "cybermetric_comparison": comparison,
        "validation_loss": {"before": baseline_full_loss, "after": after_loss["loss"],
                            "absolute_difference": after_loss["loss"] - baseline_full_loss,
                            "relative_difference": (after_loss["loss"] / baseline_full_loss - 1) if baseline_full_loss else None,
                            "valid_target_tokens": before_count, "example_count": after_loss["example_count"],
                            "max_length": after_loss["max_length"], "truncated_examples": after_loss["truncated_examples"],
                            "validation_file_sha256": sha256_file(root / "data" / "frozen_validation.jsonl"),
                            "adapter_disabled_loss": adapter_off["loss"], "adapter_disabled_absolute_drift": drift,
                            "adapter_disabled_tolerance": tolerance},
        "position_bias": {"before_mean_accuracy": baseline["benchmark"]["metrics"].get("cyclic_rotation_mean_accuracy"),
                           "after_mean_accuracy": after_metrics.get("cyclic_rotation_mean_accuracy"),
                           "before_rotations": baseline["benchmark"]["metrics"].get("cyclic_rotation_accuracy"),
                           "after_rotations": after_metrics.get("cyclic_rotation_accuracy")},
        "qualitative_review": {"status": "pending_manual_blind_review", "prompt_count": len(QUALITATIVE_PROMPTS)},
        "limitations": ["Exact and n-gram matching cannot detect paraphrase or semantic overlap.",
                        "CyberMetric derives from public sources that may also occur in training data.",
                        "Base model pretraining contamination cannot be tested here.",
                        "The qualitative set is descriptive and supports no statistical inference."],
    }
    atomic_write_json(root / "results.json", results)
    atomic_write_json(root / "post_training" / "metrics.json", {"validation_loss": after_loss, "benchmark": after_metrics})
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    return results


def execute(config_path: Path, *, output_override: str | None = None, preflight_only: bool = False) -> dict[str, Any]:
    import torch
    config = read_config(config_path)
    if output_override:
        config["output_dir"] = output_override
    config["evaluation"]["system_prompt"] = SYSTEM_PROMPT
    root = output_root_from_config(config)
    token = resolve_hf_token()
    hardware = verify_gpu()
    seed_everything(int(config["seed"]))
    dataset_meta = inspect_sources(root, config_path)
    benchmark, evaluation_hash, evaluation_revision = load_benchmark_rows(config, dataset_meta, token)
    train_records, validation_records, data_manifest = prepare_frozen_data(
        root, config, dataset_meta, benchmark, evaluation_hash, token)
    atomic_write_json(root / "dataset_metadata.json", {**dict(dataset_meta), "preparation": data_manifest})
    profile = data_manifest["profile"]
    revision, model_license = get_model_info(config["model"]["id"], config["model"]["revision"])
    if revision != config["model"]["revision"]:
        raise ExperimentError("The model revision resolved to a different SHA than the pinned configuration.")
    tokenizer = load_tokenizer(config["model"]["id"], revision, token)
    max_length = int(config["training"]["max_seq_length"])
    val_features = tokenize_records(tokenizer, validation_records, max_length)
    # Tokenize the pool to estimate the time budget after a finite pilot.
    train_features = tokenize_records(tokenizer, train_records, max_length)
    versions = package_versions()
    baseline_train_hash = data_manifest["training_file_sha256"]
    baseline = ensure_baselines(root, config, tokenizer, revision, token, benchmark, evaluation_hash,
                                evaluation_revision, validation_records, val_features, max_length,
                                profile, baseline_train_hash, hardware)
    # LoRA configuration and pilot are deliberately constructed only after all baseline artifacts are saved.
    validation_decode_check = True
    while True:
        tcfg = dict(config["training"])
        tcfg["max_seq_length"] = max_length
        local_config = dict(config); local_config["training"] = tcfg
        fps = model_fingerprints(config, tokenizer, revision, evaluation_hash,
            sha256_file(root / "data" / "frozen_validation.jsonl"), baseline_train_hash,
            max_length, profile, versions)
        preflight_fp = fingerprint({"baseline_subfingerprints": baseline["fingerprints"], "max_seq_length": max_length,
                                    "training_file_sha256": baseline_train_hash, "versions": versions})
        try:
            pilot = run_preflight(root, local_config, tokenizer, revision, train_features,
                                  config["model"]["id"], token, preflight_fp)
            break
        except torch.cuda.OutOfMemoryError:
            if max_length != int(config["training"]["max_seq_length"]):
                raise ExperimentError("The configured 512-token OOM fallback also ran out of memory.")
            fallback = int(config["training"]["oom_fallback_seq_length"])
            print(f"Pilot OOM at {max_length}; restarting baseline and preflight at {fallback} tokens.")
            import torch
            gc.collect(); torch.cuda.empty_cache()
            max_length = fallback
            val_features = tokenize_records(tokenizer, validation_records, max_length)
            train_features = tokenize_records(tokenizer, train_records, max_length)
            baseline = ensure_baselines(root, config, tokenizer, revision, token, benchmark, evaluation_hash,
                                        evaluation_revision, validation_records, val_features, max_length,
                                        profile, baseline_train_hash, hardware)
    del local_config, tcfg
    selected_records, selected_features, selection_manifest = select_training_size(
        root, config, profile, train_records, train_features, pilot, max_length)
    if preflight_only:
        atomic_write_json(root / "preflight_summary.json", {
            "passed": True, "profile": profile, "model_revision": revision,
            "benchmark_hash": evaluation_hash, "validation_hash": sha256_file(root / "data" / "frozen_validation.jsonl"),
            "selected_training_records": len(selected_records), "pilot": pilot,
            "training_selection": selection_manifest, "note": "Full QLoRA training has not started.",
        })
        print("Preflight passed. Baselines are saved; the full training run has not started.")
        return {"preflight_passed": True}
    final_train_hash = sha256_file(root / "data" / f"final_train_{max_length}.jsonl")
    run_fps = model_fingerprints(config, tokenizer, revision, evaluation_hash,
        sha256_file(root / "data" / "frozen_validation.jsonl"), final_train_hash,
        max_length, profile, versions)
    run_fingerprint = run_fps["experiment"]
    monitor_features = val_features[:min(len(val_features), int(config["training"]["validation_monitor_examples"]))]
    if not monitor_features:
        raise ExperimentError("Validation monitor subset is empty.")
    if not selected_features:
        raise ExperimentError("Training selection is empty.")
    # Update callback fingerprint used for new checkpoint markers. A different pool hash
    # cannot resume; checkpoint recovery validates this exact run fingerprint.
    adapter_dir, train_metrics = train_model(root, config, tokenizer, selected_features, monitor_features,
        config["model"]["id"], revision, token, profile, run_fingerprint)
    results = post_training_evaluation(root, config, tokenizer, revision, token, benchmark, evaluation_hash,
        evaluation_revision, val_features, baseline, adapter_dir, profile, run_fingerprint)
    experiment_meta = {
        "complete": True, "experiment_fingerprint": run_fingerprint,
        "model_id": config["model"]["id"], "model_revision": revision, "model_license": model_license or "UNVERIFIED",
        "tokenizer_revision": revision, "quantization": "4-bit NF4; double quantization; FP16 compute; SDPA",
        "dataset_manifest": data_manifest, "evaluation_hash": evaluation_hash,
        "validation_hash": sha256_file(root / "data" / "frozen_validation.jsonl"),
        "training_hash": final_train_hash, "training_selection": selection_manifest,
        "max_sequence_length": max_length, "seed": int(config["seed"]),
        "lora": {key: config["training"][key] for key in ("lora_rank", "lora_alpha", "lora_dropout", "target_modules")},
        "library_versions": versions, "hardware": hardware, "pilot": pilot, "training": train_metrics,
        "sub_fingerprints": run_fps,
    }
    atomic_write_json(root / "experiment_metadata.json", experiment_meta)
    result_path = root / "results.json"
    results["complete"] = True
    results["experiment_fingerprint"] = run_fingerprint
    atomic_write_json(result_path, results)
    required = [root / "final" / "adapter_config.json", root / "results.json", root / "samples_before_after.json",
                root / "baseline_predictions.json", root / "finetuned_predictions.json", root / "experiment_metadata.json",
                root / "dataset_metadata.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ExperimentError(f"Run artifacts are incomplete: {missing}")
    atomic_write_json(root / "COMPLETE.json", {"complete": True, "experiment_fingerprint": run_fingerprint,
                                                "required_artifacts": [str(p.relative_to(root)) for p in required]})
    print(f"Experiment completed. Results: {result_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "experiment.yaml"))
    parser.add_argument("--output-dir", help="Override CYBER_LORA_OUTPUT_DIR/config output_dir")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect sources and write dataset_metadata.json, then stop")
    parser.add_argument("--preflight-only", action="store_true", help="Save baselines, run one-step adapter pilot/reload, and stop before full training")
    args = parser.parse_args()
    try:
        if args.inspect_only:
            config = read_config(args.config)
            root = output_root_from_config({**config, "output_dir": args.output_dir or config.get("output_dir")})
            inspect_sources(root, Path(args.config).resolve())
            return 0
        execute(Path(args.config).resolve(), output_override=args.output_dir, preflight_only=args.preflight_only)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
