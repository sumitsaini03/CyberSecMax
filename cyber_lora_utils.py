"""Shared, deterministic utilities for the Cybersecurity QLoRA lab.

Heavy ML dependencies are imported only by the functions that need them so the
data and statistics helpers remain testable on a CPU-only machine.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SYSTEM_PROMPT = "You are a careful cybersecurity assistant. Answer accurately and concisely."
ANSWER_LABELS = ("A", "B", "C", "D")
QUALITATIVE_PROMPTS = (
    "Explain MITRE ATT&CK technique T1059.001 and give three detection ideas.",
    "Analyze repeated Windows event IDs 4625 followed by event ID 4624, including likely interpretation and defensive response.",
    "Write a Splunk SPL query that flags more than ten failed logins per user within five minutes.",
    "Explain the risks and remediation of an S3 bucket declared with `acl = public-read` in Terraform.",
    "Explain the CVSS v3.1 base metrics.",
)
CYBERMETRIC_USER_TEMPLATE = (
    "Question: {question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
    "Answer with a single letter (A, B, C, or D)."
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def atomic_write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_match_text(text: str, strip_punctuation: bool = False) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if strip_punctuation:
        normalized = "".join(ch for ch in normalized if not unicodedata.category(ch).startswith("P"))
        normalized = " ".join(normalized.split())
    return normalized


def validate_messages(messages: Any, *, require_system: bool = False) -> list[dict[str, str]]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} must be a mapping")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message {index} has unsupported role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be a non-empty string")
        normalized.append({"role": str(role), "content": content})
    roles = [m["role"] for m in normalized]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("conversation must contain at least one user and assistant message")
    if roles[-1] != "assistant":
        raise ValueError("conversation must end with an assistant message")
    if require_system and roles[0] != "system":
        raise ValueError("conversation must begin with the fixed system message")
    return normalized


def normalize_training_record(
    row: Mapping[str, Any], source: str, source_id: str | None = None,
) -> dict[str, Any]:
    raw_messages = row.get("messages")
    if raw_messages is None and isinstance(row.get("conversations"), list):
        raw_messages = row["conversations"]
    if not isinstance(raw_messages, (list, tuple)):
        raise ValueError("record has no messages or conversations list")
    messages = validate_messages(raw_messages)
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    messages = validate_messages(messages, require_system=True)
    row_id = source_id or row.get("source_id") or row.get("id") or row.get("prompt_id")
    if row_id is None:
        row_id = row.get("conv_title") or row.get("title")
    if row_id is None:
        # Stable fallback retains deterministic behavior but cannot provide group-aware splits.
        row_id = sha256_text(canonical_json(messages))
        group_id = None
    else:
        row_id = str(row_id)
        raw_group = row.get("source_group_id") or row.get("seed_data") or row_id
        group_value = sha256_text(str(raw_group)) if row.get("seed_data") else str(raw_group)
        group_id = f"{source}:{group_value}"
    return {
        "messages": messages,
        "source": str(source),
        "source_id": str(row_id),
        "source_group_id": group_id,
    }


def normalized_conversation_key(record: Mapping[str, Any]) -> str:
    messages = record["messages"]
    return normalize_match_text("\n".join(m["content"] for m in messages))


def deduplicate_records(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        item = dict(record)
        key = sha256_text(normalized_conversation_key(item))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(item)
    return kept, removed


def normalize_benchmark_row(row: Mapping[str, Any], row_index: int | None = None) -> dict[str, Any]:
    question = row.get("question")
    answers = row.get("answers")
    solution = row.get("solution")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"benchmark row {row_index}: question must be a non-empty string")
    if isinstance(answers, Mapping):
        normalized_answers = {str(k).strip().upper(): v for k, v in answers.items()}
    elif isinstance(answers, (list, tuple)) and len(answers) == 4:
        normalized_answers = dict(zip(ANSWER_LABELS, answers))
    else:
        raise ValueError(f"benchmark row {row_index}: answers must be a mapping or four-item list")
    if set(normalized_answers) != set(ANSWER_LABELS):
        raise ValueError(f"benchmark row {row_index}: answers must have exactly A, B, C and D")
    if any(not isinstance(normalized_answers[k], str) or not normalized_answers[k].strip() for k in ANSWER_LABELS):
        raise ValueError(f"benchmark row {row_index}: each answer must be a non-empty string")
    if not isinstance(solution, str) or solution.strip().upper() not in ANSWER_LABELS:
        raise ValueError(f"benchmark row {row_index}: solution must be one of A, B, C or D")
    return {"question": question, "answers": normalized_answers, "solution": solution.strip().upper()}


def _word_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i:i+n]) for i in range(max(0, len(words) - n + 1))}


def check_contamination(
    records: Iterable[Mapping[str, Any]], benchmark_rows: Sequence[Mapping[str, Any]],
    *, use_ngram: bool = True, jaccard_threshold: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Remove records matching benchmark question, question+key, or high 5-gram overlap.

    Records are visited one at a time. The benchmark-derived lookup is small; corpus
    text is never concatenated into one document.
    """
    questions: list[str] = []
    question_answer: list[tuple[str, str, str]] = []
    question_grams: list[set[tuple[str, ...]]] = []
    for i, raw in enumerate(benchmark_rows):
        row = normalize_benchmark_row(raw, i)
        q = normalize_match_text(row["question"])
        q_punct = normalize_match_text(row["question"], strip_punctuation=True)
        answer = normalize_match_text(row["answers"][row["solution"]])
        answer_punct = normalize_match_text(row["answers"][row["solution"]], strip_punctuation=True)
        questions.extend([q, q_punct])
        words = q.split()
        words_p = q_punct.split()
        # Removing one terminal token defines a conservative stem that still
        # retains the distinguishing wording of a complete question.
        stem = " ".join(words[:-1]) if len(words) >= 6 else q
        stem_p = " ".join(words_p[:-1]) if len(words_p) >= 6 else q_punct
        question_answer.extend([(stem, answer, False), (stem_p, answer_punct, True)])
        question_grams.append(_word_ngrams(q))
    questions = [q for q in questions if len(q) >= 8]
    exact_set = set(questions)
    keep: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    counts = {"level1_question": 0, "level2_question_and_correct_answer": 0, "level3_5gram_jaccard": 0}
    for record in records:
        item = dict(record)
        text = normalize_match_text("\n".join(m["content"] for m in item["messages"]))
        text_punct = normalize_match_text("\n".join(m["content"] for m in item["messages"]), strip_punctuation=True)
        matched_level: str | None = None
        if any(q in text for q in exact_set if len(q) >= 8) or any(q in text_punct for q in exact_set):
            matched_level = "level1_question"
        else:
            for q, answer, punctuation_stripped in question_answer:
                source_text = text_punct if punctuation_stripped else text
                if len(q) >= 8 and q in source_text and answer and answer in source_text:
                    matched_level = "level2_question_and_correct_answer"
                    break
        if matched_level is None and use_ngram:
            grams = _word_ngrams(text)
            for qgrams in question_grams:
                if not qgrams:
                    continue
                union = len(grams | qgrams)
                if union and len(grams & qgrams) / union >= jaccard_threshold:
                    matched_level = "level3_5gram_jaccard"
                    break
        if matched_level:
            counts[matched_level] += 1
            removed.append(item)
        else:
            keep.append(item)
    counts["benchmark_questions_examined"] = len(benchmark_rows)
    counts["training_records_examined"] = len(keep) + len(removed)
    counts["training_records_removed"] = len(removed)
    counts["final_training_records"] = len(keep)
    return keep, removed, counts


def split_records(
    records: Sequence[Mapping[str, Any]], *, seed: int = 42, validation_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if not records:
        raise ValueError("cannot split an empty dataset")
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        group_id = record.get("source_group_id")
        if group_id:
            groups.setdefault(str(group_id), []).append(index)
    if groups and sum(len(indices) for indices in groups.values()) == len(records):
        unique_ids = list(groups)
        rng.shuffle(unique_ids)
        target = max(1, round(len(records) * validation_fraction)) if len(records) > 1 else 0
        val_ids: set[str] = set()
        val_size = 0
        for group_id in unique_ids:
            if val_size >= target:
                break
            val_ids.add(group_id)
            val_size += len(groups[group_id])
        train = [dict(r) for r in records if str(r.get("source_group_id")) not in val_ids]
        valid = [dict(r) for r in records if str(r.get("source_group_id")) in val_ids]
        return train, valid, "source_id"
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = max(1, round(len(records) * validation_fraction)) if len(records) > 1 else 0
    val_indices = set(indices[:n_val])
    train = [dict(r) for i, r in enumerate(records) if i not in val_indices]
    valid = [dict(r) for i, r in enumerate(records) if i in val_indices]
    return train, valid, "random_record"


def render_chat(tokenizer: Any, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt,
    )


def tokenize_supervised_example(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], *, max_length: int = 1024,
) -> dict[str, Any]:
    """Tokenize chat once and supervise assistant body plus each assistant <|im_end|>.

    Prefixes are not necessarily token-prefixes for BPE tokenizers: tokenizing a
    string by itself can produce different IDs than tokenizing it as part of a
    longer string. Use character offsets from the full fast-tokenizer result so
    the loss mask follows the actual input IDs.
    """
    messages = validate_messages(messages, require_system=True)
    rendered = render_chat(tokenizer, messages)
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
    except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError("fast tokenizer offset mappings are required for safe assistant loss masking") from exc
    if len(ids) != len(offsets):
        raise ValueError("tokenizer returned different counts of token IDs and character offsets")
    im_start = getattr(tokenizer, "im_start_token", "<|im_start|>")
    im_end = getattr(tokenizer, "im_end_token", "<|im_end|>")
    search_from = 0
    assistant_spans: list[tuple[int, int]] = []
    for message in messages:
        header = f"{im_start}{message['role']}\n"
        header_at = rendered.find(header, search_from)
        if header_at < 0:
            raise ValueError("chat template output could not be aligned to message headers")
        content_start = header_at + len(header)
        content_end = content_start + len(message["content"])
        if rendered[content_start:content_end] != message["content"]:
            raise ValueError("chat template altered message text; refusing unsafe loss-mask alignment")
        end_at = content_end
        if not rendered.startswith(im_end, end_at):
            raise ValueError("chat template did not place the expected end marker after message content")
        end_marker_end = end_at + len(im_end)
        if message["role"] == "assistant":
            assistant_spans.append((content_start, end_marker_end))
        search_from = end_marker_end
    labels = [-100] * len(ids)
    for token_index, (token_start, token_end) in enumerate(offsets):
        token_start, token_end = int(token_start), int(token_end)
        if token_start == token_end:
            continue
        if token_start < 0 or token_end < token_start or token_end > len(rendered):
            raise ValueError("tokenizer returned an invalid character offset")
        overlaps = [span for span in assistant_spans
                    if token_start < span[1] and token_end > span[0]]
        if not overlaps:
            continue
        if len(overlaps) != 1 or token_start < overlaps[0][0] or token_end > overlaps[0][1]:
            raise ValueError(
                "tokenizer token crosses an assistant loss-mask boundary; refusing to guess "
                f"(character offsets {token_start}:{token_end})"
            )
        labels[token_index] = ids[token_index]
    if max_length < 1:
        raise ValueError("max_length must be positive")
    truncated = len(ids) > max_length
    ids, labels = ids[:max_length], labels[:max_length]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids), "truncated": truncated}


def assert_supervised_decode(tokenizer: Any, messages: Sequence[Mapping[str, str]], feature: Mapping[str, Any]) -> None:
    """Check the assistant-token contract on each complete, untruncated turn."""
    if feature.get("truncated"):
        raise ValueError("decode assertion requires an untruncated sample")
    im_end = getattr(tokenizer, "im_end_token", "<|im_end|>")
    ids, labels = feature["input_ids"], feature["labels"]
    # Re-align in order: each assistant content followed by one end marker.
    supervised = [token for token, label in zip(ids, labels) if label != -100]
    decoded = tokenizer.decode(supervised, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    expected = "".join(m["content"] + im_end for m in messages if m["role"] == "assistant")
    if decoded != expected:
        raise AssertionError(f"supervised decode mismatch: expected {expected!r}, got {decoded!r}")


def benchmark_prompt(row: Mapping[str, Any], *, rotation: int = 0) -> tuple[str, str]:
    row = normalize_benchmark_row(row)
    r = rotation % 4
    labels = ANSWER_LABELS
    rotated_answers = {labels[i]: row["answers"][labels[(i + r) % 4]] for i in range(4)}
    original_index = labels.index(row["solution"])
    rotated_solution = labels[(original_index - r) % 4]
    return CYBERMETRIC_USER_TEMPLATE.format(question=row["question"], **rotated_answers), rotated_solution


def validate_answer_tokens(tokenizer: Any, prompt_text: str) -> dict[str, Any]:
    base_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    candidate_ids: dict[str, int] = {}
    single = True
    for label in ANSWER_LABELS:
        candidate = list(tokenizer.encode(prompt_text + label, add_special_tokens=False))
        if len(candidate) != len(base_ids) + 1 or candidate[:-1] != base_ids:
            single = False
            continue
        candidate_ids[label] = int(candidate[-1])
        decoded = tokenizer.decode([candidate[-1]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != label:
            single = False
    if single and (len(set(candidate_ids.values())) != 4 or set(candidate_ids) != set(ANSWER_LABELS)):
        single = False
    return {"strategy": "single_token" if single else "full_sequence_loglikelihood", "candidate_ids": candidate_ids if single else {}}


def score_benchmark_row(
    model: Any, tokenizer: Any, row: Mapping[str, Any], *, row_index: int,
    rotation: int = 0, scoring: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    normalized = normalize_benchmark_row(row, row_index)
    user_text, solution = benchmark_prompt(normalized, rotation=rotation)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    prompt_text = render_chat(tokenizer, messages, add_generation_prompt=True)
    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    strategy = dict(scoring or validate_answer_tokens(tokenizer, prompt_text))
    device = getattr(model, "device", None) or next(model.parameters()).device
    with torch.no_grad():
        if strategy["strategy"] == "single_token":
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[0, -1].float()
            candidate_ids = strategy["candidate_ids"]
            scores = {label: float(logits[int(candidate_ids[label])].item()) for label in ANSWER_LABELS}
        else:
            scores = {}
            for label in ANSWER_LABELS:
                full_ids = list(tokenizer.encode(prompt_text + label, add_special_tokens=False))
                common = 0
                for left, right in zip(prompt_ids, full_ids):
                    if left != right:
                        break
                    common += 1
                if common == 0 or common == len(full_ids):
                    raise ValueError(f"benchmark row {row_index}: cannot align full-sequence label {label}")
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                logits = model(input_ids=input_ids).logits[0].float()
                target = input_ids[0, common:]
                log_probs = torch.log_softmax(logits[common - 1:len(full_ids) - 1], dim=-1)
                scores[label] = float(log_probs.gather(-1, target.unsqueeze(-1)).sum().item())
    predicted = max(ANSWER_LABELS, key=lambda label: (scores[label], -ANSWER_LABELS.index(label)))
    return {
        "row_index": row_index,
        "predicted": predicted,
        "expected": solution,
        "solution": solution,
        "correct": predicted == solution,
        "invalid": False,
        "scores": scores,
        "rotation": rotation % 4,
        "scoring_strategy": strategy["strategy"],
    }


def score_benchmark(
    model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, include_rotations: bool = True,
    batch_size: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("CyberMetric evaluation set is empty")
    first_user, _ = benchmark_prompt(normalize_benchmark_row(rows[0], 0))
    first_prompt = render_chat(tokenizer, [
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": first_user},
    ], add_generation_prompt=True)
    strategy = validate_answer_tokens(tokenizer, first_prompt)
    def score_rotation(rotation: int) -> list[dict[str, Any]]:
        if strategy["strategy"] != "single_token":
            return [score_benchmark_row(model, tokenizer, row, row_index=i, rotation=rotation, scoring=strategy)
                    for i, row in enumerate(rows)]
        import torch
        device = getattr(model, "device", None) or next(model.parameters()).device
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            encoded: list[list[int]] = []
            targets: list[str] = []
            for offset, row in enumerate(chunk):
                user, solution = benchmark_prompt(row, rotation=rotation)
                prompt = render_chat(tokenizer, [
                    {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user},
                ], add_generation_prompt=True)
                encoded.append(list(tokenizer.encode(prompt, add_special_tokens=False)))
                targets.append(solution)
            width = max(map(len, encoded))
            input_values = [ids + [pad_id] * (width - len(ids)) for ids in encoded]
            attention_values = [[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded]
            with torch.no_grad():
                logits = model(
                    input_ids=torch.tensor(input_values, dtype=torch.long, device=device),
                    attention_mask=torch.tensor(attention_values, dtype=torch.long, device=device),
                ).logits.float()
                for local_index, (ids, solution) in enumerate(zip(encoded, targets)):
                    final_logits = logits[local_index, len(ids) - 1]
                    scores = {label: float(final_logits[int(strategy["candidate_ids"][label])].item())
                              for label in ANSWER_LABELS}
                    predicted = max(ANSWER_LABELS, key=lambda label: (scores[label], -ANSWER_LABELS.index(label)))
                    outputs.append({
                        "row_index": start + local_index, "predicted": predicted, "expected": solution,
                        "solution": solution, "correct": predicted == solution, "invalid": False,
                        "scores": scores, "rotation": rotation % 4, "scoring_strategy": strategy["strategy"],
                    })
            del logits
        return outputs

    model.eval()
    predictions = score_rotation(0)
    metric: dict[str, Any] = {
        "accuracy": sum(row["correct"] for row in predictions) / len(predictions),
        "correct": sum(row["correct"] for row in predictions),
        "n": len(predictions),
        "wilson_95": wilson_interval(sum(row["correct"] for row in predictions), len(predictions)),
        "invalid_predictions": sum(row["invalid"] for row in predictions),
        "scoring_strategy": strategy["strategy"],
    }
    if include_rotations:
        rotation_acc = []
        for rotation in range(4):
            rotated = score_rotation(rotation)
            rotation_acc.append(sum(row["correct"] for row in rotated) / len(rotated))
        metric["cyclic_rotation_accuracy"] = rotation_acc
        metric["cyclic_rotation_mean_accuracy"] = sum(rotation_acc) / len(rotation_acc)
    return predictions, metric


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def exact_mcnemar_p(before: Sequence[bool], after: Sequence[bool]) -> float:
    if len(before) != len(after):
        raise ValueError("paired predictions must have equal lengths")
    b = sum(bool(x) and not bool(y) for x, y in zip(before, after))
    c = sum(not bool(x) and bool(y) for x, y in zip(before, after))
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(
    before: Sequence[bool], after: Sequence[bool], *, n_resamples: int = 10_000, seed: int = 42,
) -> tuple[float, float]:
    if len(before) != len(after) or not before:
        raise ValueError("paired non-empty predictions must have equal lengths")
    differences = [int(bool(y)) - int(bool(x)) for x, y in zip(before, after)]
    rng = random.Random(seed)
    n = len(differences)
    estimates = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)]
    estimates.sort()
    return estimates[int(0.025 * n_resamples)], estimates[min(n_resamples - 1, int(0.975 * n_resamples))]


def compare_predictions(before_rows: Sequence[Mapping[str, Any]], after_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(before_rows) != len(after_rows) or not before_rows:
        raise ValueError("prediction files must have the same non-zero number of rows")
    b_ok = [bool(row["correct"]) for row in before_rows]
    a_ok = [bool(row["correct"]) for row in after_rows]
    n = len(b_ok)
    before_n = sum(b_ok)
    after_n = sum(a_ok)
    wrong_right = sum(not b and a for b, a in zip(b_ok, a_ok))
    right_wrong = sum(b and not a for b, a in zip(b_ok, a_ok))
    before_dist = Counter(str(r["predicted"]) for r in before_rows)
    after_dist = Counter(str(r["predicted"]) for r in after_rows)
    answer_key_dist = Counter(str(r.get("solution", r.get("expected", "?"))) for r in before_rows)
    return {
        "n": n,
        "before": {"correct": before_n, "accuracy": before_n / n, "wilson_95": wilson_interval(before_n, n)},
        "after": {"correct": after_n, "accuracy": after_n / n, "wilson_95": wilson_interval(after_n, n)},
        "absolute_change_pp": (after_n - before_n) * 100 / n,
        "relative_change": ((after_n / before_n) - 1) if before_n else None,
        "wrong_to_right": wrong_right,
        "right_to_wrong": right_wrong,
        "unchanged_correct": sum(b and a for b, a in zip(b_ok, a_ok)),
        "unchanged_incorrect": sum(not b and not a for b, a in zip(b_ok, a_ok)),
        "invalid_predictions": {"before": sum(r.get("invalid", False) for r in before_rows), "after": sum(r.get("invalid", False) for r in after_rows)},
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(b_ok, a_ok),
        "paired_bootstrap_95_ci_accuracy_difference": paired_bootstrap_ci(b_ok, a_ok),
        "letter_distributions": {
            "answer_key": {k: answer_key_dist.get(k, 0) for k in ANSWER_LABELS},
            "before": {k: before_dist.get(k, 0) for k in ANSWER_LABELS},
            "after": {k: after_dist.get(k, 0) for k in ANSWER_LABELS},
        },
    }


def token_weighted_loss_from_log_probs(log_probs: Sequence[float], valid_count: int) -> float:
    if valid_count <= 0 or len(log_probs) != valid_count:
        raise ValueError("log_probs must contain exactly one value for every valid target token")
    return -sum(float(value) for value in log_probs) / valid_count


def manual_validation_loss(model: Any, tokenizer: Any, examples: Sequence[Mapping[str, Any]], *, batch_size: int = 1) -> dict[str, Any]:
    """Float32 token-weighted causal loss; ignores all -100 labels and padding."""
    import torch
    import torch.nn.functional as F

    if not examples:
        raise ValueError("validation set is empty")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        model.eval()
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            max_len = max(len(x["input_ids"]) for x in batch)
            input_ids = [list(x["input_ids"]) + [pad_id] * (max_len - len(x["input_ids"])) for x in batch]
            labels = [list(x["labels"]) + [-100] * (max_len - len(x["labels"])) for x in batch]
            mask = [[1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"])) for x in batch]
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            attention = torch.tensor(mask, dtype=torch.long, device=device)
            logits = model(input_ids=input_tensor, attention_mask=attention).logits
            shifted_logits = logits[:, :-1, :].float()
            shifted_targets = label_tensor[:, 1:]
            valid = shifted_targets.ne(-100)
            count = int(valid.sum().item())
            if count:
                log_probabilities = F.log_softmax(shifted_logits, dim=-1)
                safe_targets = shifted_targets.masked_fill(~valid, 0)
                token_log_probs = log_probabilities.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
                total_loss += float((-token_log_probs.masked_select(valid).sum()).item())
                total_tokens += count
    if total_tokens == 0:
        raise ValueError("validation examples contain no supervised target tokens after causal shift")
    return {
        "loss": total_loss / total_tokens,
        "valid_target_tokens": total_tokens,
        "example_count": len(examples),
        "max_length": max(len(x["input_ids"]) for x in examples),
        "truncated_examples": sum(bool(x.get("truncated")) for x in examples),
    }


def validate_baseline_artifact(path: str | Path, expected_fingerprint: str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if artifact.get("fingerprint") != expected_fingerprint or artifact.get("complete") is not True:
        return None
    return artifact


def assert_same_evaluation_hash(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before.get("evaluation_hash") != after.get("evaluation_hash"):
        raise ValueError("before/after evaluation file hashes differ")


def output_root_from_config(config: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    value = environ.get("CYBER_LORA_OUTPUT_DIR") or config.get("output_dir") or "/content/drive/MyDrive/cyber-lora"
    path = Path(value).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise OSError(f"Experiment output is not writable: {path}. Mount Google Drive and retry. Original error: {exc}") from exc
    return path


def valid_checkpoint(path: str | Path, expected_fingerprint: str) -> bool:
    path = Path(path)
    marker = path / "cyber_lora_checkpoint.json"
    complete_files = (
        (path / "adapter_config.json").is_file()
        and ((path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file())
        and (path / "trainer_state.json").is_file()
    )
    if not complete_files or not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("fingerprint") == expected_fingerprint and data.get("complete") is True


def latest_valid_checkpoint(checkpoint_dir: str | Path, expected_fingerprint: str) -> Path | None:
    root = Path(checkpoint_dir)
    if not root.exists():
        return None
    candidates = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1, reverse=True)
    return next((path for path in candidates if valid_checkpoint(path, expected_fingerprint)), None)


def mark_checkpoint(path: str | Path, experiment_fingerprint: str, *, complete: bool = True) -> None:
    path = Path(path)
    if complete and not ((path / "adapter_config.json").is_file()
                         and ((path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file())
                         and (path / "trainer_state.json").is_file()):
        raise ValueError(f"Cannot mark incomplete adapter checkpoint as complete: {path}")
    atomic_write_json(path / "cyber_lora_checkpoint.json", {"fingerprint": experiment_fingerprint, "complete": complete})
