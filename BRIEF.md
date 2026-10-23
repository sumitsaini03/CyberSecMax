# Cybersecurity QLoRA Lab — Project Brief

**Version:** 2.0 | **Date:** 2026-10-23 | **Status:** Primary specification (preserve unchanged once committed)

---

## 0. How to read this brief

- This brief defines **what experiment is run and how it is judged**. The implementation prompt defines how to build, test and publish the repository. If the two disagree, follow the more explicit one and record the deviation in `VERIFICATION_REPORT.md`.
- Anything marked **UNVERIFIED** was not confirmed from a primary source on 2026-10-23 and must be checked by `scripts/inspect_datasets.py` (and recorded in `dataset_metadata.json`) before it is relied on.
- The desk-check behind this brief used public web pages only. No dataset rows were inspected and no code was run.
- Values in `code` font are exact configuration values.

---

## 1. Research question and pre-registration

**Question.** Does QLoRA fine-tuning on defensive-cybersecurity instruction data improve the cybersecurity knowledge and defensive reasoning of `Qwen/Qwen2.5-7B-Instruct`, measured under identical conditions before and after training?

**Endpoints (fixed before any training run):**

| Role | Endpoint | Analysis |
|---|---|---|
| **Primary** | CyberMetric-500 accuracy, before vs after | Paired exact McNemar test (two-sided, α = 0.05) plus paired bootstrap 95% CI for the accuracy difference (10,000 resamples, seed 42) |
| Secondary | Token-weighted validation loss, before vs after | Absolute and relative difference on the identical frozen validation set |
| Secondary | Letter-position robustness | CyberMetric-500 re-scored with cyclically permuted options (§7.4) |
| Descriptive | Five fixed qualitative prompts | Pre-registered rubric (§10); counts only, no inference |

**Expectations recorded in advance (not success criteria):**

1. The base model is already instruction-tuned and the benchmark has been public since 2024, so the baseline may be high and headroom limited. The baseline is unknown until measured.
2. With n = 500, a single accuracy estimate has a 95% interval of roughly ±3 to ±4 percentage points. Illustrative arithmetic: if 12% of questions flip between models (about 60 discordant pairs), a net gain of 15 questions (+3.0 pp) gives z ≈ 15/√60 ≈ 1.9, which is borderline. Smaller effects cannot be separated from noise on this benchmark.
3. In-distribution validation loss will very likely fall after training on that distribution. That alone is **not** evidence of better cybersecurity capability.

**The experiment succeeds when it is valid, reproducible and honestly reported. It does not succeed or fail based on the direction of the result.**

---

## 2. Scope and safety policy

**In scope (defensive focus):** security operations and alert interpretation; threat intelligence and security knowledge; vulnerability analysis; misconfiguration and remediation; defensive investigation; secure coding; cybersecurity question answering.

**Out of scope:** exploit development, weaponised payloads, malware authoring, evasion tooling, and any dataset added only to increase row count.

**Scope filter (required for any source that may contain offensive content).** The RedSage materials explicitly contain offensive-security content (its own responsible-use note says so), so the filter is mandatory for Tier 1:

1. Define keyword and pattern rules for offensive-only content (exploit code, payload generation, C2/implant development, credential-theft tooling usage, evasion of defenses) in `configs/scope_filter.yaml`.
2. Apply the rules, record how many records each rule removed, and save 100 randomly sampled removed and 100 randomly sampled retained records for manual review (kept locally, not committed if the source forbids redistribution).
3. Report counts per rule in `dataset_metadata.json`.
4. If the filter leaves fewer records than the profile needs, reduce N and document it. Do not relax the filter to reach a target size.

Keyword filtering is imperfect. The report must say so.

---

## 3. Model, quantization and hardware

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct`, pinned to a commit SHA recorded at first load (`revision=`). The license is expected to be Apache-2.0 — **UNVERIFIED**, confirm from the model card and record it. |
| Quantization | 4-bit NF4, double quantization on, compute dtype FP16 |
| Attention | `sdpa` (T4 is a Turing GPU; FlashAttention-2 is not supported) |
| Target hardware | Free Colab GPU, expected Tesla T4 (~15 GB VRAM). **Verify at runtime; never assume.** |
| Precision | FP16 on T4. BF16 is not supported on T4. |
| Substitution rule | If 7B does not fit, adjust sequence length, accumulation, checkpointing or optimizer first. Do not silently switch to a smaller model. |

Free Colab sessions have limited duration and can disconnect. Every stage must be resumable (§9).

---

## 4. Data sources

Desk-check on 2026-10-23. "Verified" means seen on a public page; row counts and schemas still need confirmation by script.

| Source | Identifier | What the desk-check found | Status |
|---|---|---|---|
| Tier 1: RedSage-Conv | listed in the RedSage collection (exact repo id **UNVERIFIED**) | RedSage's GitHub roadmap marks the Conv release as done. The README describes it as agentically generated multi-turn dialogues (the project cites ~266K). The `data/README.md` covering provenance, dedup and licensing is still unchecked on the roadmap. | License, schema, splits, size, revision: **UNVERIFIED** |
| Tier 2: Primus-Instruct | `trendmicro-ailab/Primus-Instruct` | Gated: publicly accessible but requires accepting conditions and a Hugging Face login. Under 1K rows per the dataset viewer. `messages`-style role/content turns. License ODC-By. Responses were generated by GPT-4o per the card. | Exact count, revision, split names: **UNVERIFIED** |
| Tier 3: Primus-Reasoning | `trendmicro-ailab/Primus-Reasoning` (referenced in the Llama-Primus-Reasoning model card metadata; the dataset page itself was not opened) | Reasoning traces generated by a stronger model (the model card names o1-preview) | Everything: **UNVERIFIED** |
| Benchmark: CyberMetric | `RISys-Lab/Benchmarks_CyberSec_CyberMetrics`, config `cyberMetric_500` | Mirror of the original CyberMetric release. The card lists configs 80/500/2000/10000, **a single `test` split per config**, license `other`. Fields are `question`, `answers` (a mapping keyed `A`–`D`, not a list), `solution`. | Row count of config 500 (card says 500), revision, hash: confirm by script |

**Note on sibling benchmark repos.** The related RISys-Lab MCQ benchmark uses `val` (10 rows) and `test` (5,000 rows) splits. Split names differ between repos, so never rely on the first split returned by `load_dataset`.

### 4.1 Profile selection (decision rule, evaluated once, before training)

1. Run `scripts/inspect_datasets.py` and write `dataset_metadata.json`.
2. **Profile T1** is chosen only if all hold: Tier 1 downloads without breaking its terms; its license is documented and permits this use; it normalizes to the message schema; the scope filter retains enough records.
3. Otherwise **Profile T2** is chosen if Primus-Instruct is accessible with the user's own token and accepted terms.
4. Otherwise stop, report the blocker, and do not substitute another source.
5. **Tier 3 is excluded from the default experiment.** Its long reasoning traces are poorly matched to a 1,024-token cap and to a single-letter scoring metric. It may be used only in a separately labelled follow-up experiment after provenance and suitability are verified.

### 4.2 Profiles

| | Profile T1 | Profile T2 |
|---|---|---|
| Source | RedSage-Conv (scope-filtered) | Primus-Instruct |
| Final mix size | up to 20,000 records, seed 42 | all cleaned records (<1K) |
| General replay | ≈15% of the final mix (≈3,000 of 20,000), from one verified permissively licensed general instruction dataset (candidate: `HuggingFaceH4/ultrachat_200k`, **UNVERIFIED**). Omit and document if none can be verified. | none |
| Split | 90/10, group-aware (§5) | 90/10 |
| Epochs | 1 | up to 3, best-validation-loss checkpoint |
| Learning rate | `1e-4` | `2e-4` |
| Checkpoint reported | final | best by in-training validation loss |

**Sample-size rule for T1.** The 20,000 target is a ceiling. After the pilot (§8), compute the projected total training time from measured tokens per second. If it exceeds the budget of one Colab session plus recovery, reduce N **before the run starts**, record the reason, and never change N after seeing any result.

---

## 5. Data preparation

Every step is deterministic with seed 42. Row counts are logged before and after each step.

1. **Normalize to** `{"messages": [{"role": ..., "content": ...}, ...], "source": ..., "source_id": ...}`.
2. **Validate:** roles in `{system, user, assistant}`, non-empty string content, conversation ends with an assistant turn, at least one user and one assistant turn. Remove malformed records and count them.
3. **System prompt:** if a record has no system message, prepend the fixed `SYSTEM_PROMPT` (§6.2). This prevents the Qwen chat template from silently injecting its own default system text at training time.
4. **Scope filter** (§2).
5. **Exact deduplication** on the normalized concatenation of all message contents.
6. **Contamination check** (§5.1).
7. **Split 90/10.** For multi-turn dialogues, split **by `source_id` group** where such a field exists (dialogues generated from the same seed document are near-duplicates, and a random split would leak and flatter validation loss). Fall back to a random split only if no grouping field exists, and record that.
8. **Freeze** the validation file and record its SHA-256. The training and validation files are never regenerated after the baseline is measured.
9. Keep processed benchmark data separate from training data.

### 5.1 Contamination check

Streaming, one training record at a time, no concatenation of the corpus.

- **Normalization:** Unicode NFKC, case-fold, collapse whitespace. A second pass also strips punctuation.
- **Level 1 (required):** remove any training record whose normalized text contains a normalized CyberMetric question.
- **Level 2 (required):** remove any record containing both the normalized question stem and the normalized text of the correct answer.
- **Level 3 (recommended):** flag records whose word 5-gram Jaccard overlap with a benchmark question is ≥ 0.8; remove and report them.
- **Never modify, drop or rewrite benchmark questions.**
- **Report:** benchmark questions examined, training records examined, records removed per level, final training count, matching rules, and limitations.

**Limitations to state in the report:** exact and n-gram matching cannot detect paraphrases or semantic overlap; CyberMetric was generated from public sources (NIST documents, RFCs, books, papers) that may also appear in training corpora; and the base model may have seen CyberMetric during its own pretraining. That last point cannot be tested here and could inflate the baseline.

---

## 6. Tokenization, loss masking and validation loss

### 6.1 One tokenization function for everything

Training, in-training evaluation and the reported validation loss all use the **same** function. Do not rely on a trainer's `assistant_only_loss` option: it needs chat templates with generation markers, and Qwen's stock template may not have them (verify against the installed tokenizer).

- Render with the model's chat template. Maximum length `1024` (see §8 for the fallback). When a conversation exceeds the cap, retain the final assistant answer and as much recent context as fits; if the answer itself exceeds the cap, retain its beginning. Count and report truncated examples; do not drop them. This avoids batches with no supervised answer tokens.
- **Supervised tokens:** the content of every assistant turn **plus its closing `<|im_end|>` token**. All other positions get label `-100`.
- Required test: decode the supervised tokens of a sample and assert they equal the assistant text plus the end token.

### 6.2 Fixed prompt constants

```
SYSTEM_PROMPT = "You are a careful cybersecurity assistant. Answer accurately and concisely."
```

The same constant is used in training records without a system message, validation, CyberMetric scoring and qualitative generation.

### 6.3 Validation-loss function

One deterministic function, called for both the original and the fine-tuned model. **Do not use `trainer.evaluate()` for reported numbers.** A trainer's internal evaluation may be used during training for checkpoint selection only.

- `model.eval()`, `torch.no_grad()`, same frozen examples, same tokenized ids and truncation.
- Standard causal shift (logits at position *t* predict token *t+1*). Mask padding and every `-100` label.
- Cross-entropy computed from **float32** log-probabilities, summed over valid target tokens and divided by the total number of valid target tokens across the whole validation set (token-weighted, not a mean of batch means).
- Record: example count, valid target-token count, max length, number truncated, validation-file hash, loss before, loss after, absolute difference.
- The count of valid target tokens must be **identical** before and after training. Assert it.
- Unit test on a tiny synthetic case with a hand-computed expected loss.
- **In-training monitoring** uses a frozen 300-example subset of the validation set to keep cost down. Reported numbers use the full set.

### 6.4 Sanity checks that tie the pipeline together

After training, load the adapter on a fresh 4-bit base and:

1. Evaluate with the adapter **enabled** (this is the reported "after").
2. Evaluate with the adapter **disabled** and assert the loss matches the baseline within a stated tolerance. This detects evaluation-state drift.
3. Evaluate the adapter **unmerged**. Do not merge into a 4-bit base for the reported numbers, because merging changes the numerics.

---

## 7. CyberMetric-500 evaluation

### 7.1 Prompt

User message (fixed template):

```
Question: {question}
A) {answers[A]}
B) {answers[B]}
C) {answers[C]}
D) {answers[D]}

Answer with a single letter (A, B, C, or D).
```

Applied with the chat template, `SYSTEM_PROMPT`, and `add_generation_prompt=True`. The prompt therefore ends at the assistant-turn boundary.

### 7.2 Scoring

Deterministic next-token scoring. Take the logits at the final prompt position, compare the four candidate letter tokens, and predict the argmax.

**Mandatory tokenizer validation** (fail loudly if any check fails):

- Each of `A`, `B`, `C`, `D` encodes to exactly one token **at the answer boundary**, i.e. `tokenize(prompt + letter)` equals `tokenize(prompt) + [letter_id]`. Standalone encodings are not sufficient.
- The four ids are distinct and decode to the intended letters.
- If the single-token check fails, use full-sequence log-likelihood of the label string instead, consistently for both models, and document the deviation.

`answers` may arrive as a mapping or a list. Normalize it and assert exactly four options labelled `A`–`D`, and that `solution` is one of them. Reject malformed rows with an informative error rather than skipping them silently.

### 7.3 Required metrics

Accuracy before and after with Wilson 95% intervals; absolute change in percentage points; relative change; wrong→right, right→wrong, unchanged-correct and unchanged-incorrect counts; invalid predictions (zero by construction in logit scoring, still reported); exact McNemar p-value on the discordant pairs; paired bootstrap 95% CI; predicted-letter distribution against the answer-key distribution; full per-question results.

### 7.4 Position-bias control (secondary)

Fine-tuning can shift the model's prior over answer letters. Re-score the benchmark with the four cyclic rotations of the option order (remapping the solution each time), and report mean accuracy across rotations alongside the original-order result. The primary endpoint stays the original-order result.

### 7.5 Optional cross-check

RedSage publishes a `lighteval` runner with a `cybermetrics:500` task. Running it on the base model is an optional cross-check of the harness. Differences in prompt and scoring are expected, so compare for plausibility only, not for equality.

### 7.6 Hygiene

CyberMetric is never used to choose data, epochs, learning rate, checkpoints or prompts. Hyperparameters are decided with validation loss and training diagnostics only.

---

## 8. Training configuration

| Parameter | Value |
|---|---|
| LoRA rank / alpha / dropout / bias | 16 / 32 / 0.05 / `none` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (assert each exists) |
| Per-device batch / accumulation / effective batch | 1 / 16 / 16 |
| Learning rate | per profile (§4.2) |
| Scheduler / warmup ratio | cosine / 0.03 |
| Weight decay / gradient clipping | 0.01 / 0.3 |
| Optimizer | paged AdamW 8-bit if supported, otherwise documented fallback |
| Gradient checkpointing | on (`use_reentrant=False`) |
| Max sequence length | 1024 |
| OOM fallback | 512 with batch size 1 |
| Seed | 42 |
| Epochs | per profile (§4.2) |

**Rules:**

1. **The baseline is measured before any LoRA layer or trainer object exists.** Order: tokenizer, 4-bit model, tokenizer validation, frozen data, CyberMetric baseline, validation-loss baseline, five qualitative responses, save artifacts, **then** build the LoRA config and trainer.
2. **If the OOM fallback changes the sequence length, the baseline validation loss is invalid** (different truncation). Recompute it under the final length and store it under its new sub-fingerprint (§9). The CyberMetric baseline does not depend on training length and stays valid.
3. **Trainer choice.** TRL `SFTTrainer` or the plain Hugging Face `Trainer` with the §6.1 collator are both acceptable, provided the masking is identical. Prefer the plain `Trainer` if the installed TRL version's argument names (sequence-length parameter, `processing_class` vs `tokenizer`, evaluation-strategy name, `dtype` vs `torch_dtype`) cannot be resolved with `inspect.signature` feature detection.
4. **Dependency versions** are pinned in `requirements-colab.txt` to a set that actually passed the pilot. No unbounded upgrades.
5. **Preflight (all must pass before the full run):** CUDA available; GPU name and free VRAM printed; 4-bit load succeeds; all target modules found; only adapter parameters are trainable (print the count); a short pilot runs with finite loss; an adapter checkpoint saves and reloads; tokens per second measured and used for the T1 time budget.
6. **Checkpoint selection** uses in-training validation loss only (T2 best checkpoint; T1 final checkpoint unless the loss diverged, in which case report that and the last finite checkpoint).

---

## 9. Persistence, fingerprinting and recovery

**Output root** is configurable via config or environment variable. The default is `/content/drive/MyDrive/cyber-lora/`.

```
cyber-lora/
├── baseline/            # written once, never overwritten by post-training results
├── checkpoints/         # keep the last 2 only (save_total_limit=2)
├── final/               # final PEFT adapter
├── post_training/
├── results.json
├── samples_before_after.json
├── baseline_predictions.json
├── finetuned_predictions.json
├── experiment_metadata.json
└── dataset_metadata.json
```

**Experiment fingerprint.** A SHA-256 over: model id and revision; tokenizer id and revision; quantization config; evaluation-file hash; validation-file hash; training-file hash; `SYSTEM_PROMPT`; prompt template; chat-template hash; scoring config; max sequence length; seed; decoding settings; profile name; and the major library versions (torch, transformers, peft, bitsandbytes, and TRL if used).

- Each baseline component stores its own sub-fingerprint and is reused independently. The **CyberMetric baseline** depends on model, tokenizer, quantization, evaluation-file hash, prompt template, `SYSTEM_PROMPT`, chat-template hash, scoring config, decoding settings and library versions. The **validation-loss baseline** additionally depends on the validation-file hash, max sequence length, masking function and seed. The **qualitative baseline** depends on the prompts, `SYSTEM_PROMPT`, decoding settings and the model inputs.
- A component is reused **only** if its stored sub-fingerprint equals the current one. Otherwise recompute that component. Never overwrite a valid baseline with post-training output.
- Write files atomically (temp file then rename). A run directory carries a `COMPLETE.json` marker only when every artifact exists and reloads. An incomplete checkpoint is never reported as success.
- Check that Drive is mounted and writable first. If not, fail with recovery instructions.
- Resume training from the latest valid checkpoint after verifying its fingerprint.
- Never write tokens or secrets into metadata, logs, notebook outputs or commits. Use Colab Secrets or an interactive prompt for the Hugging Face token.

---

## 10. Qualitative evaluation

Five fixed prompts, identical wording, `SYSTEM_PROMPT`, chat template, greedy decoding (`do_sample=False`), `max_new_tokens=300`, before and after. Prompts are never edited after outputs are seen. A changed prompt means a new experiment.

1. Explain MITRE ATT&CK technique T1059.001 and give three detection ideas.
2. Analyze repeated Windows event IDs 4625 followed by event ID 4624, including likely interpretation and defensive response.
3. Write a Splunk SPL query that flags more than ten failed logins per user within five minutes.
4. Explain the risks and remediation of an S3 bucket declared with `acl = public-read` in Terraform.
5. Explain the CVSS v3.1 base metrics.

**Rubric (written now, before any outputs exist).** Each criterion is pass/fail. Outputs from the two models are shuffled and relabelled so the rater is blind to which is which. Greedy output is capped at 300 tokens, so a truncated answer is marked "truncated" and scored only on what it contains.

| # | Pass criteria |
|---|---|
| 1 | Identifies T1059.001 as the PowerShell sub-technique of Command and Scripting Interpreter; gives three plausible, distinct detection ideas (for example script-block or module logging, encoded-command process creation, AMSI or EDR telemetry); invents no sub-technique names |
| 2 | Identifies 4625 as failed logon and 4624 as successful logon; interprets failures followed by success as possible brute force or password spraying with a successful compromise; mentions checking logon type, source and account; gives defensive actions (credential reset, MFA, source blocking, review of post-logon activity, lockout policy) |
| 3 | Valid SPL pipeline syntax; filters failed-logon events; buckets by a 5-minute window; groups by user; uses a threshold of strictly greater than 10 |
| 4 | States the exposure risk (public read of objects); gives remediation (private ACL or ACLs disabled, S3 Block Public Access, least-privilege bucket policy); notes safer patterns for intentionally public content; makes no false claims about Terraform or AWS behavior |
| 5 | Lists all eight base metrics (Attack Vector, Attack Complexity, Privileges Required, User Interaction, Scope, Confidentiality, Integrity, Availability) with correct value sets; if severity bands are given they are correct |

Also record, for each output: malformed or looping text, any unsafe or offensive-only content, and any regression relative to the baseline response. Five prompts support illustration only. Do not draw statistical conclusions from them.

---

## 11. Reporting rules

State results exactly as measured. Pick the matching statement:

| Outcome | Required wording |
|---|---|
| McNemar p < 0.05 and CI excludes 0, accuracy up | "Measured improvement on CyberMetric-500 under this protocol" and state size, CI, and limitations |
| Otherwise, either direction | "No statistically detectable change on CyberMetric-500 (n = 500)", with the point estimate and CI |
| p < 0.05 and accuracy down | "Measured degradation", followed by an investigation (training logs, overfitting, learning rate, data quality) |

Always report: the actual sample count, evaluation hash, model and dataset revisions, fingerprint, validation-loss change with its token count, the position-robustness result, the qualitative rubric counts, and every limitation in §5.1 and §1. A lower validation loss alongside unchanged or lower benchmark accuracy is a legitimate finding and is reported as such. Do not claim generalization beyond the evaluated tasks.

If the result is null or negative, propose one justified next experiment (for example a different learning rate chosen from validation loss, a rank or target-module ablation, cleaner or larger data, or a different benchmark). Do not run it silently.

---

## 12. Integrity rules (apply to every step)

1. Never fabricate scores, row counts, dataset properties, test results, training progress, or publication status. Unknown means `UNVERIFIED`.
2. Never alter benchmark questions, scoring logic or the test split to move a result.
3. Never use benchmark performance to tune anything.
4. Never present a synthetic or tiny-model test as a result for Qwen2.5-7B-Instruct.
5. Never commit secrets, gated data, model weights, or placeholder numbers shaped like real measurements.
6. Keep runtime assertions on. Missing columns, unknown configs, invalid labels, no GPU, incompatible tokenizers and invalid checkpoints must fail with a clear message.

---

## 13. Licensing and terms notes (not legal advice)

- Primus-Instruct is ODC-By (attribution required). Its responses were generated by GPT-4o and Primus-Reasoning's by an OpenAI reasoning model. The upstream model provider's terms of use may place conditions on using such outputs to train other models. The user should read them and decide. Record the decision in `VERIFICATION_REPORT.md`.
- The CyberMetric mirror lists license `other`. Check the original CyberMetric repository's terms before redistributing any benchmark text. Do not commit benchmark questions unless the terms permit it. Predictions identified by row index are safe.
- Check RedSage-Conv's dataset card for license and any restriction before use. The roadmap shows its provenance and licensing page was still pending when checked.
- The project license covers original code only. It does not override the base-model, dataset or benchmark terms. The trained adapter inherits the base model's license terms.
- Add a `NOTICE` or README section with required attributions (CyberMetric, Primus, RedSage if used, QLoRA, Qwen2.5).

---

## 14. Acceptance criteria (binary, none depends on result direction)

- [ ] Tokenizer validation passes for the four answer letters at the answer boundary.
- [ ] The baseline (CyberMetric, validation loss, five responses) was saved **before** any LoRA or trainer object was created, with its fingerprint.
- [ ] The validation-loss function passes the synthetic test, and the valid target-token count is identical before and after.
- [ ] The adapter-disabled check reproduces the baseline loss within tolerance.
- [ ] Contamination check ran; counts and limitations are reported; the benchmark is untouched.
- [ ] Scope filter ran; counts are reported (Tier 1 only).
- [ ] Training completed or was cleanly resumed; the adapter saves, reloads and a `COMPLETE.json` exists.
- [ ] Post-training metrics were computed under the identical fingerprint-compatible conditions.
- [ ] `results.json`, both prediction files, `samples_before_after.json`, and both metadata files exist and contain no fabricated values.
- [ ] All eight required CyberMetric metrics plus the McNemar p-value and bootstrap CI are reported.
- [ ] Unit tests and notebook validation pass in CI; every unrun check is marked `UNVERIFIED`.
- [ ] README and `VERIFICATION_REPORT.md` separate measured results from pending ones.

---

## 15. Open items to resolve with the inspection script

1. RedSage-Conv repository id, license, splits, schema, size, `source_id`-like grouping field, and revision.
2. Primus-Instruct exact row count, split names, schema, revision, and the user's gated-access status.
3. Primus-Reasoning existence, license, schema (only if a follow-up experiment is planned).
4. The general-instruction replay dataset (identifier, license, revision).
5. CyberMetric `cyberMetric_500` row count, revision, file hash, and answer-key distribution.
6. Qwen2.5-7B-Instruct revision SHA and license text.
7. Whether the installed tokenizer's chat template contains generation markers and whether it injects a default system message when none is given.
8. Tested dependency set for the Colab runtime on the day of the run.

---

## 16. References

- Tihanyi et al., *CyberMetric: A Benchmark Dataset based on Retrieval-Augmented Generation for Evaluating LLMs in Cybersecurity Knowledge*, arXiv:2402.07688.
- Yu et al., *Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training*, arXiv:2502.11191.
- Suryanto et al., *RedSage: A Cybersecurity Generalist LLM*, ICLR 2026, arXiv:2601.22159. Code and data collection: https://github.com/RISys-Lab/RedSage
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314.
- CyberMetric mirror: https://huggingface.co/datasets/RISys-Lab/Benchmarks_CyberSec_CyberMetrics
- Primus-Instruct: https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct
- RedSage-Seed card (taxonomy and configs of the seed data behind Conv): https://huggingface.co/datasets/RISys-Lab/RedSage-Seed
