# Cybersecurity QLoRA Lab

Reproducible, defensive cybersecurity fine-tuning experiment for `Qwen/Qwen2.5-7B-Instruct`. The project compares the original model with a QLoRA adapter under a frozen CyberMetric-500 protocol and reports measured results without requiring an accuracy or validation-loss improvement.

**Current experiment status:** pending. The local authoring environment has an NVIDIA GeForce RTX 3050 (6 GiB), but no PyTorch/Transformers runtime, model weights, or authorized gated dataset token. No Qwen baseline, fine-tuning run, benchmark result, or adapter is included or implied.

## What is in this repository

- `BRIEF.md`: original experiment brief, copied unchanged.
- `train_cyber_lora.py`: end-to-end orchestration, including source inspection, frozen data, baseline, preflight, training, recovery, and post-training evaluation.
- `cyber_lora_utils.py`: schema checks, filtering and contamination controls, tokenizer validation, token-weighted loss, benchmark metrics, fingerprints, atomic I/O, and checkpoint helpers.
- `SecurityMaxx.ipynb`: Colab workflow that calls the primary script.
- `scripts/inspect_datasets.py`: live metadata/access inspection; benchmark rows are read only to compute their hash and are not copied to the repository.
- `configs/experiment.yaml`, `configs/scope_filter.yaml`: experiment settings and the Tier 1 scope rules.
- `tests/`: synthetic unit tests; these do not establish model performance or GPU compatibility.
- `.github/workflows/ci.yml`: CPU-only compile, YAML, notebook, and synthetic test checks.
- `VERIFICATION_REPORT.md`, `dataset_metadata.json`, and `NOTICE`: audit notes, the dated public-source inventory, and external attribution/terms notes.

## Model and data sources

The model is pinned to the observed Qwen Hub commit `a09a35458c702b33eeacc393d103063234e8bc28`. Its card reports Apache-2.0. The tokenizer and model load at that same revision, with 4-bit NF4 quantization, double quantization, FP16 compute, and SDPA attention.

The data decision is made once during preflight:

| Profile | Source | Decision conditions | Training plan |
|---|---|---|---|
| T1 | `RISys-Lab/RedSage-Conv` | Accessible, documented compatible license, approved access conditions, and scope filter retains data | Up to 20,000 records total, targeting 15% verified general replay, one epoch, `1e-4` learning rate |
| T2 | `trendmicro-ailab/Primus-Instruct` | User token has accepted its gated conditions and the user has separately reviewed the GPT-4o output terms | All cleaned records, up to three epochs, `2e-4` learning rate |

Tier 3 `Primus-Reasoning` is inspected but excluded. At the time of the source audit, RedSage-Conv and Primus-Instruct both required Hugging Face access acceptance. RedSage's current card did not state a dataset license, so it is not eligible by default. Primus-Instruct's card states ODC-By and describes GPT-4o-generated responses; exact file splits and row count still require authenticated inspection. The training flags in `configs/experiment.yaml` default to `false`. The pipeline stops instead of silently substituting a source if no profile is eligible.

CyberMetric is loaded from `RISys-Lab/Benchmarks_CyberSec_CyberMetrics`, config `cyberMetric_500`, explicit `test` split. Its card lists 500 rows, fields `question`, `answers` (`A`–`D` struct), and `solution`, with license `other`. The mirror points to the original CyberMetric project; the original release terms should be reviewed before redistribution. This repository stores only row-indexed predictions, answer letters, metrics, and hashes, not benchmark questions.

The brief's candidate general replay source, `HuggingFaceH4/ultrachat_200k`, is now verified from its public card as MIT licensed with a `train_sft` split of 207,865 examples and `messages` fields. The T1 pipeline targets roughly 15% from that split when its live revision/schema check passes. It records the source counts and omits replay if runtime loading fails; it does not inflate the T1 ceiling.

See [dataset_metadata.json](dataset_metadata.json) for the dated inventory and [NOTICE](NOTICE) for attribution and terms reminders. Live inspection output is written separately under the experiment output directory.

## Method

1. Inspect the configured Hugging Face sources and resolve immutable revisions.
2. Load and validate the 500-row benchmark; keep it separate from training.
3. Normalize conversations, add the fixed system message where missing, reject malformed rows, deduplicate, apply Tier 1 scope filtering when required, and run streaming contamination checks.
4. Make a deterministic 90/10 split. Use source group IDs when present; otherwise use a seeded record-level split. Freeze and hash both files.
5. Load the original 4-bit base model and save baseline CyberMetric predictions, full validation loss, five greedy qualitative responses, and component fingerprints **before** constructing LoRA or a Trainer.
6. Run a one-step LoRA pilot, require finite loss/gradients, save and reload the adapter, measure throughput, and choose the final training count before full training.
7. Train the adapter with gradient checkpointing, per-device batch size 1, accumulation 16, rank 16, alpha 32, dropout 0.05, and the profile-specific rate/epoch settings.
8. Load a fresh base plus the unmerged adapter. Verify adapter-disabled validation loss matches baseline, evaluate the adapter-enabled model with the same token IDs, and save full prediction records and paired metrics.

All training/evaluation tokenization uses the chat template and supervises assistant content plus `<|im_end|>`. Validation loss uses causal shifting, float32 log probabilities, masks `-100` labels/padding, and divides the total negative log probability by total valid target tokens. CyberMetric first validates one-token label IDs at the answer boundary; it falls back to full-sequence label likelihood when necessary. Its primary endpoint is original-order accuracy; four cyclic option rotations are secondary.

## Google Colab setup

1. Open the prepared `SecurityMaxx.ipynb` in Colab. The repository ZIP should extract to `/content/cybersecurity-qlora-lab`.
2. Select a GPU runtime. The notebook prints `nvidia-smi`; the script verifies CUDA, GPU name, and free VRAM.
3. Run the pinned dependency cell. PyTorch is inherited from the Colab runtime to preserve its CUDA-matched build; its actual version is recorded. The remaining pinned package set is a **candidate**, not a tested stack; it must pass the runtime preflight.
4. Mount Google Drive. The default output directory is `/content/drive/MyDrive/cyber-lora`; it can be overridden with `CYBER_LORA_OUTPUT_DIR`.
5. If using a gated source, add a Colab Secret named `HF_TOKEN` (never commit a token). Accept the source's conditions in your own Hugging Face account.
6. Review the source license and usage conditions. RedSage-Conv is not eligible unless its current card documents an acceptable license. For Primus-Instruct, decide whether training on GPT-4o-generated responses is permitted under the applicable provider terms. Only then set the related acceptance flags in `configs/experiment.yaml`.
7. Run dataset inspection, then the baseline and pilot:

   ```bash
   python scripts/inspect_datasets.py --config configs/experiment.yaml --output "$CYBER_LORA_OUTPUT_DIR/dataset_metadata.json"
   python train_cyber_lora.py --config configs/experiment.yaml --preflight-only
   ```

8. Review `dataset_metadata.json`, `baseline/`, and `preflight_summary.json`. If eligible and all preflight checks pass, start the full run:

   ```bash
   python train_cyber_lora.py --config configs/experiment.yaml
   ```

The script checks Drive writability and refuses to overwrite frozen data with a different source, configuration, or hash. After an interrupted session, mount Drive and rerun the same command. Only checkpoints with a matching fingerprint marker are eligible for recovery. If the 1,024-token pilot runs out of memory, the pipeline reduces sequence length to 512 and recomputes the validation baseline before constructing the adapter again. If that also fails, it stops.

The supplied project Colab notebook is a separate, prepared workflow. If you use an existing blank Colab notebook, import `SecurityMaxx.ipynb` through **File → Upload notebook**, then upload/extract the ZIP in the Files pane so the expected project directory exists.

## Results and outputs

The default Colab output layout is:

```text
cyber-lora/
├── baseline/
├── data/                         # local frozen data; never commit
├── checkpoints/                  # keep the last two
├── final/                        # final PEFT adapter, unmerged
├── post_training/
├── results.json
├── samples_before_after.json
├── baseline_predictions.json
├── finetuned_predictions.json
├── experiment_metadata.json
├── dataset_metadata.json
└── COMPLETE.json                 # only after required artifacts/reload checks pass
```

The primary comparison includes Wilson intervals, absolute and relative change, discordant-pair counts, exact two-sided McNemar p-value, paired bootstrap 95% interval (10,000 resamples, seed 42), answer/prediction letter distributions, invalid prediction count, and row-indexed predictions. Secondary results include token-weighted loss and cyclic option rotations. The script saves qualitative outputs but leaves their blinded rubric rating pending for manual review.

No benchmark result or trained adapter exists in this source repository. `results.json` and prediction files are produced only by a successful real run. A lower validation loss is not evidence by itself of improved cybersecurity capability. Report null or negative changes as measured.

## Reproducibility and limitations

- Seed: 42. Model, tokenizer, data, prompt, chat-template, scoring, sequence length, and major package versions contribute to fingerprints.
- Baseline components are fingerprinted and reused independently; invalid/incomplete files are not reused.
- Validation file is frozen before baseline and cannot be regenerated in the same output directory.
- Scope filtering is imperfect regex/keyword triage; it can miss offensive material and remove benign defensive discussion. Tier 1 stores random removed/retained review samples locally and excludes them from Git.
- Exact text and 5-gram checks do not identify semantic paraphrases. CyberMetric is derived from public cybersecurity sources, and pretraining exposure cannot be measured here.
- The fixed single-letter CyberMetric prompt differs from the original project's XML answer prompt. This follows the pre-registered experiment protocol and is recorded as a protocol choice, not an attempt to reproduce the upstream evaluator exactly.
- The 7B 4-bit configuration is not proven to fit the free Colab GPU until the pilot runs. Free runtime duration and actual throughput vary.
- The five qualitative prompts support illustration only; they are not a statistical endpoint.

## Development

Use Python 3.11+ for the lightweight local checks:

```bash
python -m pip install -r requirements-test.txt
python -m compileall -q train_cyber_lora.py cyber_lora_utils.py scripts tests
python scripts/validate_notebook.py SecurityMaxx.ipynb
python -m pytest -q
```

CI intentionally excludes model downloads, gated datasets, and GPU training.

## Contributing

Keep benchmark data and model weights out of pull requests. Changes to prompts, splits, scoring, or hyperparameters need a documented protocol deviation and a new experiment fingerprint. Add synthetic tests for data/statistics behavior and report unverified GPU or dataset assumptions plainly.
