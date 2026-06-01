# IML_TalkieBridge

Introduction of Machine Learning term project repository.

This project implements the proposal in `PROJECT_PROPOSAL.md`: an
anachronism-aware prompt rewriting system that converts modern multiple-choice
questions into era-neutral functional descriptions for evaluating Talkie-1930.

Talkie-1930 is not the model we are building. It is only a fixed downstream
evaluator. The project we are building is the prompt rewriting and evaluation
pipeline in front of it.

## Current Status

The main implementation lives in `src/talkie_bridge/`. The repository does not
vendor the original `talkie` package and does not keep prototype scripts in the
committed project tree. Talkie is treated as an external fixed evaluator.

## Repository Layout

```text
.
├─ PROJECT_PROPOSAL.md          # Project proposal and source of truth
├─ DATASET_SCHEMA.md            # Dataset and dictionary schema
├─ pyproject.toml               # Minimal project metadata
├─ src/talkie_bridge/           # Proposal-centered rewriting/evaluation pipeline
├─ tests/                       # Unit and pipeline tests
├─ data/*_example.*             # Example dataset and dictionary files
├─ input_data/*_example.*       # Example manual response sheet
└─ results/*_example.*          # Example output artifacts
```

Generated `data/`, `input_data/`, `cache/`, and `results/` files are ignored by
default, except files with `_example` in the filename.

## Pipeline Commands

Install the project in editable mode if you want `python -m talkie_bridge.cli`
to work without setting `PYTHONPATH`:

```powershell
python -m pip install -e .
```

Generate rewrites and prompts only:

```powershell
python -m talkie_bridge.cli rewrite-only
```

This command reads `data/generated_questions.jsonl`. It does not auto-generate
final experiment data. See `DATASET_SCHEMA.md` for the fields to fill.

Use the committed example files only as templates:

```powershell
Copy-Item data/generated_questions_example.jsonl data/generated_questions.jsonl
Copy-Item data/modern_terms_dictionary_example.json data/modern_terms_dictionary.json
Copy-Item data/primitive_dictionary_example.json data/primitive_dictionary.json
```

Then replace the copied dataset with the real 100-150 item human-validated
dataset before making final claims.

Create a local mock dataset for smoke testing:

```powershell
python -m talkie_bridge.cli init-mock-data --force
```

Prepare a CSV for manual Talkie response collection. The default evaluates the
test split only; use `--eval-split all` only for diagnostics or collection runs
where you intentionally want every split:

```powershell
python -m talkie_bridge.cli prepare-manual
```

Use a predeclared concept dictionary instead of train-derived resources:

```powershell
python -m talkie_bridge.cli prepare-manual --concept-dictionary-json data/modern_terms_dictionary.json --primitive-dictionary-json data/primitive_dictionary.json
```

Dictionary files are never auto-loaded by filename. They are used only when both
paths are supplied explicitly.
Files named `mock_*dictionary.json` are blocked in evaluation unless
`--allow-mock-dictionary` is supplied for diagnostics.

Evaluate manually pasted Talkie responses:

```powershell
python -m talkie_bridge.cli evaluate-manual --manual-response-csv input_data/manual_talkie_input_sheet.csv
```

`input_data/manual_talkie_input_sheet_example.csv` shows the expected manual
sheet columns. Final evaluation should use the generated
`input_data/manual_talkie_input_sheet.csv` with real Talkie responses pasted into
`raw_response_manual`.

Manual evaluation fails by default if prompt hashes are missing/mismatched or if
responses are blank. Use `--allow-hash-mismatch` or `--allow-missing-responses`
only for diagnostics.

Optionally call the unofficial Talkie web SSE endpoint:

```powershell
python -m talkie_bridge.cli run-api --provider unofficial-api
```

Manual CSV mode is the primary reproducibility path. The unofficial API mode is
provided only as an opt-in convenience path and is labeled as such in generated
reports.

Generate a static demo page for one item:

```powershell
python -m talkie_bridge.cli demo --item-id q001 --out results/demo.html
```

The proposed condition uses a small dependency-free primitive bottleneck
autoencoder when enough train/dev data exists. With tiny mock data it falls back
to deterministic primitive rewriting and records that fallback in `cache/`.

## Setup

Use a normal Python virtual environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Local environments, Python caches, logs, model checkpoints, large recordings,
and generated dataset/result/cache files are ignored by git unless their
filenames include `_example`.
