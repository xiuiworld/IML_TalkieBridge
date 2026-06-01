# IML_TalkieBridge

Introduction of Machine Learning term project repository.

This project implements the proposal in `PROJECT_PROPOSAL.md`: an
anachronism-aware prompt rewriting system that converts modern multiple-choice
questions into era-neutral functional descriptions for evaluating Talkie-1930.

Talkie-1930 is not the model we are building. It is only a fixed downstream
evaluator. The project we are building is the prompt rewriting and evaluation
pipeline in front of it.

## Current Status

The main implementation now lives in `src/talkie_bridge/`. The old
`TermProject_team2.py` file is kept only as a prototype/reference under
`prototypes/team2_pipeline/`. Do not treat it as the current project
implementation or source of truth.

## Repository Layout

```text
.
├─ PROJECT_PROPOSAL.md          # Project proposal and source of truth
├─ pyproject.toml               # Minimal project metadata
├─ src/talkie_bridge/           # Proposal-centered rewriting/evaluation pipeline
├─ prototypes/team2_pipeline/   # Old prototype, not the main implementation
├─ prototypes/talkie_web_api/   # Old unofficial Talkie API prototype
└─ docs/html/                   # Proposal/explanation HTML exports
```

The original `talkie` implementation is not vendored into this project
repository. Talkie is treated as an external fixed evaluator.

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

Create a local mock dataset for schema reference:

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
and generated dataset/result/cache directories are ignored by git.
