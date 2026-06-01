# Team 2 Prototype Pipeline

This folder contains an old end-to-end prototype script. It is preserved only as
reference material and is not the current implementation source of truth for
`IML_TalkieBridge`.

It was useful as a proof of concept because it connected the full proposed
flow once: synthetic four-choice question generation, modern-term detection,
primitive mapping, autoencoder-based rewriting, validation/repair, Talkie web
evaluation, and metric/report generation.

It should not be reused as the final experiment code. The prototype has known
methodological problems:

- The dataset is synthetic and template-driven from `build_concept_library()`.
- Question generation, answer choices, and rewrite primitives all come from the
  same concept library, creating leakage and answer-hint risk.
- The autoencoder is trained and evaluated on the same dataset.
- The intended ablations are mostly missing; it is effectively raw vs proposed.
- The mapper already performs direct primitive lookup, so the autoencoder's
  independent contribution is unclear.
- `run_web_grid` tunes hyperparameters using the same Talkie responses later
  treated as final evaluation.
- The mock evaluator is biased toward proposed prompts and cannot support
  performance claims.
- The Talkie web integration scrapes an unofficial web UI, so it is brittle and
  hard to reproduce.
- Response parsing can guess labels from token overlap when Talkie gives
  explanation-style answers.
- Statistical analysis is incomplete compared with the proposal.

Use this folder only to recover ideas or inspect discarded behavior.

## Run Manually If Needed

Run commands from the project root so generated artifacts are written to the
shared project folders:

```powershell
python prototypes/team2_pipeline/TermProject_team2.py --mode simulate --n_items 20 --out_dir results_smoke
```

Old web experiment shape:

```powershell
python prototypes/team2_pipeline/TermProject_team2.py --mode run_web_grid --n_items 100 --grid_eval_items 20 --out_dir results_real_talkie --headless
```

## Modes

| Mode | Purpose |
|---|---|
| `simulate` | Fast mock Talkie pipeline verification |
| `simulate_grid` | Mock Talkie hyperparameter grid search |
| `rewrite_only` | Generate rewritten prompts without Talkie calls |
| `prepare_manual` | Export manual Talkie input sheet |
| `evaluate_manual` | Evaluate manually pasted Talkie responses |
| `run_web` | Use Playwright to drive the Talkie web UI |
| `run_web_grid` | Real Talkie web run with Autoencoder grid search |

## Notes

- `run_web` and `run_web_grid` require Playwright.
- The script writes generated artifacts to root-level `data/`, `input_data/`, `cache/`, and `results*/` directories.
- Do not use this file as the main project architecture.
- Use it only to recover ideas, examples, or discarded prototype behavior.
