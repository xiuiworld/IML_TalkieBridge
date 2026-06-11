"""Standalone cached live demo generation for presentation use."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Sequence

from talkie_bridge.data_schema import json_loads


DEFAULT_DEMO_CASES = (
    {
        "comparison": "proposed_vs_raw",
        "winner_condition": "proposed",
        "title": "Era-neutral rewrite improves a raw modern prompt",
        "takeaway": "This single cached example shows the full project pipeline without any live API call.",
    },
)


def write_cached_live_demo(
    path: Path,
    *,
    prepared_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    judge_rows: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    max_examples: int = 1,
) -> None:
    examples = build_demo_examples(
        prepared_rows=prepared_rows,
        response_rows=response_rows,
        judge_rows=judge_rows,
        max_examples=max_examples,
    )
    if not examples:
        raise ValueError("No matching cached judge examples were found for the live demo.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_demo_html(example=examples[0], metrics=[_metric_payload(row) for row in metric_rows]),
        encoding="utf-8",
    )


def build_demo_examples(
    *,
    prepared_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    judge_rows: Sequence[dict[str, Any]],
    max_examples: int = 1,
) -> list[dict[str, Any]]:
    prompts = {
        (str(row.get("item_id", "")), str(row.get("condition", ""))): row
        for row in prepared_rows
    }
    responses = {
        (str(row.get("item_id", "")), str(row.get("condition", ""))): row
        for row in response_rows
    }
    selected: list[dict[str, Any]] = []
    for target in DEFAULT_DEMO_CASES:
        row = _first_matching_judge_row(
            judge_rows,
            comparison=target["comparison"],
            winner_condition=target["winner_condition"],
            used_item_ids={str(example["item_id"]) for example in selected},
        )
        if row is None:
            continue
        selected.append(_example_payload(row, target, prompts=prompts, responses=responses))
        if len(selected) >= max_examples:
            break
    if len(selected) < max_examples:
        used = {str(example["item_id"]) for example in selected}
        for row in judge_rows:
            if str(row.get("item_id", "")) in used:
                continue
            selected.append(
                _example_payload(
                    row,
                    {
                        "comparison": str(row.get("comparison", "")),
                        "winner_condition": str(row.get("winner_condition", "")),
                        "title": str(row.get("comparison", "Demo example")),
                        "takeaway": "This pair shows what the judge saw during the completed run.",
                    },
                    prompts=prompts,
                    responses=responses,
                )
            )
            used.add(str(row.get("item_id", "")))
            if len(selected) >= max_examples:
                break
    return selected


def _first_matching_judge_row(
    rows: Sequence[dict[str, Any]],
    *,
    comparison: str,
    winner_condition: str,
    used_item_ids: set[str],
) -> dict[str, Any] | None:
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if item_id in used_item_ids:
            continue
        if row.get("comparison") == comparison and row.get("winner_condition") == winner_condition:
            return row
    return None


def _example_payload(
    row: dict[str, Any],
    target: dict[str, str],
    *,
    prompts: dict[tuple[str, str], dict[str, Any]],
    responses: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    item_id = str(row.get("item_id", ""))
    proposed_prompt = prompts.get((item_id, "proposed"), {})
    raw_prompt = prompts.get((item_id, "raw"), {})
    condition_a = str(row.get("condition_a", ""))
    condition_b = str(row.get("condition_b", ""))
    response_a = responses.get((item_id, condition_a), {})
    response_b = responses.get((item_id, condition_b), {})
    return {
        "title": target["title"],
        "takeaway": target["takeaway"],
        "item_id": item_id,
        "domain": str(row.get("domain", "")),
        "comparison": str(row.get("comparison", "")),
        "winner_condition": str(row.get("winner_condition", "")),
        "original_question": str(raw_prompt.get("open_question") or raw_prompt.get("original_question") or row.get("question", "")),
        "rewritten_question": str(proposed_prompt.get("rewritten_question") or row.get("question", "")),
        "expected_mechanism": str(row.get("expected_mechanism", "")),
        "detected_terms": _list_value(proposed_prompt.get("detected_terms", [])),
        "mapped_primitives": _list_value(proposed_prompt.get("mapped_primitives", [])),
        "response_a_condition": condition_a,
        "response_b_condition": condition_b,
        "response_a": str(row.get("response_a") or response_a.get("raw_response", "")),
        "response_b": str(row.get("response_b") or response_b.get("raw_response", "")),
        "winner": str(row.get("winner", "")),
        "rationale": str(row.get("rationale", "")),
        "scores": {
            "task relevance": _score_pair(row, "task_relevance"),
            "functional correctness": _score_pair(row, "functional_correctness"),
            "era neutrality": _score_pair(row, "era_neutrality"),
            "anachronism handling": _score_pair(row, "anachronism_handling"),
            "usefulness": _score_pair(row, "usefulness"),
            "leakage risk": _score_pair(row, "leakage_risk"),
        },
    }


def _metric_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison": str(row.get("comparison", "")),
        "condition": str(row.get("condition", "")),
        "baseline_condition": str(row.get("baseline_condition", "")),
        "n_pairs": int(float(row.get("n_pairs") or 0)),
        "condition_wins": int(float(row.get("condition_wins") or 0)),
        "baseline_wins": int(float(row.get("baseline_wins") or 0)),
        "ties": int(float(row.get("ties") or 0)),
        "condition_win_rate_excluding_ties": _float_value(row.get("condition_win_rate_excluding_ties")),
    }


def _score_pair(row: dict[str, Any], metric: str) -> dict[str, int]:
    return {
        "A": int(float(row.get(f"{metric}_a") or 0)),
        "B": int(float(row.get(f"{metric}_b") or 0)),
    }


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _list_value(value: Any) -> list[str]:
    loaded = json_loads(value, value)
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    if isinstance(loaded, str) and loaded.strip():
        return [loaded.strip()]
    return []


def _render_demo_html(*, example: dict[str, Any], metrics: Sequence[dict[str, Any]]) -> str:
    del metrics
    winner_is_proposed = example["winner_condition"] == "proposed"
    highlighted_question = _highlight_terms(example["original_question"], example["detected_terms"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TalkieBridge Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #596a75;
      --page: #f3f6f8;
      --panel: #ffffff;
      --line: #d6dfe6;
      --blue: #245b93;
      --green: #21734f;
      --gold: #8b6714;
      --red: #9d3f3f;
      --soft-blue: #eef6fc;
      --soft-green: #eef8f2;
      --soft-gold: #fff8e8;
      --shadow: 0 18px 36px rgba(23, 32, 38, 0.10);
      --shadow-active: 0 22px 44px rgba(36, 91, 147, 0.18);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.42;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 22px clamp(18px, 4vw, 48px);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(26px, 3.2vw, 44px);
      line-height: 1.06;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 1040px;
      font-size: 16px;
    }}
    main {{
      width: min(1280px, calc(100% - 32px));
      margin: 18px auto 34px;
    }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }}
    .step-dot {{
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 700;
      color: var(--muted);
      min-height: 44px;
    }}
    .step-dot.active {{
      border-color: var(--blue);
      color: var(--blue);
      background: var(--soft-blue);
    }}
    .stage {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      overflow: hidden;
    }}
    .stage-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .zone {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
      transition: transform 420ms ease, opacity 420ms ease, border-color 420ms ease, background-color 420ms ease, box-shadow 420ms ease;
      position: relative;
    }}
    .zone.clickable {{
      cursor: pointer;
    }}
    .zone.clickable:hover {{
      border-color: var(--blue);
      box-shadow: var(--shadow-active);
      transform: translateY(-2px);
    }}
    .zone.locked {{
      opacity: 0.28;
      cursor: not-allowed;
    }}
    .zone.locked:hover {{
      border-color: var(--line);
      box-shadow: none;
      transform: none;
    }}
    .tap-hint {{
      position: absolute;
      right: 12px;
      top: 12px;
      background: var(--blue);
      color: #ffffff;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    .zone h2 {{
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .eyebrow {{
      display: flex;
      gap: 7px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .pill {{
      border: 1px solid var(--line);
      background: #f7fafc;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }}
    .box {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      padding: 11px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .term-hit {{
      border-radius: 4px;
      padding: 1px 3px;
      transition: background-color 360ms ease, box-shadow 360ms ease;
    }}
    body.reveal-detect .term-hit {{
      background: #fff1a8;
      box-shadow: 0 0 0 3px rgba(255, 213, 80, 0.35);
      animation: pulse 900ms ease 1;
    }}
    @keyframes pulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(255, 213, 80, 0.0); }}
      45% {{ box-shadow: 0 0 0 7px rgba(255, 213, 80, 0.45); }}
      100% {{ box-shadow: 0 0 0 3px rgba(255, 213, 80, 0.35); }}
    }}
    .detect-strip {{
      opacity: 0;
      transform: translateY(-10px);
      transition: transform 420ms ease, opacity 420ms ease;
    }}
    body.reveal-detect .detect-strip {{
      opacity: 1;
      transform: translateY(0);
    }}
    .box + .caption, .caption + .box, .terms + .caption {{
      margin-top: 10px;
    }}
    .caption {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .terms {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .terms li {{
      border: 1px solid var(--line);
      background: var(--soft-blue);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
    }}
    .rewrite-zone, .talkie-zone, .judge-zone {{
      opacity: 0.18;
      transform: translateY(18px);
    }}
    body.reveal-rewrite .rewrite-zone,
    body.reveal-talkie .talkie-zone,
    body.reveal-judge .judge-zone {{
      opacity: 1;
      transform: translateY(0);
      border-color: rgba(36, 91, 147, 0.45);
    }}
    .rewrite-arrow {{
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      color: var(--blue);
      font-weight: 700;
    }}
    .rewrite-arrow::before,
    .rewrite-arrow::after {{
      content: "";
      height: 2px;
      background: var(--line);
    }}
    body.reveal-rewrite .rewrite-arrow::before,
    body.reveal-rewrite .rewrite-arrow::after {{
      background: var(--blue);
    }}
    .responses {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .answer {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }}
    .answer h3 {{
      margin: 0;
      padding: 10px 12px;
      background: #f5f8fa;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      letter-spacing: 0;
    }}
    .answer p {{
      margin: 0;
      padding: 12px;
      min-height: 116px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .answer {{
      opacity: 0;
      transform: translateX(-20px);
      transition: transform 480ms ease, opacity 480ms ease;
    }}
    .answer:nth-child(2) {{
      transform: translateX(20px);
      transition-delay: 120ms;
    }}
    body.reveal-talkie .answer {{
      opacity: 1;
      transform: translateX(0);
    }}
    .winner {{
      border-left: 5px solid {("var(--green)" if winner_is_proposed else "var(--gold)")};
      background: {("var(--soft-green)" if winner_is_proposed else "var(--soft-gold)")};
      border-radius: 6px;
      padding: 12px;
      font-weight: 700;
      margin-bottom: 12px;
    }}
    .rationale {{
      margin-bottom: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 7px 6px;
      vertical-align: top;
    }}
    th {{
      background: #f5f8fa;
      color: var(--muted);
      font-weight: 700;
    }}
    .metric-wrap {{
      overflow-x: auto;
      max-width: 100%;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .stage-grid, .flow {{
        grid-template-columns: 1fr;
      }}
      .responses {{
        grid-template-columns: 1fr;
      }}
      table {{
        min-width: 520px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TalkieBridge Demo</h1>
    <p class="subtitle">Click the next highlighted card to run the pipeline: detect a modern term, rewrite it into era-neutral primitives, compare Talkie outputs, then reveal the blind judge result.</p>
  </header>
  <main>
    <section class="flow" aria-label="Pipeline overview">
      <div class="step-dot" data-step-dot="0">1. Modern input</div>
      <div class="step-dot" data-step-dot="1">2. Detect + rewrite</div>
      <div class="step-dot" data-step-dot="2">3. Talkie outputs</div>
      <div class="step-dot" data-step-dot="3">4. Blind judge</div>
    </section>
    <section class="stage">
      <div class="stage-grid">
      <article class="zone input-zone">
        <div class="eyebrow">
          <span class="pill">{_e(example["item_id"])}</span>
          <span class="pill">{_e(example["domain"])}</span>
          <span class="pill">{_e(example["comparison"])}</span>
        </div>
        <h2>Input</h2>
        <div class="caption">Original open-ended question</div>
        <div class="box">{highlighted_question}</div>
        <div class="detect-strip">
          <div class="caption">Detected modern terms</div>
          <ul class="terms">{_term_items(example["detected_terms"])}</ul>
        </div>
      </article>
      <article class="zone rewrite-zone locked" data-step-card="1" role="button" tabindex="0" aria-label="Run era-neutral rewrite">
        <span class="tap-hint">Click to rewrite</span>
        <h2>Rewrite</h2>
        <div class="caption">Primitive bottleneck terms</div>
        <ul class="terms">{_term_items(example["mapped_primitives"])}</ul>
        <div class="rewrite-arrow">rewrite</div>
        <div class="caption">Primitive rewrite</div>
        <div class="box">{_e(example["rewritten_question"])}</div>
      </article>
      <article class="zone talkie-zone wide locked" data-step-card="2" role="button" tabindex="0" aria-label="Run cached Talkie comparison">
        <span class="tap-hint">Click to compare</span>
        <h2>Talkie Outputs</h2>
        <div class="caption">Two prompts are sent to the same fixed Talkie model; the outputs are shown below.</div>
        <div class="responses">
          <section class="answer">
            <h3>Answer A: {_e(example["response_a_condition"])} prompt</h3>
            <p>{_e(example["response_a"])}</p>
          </section>
          <section class="answer">
            <h3>Answer B: {_e(example["response_b_condition"])} prompt</h3>
            <p>{_e(example["response_b"])}</p>
          </section>
        </div>
      </article>
      <article class="zone judge-zone wide locked" data-step-card="3" role="button" tabindex="0" aria-label="Reveal blind judge result">
        <span class="tap-hint">Click to judge</span>
        <h2>Judge Result</h2>
        <div class="winner">Judge picked {_e(example["winner"])}. Winner after unblinding: {_e(example["winner_condition"])}.</div>
        <div class="caption">Judge rationale</div>
        <div class="box rationale">{_e(example["rationale"])}</div>
        <div class="caption">Rubric scores</div>
        <div class="metric-wrap">{_score_table(example["scores"])}</div>
      </article>
      </div>
    </section>
  </main>
  <script>
    const maxStep = 3;
    let step = 0;
    const body = document.body;
    const cards = Array.from(document.querySelectorAll('[data-step-card]'));

    function setStep(value) {{
      step = Math.max(0, Math.min(maxStep, value));
      body.classList.toggle('reveal-detect', step >= 1);
      body.classList.toggle('reveal-rewrite', step >= 1);
      body.classList.toggle('reveal-talkie', step >= 2);
      body.classList.toggle('reveal-judge', step >= 3);
      document.querySelectorAll('[data-step-dot]').forEach((item) => {{
        const itemStep = Number(item.getAttribute('data-step-dot'));
        item.classList.toggle('active', itemStep <= step);
      }});
      cards.forEach((card) => {{
        const cardStep = Number(card.getAttribute('data-step-card'));
        const isUnlocked = cardStep <= step + 1;
        const isNext = cardStep === step + 1;
        card.classList.toggle('locked', !isUnlocked);
        card.classList.toggle('clickable', isUnlocked);
        const hint = card.querySelector('.tap-hint');
        if (hint) {{
          hint.style.display = isNext && step < maxStep ? 'inline-block' : 'none';
        }}
      }});
    }}

    cards.forEach((card) => {{
      card.addEventListener('click', () => {{
        const cardStep = Number(card.getAttribute('data-step-card'));
        if (cardStep <= step + 1) {{
          setStep(cardStep);
        }}
      }});
      card.addEventListener('keydown', (event) => {{
        if (event.key === 'Enter' || event.key === ' ') {{
          event.preventDefault();
          card.click();
        }}
      }});
    }});
    setStep(0);
  </script>
</body>
</html>
"""


def _score_table(scores: dict[str, dict[str, int]]) -> str:
    rows = ["<table><tr><th>Rubric</th><th>A</th><th>B</th></tr>"]
    for metric, score in scores.items():
        rows.append(f"<tr><td>{_e(metric)}</td><td>{score['A']}</td><td>{score['B']}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _metric_table(metrics: Sequence[dict[str, Any]]) -> str:
    rows = ["<table><tr><th>Comparison</th><th>Wins</th><th>Losses</th><th>Ties</th><th>Win rate</th></tr>"]
    for metric in metrics:
        rows.append(
            "<tr>"
            f"<td>{_e(metric['comparison'])}</td>"
            f"<td>{metric['condition_wins']}</td>"
            f"<td>{metric['baseline_wins']}</td>"
            f"<td>{metric['ties']}</td>"
            f"<td>{100.0 * metric['condition_win_rate_excluding_ties']:.1f}%</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _term_items(values: Sequence[str]) -> str:
    terms = list(values) or ["None recorded"]
    return "".join(f"<li>{_e(value)}</li>" for value in terms)


def _highlight_terms(text: Any, terms: Sequence[str]) -> str:
    escaped = _e(text)
    sorted_terms = [term for term in sorted(set(terms), key=len, reverse=True) if term.strip()]
    for term in sorted_terms:
        pattern = re.compile(re.escape(_e(term)), flags=re.IGNORECASE)
        escaped = pattern.sub(lambda match: f'<mark class="term-hit">{match.group(0)}</mark>', escaped)
    return escaped


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)
