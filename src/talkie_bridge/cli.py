"""Command line entry point for the TalkieBridge experiment pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

from talkie_bridge.analysis import (
    add_length_control_diagnostics,
    add_proposed_difference_diagnostics,
    compute_dataset_quality,
    write_dataset_quality_markdown,
    write_demo_html,
    write_qualitative_examples,
)
from talkie_bridge.autoencoder import train_select_autoencoder
from talkie_bridge.clients import UnofficialTalkieApiClient, load_manual_response_records
from talkie_bridge.data_schema import (
    CONDITIONS,
    LABELS,
    DatasetItem,
    RunPaths,
    infer_split,
    json_dumps,
    read_csv_dicts,
    read_json,
    read_jsonl,
    resolve_under,
    write_csv_dicts,
    write_json,
    write_jsonl,
)
from talkie_bridge.detector import BernoulliModernTermClassifier, build_modern_token_examples
from talkie_bridge.live_demo import write_cached_live_demo
from talkie_bridge.metrics import (
    compute_component_metrics,
    compute_condition_metrics,
    compute_domain_metrics,
    compute_key_comparisons,
    compute_paired_tests,
)
from talkie_bridge.open_ended import (
    build_judge_pair_rows,
    build_open_ended_response_rows,
    compute_judge_pairwise_metrics,
    item_with_open_question,
    judge_input_rows,
    judge_integrity_rows,
    open_ended_prompt_row_for_item,
    parse_judge_rows,
    write_response_quality_report,
)
from talkie_bridge.primitive import (
    build_dictionaries_from_dataset,
    generate_dataset,
)
from talkie_bridge.prompting import build_mcq_prompt, normalize_choice, prompt_hash
from talkie_bridge.reports import write_report
from talkie_bridge.rewriter import build_generator


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = RunPaths.from_root(
        Path.cwd(),
        data_dir=args.data_dir,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
    )
    paths.ensure()

    if args.command == "init-mock-data":
        dataset_path = write_mock_dataset(args, paths)
        print(f"Mock dataset: {dataset_path}")
        return 0

    if args.command == "rewrite-only":
        prompt_rows = prepare_prompt_rows(args, paths)
        write_prepared_artifacts(paths, prompt_rows)
        print(f"Prepared prompts: {paths.out_dir / 'prepared_prompts.csv'}")
        return 0

    if args.command == "prepare-manual":
        prompt_rows = prepare_prompt_rows(args, paths)
        write_prepared_artifacts(paths, prompt_rows)
        manual_path = write_manual_sheet(paths, prompt_rows)
        print(f"Manual Talkie sheet: {manual_path}")
        return 0

    if args.command == "prepare-open-ended":
        prompt_rows = prepare_open_ended_prompt_rows(args, paths)
        write_open_ended_prepared_artifacts(paths, prompt_rows)
        manual_path = write_open_ended_manual_sheet(paths, prompt_rows)
        print(f"Open-ended Talkie sheet: {manual_path}")
        return 0

    if args.command == "evaluate-manual":
        if not args.allow_mock_dictionary:
            reject_mock_dictionary_paths(args)
        prompt_rows = prepare_prompt_rows(args, paths)
        write_prepared_artifacts(paths, prompt_rows)
        manual_path = Path(args.manual_response_csv)
        if not manual_path.is_absolute():
            manual_path = paths.root / manual_path
            if not manual_path.exists():
                manual_path = paths.input_dir / Path(args.manual_response_csv).name
        responses = load_manual_response_records(manual_path)
        result_rows = build_result_rows(prompt_rows, responses, provider="manual_talkie_csv")
        enforce_manual_integrity(result_rows, args)
        write_evaluation_outputs(paths, prompt_rows, result_rows, provider="manual_talkie_csv", seed=args.seed)
        print(f"Report: {paths.out_dir / 'report.md'}")
        return 0

    if args.command == "evaluate-open-ended-manual":
        if not args.allow_mock_dictionary:
            reject_mock_dictionary_paths(args)
        prompt_rows = prepare_open_ended_prompt_rows(args, paths)
        write_open_ended_prepared_artifacts(paths, prompt_rows)
        manual_path = Path(args.manual_response_csv)
        if not manual_path.is_absolute():
            manual_path = paths.root / manual_path
            if not manual_path.exists():
                manual_path = paths.input_dir / Path(args.manual_response_csv).name
        responses = load_manual_response_records(manual_path)
        response_rows = build_open_ended_response_rows(prompt_rows, responses, provider="manual_talkie_csv")
        enforce_manual_integrity(response_rows, args)
        write_open_ended_response_outputs(paths, prompt_rows, response_rows, provider="manual_talkie_csv", seed=args.seed)
        print(f"Open-ended judge sheet: {paths.input_dir / 'open_ended_judge_input_sheet.csv'}")
        return 0

    if args.command == "run-api":
        if args.provider != "unofficial-api":
            raise ValueError("Only --provider unofficial-api is supported for run-api.")
        prompt_rows = prepare_prompt_rows(args, paths)
        write_prepared_artifacts(paths, prompt_rows)
        if not args.allow_mock_dictionary:
            reject_mock_dictionary_paths(args)
        progress = ApiRunProgress(total=len(prompt_rows), label="MCQ Talkie API")
        client = UnofficialTalkieApiClient(
            cache_path=paths.cache_dir / "talkie_unofficial_api_cache.jsonl",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            request_delay=args.request_delay,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            retry_max_delay=args.retry_max_delay,
            event_callback=progress.client_event,
        )
        progress.begin()
        responses: dict[tuple[str, str], dict[str, str]] = {}
        for index, row in enumerate(prompt_rows, start=1):
            progress.start_item(index, row)
            answer = client.ask(str(row["prompt"]))
            progress.finish_item(index, row, answer)
            responses[(str(row["item_id"]), str(row["condition"]))] = {
                "raw_response": str(answer.get("raw_response", "")),
                "prompt_hash": str(row.get("prompt_hash", "")),
            }
        progress.finish_run()
        result_rows = build_result_rows(prompt_rows, responses, provider="unofficial_talkie_api")
        write_evaluation_outputs(paths, prompt_rows, result_rows, provider="unofficial_talkie_api", seed=args.seed)
        print(f"Report: {paths.out_dir / 'report.md'}")
        return 0

    if args.command == "run-open-ended-api":
        if args.provider != "unofficial-api":
            raise ValueError("Only --provider unofficial-api is supported for run-open-ended-api.")
        prompt_rows = prepare_open_ended_prompt_rows(args, paths)
        write_open_ended_prepared_artifacts(paths, prompt_rows)
        if not args.allow_mock_dictionary:
            reject_mock_dictionary_paths(args)
        progress = ApiRunProgress(total=len(prompt_rows), label="Open-ended Talkie API")
        client = UnofficialTalkieApiClient(
            cache_path=paths.cache_dir / "talkie_unofficial_api_cache_open_ended.jsonl",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            request_delay=args.request_delay,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            retry_max_delay=args.retry_max_delay,
            event_callback=progress.client_event,
        )
        progress.begin()
        responses: dict[tuple[str, str], dict[str, str]] = {}
        for index, row in enumerate(prompt_rows, start=1):
            progress.start_item(index, row)
            answer = client.ask(str(row["prompt"]))
            progress.finish_item(index, row, answer)
            responses[(str(row["item_id"]), str(row["condition"]))] = {
                "raw_response": str(answer.get("raw_response", "")),
                "prompt_hash": str(row.get("prompt_hash", "")),
            }
        progress.finish_run()
        response_rows = build_open_ended_response_rows(prompt_rows, responses, provider="unofficial_talkie_api")
        write_open_ended_response_outputs(paths, prompt_rows, response_rows, provider="unofficial_talkie_api", seed=args.seed)
        print(f"Open-ended responses: {paths.out_dir / 'open_ended_responses.csv'}")
        print(f"Open-ended judge sheet: {paths.input_dir / 'open_ended_judge_input_sheet.csv'}")
        return 0

    if args.command == "prepare-open-ended-judge":
        response_path = resolve_run_file(paths, args.open_ended_response_csv, default_dir=paths.out_dir)
        response_rows = read_csv_dicts(response_path)
        pair_rows = build_judge_pair_rows(response_rows, seed=args.seed)
        write_open_ended_judge_sheets(paths, pair_rows)
        print(f"Open-ended judge sheet: {paths.input_dir / 'open_ended_judge_input_sheet.csv'}")
        return 0

    if args.command == "evaluate-open-ended-judge":
        pair_path = resolve_run_file(paths, args.open_ended_judge_pairs_csv, default_dir=paths.out_dir)
        judge_path = resolve_run_file(paths, args.judge_response_csv, default_dir=paths.input_dir)
        pair_rows = read_csv_dicts(pair_path)
        judge_records = load_judge_records(judge_path)
        parsed_rows = parse_judge_rows(pair_rows, judge_records)
        if not args.allow_hash_mismatch:
            enforce_judge_integrity(parsed_rows)
        write_open_ended_judge_outputs(paths, parsed_rows)
        print(f"Open-ended response quality report: {paths.out_dir / 'open_ended_response_quality.md'}")
        return 0

    if args.command == "demo":
        prompt_rows = prepare_prompt_rows(args, paths)
        write_prepared_artifacts(paths, prompt_rows)
        out_path = resolve_under(paths.root, args.out)
        write_demo_html(out_path, prompt_rows, args.item_id)
        print(f"Demo: {out_path}")
        return 0

    if args.command == "demo-open-ended":
        prepared_path = resolve_run_file(paths, args.prepared_csv, default_dir=paths.out_dir)
        responses_path = resolve_run_file(paths, args.responses_csv, default_dir=paths.out_dir)
        judge_scores_path = resolve_run_file(paths, args.judge_scores_csv, default_dir=paths.out_dir)
        metrics_path = resolve_run_file(paths, args.metrics_csv, default_dir=paths.out_dir)
        out_path = resolve_under(paths.root, args.out)
        write_cached_live_demo(
            out_path,
            prepared_rows=read_csv_dicts(prepared_path),
            response_rows=read_csv_dicts(responses_path),
            judge_rows=read_csv_dicts(judge_scores_path),
            metric_rows=read_csv_dicts(metrics_path),
            max_examples=args.max_examples,
        )
        print(f"Open-ended live demo: {out_path}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


class ApiRunProgress:
    def __init__(self, *, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.started_at = 0.0
        self.item_started_at = 0.0
        self.completed = 0
        self.current_item = ""
        self.cache_hit = False
        self.response_chars = 0

    def begin(self) -> None:
        self.started_at = time.monotonic()
        print(f"[start] {self.label}: {self.total} prompts", flush=True)

    def start_item(self, index: int, row: dict[str, Any]) -> None:
        self.item_started_at = time.monotonic()
        self.current_item = f"{row.get('item_id', '')}/{row.get('condition', '')}"
        self.cache_hit = False
        self.response_chars = 0
        percent = (index - 1) / self.total * 100 if self.total else 0.0
        print(f"[start] {index}/{self.total} ({percent:.1f}%) {self.current_item}", flush=True)

    def client_event(self, event: dict[str, Any]) -> None:
        name = event.get("event")
        if name == "cache_hit":
            self.cache_hit = True
        elif name == "response_complete":
            self.response_chars = int(event.get("response_chars") or 0)
        elif name == "retry_wait":
            reason = str(event.get("reason") or "request failed")
            attempt = event.get("attempt")
            max_retries = event.get("max_retries")
            delay = float(event.get("delay_seconds") or 0.0)
            print(
                f"[retry] {self.current_item}: {reason}; "
                f"retry {attempt}/{max_retries} after {_format_duration(delay)}",
                flush=True,
            )
        elif name == "request_failed":
            reason = str(event.get("reason") or "request failed")
            print(f"[failed] {self.current_item}: {reason}", flush=True)
        elif name == "request_delay":
            delay = float(event.get("delay_seconds") or 0.0)
            if delay >= 5:
                print(f"[delay] {self.current_item}: waiting {_format_duration(delay)} before next request", flush=True)

    def finish_item(self, index: int, row: dict[str, Any], answer: dict[str, Any]) -> None:
        del row
        self.completed = index
        item_elapsed = time.monotonic() - self.item_started_at
        elapsed = time.monotonic() - self.started_at
        remaining = max(self.total - self.completed, 0)
        avg = elapsed / self.completed if self.completed else 0.0
        eta = avg * remaining
        percent = self.completed / self.total * 100 if self.total else 100.0
        cache_hit = bool(answer.get("cache_hit", self.cache_hit))
        response_chars = int(answer.get("response_chars") or self.response_chars or len(str(answer.get("raw_response", ""))))
        print(
            f"[done] {self.completed}/{self.total} ({percent:.1f}%) {self.current_item}; "
            f"item={_format_duration(item_elapsed)}; elapsed={_format_duration(elapsed)}; "
            f"eta={_format_duration(eta)}; cache={'yes' if cache_hit else 'no'}; chars={response_chars}",
            flush=True,
        )

    def finish_run(self) -> None:
        elapsed = time.monotonic() - self.started_at
        print(f"[complete] {self.label}: {self.completed}/{self.total} prompts in {_format_duration(elapsed)}", flush=True)


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Era-neutral prompt rewriting and Talkie evaluation pipeline.")
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--seed", type=int, default=13)
    parent.add_argument("--target-year", type=int, default=1930)
    parent.add_argument("--dataset-jsonl", default="data/generated_questions.jsonl")
    parent.add_argument("--limit", type=int, default=None, help="Optional item limit when reading the dataset.")
    parent.add_argument("--eval-split", choices=["all", "train", "dev", "test"], default="test")
    parent.add_argument("--concept-dictionary-json", default="")
    parent.add_argument("--primitive-dictionary-json", default="")
    parent.add_argument("--data-dir", default="data")
    parent.add_argument("--input-dir", default="input_data")
    parent.add_argument("--out-dir", default="results")
    parent.add_argument("--cache-dir", default="cache")

    sub = parser.add_subparsers(dest="command", required=True)
    init_mock = sub.add_parser("init-mock-data", parents=[parent], help="Write an explicit mock dataset file for schema reference.")
    init_mock.add_argument("--n-items", type=int, default=3)
    init_mock.add_argument("--force", action="store_true", help="Overwrite an existing dataset file.")
    sub.add_parser("rewrite-only", parents=[parent], help="Generate rewrites and prompts without Talkie responses.")
    sub.add_parser("prepare-manual", parents=[parent], help="Generate a CSV sheet for manual Talkie response collection.")
    sub.add_parser("prepare-open-ended", parents=[parent], help="Generate open-ended Talkie prompts and a manual response sheet.")
    eval_manual = sub.add_parser("evaluate-manual", parents=[parent], help="Evaluate responses pasted into a manual CSV.")
    eval_manual.add_argument("--manual-response-csv", default="input_data/manual_talkie_input_sheet.csv")
    eval_manual.add_argument("--allow-hash-mismatch", action="store_true")
    eval_manual.add_argument("--allow-missing-responses", action="store_true")
    eval_manual.add_argument("--allow-mock-dictionary", action="store_true")
    eval_open = sub.add_parser("evaluate-open-ended-manual", parents=[parent], help="Evaluate collected open-ended Talkie responses and prepare blind judge pairs.")
    eval_open.add_argument("--manual-response-csv", default="input_data/open_ended_talkie_input_sheet.csv")
    eval_open.add_argument("--allow-hash-mismatch", action="store_true")
    eval_open.add_argument("--allow-missing-responses", action="store_true")
    eval_open.add_argument("--allow-mock-dictionary", action="store_true")

    run_api = sub.add_parser("run-api", parents=[parent], help="Collect responses through the optional unofficial Talkie API.")
    run_api.add_argument("--provider", choices=["unofficial-api"], required=True)
    run_api.add_argument("--temperature", type=float, default=0.0)
    run_api.add_argument("--max-tokens", type=int, default=8)
    run_api.add_argument("--timeout", type=int, default=120)
    run_api.add_argument("--request-delay", type=float, default=1.0, help="Seconds to wait after each uncached API response.")
    run_api.add_argument("--max-retries", type=int, default=6, help="Maximum retry attempts for 429 and transient server errors.")
    run_api.add_argument("--retry-base-delay", type=float, default=30.0, help="Initial retry delay in seconds when the API does not send Retry-After.")
    run_api.add_argument("--retry-max-delay", type=float, default=300.0, help="Maximum retry delay in seconds.")
    run_api.add_argument("--allow-mock-dictionary", action="store_true")
    run_open = sub.add_parser("run-open-ended-api", parents=[parent], help="Collect open-ended Talkie responses through the optional unofficial Talkie API.")
    run_open.add_argument("--provider", choices=["unofficial-api"], required=True)
    run_open.add_argument("--temperature", type=float, default=0.0)
    run_open.add_argument("--max-tokens", type=int, default=96)
    run_open.add_argument("--timeout", type=int, default=120)
    run_open.add_argument("--request-delay", type=float, default=1.0, help="Seconds to wait after each uncached API response.")
    run_open.add_argument("--max-retries", type=int, default=6, help="Maximum retry attempts for 429 and transient server errors.")
    run_open.add_argument("--retry-base-delay", type=float, default=30.0, help="Initial retry delay in seconds when the API does not send Retry-After.")
    run_open.add_argument("--retry-max-delay", type=float, default=300.0, help="Maximum retry delay in seconds.")
    run_open.add_argument("--allow-mock-dictionary", action="store_true")
    prep_judge = sub.add_parser("prepare-open-ended-judge", parents=[parent], help="Create blind judge sheets from collected open-ended responses.")
    prep_judge.add_argument("--open-ended-response-csv", default="results/open_ended_responses.csv")
    eval_judge = sub.add_parser("evaluate-open-ended-judge", parents=[parent], help="Parse filled judge outputs and compute pairwise response-quality metrics.")
    eval_judge.add_argument("--open-ended-judge-pairs-csv", default="results/open_ended_judge_pairs_unblinded.csv")
    eval_judge.add_argument("--judge-response-csv", default="input_data/open_ended_judge_input_sheet.csv")
    eval_judge.add_argument("--allow-hash-mismatch", action="store_true")
    demo = sub.add_parser("demo", parents=[parent], help="Write a static HTML demo for one dataset item.")
    demo.add_argument("--item-id", required=True)
    demo.add_argument("--out", default="results/demo.html")
    demo_open = sub.add_parser("demo-open-ended", parents=[parent], help="Write a static cached live demo for the completed open-ended run.")
    demo_open.add_argument("--prepared-csv", default="results/open_ended_prepared_prompts.csv")
    demo_open.add_argument("--responses-csv", default="results/open_ended_responses.csv")
    demo_open.add_argument("--judge-scores-csv", default="results/open_ended_judge_scores.csv")
    demo_open.add_argument("--metrics-csv", default="results/open_ended_pairwise_metrics.csv")
    demo_open.add_argument("--max-examples", type=int, default=1)
    demo_open.add_argument("--out", default="results/open_ended_live_demo.html")
    return parser


def prepare_prompt_rows(args: argparse.Namespace, paths: RunPaths) -> list[dict[str, Any]]:
    dataset = load_dataset(paths, dataset_jsonl=args.dataset_jsonl, limit=args.limit, seed=args.seed)
    train_items = [item for item in dataset if item.split == "train"]
    modern_dictionary, primitive_dictionary, resource_source = load_generation_resources(args, paths, train_items)
    _full_modern_dictionary, validation_primitive_dictionary = build_dictionaries_from_dataset(dataset)
    if not modern_dictionary or not primitive_dictionary:
        raise ValueError("Train split must provide enough gold_anachronism_terms, forbidden_terms, required_primitives, and primitive_phrase to build generation resources.")
    if not validation_primitive_dictionary:
        raise ValueError("Dataset must provide required_primitives and primitive_phrase for validation metrics.")
    write_json(paths.data_dir / "modern_terms_dictionary.json", modern_dictionary)
    write_json(paths.data_dir / "primitive_dictionary.json", primitive_dictionary)
    write_json(paths.data_dir / "validation_primitive_dictionary.json", validation_primitive_dictionary)
    write_json(
        paths.cache_dir / "generation_resource_source.json",
        {
            "source": resource_source,
            "concept_dictionary_json": args.concept_dictionary_json,
            "primitive_dictionary_json": args.primitive_dictionary_json,
            "note": "Generated prompts use explicit predeclared dictionaries only when CLI paths are provided; otherwise train split annotations are used.",
        },
    )
    classifier = BernoulliModernTermClassifier()
    classifier.fit(
        build_modern_token_examples(
            [item.original_question for item in train_items],
            [[*item.gold_anachronism_terms, *item.forbidden_terms] for item in train_items],
        )
    )
    autoencoder, autoencoder_selection = train_select_autoencoder(train_items + [item for item in dataset if item.split == "dev"], primitive_dictionary, seed=args.seed)
    autoencoder_info = autoencoder_selection.__dict__
    if autoencoder is not None:
        write_json(paths.cache_dir / "primitive_autoencoder.json", autoencoder.to_dict())
    write_json(paths.cache_dir / "primitive_autoencoder_selection.json", autoencoder_info)
    write_json(paths.cache_dir / "modern_token_classifier.json", classifier.to_dict())
    generator = build_generator(
        modern_dictionary,
        primitive_dictionary,
        validation_primitive_dictionary=validation_primitive_dictionary,
        token_classifier=classifier,
        autoencoder=autoencoder,
        autoencoder_info=autoencoder_info,
    )
    output_dataset = [item for item in dataset if args.eval_split == "all" or item.split == args.eval_split]
    rows: list[dict[str, Any]] = []
    for item in output_dataset:
        artifact_by_condition = {artifact.condition: artifact for artifact in generator.rewrite_item(item)}
        rows.append(prompt_row_for_item(item, condition="raw", question=item.original_question, artifact=None))
        for condition in CONDITIONS:
            if condition == "raw":
                continue
            artifact = artifact_by_condition[condition]
            rows.append(prompt_row_for_item(item, condition=condition, question=artifact.rewritten_question, artifact=artifact))
    return rows


def prepare_open_ended_prompt_rows(args: argparse.Namespace, paths: RunPaths) -> list[dict[str, Any]]:
    dataset = load_dataset(paths, dataset_jsonl=args.dataset_jsonl, limit=args.limit, seed=args.seed)
    train_items = [item for item in dataset if item.split == "train"]
    modern_dictionary, primitive_dictionary, resource_source = load_generation_resources(args, paths, train_items)
    _full_modern_dictionary, validation_primitive_dictionary = build_dictionaries_from_dataset(dataset)
    if not modern_dictionary or not primitive_dictionary:
        raise ValueError("Train split must provide enough annotations or explicit dictionaries for open-ended prompt generation.")
    if not validation_primitive_dictionary:
        raise ValueError("Dataset must provide required_primitives and primitive_phrase for validation metrics.")
    write_json(paths.data_dir / "modern_terms_dictionary.json", modern_dictionary)
    write_json(paths.data_dir / "primitive_dictionary.json", primitive_dictionary)
    write_json(paths.data_dir / "validation_primitive_dictionary.json", validation_primitive_dictionary)
    write_json(
        paths.cache_dir / "generation_resource_source.json",
        {
            "source": resource_source,
            "concept_dictionary_json": args.concept_dictionary_json,
            "primitive_dictionary_json": args.primitive_dictionary_json,
            "note": "Open-ended prompts use explicit predeclared dictionaries only when CLI paths are provided; otherwise train split annotations are used.",
        },
    )
    classifier = BernoulliModernTermClassifier()
    classifier.fit(
        build_modern_token_examples(
            [item.original_question for item in train_items],
            [[*item.gold_anachronism_terms, *item.forbidden_terms] for item in train_items],
        )
    )
    autoencoder, autoencoder_selection = train_select_autoencoder(train_items + [item for item in dataset if item.split == "dev"], primitive_dictionary, seed=args.seed)
    autoencoder_info = autoencoder_selection.__dict__
    if autoencoder is not None:
        write_json(paths.cache_dir / "primitive_autoencoder.json", autoencoder.to_dict())
    write_json(paths.cache_dir / "primitive_autoencoder_selection.json", autoencoder_info)
    write_json(paths.cache_dir / "modern_token_classifier.json", classifier.to_dict())
    generator = build_generator(
        modern_dictionary,
        primitive_dictionary,
        validation_primitive_dictionary=validation_primitive_dictionary,
        token_classifier=classifier,
        autoencoder=autoencoder,
        autoencoder_info=autoencoder_info,
    )
    output_dataset = [item for item in dataset if args.eval_split == "all" or item.split == args.eval_split]
    rows: list[dict[str, Any]] = []
    for item in output_dataset:
        open_item = item_with_open_question(item)
        artifact_by_condition = {artifact.condition: artifact for artifact in generator.rewrite_open_item(open_item)}
        rows.append(open_ended_prompt_row_for_item(open_item, condition="raw", question=open_item.original_question, artifact=None))
        for condition in CONDITIONS:
            if condition == "raw":
                continue
            artifact = artifact_by_condition[condition]
            rows.append(open_ended_prompt_row_for_item(open_item, condition=condition, question=artifact.rewritten_question, artifact=artifact))
    return rows


def load_dataset(paths: RunPaths, *, dataset_jsonl: str | Path, limit: int | None = None, seed: int = 13) -> list[DatasetItem]:
    data_path = resolve_under(paths.root, dataset_jsonl)
    rows = read_jsonl(data_path)
    if not rows:
        raise FileNotFoundError(
            f"Dataset not found or empty: {data_path}. "
            "Place your validated JSONL there, or run `python -m talkie_bridge.cli init-mock-data` for a schema example."
        )
    if limit is not None:
        rows = rows[:limit]
    dataset = [DatasetItem.from_dict(row) for row in rows]
    normalized: list[DatasetItem] = []
    for item in dataset:
        split = item.split if item.split in {"train", "dev", "test"} else infer_split(item.id, seed=seed)
        normalized.append(
            DatasetItem(
                id=item.id,
                domain=item.domain,
                original_question=item.original_question,
                choices=item.choices,
                gold_answer=item.gold_answer,
                gold_anachronism_terms=item.gold_anachronism_terms,
                forbidden_terms=item.forbidden_terms,
                required_primitives=item.required_primitives,
                primitive_phrase=item.primitive_phrase,
                human_validated=item.human_validated,
                target_year=item.target_year,
                split=split,
                concept_term=item.concept_term,
                task=item.task,
                open_question=item.open_question,
                expected_mechanism=item.expected_mechanism,
                judge_reference_points=item.judge_reference_points,
                leakage_sensitive_terms=item.leakage_sensitive_terms,
                answer_leakage_terms=item.answer_leakage_terms,
                dataset_version=item.dataset_version,
                review_status=item.review_status,
                review_notes=item.review_notes,
            )
        )
    return normalized


def load_generation_resources(
    args: argparse.Namespace,
    paths: RunPaths,
    train_items: Sequence[DatasetItem],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    concept_path = resolve_optional_path(paths.root, args.concept_dictionary_json)
    primitive_path = resolve_optional_path(paths.root, args.primitive_dictionary_json)
    if concept_path and primitive_path:
        return read_json(concept_path), read_json(primitive_path), "predeclared_dictionary"
    if concept_path or primitive_path:
        raise ValueError("Provide both --concept-dictionary-json and --primitive-dictionary-json, or neither.")

    modern_dictionary, primitive_dictionary = build_dictionaries_from_dataset(train_items)
    return modern_dictionary, primitive_dictionary, "train_split_annotations"


def resolve_optional_path(root: Path, value: str) -> Path | None:
    if not value:
        return None
    return resolve_under(root, value)


def resolve_run_file(paths: RunPaths, value: str | Path, *, default_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    root_candidate = paths.root / candidate
    default_candidate = default_dir / candidate.name
    if str(candidate).replace("\\", "/").startswith(("results/", "input_data/")):
        return default_candidate
    if root_candidate.exists():
        return root_candidate
    return default_candidate


def write_mock_dataset(args: argparse.Namespace, paths: RunPaths) -> Path:
    dataset_path = resolve_under(paths.root, args.dataset_jsonl)
    if dataset_path.exists() and not args.force:
        raise FileExistsError(f"Dataset already exists: {dataset_path}. Use --force to overwrite it.")
    split_cycle = ("train", "dev", "test")
    dataset = [
        DatasetItem(
            id=item.id,
            domain=item.domain,
            original_question=item.original_question,
            choices=item.choices,
            gold_answer=item.gold_answer,
            gold_anachronism_terms=item.gold_anachronism_terms,
            forbidden_terms=item.forbidden_terms,
            required_primitives=item.required_primitives,
            primitive_phrase=item.primitive_phrase,
            human_validated=item.human_validated,
            target_year=item.target_year,
            split=split_cycle[index % len(split_cycle)],
        )
        for index, item in enumerate(generate_dataset(n_items=args.n_items, target_year=args.target_year, seed=args.seed))
    ]
    write_jsonl(dataset_path, [item.to_dict() for item in dataset])
    modern_dictionary, primitive_dictionary = build_dictionaries_from_dataset(dataset)
    write_json(paths.data_dir / "mock_modern_terms_dictionary.json", modern_dictionary)
    write_json(paths.data_dir / "mock_primitive_dictionary.json", primitive_dictionary)
    return dataset_path


def prompt_row_for_item(item: DatasetItem, *, condition: str, question: str, artifact: Any | None) -> dict[str, Any]:
    validation = artifact.rewrite_validation if artifact else {"pass": True}
    detected_terms = artifact.detected_terms if artifact else []
    mapped_primitives = artifact.mapped_primitives if artifact else []
    repair_attempts = artifact.repair_attempts if artifact else 0
    pass_validation = artifact.pass_validation if artifact else True
    detector_token_scores = artifact.detector_token_scores if artifact else []
    model_info = artifact.model_info if artifact else {}
    prompt = build_mcq_prompt(question, item.choices)
    return {
        "item_id": item.id,
        "id": item.id,
        "domain": item.domain,
        "split": item.split,
        "condition": condition,
        "prompt": prompt,
        "prompt_hash": prompt_hash(prompt),
        "original_question": item.original_question,
        "rewritten_question": question,
        "choices": item.choices,
        "choice_A": item.choices["A"],
        "choice_B": item.choices["B"],
        "choice_C": item.choices["C"],
        "choice_D": item.choices["D"],
        "gold_answer": item.gold_answer,
        "gold_anachronism_terms": item.gold_anachronism_terms,
        "forbidden_terms": item.forbidden_terms,
        "required_primitives": item.required_primitives,
        "primitive_phrase": item.primitive_phrase,
        "human_validated": item.human_validated,
        "detected_terms": detected_terms,
        "detector_token_scores": detector_token_scores,
        "mapped_primitives": mapped_primitives,
        "model_info": model_info,
        "rewrite_validation": validation,
        "repair_attempts": repair_attempts,
        "pass_validation": pass_validation,
    }


def write_prepared_artifacts(paths: RunPaths, prompt_rows: Sequence[dict[str, Any]]) -> None:
    write_csv_dicts(paths.out_dir / "prepared_prompts.csv", list(prompt_rows))
    write_jsonl(paths.out_dir / "prepared_prompts.jsonl", [_json_ready(row) for row in prompt_rows])

    raw_rows = [row for row in prompt_rows if row["condition"] == "raw"]
    rewrite_rows = [row for row in prompt_rows if row["condition"] != "raw"]
    write_csv_dicts(paths.input_dir / "raw_4choice_questions.csv", raw_rows)
    write_jsonl(paths.input_dir / "raw_4choice_questions.jsonl", [_json_ready(row) for row in raw_rows])
    write_csv_dicts(paths.input_dir / "era_neutral_preprocessed_questions.csv", rewrite_rows)
    write_jsonl(paths.input_dir / "era_neutral_preprocessed_questions.jsonl", [_json_ready(row) for row in rewrite_rows])
    write_csv_dicts(paths.out_dir / "component_metrics.csv", compute_component_metrics(_csv_ready_rows(prompt_rows)))
    quality_rows = compute_dataset_quality(_csv_ready_rows(prompt_rows))
    add_length_control_diagnostics(_csv_ready_rows(prompt_rows), quality_rows)
    add_proposed_difference_diagnostics(_csv_ready_rows(prompt_rows), quality_rows)
    write_csv_dicts(paths.out_dir / "dataset_quality.csv", quality_rows)
    write_dataset_quality_markdown(paths.out_dir / "dataset_quality.md", quality_rows)


def write_manual_sheet(paths: RunPaths, prompt_rows: Sequence[dict[str, Any]]) -> Path:
    manual_rows = [
        {
            "item_id": row["item_id"],
            "condition": row["condition"],
            "prompt_hash": row["prompt_hash"],
            "prompt": row["prompt"],
            "raw_response_manual": "",
        }
        for row in prompt_rows
    ]
    manual_path = paths.input_dir / "manual_talkie_input_sheet.csv"
    fieldnames = ["item_id", "condition", "prompt_hash", "prompt", "raw_response_manual"]
    write_csv_dicts(manual_path, manual_rows, fieldnames)
    write_csv_dicts(paths.out_dir / "manual_talkie_input_sheet.csv", manual_rows, fieldnames)
    return manual_path


def write_open_ended_prepared_artifacts(paths: RunPaths, prompt_rows: Sequence[dict[str, Any]]) -> None:
    write_csv_dicts(paths.out_dir / "open_ended_prepared_prompts.csv", list(prompt_rows))
    write_jsonl(paths.out_dir / "open_ended_prepared_prompts.jsonl", [_json_ready(row) for row in prompt_rows])

    raw_rows = [row for row in prompt_rows if row["condition"] == "raw"]
    rewrite_rows = [row for row in prompt_rows if row["condition"] != "raw"]
    write_csv_dicts(paths.input_dir / "open_ended_raw_questions.csv", raw_rows)
    write_jsonl(paths.input_dir / "open_ended_raw_questions.jsonl", [_json_ready(row) for row in raw_rows])
    write_csv_dicts(paths.input_dir / "open_ended_preprocessed_questions.csv", rewrite_rows)
    write_jsonl(paths.input_dir / "open_ended_preprocessed_questions.jsonl", [_json_ready(row) for row in rewrite_rows])
    write_csv_dicts(paths.out_dir / "open_ended_component_metrics.csv", compute_component_metrics(_csv_ready_rows(prompt_rows)))
    quality_rows = compute_dataset_quality(_csv_ready_rows(prompt_rows))
    add_length_control_diagnostics(_csv_ready_rows(prompt_rows), quality_rows)
    add_proposed_difference_diagnostics(_csv_ready_rows(prompt_rows), quality_rows)
    write_csv_dicts(paths.out_dir / "open_ended_dataset_quality.csv", quality_rows)
    write_dataset_quality_markdown(paths.out_dir / "open_ended_dataset_quality.md", quality_rows)


def write_open_ended_manual_sheet(paths: RunPaths, prompt_rows: Sequence[dict[str, Any]]) -> Path:
    manual_rows = [
        {
            "item_id": row["item_id"],
            "condition": row["condition"],
            "prompt_hash": row["prompt_hash"],
            "prompt": row["prompt"],
            "raw_response_manual": "",
        }
        for row in prompt_rows
    ]
    manual_path = paths.input_dir / "open_ended_talkie_input_sheet.csv"
    fieldnames = ["item_id", "condition", "prompt_hash", "prompt", "raw_response_manual"]
    write_csv_dicts(manual_path, manual_rows, fieldnames)
    write_csv_dicts(paths.out_dir / "open_ended_talkie_input_sheet.csv", manual_rows, fieldnames)
    return manual_path


def build_result_rows(
    prompt_rows: Sequence[dict[str, Any]],
    responses: dict[tuple[str, str], dict[str, str] | str],
    *,
    provider: str,
) -> list[dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    for row in prompt_rows:
        response_record = responses.get((str(row["item_id"]), str(row["condition"])), {})
        if isinstance(response_record, str):
            raw_response = response_record
            response_prompt_hash = ""
        else:
            raw_response = response_record.get("raw_response", "")
            response_prompt_hash = response_record.get("prompt_hash", "")
        choices = row.get("choices")
        if not isinstance(choices, dict):
            choices = {label: str(row.get(f"choice_{label}", "")) for label in LABELS}
        parsed = normalize_choice(raw_response, choices)
        correct = parsed == row["gold_answer"]
        current_prompt_hash = str(row.get("prompt_hash", ""))
        hash_match = bool(response_prompt_hash) and response_prompt_hash == current_prompt_hash
        result_rows.append(
            {
                "item_id": row["item_id"],
                "domain": row["domain"],
                "split": row.get("split", ""),
                "condition": row["condition"],
                "gold_answer": row["gold_answer"],
                "raw_response": raw_response,
                "parsed_answer": parsed,
                "correct": correct,
                "provider": provider,
                "prompt_hash": current_prompt_hash,
                "response_prompt_hash": response_prompt_hash,
                "prompt_hash_match": hash_match,
                "rewrite_validation": row.get("rewrite_validation", {}),
                "repair_attempts": row.get("repair_attempts", 0),
                "prompt": row["prompt"],
            }
        )
    return result_rows


def write_open_ended_response_outputs(
    paths: RunPaths,
    prompt_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
    *,
    provider: str,
    seed: int,
) -> None:
    write_csv_dicts(paths.out_dir / "open_ended_responses.csv", list(response_rows))
    write_jsonl(paths.out_dir / "open_ended_responses.jsonl", [_json_ready(row) for row in response_rows])
    write_csv_dicts(paths.out_dir / "open_ended_response_integrity.csv", compute_response_integrity(response_rows))
    write_csv_dicts(paths.out_dir / "open_ended_component_metrics.csv", compute_component_metrics(_csv_ready_rows(prompt_rows)))
    pair_rows = build_judge_pair_rows(response_rows, seed=seed)
    write_open_ended_judge_sheets(paths, pair_rows)
    write_json(
        paths.out_dir / "open_ended_run_metadata.json",
        {
            "provider": provider,
            "n_items": len({row["item_id"] for row in response_rows}),
            "n_response_rows": len(response_rows),
            "n_judge_pairs": len(pair_rows),
            "note": "Judge prompts are blind: condition labels are stored in the unblinded analysis file but not included in the judge input sheet.",
        },
    )


def write_open_ended_judge_sheets(paths: RunPaths, pair_rows: Sequence[dict[str, Any]]) -> None:
    write_csv_dicts(paths.out_dir / "open_ended_judge_pairs_unblinded.csv", list(pair_rows))
    write_jsonl(paths.out_dir / "open_ended_judge_pairs_unblinded.jsonl", [_json_ready(row) for row in pair_rows])
    input_rows = judge_input_rows(pair_rows)
    fieldnames = ["pair_id", "judge_prompt_hash", "judge_prompt", "judge_raw_output"]
    write_csv_dicts(paths.input_dir / "open_ended_judge_input_sheet.csv", input_rows, fieldnames)
    write_csv_dicts(paths.out_dir / "open_ended_judge_input_sheet.csv", input_rows, fieldnames)


def write_open_ended_judge_outputs(paths: RunPaths, parsed_rows: Sequence[dict[str, Any]]) -> None:
    metrics = compute_judge_pairwise_metrics(parsed_rows)
    integrity = judge_integrity_rows(parsed_rows)
    write_csv_dicts(paths.out_dir / "open_ended_judge_scores.csv", list(parsed_rows))
    write_jsonl(paths.out_dir / "open_ended_judge_scores.jsonl", [_json_ready(row) for row in parsed_rows])
    write_csv_dicts(paths.out_dir / "open_ended_pairwise_metrics.csv", metrics)
    write_csv_dicts(paths.out_dir / "open_ended_judge_integrity.csv", integrity)
    (paths.out_dir / "open_ended_response_quality.md").write_text(
        write_response_quality_report(metrics=metrics, integrity=integrity),
        encoding="utf-8",
    )


def write_evaluation_outputs(
    paths: RunPaths,
    prompt_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[dict[str, Any]],
    *,
    provider: str,
    seed: int,
) -> None:
    csv_prompt_rows = _csv_ready_rows(prompt_rows)
    final_metrics = compute_condition_metrics(result_rows)
    test_result_rows = [row for row in result_rows if row.get("split") == "test"]
    test_metrics = compute_condition_metrics(test_result_rows) if test_result_rows else []
    component_metrics = compute_component_metrics(csv_prompt_rows)
    domain_metrics = compute_domain_metrics(result_rows)
    paired_tests = compute_paired_tests(result_rows, seed=seed)
    test_paired_tests = compute_paired_tests(test_result_rows, seed=seed) if test_result_rows else []
    key_comparisons = compute_key_comparisons(result_rows, seed=seed)
    test_key_comparisons = compute_key_comparisons(test_result_rows, seed=seed) if test_result_rows else []
    quality_rows = compute_dataset_quality(csv_prompt_rows)
    add_length_control_diagnostics(csv_prompt_rows, quality_rows)
    add_proposed_difference_diagnostics(csv_prompt_rows, quality_rows)
    integrity_rows = compute_response_integrity(result_rows)
    resource_source = read_json(paths.cache_dir / "generation_resource_source.json") if (paths.cache_dir / "generation_resource_source.json").exists() else {}

    write_csv_dicts(paths.out_dir / "per_item_results.csv", list(result_rows))
    write_jsonl(paths.out_dir / "per_item_results.jsonl", [_json_ready(row) for row in result_rows])
    write_csv_dicts(paths.out_dir / "final_metrics.csv", final_metrics)
    if test_metrics:
        write_csv_dicts(paths.out_dir / "final_metrics_test.csv", test_metrics)
    write_csv_dicts(paths.out_dir / "component_metrics.csv", component_metrics)
    write_csv_dicts(paths.out_dir / "domain_metrics.csv", domain_metrics)
    write_csv_dicts(paths.out_dir / "paired_tests.csv", paired_tests)
    write_csv_dicts(paths.out_dir / "key_comparisons.csv", key_comparisons)
    if test_paired_tests:
        write_csv_dicts(paths.out_dir / "paired_tests_test.csv", test_paired_tests)
    if test_key_comparisons:
        write_csv_dicts(paths.out_dir / "key_comparisons_test.csv", test_key_comparisons)
    write_csv_dicts(paths.out_dir / "response_integrity.csv", integrity_rows)
    write_csv_dicts(paths.out_dir / "dataset_quality.csv", quality_rows)
    write_dataset_quality_markdown(paths.out_dir / "dataset_quality.md", quality_rows)
    write_qualitative_examples(paths.out_dir / "qualitative_examples.md", prompt_rows, result_rows)
    write_report(
        paths.out_dir / "report.md",
        provider=provider,
        n_items=len({row["item_id"] for row in result_rows}),
        final_metrics=final_metrics,
        test_metrics=test_metrics,
        component_metrics=component_metrics,
        dataset_quality=quality_rows,
        domain_metrics=domain_metrics,
        response_integrity=integrity_rows,
        resource_source=resource_source,
        paired_tests=paired_tests,
        test_paired_tests=test_paired_tests,
        key_comparisons=key_comparisons,
        test_key_comparisons=test_key_comparisons,
    )


def _csv_ready_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: json_dumps(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}
        for row in rows
    ]


def _json_ready(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def compute_response_integrity(result_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(result_rows)
    matches = sum(1 for row in result_rows if row.get("prompt_hash_match"))
    missing = sum(1 for row in result_rows if not row.get("response_prompt_hash"))
    blank = sum(1 for row in result_rows if not str(row.get("raw_response", "")).strip())
    return [
        {"metric": "prompt_hash_match_rate", "value": matches / n if n else 0.0},
        {"metric": "missing_response_prompt_hash_count", "value": missing},
        {"metric": "blank_response_count", "value": blank},
        {"metric": "response_completion_rate", "value": (n - blank) / n if n else 0.0},
        {"metric": "n_rows", "value": n},
    ]


def enforce_manual_integrity(result_rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    bad_hash = [row for row in result_rows if not row.get("prompt_hash_match")]
    blank = [row for row in result_rows if not str(row.get("raw_response", "")).strip()]
    if bad_hash and not args.allow_hash_mismatch:
        raise ValueError(
            f"Manual response CSV has {len(bad_hash)} missing or mismatched prompt_hash values. "
            "Use --allow-hash-mismatch only for diagnostic runs."
        )
    if blank and not args.allow_missing_responses:
        raise ValueError(
            f"Manual response CSV has {len(blank)} blank responses. "
            "Fill raw_response_manual, or use --allow-missing-responses only for diagnostics."
        )


def load_judge_records(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for row in read_csv_dicts(path):
        pair_id = row.get("pair_id", "")
        if pair_id:
            records[pair_id] = {
                "judge_prompt_hash": row.get("judge_prompt_hash", ""),
                "judge_raw_output": row.get("judge_raw_output", ""),
            }
    return records


def enforce_judge_integrity(parsed_rows: Sequence[dict[str, Any]]) -> None:
    bad_hash = [row for row in parsed_rows if not row.get("judge_prompt_hash_match")]
    blank = [row for row in parsed_rows if not str(row.get("judge_raw_output", "")).strip()]
    unparsable = [row for row in parsed_rows if row.get("winner") not in {"A", "B", "Tie"}]
    if bad_hash:
        raise ValueError(
            f"Judge response CSV has {len(bad_hash)} missing or mismatched judge_prompt_hash values. "
            "Use --allow-hash-mismatch only for diagnostic runs."
        )
    if blank:
        raise ValueError(f"Judge response CSV has {len(blank)} blank judge outputs.")
    if unparsable:
        raise ValueError(f"Judge response CSV has {len(unparsable)} judge outputs that could not be parsed.")


def reject_mock_dictionary_paths(args: argparse.Namespace) -> None:
    paths = [str(getattr(args, "concept_dictionary_json", "")), str(getattr(args, "primitive_dictionary_json", ""))]
    if any("mock_" in path.replace("\\", "/").split("/")[-1] for path in paths if path):
        raise ValueError("Mock dictionaries cannot be used for final evaluation unless --allow-mock-dictionary is set.")


if __name__ == "__main__":
    raise SystemExit(main())
