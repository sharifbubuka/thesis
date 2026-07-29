from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.benchmarks import BenchmarkName
from src.continual import ContinualExperiment, ContinualExperimentConfig
from src.contribution import (
    ModalityContributionEstimator,
    estimate_samples,
    save_contribution_artifacts,
)
from src.data import load_benchmark_samples
from src.utils.device import resolve_device
from src.vilt import (
    assert_stage_one_result,
    create_adapter,
    load_vilt_checkpoint,
    save_stage_one_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViLT contribution and continual experiments.")
    parser.add_argument("--stage", choices=["1", "2", "3"], default="1")
    parser.add_argument(
        "--benchmark",
        choices=[item.value for item in BenchmarkName],
        default=BenchmarkName.VQAV2.value,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/results"))
    parser.add_argument("--train-samples", type=int, default=1_000)
    parser.add_argument("--validation-samples", type=int, default=250)
    parser.add_argument("--vocabulary-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--no-stage-checkpoints",
        action="store_true",
        help="Do not save the evolving model after each task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device()
    if args.stage == "3":
        config = ContinualExperimentConfig(
            train_samples_per_task=args.train_samples,
            validation_samples_per_task=args.validation_samples,
            vocabulary_size=args.vocabulary_size,
            batch_size=args.batch_size,
            epochs_per_task=args.epochs,
            learning_rate=args.learning_rate,
            save_stage_checkpoints=not args.no_stage_checkpoints,
        )
        dataframe, metrics = ContinualExperiment(
            config,
            device=device,
            output_dir=args.output_dir / "stage_3_continual_vqa",
        ).run()
        print(dataframe.to_string())
        print(asdict(metrics))
        return

    benchmark = BenchmarkName(args.benchmark)
    spec, processor, model = load_vilt_checkpoint(benchmark, device=device)
    adapter = create_adapter(spec, processor, model, device)
    sample = load_benchmark_samples(benchmark, count=1)[0]

    if args.stage == "1":
        result = adapter.predict(sample, include_internals=True)
        assert_stage_one_result(result)
        path = args.output_dir / f"stage_1_{benchmark.value}.json"
        save_stage_one_report([result], path)
        print(result.summary())
        return

    estimator = ModalityContributionEstimator(adapter)
    dataframe, results = estimate_samples(estimator, [sample], show_progress=False)
    paths = save_contribution_artifacts(
        dataframe,
        results,
        args.output_dir,
        stem=f"stage_2_{benchmark.value}",
    )
    print(dataframe.to_string(index=False))
    print({key: str(path) for key, path in paths.items()})


if __name__ == "__main__":
    main()
