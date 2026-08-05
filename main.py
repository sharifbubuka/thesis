from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarks import BenchmarkName
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
    parser = argparse.ArgumentParser(description="Run ViLT modality-contribution experiments.")
    parser.add_argument("--stage", choices=["1", "2"], default="1")
    parser.add_argument(
        "--benchmark",
        choices=[item.value for item in BenchmarkName],
        default=BenchmarkName.VQAV2.value,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device()
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
