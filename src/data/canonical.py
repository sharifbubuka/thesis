from typing import Any, Dict, List
from collections import Counter
from pathlib import Path
import json

import matplotlib.pyplot as plt

from src.utils import print_message


def normalize_answers(answer: Any) -> List[str]:
    """
    Convert dataset-specific answer formats into a list of normalized strings.

    Supported formats:
    - string, integer, float
    - dictionary containing answer-like fields
    - list of strings or dictionaries
    """

    if answer is None:
        return []

    if isinstance(answer, (str, int, float)):
        value = str(answer).strip()
        return [value] if value else []

    if isinstance(answer, dict):
        value = (
            answer.get("answer")
            or answer.get("raw_answer")
            or answer.get("text")
            or ""
        )

        value = str(value).strip()
        return [value] if value else []

    if isinstance(answer, list):
        values: List[str] = []

        for item in answer:
            values.extend(normalize_answers(item))

        return [value for value in values if value]

    value = str(answer).strip()
    return [value] if value else []


def select_representative_answer(answers: List[str]) -> str:
    """
    Select the most frequent answer as the canonical answer.

    All original valid answers are still retained in the canonical sample.
    """

    if not answers:
        return ""

    return Counter(answers).most_common(1)[0][0]


class CanonicalDatasetBuilder:
    """
    Converts raw multimodal dataset samples into a shared canonical format.

    Canonical format:

    {
        "sample_id": str,
        "dataset": str,
        "task": str,
        "image": PIL.Image,
        "question": str,
        "answer": str,
        "answers": list[str],
        "metadata": dict,
    }
    """

    def __init__(
        self,
        dataset_key: str,
        dataset_config: Dict[str, Any],
    ):
        self.dataset_key = dataset_key
        self.dataset_config = dataset_config
        self.samples: List[Dict[str, Any]] = []

    def build_dataset(
        self,
        dataset,
    ) -> List[Dict[str, Any]]:
        """
        Convert the complete raw dataset into canonical samples.
        """

        # Reset the internal list so repeated calls do not create duplicates.
        self.samples = []

        for index, raw_sample in enumerate(dataset):
            self.samples.extend(
                self.build_sample(
                    raw_sample=raw_sample,
                    index=index,
                )
            )
            
        print_message(f"Created canonical dataset {self.dataset_key} with {len(self.samples)} samples", skip_line=False)


        return self.samples

    def build_sample(
        self,
        raw_sample: Dict[str, Any],
        index: int,
    ) -> List[Dict[str, Any]]:
        """
        Route a raw sample to the correct dataset-specific builder.
        """

        task = self.dataset_config["task"]

        if task == "gqa":
            return self._build_gqa_samples(
                raw_sample=raw_sample,
                index=index,
            )

        return self._build_single_sample(
            raw_sample=raw_sample,
            index=index,
        )

    def _build_single_sample(
        self,
        raw_sample: Dict[str, Any],
        index: int,
    ) -> List[Dict[str, Any]]:
        """
        Build one canonical sample for datasets such as TextVQA and VQAv2.
        """

        image = raw_sample.get(
            self.dataset_config["image_column"]
        )

        question = str(
            raw_sample.get(
                self.dataset_config["question_column"],
                "",
            )
        ).strip()

        raw_answers = raw_sample.get(
            self.dataset_config["answer_column"]
        )

        answers = normalize_answers(raw_answers)
        answer = select_representative_answer(answers)

        if image is None or not question or not answer:
            return []

        image_id = self._extract_image_id(
            raw_sample=raw_sample,
            fallback=index,
        )

        return [
            self._make_sample(
                sample_id=f"{self.dataset_key}_{index}",
                image=image,
                question=question,
                answer=answer,
                answers=answers,
                metadata={
                    "raw_index": index,
                    "image_id": image_id,
                },
            )
        ]

    # === GQA: One Question-Answer Pair Per Image ===

    def _build_gqa_samples(
        self,
        raw_sample: Dict[str, Any],
        index: int,
    ) -> List[Dict[str, Any]]:
        """
        Build exactly one canonical question-answer sample per GQA image.

        One valid QA pair is selected deterministically from the image record.
        """

        image = raw_sample.get(
            self.dataset_config.get(
                "image_column",
                "image",
            )
        )

        qa_pairs = raw_sample.get("qa", [])

        if image is None or not qa_pairs:
            return []

        image_id = self._extract_image_id(
            raw_sample=raw_sample,
            fallback=index,
        )

        # Select the first valid QA pair for this image.
        for qa_index, qa in enumerate(qa_pairs):
            question = str(
                qa.get("question", "")
            ).strip()

            short_answer = qa.get("answer")
            full_answer = qa.get("fullAnswer")

            answers = normalize_answers(
                short_answer or full_answer
            )

            answer = select_representative_answer(answers)

            if not question or not answer:
                continue

            return [
                self._make_sample(
                    sample_id=f"{self.dataset_key}_{index}",
                    image=image,
                    question=question,
                    answer=answer,
                    answers=answers,
                    metadata={
                        "raw_index": index,
                        "qa_index": qa_index,
                        "image_id": image_id,
                        "full_answer": full_answer or "",
                        "num_available_qa_pairs": len(qa_pairs),
                    },
                )
            ]

        # No valid QA pair was found.
        return []

    def _extract_image_id(
        self,
        raw_sample: Dict[str, Any],
        fallback: int,
    ) -> str:
        """
        Extract a stable image identifier.

        This can later be used for image-disjoint train, validation,
        and test splitting.
        """

        possible_keys = (
            "image_id",
            "imageId",
            "img_id",
            "image_name",
            "id",
        )

        for key in possible_keys:
            value = raw_sample.get(key)

            if value is not None:
                return str(value)

        return str(fallback)

    def _make_sample(
        self,
        sample_id: str,
        image: Any,
        question: str,
        answer: str,
        answers: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct one canonical multimodal sample.
        """

        return {
            "sample_id": sample_id,

            # Use the stable local registry key instead of the HF repository.
            "dataset": self.dataset_key,

            "task": self.dataset_config["task"],
            "image": image,
            "question": question,

            # Representative answer for training and simple evaluation.
            "answer": answer,

            # All available human answers for VQA-style soft accuracy.
            "answers": answers,

            "metadata": metadata,
        }


class CanonicalDatasetVisualizer:
    """
    Displays canonical dataset samples and their metadata.
    """

    def __init__(
        self,
        canonical_datasets: Dict[str, List[Dict[str, Any]]],
    ):
        self.canonical_datasets = canonical_datasets

    def preview_sample(
        self,
        dataset_key: str,
        index: int,
        show_image: bool = False,
    ):
        """
        Print and optionally visualize one canonical sample.
        """

        if dataset_key not in self.canonical_datasets:
            print_message(
                f'Dataset "{dataset_key}" not found.',
                skip_line=False,
            )
            return

        dataset = self.canonical_datasets[dataset_key]

        if index < 0 or index >= len(dataset):
            print_message(
                f"Index {index} out of range.",
                skip_line=False,
            )
            return

        sample = dataset[index]

        print_message(
            f"TASK {sample['task']} | Sample INDEX {index}",
            skip_line=False,
        )

        print(f"Dataset: {sample['dataset']}")
        print(f"Task: {sample['task']}")
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"All answers: {sample.get('answers', [])}")
        print(f"Metadata: {sample['metadata']}")

        if sample["image"] is None:
            print("Image: Missing.")
            return

        if not show_image:
            print("Image: Available but not displayed.")
            return

        plt.figure(figsize=(5, 5))
        plt.imshow(sample["image"])
        plt.title(
            f"{sample['task']} | {sample['sample_id']}"
        )
        plt.axis("off")
        plt.show()


class CanonicalDatasetSerializer:
    """
    Serializes canonical dataset metadata without storing image bytes.
    """

    def __init__(
        self,
        canonical_datasets: Dict[str, List[Dict[str, Any]]],
    ):
        self.canonical_datasets = canonical_datasets

    def serialize_sample(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert one canonical sample into a JSON-serializable dictionary.
        """

        return {
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "task": sample["task"],
            "question": sample["question"],
            "answer": sample["answer"],
            "answers": sample.get(
                "answers",
                [sample["answer"]],
            ),
            "metadata": sample["metadata"],
            "has_image": sample["image"] is not None,
        }

    def save_metadata(
        self,
        output_dir: str,
        file_name: str = "canonical_metadata",
    ):
        """
        Save all canonical dataset metadata as JSON.
        """

        output_path = (
            Path(output_dir)
            / f"{file_name}.json"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serializable_data = {
            dataset_key: [
                self.serialize_sample(sample)
                for sample in samples
            ]
            for dataset_key, samples
            in self.canonical_datasets.items()
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                serializable_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f'✅ Canonical metadata saved to "{output_path}".'
        )