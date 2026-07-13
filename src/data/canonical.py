from typing import Any, Dict, List
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import json
from src.utils import print_message


def normalize_answer(answer: Any) -> str:
    """
    Convert dataset-specific answer formats into one representative string.
    """
    if answer is None:
        return ""

    if isinstance(answer, (str, int, float)):
        return str(answer).strip()

    if isinstance(answer, dict):
        value = answer.get("answer") or answer.get("raw_answer") or answer.get("text") or ""
        return str(value).strip()

    if isinstance(answer, list):
        values = []

        for item in answer:
            value = normalize_answer(item)
            if value:
                values.append(value)

        return Counter(values).most_common(1)[0][0] if values else ""

    return str(answer).strip()


class CanonicalDatasetBuilder:
    """
    Converts raw multimodal dataset samples into a shared canonical format.
    """

    def __init__(self, dataset_key: str, dataset_info: Dict[str, Any]):
        self.dataset_key = dataset_key
        self.dataset_info = dataset_info
        self.samples = []

    def build_dataset(self, dataset) -> List[Dict[str, Any]]:
        for index, raw_sample in enumerate(dataset):
            self.samples.extend(self.build_sample(raw_sample, index))

        return self.samples

    def build_sample(self, raw_sample: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        if self.dataset_key == "gqa":
            return self._build_gqa_samples(raw_sample, index)

        return self._build_single_sample(raw_sample, index)

    def _build_single_sample(self, raw_sample: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        image = raw_sample.get(self.dataset_info["image_column"])
        question = str(raw_sample.get(self.dataset_info["question_column"], "")).strip()
        answer = normalize_answer(raw_sample.get(self.dataset_info["answer_column"]))

        if image is None or not question or not answer:
            return []

        return [
            self._make_sample(
                sample_id=f"{self.dataset_key}_{index}",
                image=image,
                question=question,
                answer=answer,
                metadata={"raw_index": index},
            )
        ]

    def _build_gqa_samples(self, raw_sample: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        image = raw_sample.get("image")
        qa_pairs = raw_sample.get("qa", [])

        if image is None:
            return []

        samples = []

        for qa_index, qa in enumerate(qa_pairs):
            question = str(qa.get("question", "")).strip()
            answer = normalize_answer(qa.get("fullAnswer") or qa.get("answer"))

            if not question or not answer:
                continue

            samples.append(
                self._make_sample(
                    sample_id=f"{self.dataset_key}_{index}_{qa_index}",
                    image=image,
                    question=question,
                    answer=answer,
                    metadata={
                        "raw_index": index,
                        "qa_index": qa_index,
                        "full_answer": qa.get("fullAnswer", ""),
                    },
                )
            )

        return samples

    def _make_sample(
        self,
        sample_id: str,
        image: Any,
        question: str,
        answer: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "sample_id": sample_id,
            "dataset": self.dataset_key,
            "task": self.dataset_info["task"],
            "image": image,
            "question": question,
            "answer": answer,
            "metadata": metadata,
        }

class CanonicalDatasetVisualizer:
    """
    Visualizes canonical dataset
    """

    def __init__(self, canonical_datasets: dict):
        self.canonical_datasets = canonical_datasets
        
    def preview_sample(self, dataset_key: str, index: int, show_image: bool = False):
        if dataset_key not in self.canonical_datasets:
            print(f"🔴 Dataset \"{dataset_key}\" not found.")
            return

        dataset = self.canonical_datasets[dataset_key]
        if index < 0 or index >= len(dataset):
            print(f"🔴 Index {index} out of range.")
            return

        sample = dataset[index]

        print_message(f"TASK {sample['task']} | Sample INDEX {index}", skip_line=False)
        print(f"Task: {sample['task']}")
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Metadata: {sample['metadata']}")

        if show_image and sample["image"] is not None:
            plt.figure(figsize=(4, 4))
            plt.imshow(sample["image"])
            plt.axis("off")
            plt.show()
        else:
            print("Image: Has no image.")
                        
class CanonicalDatasetSerializer:
    """
    Serializes canonical datasets to disk.
    """

    def __init__(self, canonical_datasets: dict):
        self.canonical_datasets = canonical_datasets

    def serialize_sample(self, sample: dict):
        return {
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "task": sample["task"],
            "question": sample["question"],
            "answer": sample["answer"],
            "metadata": sample["metadata"],
            "has_image": sample["image"] is not None,
        }

    def save_metadata(self, output_dir: str, file_name: str = "canonical_metadata"):
        output_path = Path(output_dir) / f"{file_name}.json"

        serializable_data = {
            dataset_key: [
                self.serialize_sample(sample)
                for sample in samples
            ]
            for dataset_key, samples in self.canonical_datasets.items()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Canonical metadata saved to \"{output_path}\".")