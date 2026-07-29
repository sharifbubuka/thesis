from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s']")


def normalize_answer(answer: str) -> str:
    """Apply one stable normalization to labels from all VQA datasets."""
    value = answer.lower().strip()
    value = _PUNCTUATION.sub(" ", value)
    value = _ARTICLES.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


@dataclass(frozen=True, slots=True)
class AnswerVocabulary:
    answers: tuple[str, ...]
    unknown_token: str = "<unk>"

    def __post_init__(self) -> None:
        if not self.answers or self.answers[0] != self.unknown_token:
            raise ValueError("The unknown token must be the first vocabulary entry.")
        if len(set(self.answers)) != len(self.answers):
            raise ValueError("Vocabulary entries must be unique.")

    @classmethod
    def build(
        cls,
        answer_groups: Iterable[Sequence[str]],
        *,
        max_size: int,
        unknown_token: str = "<unk>",
    ) -> AnswerVocabulary:
        if max_size < 2:
            raise ValueError("max_size must leave room for at least one known answer.")
        counts: Counter[str] = Counter()
        for answers in answer_groups:
            counts.update(normalize_answer(answer) for answer in answers if normalize_answer(answer))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        labels = tuple(answer for answer, _ in ordered if answer != unknown_token)
        return cls((unknown_token, *labels[: max_size - 1]), unknown_token)

    @property
    def answer_to_id(self) -> Mapping[str, int]:
        return {answer: index for index, answer in enumerate(self.answers)}

    def encode(self, answer: str) -> int:
        return self.answer_to_id.get(normalize_answer(answer), 0)

    def decode(self, answer_id: int) -> str:
        return self.answers[answer_id]

    def to_dict(self) -> dict[str, object]:
        return {"answers": list(self.answers), "unknown_token": self.unknown_token}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> AnswerVocabulary:
        answers = values["answers"]
        if not isinstance(answers, list) or not all(isinstance(item, str) for item in answers):
            raise TypeError("answers must be a list of strings.")
        unknown = values.get("unknown_token", "<unk>")
        if not isinstance(unknown, str):
            raise TypeError("unknown_token must be a string.")
        return cls(tuple(answers), unknown)
