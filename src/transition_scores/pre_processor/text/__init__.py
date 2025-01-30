from typing import Any

from transition_scores.pre_processor.abc import PreProcessor


class TextPreProcessor(PreProcessor):
    def _prepare(
        self,
        dataset: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        dataset = super()._prepare(dataset)
        for document in dataset:
            document["text"] = " ".join(document["text"].strip().split())
        return dataset
