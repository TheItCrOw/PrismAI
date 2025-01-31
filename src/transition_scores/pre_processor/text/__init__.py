from typing import Any

from simple_dataset.dataset import Dataset
from transition_scores.pre_processor.abc import PreProcessor
from transition_scores.utils import normalize_text


class TextPreProcessor(PreProcessor):
    def _prepare(
        self,
        dataset: Dataset[str, Any],
    ) -> Dataset[str, Any]:
        dataset = super()._prepare(dataset)
        dataset.modify(normalize_text)
        dataset.filter(lambda document: document["text"], in_place=True)
        return dataset
