from typing import Any

from prismai_features.pre_processor.abc import PreProcessor
from prismai_features.utils import normalize_text
from simple_dataset.dataset import Dataset


class TextPreProcessor(PreProcessor):
    def _prepare(
        self,
        dataset: Dataset[str, Any],
    ) -> Dataset[str, Any]:
        dataset = super()._prepare(dataset)
        dataset.apply(normalize_text, "text")
        dataset.filter(lambda document: document["text"], in_place=True)
        return dataset
