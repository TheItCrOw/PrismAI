from datasets import Dataset
from transformers import BatchEncoding

from transition_scores.pre_processor.abc import PreProcessor, text_sha256


class TextPreProcessor(PreProcessor):
    """
    Simple `text` pre-processor.
    Sequences are tokenized and truncated to `max_length`.
    """

    def process(self, batch: dict[str, list]) -> BatchEncoding:
        """Process a *batch* of samples.
        Expects a dictionary with a "text" field containing a list of strings.

        Note:
            Effectively calls:
            ```py
            >>> tokenizer(
            >>>     batch["text"],
            >>>     truncation=True,
            >>>     return_length=True,
            >>>     add_special_tokens=True,
            >>> )
            ```

        Args:
            batch (dict[str, list]): Batch of samples to tokenize.

        Returns:
            BatchEncoding | dict[str, list]: Tokenized batch.
        """
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            add_special_tokens=True,
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare the `text` of the samples in a dataset.
        Adds `text_sha256` field to the dataset.

        Args:
            dataset (Dataset): A dataset containing with fields: `text: str` and `chunks: list[str]`.

        Returns:
            Dataset: Tokenized dataset. The `text` and `chunks` fields are removed.
        """
        return (
            dataset.map(text_sha256)
            .map(self.process, batched=True, remove_columns=["text", "chunks"])
            .sort("length")
            .remove_columns("length")
        )
