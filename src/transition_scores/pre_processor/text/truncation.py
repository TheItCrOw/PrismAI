from tqdm import tqdm
from transformers import BatchEncoding

from transition_scores.data import PreProcessorMetadata
from transition_scores.pre_processor.text import TextPreProcessor
from transition_scores.utils import transpose_dict_of_lists


class TruncationTextPreProcessor(TextPreProcessor):
    """
    Simple `text` pre-processor.
    Sequences are tokenized and truncated to `max_length`.
    """

    @property
    def required_fields(self) -> dict[str, type]:
        return {"text": str}

    @property
    def additional_fields(self) -> None | dict[str, type]:
        return {"text": str} if self.include_text else None

    def get_metadata(self) -> PreProcessorMetadata:
        return PreProcessorMetadata.new(
            "truncate",
            max_length=self.max_length,
        )

    def _process(self, text: list[str]) -> BatchEncoding:
        """Process a *batch* of samples.

        Note:
            Effectively calls:
            ```py
            >>> tokenizer( # doctest: +SKIP
            ...     text,
            ...     truncation=True,
            ...     return_length=True,
            ...     add_special_tokens=True,
            ... )
            ```

        Args:
            batch (dict[str, list]): Batch of samples to tokenize.

        Returns:
            BatchEncoding | dict[str, list]: Tokenized batch.
        """
        batch_encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            add_special_tokens=True,
        )

        if self.include_text:
            batch_encodings = [
                batch_encoding
                | {
                    "text": self._tokenizer.batch_decode(
                        batch_encoding["input_ids"],
                        skip_special_tokens=True,
                    )
                }
                for batch_encoding in batch_encodings
            ]

        return batch_encodings

    def pre_process(self, dataset: list[dict]) -> list[dict]:
        """Prepare the `text` of the samples in a dataset.
        Adds `text_sha256` field to the dataset.

        Args:
            dataset (list[dict]): A dataset containing with fields: `text: str` and `chunks: list[str]`.

        Returns:
            list[dict]: Tokenized dataset. The `text` and `chunks` fields are removed.
        """
        with tqdm(total=4, position=2, leave=False, desc="Pre-Processing") as tq:
            try:
                tq.set_postfix_str("Preparing Dataset")
                dataset = self._prepare(dataset)
                tq.update(1)

                tq.set_postfix_str("Tokenizing Rolling Windows")
                encodings = [
                    self._process(document.pop("text")) for document in dataset
                ]
                tq.update(1)

                tq.set_postfix_str("Exploding Samples from Encoding")
                dataset = (
                    dict(
                        **document,
                        **transposed,
                    )
                    for document, encoding in zip(dataset, encodings)
                    for transposed in transpose_dict_of_lists(encoding, iter=True)
                )
                tq.update(1)

                tq.set_postfix_str("Sorting Dataset by Length")
                dataset = self._sort_dataset_by_length(dataset)
                tq.update(1)
            except KeyError as e:
                raise KeyError(
                    f"{type(self).__name__} requires the fields: {self.required_fields}."
                ) from e

        return dataset
