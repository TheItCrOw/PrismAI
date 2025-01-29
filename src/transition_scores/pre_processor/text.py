from hashlib import sha256

from tqdm import tqdm
from transformers import BatchEncoding

from transition_scores.data import PreProcessorMetadata
from transition_scores.pre_processor.abc import PreProcessor
from transition_scores.utils import transpose_dict_of_lists


class TextPreProcessor(PreProcessor):
    """
    Simple `text` pre-processor.
    Sequences are tokenized and truncated to `max_length`.
    """

    @property
    def required_fields(self) -> dict[str, type]:
        return {"text": str}

    def get_metadata(self) -> PreProcessorMetadata:
        return PreProcessorMetadata.new(
            "text",
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
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            add_special_tokens=True,
        )

    def pre_process(self, dataset: list[dict]) -> list[dict]:
        """Prepare the `text` of the samples in a dataset.
        Adds `text_sha256` field to the dataset.

        Args:
            dataset (list[dict]): A dataset containing with fields: `text: str` and `chunks: list[str]`.

        Returns:
            list[dict]: Tokenized dataset. The `text` and `chunks` fields are removed.
        """
        with tqdm(
            total=4, desc="Pre-Processing Dataset", position=1, leave=False
        ) as tq:
            tq.set_postfix_str("Calculating Text Hash")
            text_hashes = [
                sha256(row.pop("text").encode()).hexdigest() for row in dataset
            ]
            tq.update(1)

            tq.set_postfix_str("Tokenizing Rolling Windows")
            encodings = [
                self._process(row.pop("text"))
                for row in tqdm(dataset, position=2, leave=False)
            ]
            tq.update(1)

            tq.set_postfix_str("Exploding Samples from Encoding")
            dataset = (
                dict(
                    **row,
                    **transposed,
                    text_sha256=txt_hsh,
                )
                for row, txt_hsh, encoding in zip(
                    tqdm(dataset, position=2, leave=False),
                    text_hashes,
                    encodings,
                )
                for transposed in transpose_dict_of_lists(encoding, iter=True)
            )
            tq.update(1)

            tq.set_postfix_str("Sorting Dataset by Length")
            dataset = self._sort_dataset_by_length(dataset)
            tq.update(1)

        return dataset
