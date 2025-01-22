from collections import namedtuple
from dataclasses import dataclass

from transformers import AutoConfig, BatchEncoding, PreTrainedTokenizer

EncodedSequence = namedtuple("EncodedSequence", ["input_ids", "attention_mask"])


@dataclass
class LogProbs:
    target_id: int
    target_prob: float
    top_k_id: list[int]
    top_k_probs: list[float]

    def __iter__(self):
        yield from (self.target_id, self.target_prob, self.top_k_id, self.top_k_probs)


def infer_max_length(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "n_positions"):
        return config.n_positions
    raise ValueError(f"Could not infer max length from {model_name_or_path}")


class RollingWindowChunkTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int | None = None):
        self.change_tokenizer(tokenizer, max_length)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def change_tokenizer(
        self, tokenizer: PreTrainedTokenizer, max_length: int | None = None
    ):
        """Change tokenizer and update max_length.

        Args:
            tokenizer (PreTrainedTokenizer): The new tokenizer.
            max_length (int, optional): The new max_length.
                Will try to infer max_length from the tokenizer if not given.

        Raises:
            ValueError: If max_length is not given and we could not infer it from the tokenizer.
        """
        self._tokenizer = tokenizer

        max_length = max_length or tokenizer.model_max_length
        if max_length is None:
            try:
                max_length = infer_max_length(tokenizer.name_or_path)
            except ValueError as e:
                raise ValueError(
                    f"max_length was not given and we could not infer the max_length from {type(tokenizer)}({tokenizer.name_or_path}). Please provide a max_length to the {type(self).__name__} constructor."
                ) from e
        self.max_length = max_length

    def __call__(self, chunks: list[str]) -> BatchEncoding:
        batch_encoding = BatchEncoding(
            {
                "input_ids": [],
                "attention_mask": [],
                "length": [],
                "chunk_text": [],
                "chunk_prefix_idx": [],
                "chunk_start_idx": [],
                "chunk_end_idx": [],
                "chunk_start_token_idx": [],
            }
        )

        # For each span, try to find the largest possible prefix-span that fits within the max_length.
        prefix_start = 0
        for chunk_idx in range(len(chunks)):
            while True:
                buffer = chunks[prefix_start : chunk_idx + 1]
                text = self.chunks_to_text(buffer)
                encoding = self._tokenizer(
                    text,
                    return_length=True,
                    add_special_tokens=True,
                )

                if (
                    prefix_start == chunk_idx
                    or len(encoding["input_ids"]) <= self.max_length
                ):
                    break

                prefix_start += 1

            try:
                char_idx = text.index(chunks[chunk_idx].strip())
                word_idx = encoding.char_to_word(char_idx)
                token_idx, _ = encoding.word_to_tokens(word_idx)
            except Exception as e:
                raise ValueError(
                    f"Could not find the start token of chunk {chunk_idx}:'{chunks[chunk_idx]}' in the encoding of '{text}'."
                ) from e

            batch_encoding["input_ids"].append(encoding["input_ids"])
            batch_encoding["attention_mask"].append(encoding["attention_mask"])
            batch_encoding["length"].extend(encoding["length"])
            batch_encoding["chunk_text"].append(text)
            batch_encoding["chunk_prefix_idx"].append(prefix_start)
            batch_encoding["chunk_start_idx"].append(chunk_idx)
            batch_encoding["chunk_end_idx"].append(chunk_idx + 1)
            batch_encoding["chunk_start_token_idx"].append(token_idx)

        return batch_encoding

    @staticmethod
    def chunks_to_text(chunks: list[str]) -> str:
        return " ".join(chunk.strip() for chunk in chunks)
