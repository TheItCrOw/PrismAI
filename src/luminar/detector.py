import os.path
import statistics
from argparse import Namespace
from dataclasses import dataclass

import torch
from typing_extensions import Literal
from IPython.display import display, HTML
from data_hub.sequential_data_processor import SequentialDataProcessor
from luminar.encoder import LuminarEncoder
from luminar.sequence_classifier import LuminarSequence


class LuminarSequenceDetector:

    def __init__(self,
                 model_path: str,
                 feature_agent: Literal["gpt2", "tiiuae/falcon-7b"] = "gpt2",
                 device: str = "cpu"):
        if not model_path or not os.path.exists(model_path):
            raise ValueError("Valid and existing model path must be provided for LuminarSequenceDetector.")

        print(f"Loading LuminarSequenceDetector from {model_path} to device {device}")
        self.classifier = LuminarSequence.load(model_path, device=device)
        if not self.classifier:
            raise ValueError(f"Failed to load LuminarSequence model from {model_path}")

        self.encoder = LuminarEncoder(model_name_or_path=feature_agent, device=device)
        self.data_processor = SequentialDataProcessor(self.encoder)
        self.device = device
        print("Loaded.")

    def detect(self, document: str, chunk_size: int = 256, stride: int = 256) -> dict:
        """
        Detects AI-generated sequences in a document using chunked processing to prevent GPU overflowing on long texts.

        :param document: The document to analyze.
        :param chunk_size: Number of tokens per chunk (e.g. 256).
        :param stride: Overlap between chunks for continuity (e.g. stride = chunk_size = 256 => no overlap).
        :return: A dictionary with probabilities, token-level spans, and character-level spans.
        """
        if not isinstance(document, str) or not document.strip():
            raise ValueError("Document must be a non-empty string.")

        tokenized = self.encoder.tokenize(texts=document, truncate=False)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        offset_mapping = tokenized["offset_mapping"]

        all_probs = []
        all_token_spans = []
        all_char_spans = []

        # Chunk traversal over tokenized input
        for start in range(0, len(input_ids), stride):
            end = start + chunk_size
            input_ids_chunk = input_ids[start:end]
            attention_mask_chunk = attention_mask[start:end]
            offset_mapping_chunk = offset_mapping[start:end]

            if len(input_ids_chunk) < 10:
                break  # skip tiny tail chunks

            # Encode the chunk to features
            encoded = self.encoder.rolling_process({
                "input_ids": [input_ids_chunk],
                "attention_mask": [attention_mask_chunk]
            }, max_chunks=1)

            features = torch.tensor(encoded["features"], device=self.device)

            # Compute spans within chunk length
            spans = self.data_processor.process_for_single_document(document, features.shape[1])

            # Adjust token spans by chunk offset
            adjusted_spans = [(s + start, e + start) for s, e in spans if e + start <= len(offset_mapping)]

            if not adjusted_spans:
                continue

            with torch.no_grad():
                output = self.classifier(features, [spans])
                probs = torch.sigmoid(output.logits).view(-1).cpu().numpy()

            char_spans = SequentialDataProcessor.map_token_spans_to_character_spans(
                adjusted_spans, offset_mapping
            )

            all_probs.extend(probs)
            all_token_spans.extend(adjusted_spans)
            all_char_spans.extend(char_spans)

        return {
            "avg": statistics.mean(all_probs).item() * 100,
            "probs": [float(p) * 100 for p in all_probs],
            "token_spans": all_token_spans,
            "char_spans": all_char_spans
        }

