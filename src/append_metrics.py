import traceback
from itertools import batched
from pathlib import Path

import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

from calc_transition_scores import get_argparser, parse_pre_processors
from simple_dataset.dataset import Dataset
from transition_scores.data import DocumentMetadata
from transition_scores.scorer import TransformersTransitionScorer
from transition_scores.scorer.abc import (
    _ModelOutput,
)

if Path(".env").exists():
    load_dotenv()
elif Path("../.env").exists():
    load_dotenv("../.env")


class MetricScorer(TransformersTransitionScorer):
    def process(
        self,
        dataset: Dataset | list[dict],
        pad_token_id: int,
    ) -> Dataset:
        """
        Calculate transition scores for the given pre-processed dataset.

        Args:
            dataset (Dataset): A sequence of pre-processed documents to be processed.
            pad_token_id (int): The token ID to use for padding.

        Raises:
            KeyError: If the pre-processor requires a field that is not present in the given dataset.

        Returns:
            Dataset: The processed dataset with a new "transition_scores" field.
        """
        if not isinstance(dataset, Dataset):
            dataset = Dataset(dataset)

        return dataset.update(self._generate_scores(dataset, pad_token_id))

    def _process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_token_id: int,
    ) -> list[dict]:
        """
        Process the a batch of input sequences and calculate transition scores.
        Runs a forward pass on the model and extracts the top k probabilities.

        Args:
            input_ids (torch.Tensor): A list of input sequences, each represented as a list of token IDs.
            attention_mask (torch.Tensor): A list of attention masks for each input sequence.
            pad_token_id (int): The token ID that has been used for padding.

        Returns:
            list[TransitionScores]: A list output probability tuples.
        """
        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414
        likelihoods, log_likelihoods = self._forward(input_ids, attention_mask)

        results = []
        for target_ids, likelihood, log_likelihood in zip(
            input_ids.to(self.device), likelihoods, log_likelihoods
        ):
            # Truncate the sequence to the last non-pad token
            labels = target_ids[1:].view(-1, 1)
            labels = labels[: labels.ne(pad_token_id).sum()]
            seq_length = labels.size(0)

            likelihood: torch.Tensor = likelihood[:seq_length]
            log_likelihood: torch.Tensor = log_likelihood[:seq_length]

            # Sort likelihoods and get ranks of target labels
            _, sorted_indices = torch.sort(likelihood, descending=True)
            _, target_ranks = torch.where(sorted_indices.eq(labels))

            # Get DetectLLM-LLR criterion
            target_log_probs = log_likelihood.gather(-1, labels).squeeze(-1)

            llr = self._calculate_log_likelihood_ratio(target_ranks, target_log_probs)

            # Get Fast-DetectGPT criterion
            fast_detect_gpt = self._calculate_fast_detect_gpt(
                likelihood, log_likelihood, target_log_probs
            )

            results.append(
                {
                    "values": {
                        "transition_scores.target_ids": target_ids.cpu().tolist(),
                        "metrics": [
                            {
                                "fast_detect_gpt": fast_detect_gpt,
                                "llr": llr,
                            }
                        ],
                    }
                }
            )

        return results

    def _forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            outputs: _ModelOutput = self._model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids.to(self.device),
            )

            likelihoods: torch.Tensor = outputs.logits.softmax(-1)
            log_likelihoods: torch.Tensor = outputs.logits.log_softmax(-1)

            del outputs
        return likelihoods, log_likelihoods


if __name__ == "__main__":
    args = get_argparser().parse_args()
    mongodb_batch_size = args.batch_size or args.mongodb_batch_size
    mongodb_filter_query = args.mongodb_filter or {}
    dataset_batch_size = (
        args.batch_size or args.dataset_batch_size or mongodb_batch_size
    )
    dataset_batch_size = min(dataset_batch_size, mongodb_batch_size)

    mongodb_client = MongoClient(args.mongodb_uri)
    mongodb_database = mongodb_client.get_database(args.mongodb_database)

    collected_items = mongodb_database.get_collection("collected_items")
    features_prismai = mongodb_database.get_collection("features_prismai")
    synthesized_texts = mongodb_database.get_collection("synthesized_texts")

    model = MetricScorer(
        args.model,
        batch_size=args.batch_size or args.model_batch_size,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
    )
    pre_processor = parse_pre_processors(args)

    mongodb_filter_query |= {"model.name": args.model}

    domains = args.mongodb_filter_domains or mongodb_filter_query.pop("domain", (None,))

    fields_projection = {
        "_id",
        "_ref_id",
        "domain",
        "lang",
        "type",
        "agent",
        "label",
    } | set(pre_processor.required_fields.keys())
    fields_projection = list(fields_projection)

    for domain in tqdm(domains, position=0, desc="Processing Domains"):
        if domain:
            mongodb_filter_query |= {"document.domain": domain}

        num_documents = features_prismai.count_documents(mongodb_filter_query)
        mongodb_limit = args.mongodb_limit or num_documents
        mongodb_skip = args.mongodb_skip

        for feature_documents in batched(
            tqdm(
                features_prismai.find(
                    mongodb_filter_query,
                    projection=["_id", "document._id", "document._synth_id"],
                    batch_size=mongodb_batch_size,
                    limit=mongodb_limit,
                    skip=mongodb_skip,
                ),
                total=mongodb_limit,
                desc=f"Processing Documents from {domain or 'All Domains'}",
                position=1,
                leave=False,
            ),
            dataset_batch_size,
        ):
            human_ids = []
            synth_ids = []
            human_lookup = []
            synth_lookup = []
            for features in feature_documents:
                if features["document"].get("_synth_id", None) is None:
                    human_lookup.append(features["document"]["_id"].id)
                    human_ids.append(features["_id"])
                else:
                    synth_lookup.append(features["document"]["_synth_id"].id)
                    synth_ids.append(features["_id"])

            dataset = []
            if human_lookup:
                dataset.extend(
                    collected_items.find(
                        {"_id": {"$in": human_lookup}}, projection=fields_projection
                    )
                )
            if synth_lookup:
                dataset.extend(
                    synthesized_texts.find(
                        {"_id": {"$in": synth_lookup}}, projection=fields_projection
                    )
                )

            dataset = Dataset(dataset)
            dataset.modify(
                DocumentMetadata.add_metadata_to_document,
                source_collection=args.source_collection,
            )
            dataset.update({"_id": _id} for _id in human_ids + synth_ids)
            dataset = pre_processor.pre_process(dataset)
            dataset = model.process(dataset, pre_processor.pad_token_id)
            dataset = pre_processor.post_process(dataset)

            for result in tqdm(
                dataset,
                desc="Inserting Document Batches",
                position=2,
                leave=False,
            ):
                try:
                    _id = result["_id"]
                    features_prismai.update_one(
                        {"_id": _id},
                        {"$set": result["values"]},
                    )
                except Exception:
                    traceback.print_exc()
