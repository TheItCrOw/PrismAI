import traceback
from argparse import Namespace
from itertools import batched
from pathlib import Path

import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

from calc_transition_scores import get_argparser, parse_pre_processors
from simple_dataset.dataset import Dataset
from transition_scores.scorer import TransformersTransitionScorer
from transition_scores.scorer.abc import (
    _ModelOutput,
)

if Path(".env").exists():
    load_dotenv()
elif Path("../.env").exists():
    load_dotenv("../.env")


class IntermediateLikelihoodScorer(TransformersTransitionScorer):
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
        hidden_states = self._forward(input_ids, attention_mask)

        results = []
        for target_ids, intermediate_probs in zip(input_ids, hidden_states):
            # Truncate the sequence to the last non-pad token
            labels = target_ids[1:]
            labels = labels[: labels.ne(pad_token_id).sum()]
            labels = labels.view(-1, 1).to(self.device)

            intermediate_probs = self._calculate_intermediate_probs(
                intermediate_probs, labels
            )

            results.append(
                {
                    "values": {
                        "transition_scores.target_ids": target_ids.tolist(),
                        "transition_scores.intermediate_probs": intermediate_probs.tolist(),
                    }
                }
            )

        return results

    def _forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[tuple[torch.Tensor, ...]]:
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            outputs: _ModelOutput = self._model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids.to(self.device),
                output_hidden_states=True,
            )

            # Unpack hidden states to get one list of tensors per input sequence,
            # instead of one hidden state per layer in the model
            hidden_states = zip(*[hs.cpu() for hs in outputs.hidden_states])

            del outputs
        return hidden_states


def append_il(args: Namespace, filter_query: dict, pipeline: list[dict]):
    mongodb_batch_size = args.batch_size or args.mongodb_batch_size
    dataset_batch_size = (
        args.batch_size or args.dataset_batch_size or mongodb_batch_size
    )
    dataset_batch_size = min(dataset_batch_size, mongodb_batch_size)

    mongodb_client = MongoClient(args.mongodb_uri)
    mongodb_database = mongodb_client.get_database(args.mongodb_database)

    features_prismai = mongodb_database.get_collection(args.target_collection)

    model = IntermediateLikelihoodScorer(
        args.model,
        batch_size=args.batch_size or args.model_batch_size,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
    )
    pre_processor = parse_pre_processors(args)

    num_documents = features_prismai.count_documents(filter_query)
    if not num_documents:
        return

    mongodb_total = args.mongodb_limit or num_documents

    for dataset in batched(
        tqdm(
            features_prismai.aggregate(pipeline),
            total=mongodb_total,
            desc="Processing Documents",
            position=1,
            leave=False,
        ),
        dataset_batch_size,
    ):
        dataset = Dataset(dataset)
        dataset = pre_processor.pre_process(dataset)
        dataset = model.process(dataset, pre_processor.pad_token_id)
        dataset = pre_processor.post_process(dataset)

        for result in dataset:
            try:
                _id = result["_id"]
                features_prismai.update_one(
                    {"_id": _id},
                    {"$set": result["values"]},
                )
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    args = get_argparser().parse_args()

    for domain in tqdm(
        args.mongodb_filter_domains or [None],
        desc="Processing Domains"
        if args.mongodb_filter_domains
        else "Processing All Domains",
    ):
        filter_query = {
            "model.name": args.model,
            "document.type": "source",
            # "document._synth_id": None,
        }
        if args.mongodb_filter:
            filter_query.update(args.mongodb_filter)
        if domain:
            filter_query["document.domain"] = domain

        append_il(
            args,
            filter_query,
            [
                {"$project": {"document": 1, "model": 1}},
                {"$match": filter_query},
            ]
            + ([{"$skip": args.mongodb_skip}] if args.mongodb_skip else [])
            + ([{"$limit": args.mongodb_limit}] if args.mongodb_limit else [])
            + [
                {
                    "$lookup": {
                        "from": "collected_items",
                        "localField": "document._id.$id",
                        "foreignField": "_id",
                        "as": "source",
                        "pipeline": [
                            {
                                "$project": {
                                    "_id": 1,
                                    "text": 1,
                                }
                            }
                        ],
                    }
                },
                {"$unwind": "$source"},
                {"$project": {"_id": 1, "text": "$source.text"}},
            ],
        )

        filter_query = {
            "model.name": args.model,
            "document.type": {"$ne": "source"},
            # "document._synth_id": {"$ne": None},
        }
        if args.mongodb_filter:
            filter_query.update(args.mongodb_filter)
        if domain:
            filter_query["document.domain"] = domain

        append_il(
            args,
            filter_query,
            [
                {"$project": {"document": 1, "model": 1}},
                {"$match": filter_query},
            ]
            + ([{"$skip": args.mongodb_skip}] if args.mongodb_skip else [])
            + ([{"$limit": args.mongodb_limit}] if args.mongodb_limit else [])
            + [
                {
                    "$lookup": {
                        "from": "synthesized_texts",
                        "localField": "document._synth_id.$id",
                        "foreignField": "_id",
                        "as": "source",
                        "pipeline": [
                            {
                                "$project": {
                                    "_id": 1,
                                    "text": 1,
                                }
                            }
                        ],
                    }
                },
                {"$unwind": "$source"},
                {"$project": {"_id": 1, "text": "$source.text"}},
            ],
        )
