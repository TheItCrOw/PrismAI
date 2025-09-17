import argparse
import torch
import optuna
import wandb
import traceback

from datasets import DatasetDict
from optuna.exceptions import TrialPruned
from pathlib import Path
from dataclasses import dataclass, fields
from functools import partial

from data_hub.hub import DataHub
from data_hub.sequential_data_processor import SequentialDataProcessor
from luminar.encoder import LuminarEncoder
from luminar.sequence_trainer import LuminarSequenceTrainer
from luminar.utils import (
    LuminarSequenceTrainingConfig,
    get_best_device,
    ConvolutionalLayerSpec
)

HF_TOKEN : str = (Path.home() / ".hf_token").read_text().strip()
# Different Convolution Configs we want to try
NAMED_CONV_LAYER_CONFIGS = {
    "A": (
        ConvolutionalLayerSpec(64, 5),
        ConvolutionalLayerSpec(128, 5),
        ConvolutionalLayerSpec(64, 3),
    ),
    "B": (
        ConvolutionalLayerSpec(32, 5),
        ConvolutionalLayerSpec(64, 5),
        ConvolutionalLayerSpec(32, 3),
    ),
    "C": (
        ConvolutionalLayerSpec(128, 5),
        ConvolutionalLayerSpec(256, 5),
    ),
    "D": (
        ConvolutionalLayerSpec(8, 2),
        ConvolutionalLayerSpec(32, 4),
        ConvolutionalLayerSpec(8, 2),
    ),
}

def print_sanity_check(dataset : DatasetDict):
    splits_to_check = ["train", "eval", "test"]

    total_invalid_label_count = 0
    total_length_mismatch_count = 0

    for split in splits_to_check:
        if split not in dataset:
            print(f"Split '{split}' not found in dataset. Skipping.")
            continue

        print(f"\nChecking split: {split}")
        invalid_label_count = 0
        length_mismatch_count = 0

        for i, example in enumerate(dataset[split]):
            labels = example["span_labels"]
            spans = example["sentence_token_spans"]

            # Check for invalid labels
            if any(label not in (0, 1) for label in labels):
                #print(f"Invalid labels at {split}[{i}]: {labels}")
                invalid_label_count += 1

            # Check for length mismatch
            if len(labels) != len(spans):
                print(f"Length mismatch at {split}[{i}]: {len(labels)} labels vs {len(spans)} spans")
                length_mismatch_count += 1

        print(
            f"✅ {split} summary: {invalid_label_count} invalid label entries, {length_mismatch_count} length mismatches")

        total_invalid_label_count += invalid_label_count
        total_length_mismatch_count += length_mismatch_count

    # Overall summary
    print("\nOverall Summary:")
    print(f"Total entries with invalid labels: {total_invalid_label_count}")
    print(f"Total entries with length mismatch: {total_length_mismatch_count}")

def add_args_from_dataclass(parser: argparse.ArgumentParser, config_cls: type):
    for f in fields(config_cls):
        arg_type = f.type
        if arg_type == bool:
            parser.add_argument(f'--{f.name}', action='store_true' if f.default is False else 'store_false')
        else:
            parser.add_argument(f'--{f.name}', type=arg_type, default=f.default)

def objective(trial, train_dataset, test_loader, collate_fn, device, base_config):
    """
    Objective Function for Optuna
    """
    config_dict = vars(base_config).copy()

    try:
        # Suggest hyperparameters
        apply_delta_augmentation = trial.suggest_categorical("apply_delta_augmentation", [True, False])
        apply_product_augmentation = trial.suggest_categorical("apply_product_augmentation", [True, False])

        # Invalid combination check
        if apply_delta_augmentation and apply_product_augmentation:
            raise TrialPruned("Delta and Product augmentations cannot both be True")

        config_dict["conv_layer_shapes"] = NAMED_CONV_LAYER_CONFIGS[
            trial.suggest_categorical("conv_config", list(NAMED_CONV_LAYER_CONFIGS.keys()))
        ]
        config_dict["apply_delta_augmentation"] = apply_delta_augmentation
        config_dict["apply_product_augmentation"] = apply_product_augmentation
        config_dict["projection_dim"] = trial.suggest_categorical("projection_dim", [64, 128, 256])
        config_dict["stack_spans"] = trial.suggest_categorical("stack_spans", [1, 2, 3, 4, 5])
        config_dict["lstm_hidden_dim"] = trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128, 256])
        #config_dict["learning_rate"] = trial.suggest_float("learning_rate", 6e-4, 4e-3, log=True)

        config = LuminarSequenceTrainingConfig(**config_dict)

        hf_dataset_names = [hf_dataset.split("/")[1] for hf_dataset in config.hf_dataset.split("___")]
        group_name = f"{'_'.join(hf_dataset_names)}{f":{config.domain}" if config.domain else ''}"
        tags = [tag for tag in ["optuna", "trial", *hf_dataset_names, config.domain, config.feature_agent] if tag is not None]#, config.agent, config.feature_agent]

        run = wandb.init(
            project="LuminarSeq",
            config=config.__dict__,
            reinit=True,
            name=f"trial_{trial.number}",
            group=group_name,
            tags=tags
        )

        trainer = LuminarSequenceTrainer(
            train_dataset=train_dataset,
            test_data_loader=test_loader,
            collate_fn=collate_fn,
            log_to_wandb=True,
            config=config,
            device=device
        )

        metrics, best_model = trainer.train()
        avg_f1 = metrics.get("f1_score", 0.0)
        wandb.log({"objective_f1": avg_f1})

        # Save model locally
        model_store_path = Path(config.models_root_path) / group_name / run.id
        best_model.save(model_store_path)

        # Upload to wandb as an artifact
        artifact = wandb.Artifact(name=f"luminar-sequence-{run.id}", type="model")
        artifact.add_dir(str(model_store_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        return avg_f1

    except TrialPruned as e:
        print(f"[Trial {trial.number}] Pruned: {e}")
        wandb.finish()
        raise e

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Trial {trial.number}] Failed with exception:\n{tb}")
        if wandb.run:
            wandb.log({"error": str(e), "traceback": tb})
        wandb.finish()
        raise TrialPruned(f"[Trial {trial.number}] Pruned: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args_from_dataclass(parser, LuminarSequenceTrainingConfig)
    args = parser.parse_args()
    config = LuminarSequenceTrainingConfig(**vars(args))

    device = get_best_device()
    print(f"Starting new training on device {device} with training config :")
    print(config.json())

    # Initialize encoder and data
    print("Loading and setting up the data...")
    luminar_encoder = LuminarEncoder(max_len=config.feature_len, model_name_or_path=config.feature_agent, device=device)
    data_hub = DataHub(HF_TOKEN)

    hf_datasets = config.hf_dataset.split("___") if config.hf_dataset else [config.hf_dataset]

    # If we have domain filters, we apply them onto the datasets
    domains = config.domain.split("___") if config.domain else None
    filters = None
    if domains is not None:
        for domain in domains:
            filters = {"domain": domain} if filters is None else {**filters, "domain": domain}

    dataset_dict = DataHub.concat_splits(data_hub.get_many_splits(hf_datasets,
                                                                  min_length=config.min_character_length,
                                                                  filter_by=filters))
    print(dataset_dict)

    print("Processing the dataset for sequential training...")
    data_processor = SequentialDataProcessor(luminar_encoder)
    dataset_dict = data_processor.process_for_training(dataset_dict)
    print(dataset_dict)

    print("Transforming dataset to LuminarSequenceDataset")
    train_dataset, test_dataset, test_loader = data_processor.dataset_to_luminar_sequence_dataset(dataset_dict)

    print("Loaded the data, initializing the training study...")
    # Partial application to pass shared variables into Optuna
    study = optuna.create_study(direction="maximize")
    objective_fn = partial(objective,
                           train_dataset=train_dataset,
                           test_loader=test_loader,
                           collate_fn=data_processor.collate_fn,
                           device=device,
                           base_config=config)

    print("Running Optuna study:")
    study.optimize(objective_fn, n_trials=50)

    print("Best trial:", study.best_trial.params)
