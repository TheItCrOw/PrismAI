import argparse
import torch
import optuna
import wandb
import traceback

from optuna.exceptions import TrialPruned
from pathlib import Path
from dataclasses import dataclass, fields
from functools import partial

from luminar.encoder import LuminarEncoder
from luminar.sequence_trainer import LuminarSequenceTrainer
from luminar.utils import (
    LuminarSequenceTrainingConfig,
    get_best_device,
    SequentialDataService,
    ConvolutionalLayerSpec
)

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
        config_dict["projection_dim"] = trial.suggest_categorical("projection_dim", [16, 32, 64, 128])
        config_dict["stack_spans"] = trial.suggest_categorical("stack_spans", [0, 1, 2, 3])
        config_dict["lstm_hidden_dim"] = trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128, 256])
        config_dict["learning_rate"] = trial.suggest_float("learning_rate", 6e-4, 4e-3, log=True)

        config = LuminarSequenceTrainingConfig(**config_dict)

        run = wandb.init(
            project="Luminar",
            config=config.__dict__,
            reinit=True,
            name=f"trial_{trial.number}",
            group=f"{config.domain}_study",
            tags=["optuna", "trial", config.domain, config.agent, config.feature_agent]
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
        model_store_path = Path(config.models_root_path) / config.domain / wandb.run.id
        best_model.save(model_store_path)

        # Upload to wandb as an artifact
        artifact = wandb.Artifact(name=f"luminar-sequence-{wandb.run.id}", type="model")
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
    luminar_encoder = LuminarEncoder(max_len=config.feature_len)
    data_service = SequentialDataService(luminar_encoder, config.batch_size, config.feature_len)

    domains = config.domain.split("___")
    dataset_paths = []
    for domain in domains:
        dataset_paths.append(Path(config.dataset_root_path) / config.agent / config.feature_agent / domain)
    dataset_dict = data_service.load_multiple_datasets(dataset_paths)
    print(dataset_dict)

    print("Transforming dataset to LuminarSequenceDataset")
    train_dataset, test_dataset, test_loader = data_service.dataset_to_luminar_sequence_dataset(dataset_dict)
    print("Loaded the data, initializing the training study...")

    # Partial application to pass shared variables into Optuna
    study = optuna.create_study(direction="maximize")
    objective_fn = partial(objective,
                           train_dataset=train_dataset,
                           test_loader=test_loader,
                           collate_fn=data_service._collate_fn,
                           device=device,
                           base_config=config)
    print("Running Optuna study:")
    study.optimize(objective_fn, n_trials=50)

    print("Best trial:", study.best_trial.params)
