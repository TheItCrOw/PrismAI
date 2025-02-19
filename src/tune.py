import argparse
import json
import os
from functools import partial

import torch
from dotenv import load_dotenv
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from luminar.document.data import (
    DocumentClassificationDataModule,
)
from luminar.document.model import CNNDocumentClassficationModel
from luminar.features import FeatureExtractor, OneDimFeatures, Slicer, TwoDimFeatures
from luminar.mongo import PrismaiDataset

load_dotenv("../env")


def train_func(config):
    slicer = config["slicer"](config["feature_dim"])
    match config["n_dim"]:
        case 1:
            config["feature_dim"] = OneDimFeatures(config["feature_dim"])
            featurizer = config["featurizer"]()
        case 2:
            config["feature_dim"] = TwoDimFeatures(
                config["feature_dim"], config["second_dim"]
            )
            featurizer = config["featurizer"](config["second_dim"])
        case _:
            raise ValueError(f"Unknown n_dim {config['n_dim']}")

    seed_everything(config["seed"])
    model = CNNDocumentClassficationModel(
        feature_dim=config["feature_dim"],
        projection_dim=config["projection_dim"],
        conv_layer_shapes=config["conv_layer_shapes"],
        learning_rate=config["lr"],
        warmup_steps=config.get("warmup_steps", 100),
        train_batch_size=config["batch_size"],
        eval_batch_size=config["batch_size"],
        second_dim_as_channels=config.get("second_dim_as_channels", False),
    )
    model.forward(torch.randn((config["batch_size"], *config["feature_dim"])))

    # dataset = PrismaiDataset(
    #     mongo_db_connection=os.environ.get("MONGO_DB_CONNECTION"),
    #     database="prismai",
    #     collection="features_prismai",
    #     domain=None,
    #     # domain=config["domain"],
    #     lang=config.get("lang", None),
    # )

    dataset = PrismaiDataset(
        mongo_db_connection=os.environ.get("MONGO_DB_CONNECTION"),
        database="prismai",
        collection="features_prismai",
        domain=None,
        # domain=config["domain"],
        lang=None,  # config.get("lang", None),
    )
    datamodule = DocumentClassificationDataModule(
        dataset,
        slicer=slicer,
        featurizer=featurizer,
        feature_dim=config["feature_dim"],
        train_batch_size=config["batch_size"],
        eval_batch_size=config["batch_size"],
        num_samples=config.get("num_samples", None),
    )

    trainer = Trainer(
        max_epochs=50,
        logger=pl_loggers.TensorBoardLogger(
            save_dir="logs/param_search/",
            name="all_domains",
        ),
        gradient_clip_val=config.get("gradient_clip_val", 0.5),
        deterministic=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            RayTrainReportCallback(),
        ],
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=datamodule)

    (metrics,) = trainer.test(model, datamodule=datamodule)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        choices=["gpt2", "meta-llama/Llama-3.2-1B"],
    )

    parser.add_argument(
        "featurizer",
        type=str,
        choices=["likelihood", "llr", "top_k_ratio", "intermediate"],
    )

    # parser.add_argument(
    #     "--domains",
    #     type=str,
    #     nargs="*",
    #     default=(
    #         "arxiv_papers",
    #         "blog_authorship_corpus",
    #         "cnn_news",
    #         "euro_court_cases",
    #         "gutenberg",
    #         "house_of_commons",
    #         "student_essays",
    #         "bundestag",
    #         "spiegel_articles",
    #     ),
    # )

    args = parser.parse_args()

    # match args.domains:
    #     case ["all"]:
    #         domain_filter = {
    #             "$in": [
    #                 "blog_authorship_corpus",
    #                 "student_essays",
    #                 "cnn_news",
    #                 "euro_court_cases",
    #                 "house_of_commons",
    #                 "arxiv_papers",
    #                 "gutenberg",
    #                 "bundestag",
    #                 "spiegel_articles",
    #             ]
    #         }
    #     case [] | ["None" | "none"]:
    #         domain_filter = None
    #     case [domain]:
    #         domain_filter = domain
    #     case domains:
    #         domain_filter = {"$in": domains}

    search_space = {
        "seed": 42,
        "conv_layer_shapes": tune.choice(
            [
                [(16, 5), (32, 3), (16, 3)],
                [(32, 5), (64, 3), (32, 3)],
                [(64, 5), (128, 3), (64, 3)],
                [(64, 5), (128, 3), (128, 3), (64, 3)],
                # SeqXGPT Layer Configuration
                [(64, 5), (128, 3), (128, 3), (128, 3), (64, 3)],
            ]
        ),
        "projection_dim": tune.choice([None, 32, 64, 128, 256]),
        "lr": 0.0001,
        "batch_size": 32,
        # "lr": tune.loguniform(5e-4, 1e-5),
        # "batch_size": tune.choice([32, 64]),
        "slicer": tune.choice(
            [
                # Slicer.First,
                # Slicer.Random,
                # partial(Slicer.RandomMultiple, multiple=2, infer_slice_size=True),
                # partial(Slicer.RandomMultiple, multiple=4, infer_slice_size=True),
                # partial(Slicer.RandomMultiple, multiple=8, infer_slice_size=True),
                # partial(Slicer.RandomMultiple, multiple=16, infer_slice_size=True),
                partial(
                    Slicer.RandomMultiple, multiple=2, stride=16, infer_slice_size=True
                ),
                partial(
                    Slicer.RandomMultiple, multiple=2, stride=32, infer_slice_size=True
                ),
                partial(
                    Slicer.RandomMultiple, multiple=4, stride=16, infer_slice_size=True
                ),
                partial(
                    Slicer.RandomMultiple, multiple=4, stride=32, infer_slice_size=True
                ),
                partial(
                    Slicer.RandomMultiple, multiple=8, stride=16, infer_slice_size=True
                ),
                partial(
                    Slicer.RandomMultiple, multiple=8, stride=32, infer_slice_size=True
                ),
            ]
        ),
        # "num_samples": tune.choice([None, 4, 8]),
        # "warmup_steps": tune.randint(0, 132),
    }

    match args.featurizer:
        case "likelihood" | "llr":
            search_space |= {
                "n_dim": 1,
                "feature_dim": tune.choice([64, 128, 256]),
                "featurizer": (
                    FeatureExtractor.Likelihood
                    if args.featurizer == "likelihood"
                    else FeatureExtractor.LogLikelihoodLogRankRatio
                ),
            }
        case "top_k_ratio":
            search_space |= {
                "n_dim": 2,
                "feature_dim": tune.choice([64, 128, 256]),
                "second_dim": tune.choice([4, 8, 16]),
                "featurizer": FeatureExtractor.LikelihoodTopkLikelihoodRatio,
                "second_dim_as_channels": True,
                # "second_dim_as_channels": tune.choice([True, False]),
            }
        case "intermediate":
            match args.model:
                case "gpt2":
                    num_layers = [13, 9]  # , 7, 5
                case llama if "llama" in args.model.lower():
                    num_layers = [17, 13, 9]  # , 7, 5
                case _:
                    raise ValueError(f"Unknown model {args.model}")

            search_space |= {
                "n_dim": 2,
                "feature_dim": tune.choice([64, 128, 256]),
                "second_dim": tune.choice(num_layers),
                "featurizer": FeatureExtractor.IntermediateLikelihood,
                "second_dim_as_channels": True,
                # "second_dim_as_channels": tune.choice([True, False]),
            }

    dataset = PrismaiDataset(
        mongo_db_connection=os.environ.get("MONGO_DB_CONNECTION"),
        database="prismai",
        collection="features_prismai",
        domain=None,
        # domain=config["domain"],
        lang=None,  # config.get("lang", None),
    )
    if not dataset.get_cache_file().exists():
        dataset.load()
    del dataset

    scaling_config = ScalingConfig(
        num_workers=4, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 0.25}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_acc@0.5",
            checkpoint_score_order="max",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    num_samples = 100
    scheduler = ASHAScheduler(max_t=5, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_acc@0.5",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()

    print(results)
    print(json.dumps(results, indent=2))
