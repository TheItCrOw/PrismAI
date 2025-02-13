import argparse
import os
import traceback
from itertools import product

from dotenv import load_dotenv
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from luminar.document.data import DocumentClassificationDataModule
from luminar.document.model import DocumentClassficationModel
from luminar.features import OneDimFeatures
from luminar.mongo import PrismaiDataset

load_dotenv("../env")
seed_everything(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "domains",
        type=str,
        nargs="+",
        default=(
            "blog_authorship_corpus",
            "student_essays",
            "cnn_news",
            "bundestag",
            "euro_court_cases",
            "house_of_commons",
        ),
        choices=(
            (
                "blog_authorship_corpus",
                "student_essays",
                "cnn_news",
                "bundestag",
                "euro_court_cases",
                "house_of_commons",
                "spiegel_articles",
                "arxiv_papers",
                "gutenberg",
            )
        ),
    )
    args = parser.parse_args()

    for domain in args.domains:
        for feature_size in (32, 64, 128, 256):
            feature_size = OneDimFeatures(feature_size)
            try:
                db = PrismaiDataset(
                    os.environ.get("MONGO_DB_CONNECTION"),
                    "prismai",
                    "collected_items",
                    "synthesized_texts",
                    "log_likelihoods",
                    domain=domain,
                    source_collection_limit=1500,
                )
                seed_everything(42)
                dm = DocumentClassificationDataModule(db, feature_size)
                for lr, pdim, wus in product(
                    (0.0005, 0.0001), (8, 16, 32, 64, 128, 256), (0, 100)
                ):
                    try:
                        seed_everything(42)
                        model = DocumentClassficationModel(
                            feature_size,
                            projection_dim=pdim,
                            learning_rate=lr,
                            warmup_steps=wus,
                        )
                        trainer = Trainer(
                            max_epochs=50,
                            logger=pl_loggers.TensorBoardLogger(
                                save_dir="logs/",
                                name=domain,
                            ),
                            gradient_clip_val=0.5,
                            deterministic=True,
                            callbacks=[EarlyStopping(monitor="roc_auc", mode="max")],
                        )
                        # tuner = Tuner(trainer)
                        trainer.fit(model, dm)
                        (metrics,) = trainer.validate(
                            model, dataloaders=[dm.val_dataloader()]
                        )
                    except Exception:
                        traceback.print_exc()
            except Exception:
                traceback.print_exc()
