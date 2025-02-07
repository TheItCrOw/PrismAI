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
from luminar.mongo import MongoDBAdapter

load_dotenv("../env")
seed_everything(42)
# for bs, fs in product((32, 64, 128), (128, 256, 512)):
for feature_size in (64, 128, 256):
    feature_size = OneDimFeatures(feature_size)
    for domain in (
        # "arxiv_papers",
        # "blog_authorship_corpus",
        # "bundestag",
        # "cnn_news",
        # "euro_court_cases",
        # "gutenberg",
        # "house_of_commons",
        # "spiegel_articles",
        "student_essays",
    ):
        try:
            db = MongoDBAdapter(
                os.environ.get("MONGO_DB_CONNECTION"),
                "prismai",
                "collected_items",
                "synthesized_texts",
                "transition_scores",
                domain=domain,
                source_collection_limit=1500,
            )
            seed_everything(42)
            dm = DocumentClassificationDataModule(db, feature_size)
            for lr, pdim in product((0.0005, 0.0001), (32, 64, 128)):
                try:
                    # for lr, wu, pr in product((1e-3, 1e-4, 1e-5), (0, 50, 100), (64, 128, 256)):
                    seed_everything(42)
                    model = DocumentClassficationModel(
                        dm.feature_selection.effective_shape(),
                        projection_dim=pdim,
                        learning_rate=lr,
                        warmup_steps=100,
                    )
                    trainer = Trainer(
                        max_epochs=50,
                        logger=pl_loggers.TensorBoardLogger(
                            save_dir="logs/",
                            name=domain,
                        ),
                        gradient_clip_val=0.5,
                        deterministic=True,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
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
