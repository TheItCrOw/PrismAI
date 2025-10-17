import argparse
from pathlib import Path

import ray
import torch
from datasets import DatasetDict, load_dataset
from ray.util import ActorPool
from tqdm import tqdm

from luminar.encoder import LuminarEncoder

HF_TOKEN = (Path.home() / ".hf_token").read_text().strip()


@ray.remote(
    num_gpus=1,
    # scheduling_strategy="SPREAD",
)
class LuminarEncoderActor(LuminarEncoder):
    def __init__(
        self,
        model_name_or_path="gpt2",
        max_len=1024,
        low_memory=False,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        get_lm_head_from_model_fn=None,
    ):
        super().__init__(
            model_name_or_path,
            max_len,
            low_memory,
            device,
            get_lm_head_from_model_fn,
        )
        self.model_str = Path(model_name_or_path.replace(":", "-")).stem

    def run(self, config_name: str, batch_size: int = 64):
        dataset: DatasetDict = load_dataset(
            "liberi-luminaris/PrismAI",
            config_name,
        )  # type: ignore
        dataset = dataset.map(
            self.tokenize,
            input_columns=["text"],
            batched=True,
            batch_size=1024,
            desc="Tokenizing",
            remove_columns=["text"],
        )
        dataset = dataset.sort("length")
        dataset = dataset.map(
            self.process,
            batched=True,
            batch_size=batch_size,
            desc="Encoding",
            remove_columns=["input_ids", "attention_mask"],
        )
        dataset.push_to_hub(
            f"liberi-luminaris/PrismAI-encoded-{self.model_str}",
            config_name,
            token=HF_TOKEN,
            private=True,
        )


DEFAULT_DOMAINS = (
    "blog_authorship_corpus",
    "student_essays",
    "cnn_news",
    "euro_court_cases",
    "house_of_commons",
    "arxiv_papers",
    "gutenberg_en",
    "en",
    "bundestag",
    "spiegel_articles",
    "gutenberg_de",
    "de",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=DEFAULT_DOMAINS,
        choices=list(sorted((DEFAULT_DOMAINS))),
    )
    parser.add_argument(
        "--en",
        action="store_const",
        const=(
            "blog_authorship_corpus",
            "student_essays",
            "cnn_news",
            "euro_court_cases",
            "house_of_commons",
            "arxiv_papers",
            "gutenberg_en",
            "en",
        ),
        dest="domains",
    )
    parser.add_argument(
        "--de",
        action="store_const",
        const=(
            "bundestag",
            "spiegel_articles",
            "gutenberg_de",
            "de",
        ),
        dest="domains",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path to use for encoding",
    )
    parser.add_argument(
        "-l",
        "--max_len",
        type=int,
        default=1024,
        help="Maximum sequence length for encoding (default: 1024)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "-p",
        "--pool_size",
        type=int,
        default=4,
        help="Size of the actor pool for parallel encoding",
    )

    args = parser.parse_args()

    devices = args.devices.split(",")
    if len(devices) == 1:
        devices = devices[0]
    else:
        devices = tuple(devices)

    pool = ActorPool(
        [
            LuminarEncoderActor.remote(args.model, max_len=args.max_len, device=devices)
            for _ in range(args.pool_size)
        ]
    )
    it = pool.map_unordered(
        lambda actor, domain: actor.run.remote(
            f"{domain}-fulltext", batch_size=args.batch_size
        ),
        args.domains,
    )
    for _ in tqdm(it, total=len(args.domains), desc="Encoding Datasets"):
        pass
