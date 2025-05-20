import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset
from tqdm import tqdm

from luminar.encoder import LuminarEncoder

HF_TOKEN = (Path.home() / ".hf_token").read_text().strip()

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
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
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
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for encoding",
    )

    args = parser.parse_args()

    devices = args.devices.split(",")
    if len(devices) == 1:
        devices = devices[0]
    else:
        devices = tuple(devices)

    encoder = LuminarEncoder("gpt2", max_len=1024, device=devices)

    tq = tqdm(
        [f"{domain}-fulltext" for domain in args.domains], desc="Encoding Datasets"
    )
    for config_name in tq:
        tq.set_postfix_str(f"domain: {config_name}")

        dataset: DatasetDict = load_dataset("liberi-luminaris/PrismAI", config_name)  # type: ignore
        dataset = dataset.map(
            encoder.tokenize,
            input_columns=["text"],
            batched=True,
            batch_size=1024,
            desc="Tokenizing",
            remove_columns=["text"],
        )
        dataset = dataset.sort("length")
        dataset = dataset.map(
            encoder.process,
            batched=True,
            batch_size=args.batch_size,
            desc="Encoding",
            remove_columns=["input_ids", "attention_mask"],
        )
        dataset.push_to_hub(
            "liberi-luminaris/PrismAI-encoded-gpt2",
            config_name,
            token=HF_TOKEN,
            private=True,
        )
