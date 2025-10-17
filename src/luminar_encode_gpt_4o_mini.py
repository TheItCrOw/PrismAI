from datasets import DatasetDict, load_dataset
from tqdm import tqdm

from luminar.encoder import LuminarEncoder

HF_TOKEN = open("/home/staff_homes/mastoeck/.hf_token").read().strip()


encoder = LuminarEncoder("gpt2", max_len=1024, device=("cuda:0", "cuda:1"))

configurations = [
    f"{domain}-fulltext-{agent}"
    for agent in (
        "gpt_4o_mini",
        # "gemma2_9b",
    )
    for domain in tqdm(
        [
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
        ]
    )
]
for config_name in tqdm(configurations, desc="Encoding Datasets"):
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
        batch_size=256,
        desc="Encoding",
        remove_columns=["input_ids", "attention_mask"],
    )
    dataset.push_to_hub(
        "liberi-luminaris/PrismAI-encoded-gpt2",
        config_name,
        token=HF_TOKEN,
        private=True,
    )
