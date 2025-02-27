import logging
import pickle
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer
from datasets import Dataset
from speedy_utils.all import *

logger = logging.getLogger(__name__)


class TokenClassificationDataset:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def chunks_to_labels(self, chunks: List[Dict]) -> Dict:
        """Convert chunks to token classification labels."""
        full_text = "".join([chunk["text"] for chunk in chunks])
        token_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        token_labels = [0] * len(token_ids)

        # Create character-to-chunk mapping
        char_to_chunk_idx = []
        for i, chunk in enumerate(chunks):
            char_to_chunk_idx.extend([i] * len(chunk["text"]))

        # Map tokens to chunks
        position = 0
        for i, token_id in enumerate(token_ids):
            token_text = self.tokenizer.decode([token_id])
            chunk_indices = []
            for j in range(min(len(token_text), len(char_to_chunk_idx) - position)):
                char_pos = position + j
                if char_pos < len(char_to_chunk_idx):
                    chunk_indices.append(char_to_chunk_idx[char_pos])

            if chunk_indices:
                most_common_idx = max(set(chunk_indices), key=chunk_indices.count)
                token_labels[i] = 1 if chunks[most_common_idx]["need_translate"] else 0

            position += len(token_text)

        return {"text": full_text, "input_ids": token_ids, "labels": token_labels}

    def load_and_process_data(
        self,
        data_path: str,
        test_ratio: float = 0.1,
    ):
        """Load data and convert to HuggingFace dataset format, returning train and test splits."""
        data = self.process_and_cache(data_path)
        dataset = Dataset.from_list(data)

        # Split the dataset into train and test sets
        if test_ratio:
            train_test_split = dataset.train_test_split(test_size=test_ratio)
            train_dataset = train_test_split["train"]
            test_dataset = train_test_split["test"]

            return train_dataset, test_dataset
        return dataset

    def preprocess(self, data_path, sample_size, fold):
        logger.info(f"Loading data from {data_path}, {fold=}")
        data_df = load_by_ext(data_path)
        if fold:
            data_df = data_df[fold[0] :: fold[1]]
        if sample_size:
            data_df = data_df[:sample_size]

        def f(name):
            row = data_df.loc[name]
            return self.chunks_to_labels(row["chunks"])

        data = multi_thread(f, data_df.index.tolist())

        if not self.max_length:
            max_len = max([len(x["input_ids"]) for x in data])
            self.max_length = 8 * (max_len // 8 + 1)  # Make divisible by 8

        logger.info(f"Created dataset with {len(data)} examples")
        return data

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset."""
        return dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            ),
            batched=True,
        )

    def process_and_cache(self, data_path: str):
        cache_path = f"{data_path}.cache.pkl"
        lock_path = f"{cache_path}.lock"

        if os.path.exists(lock_path):
            logger.info("Cache is being processed by another process. Waiting...")
            while os.path.exists(lock_path):
                time.sleep(1)
            logger.info(
                "Cache processing completed by another process. Loading cache..."
            )

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                outputs = pickle.load(f)
        else:
            try:
                open(lock_path, "w").close()
                f = lambda fold_id: self.preprocess(
                    data_path, fold=(fold_id, 100), sample_size=None
                )
                outputs = multi_process(f, range(100), workers=100)
                outputs = [item for sublist in outputs for item in sublist]
                with open(cache_path, "wb") as f:
                    pickle.dump(outputs, f)
            finally:
                os.remove(lock_path)

        print("Num of outputs:", len(outputs))
        print("First output:", outputs[0])
        return outputs


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, nargs=2, default=None)
    args = parser.parse_args()
    path = "../localization/data/preprocess_242k_source_target_chunks.pkl"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tcd = TokenClassificationDataset(tokenizer, max_length=2048)
    tcd.process_and_cache(path)
