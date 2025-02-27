from lclzt.data_types import TranslateItem
from speedy_utils import *
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

import evaluate


model = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)


def chunks_to_labels(chunks):
    """
    Convert chunks to token classification labels with improved accuracy.
    """
    full_text = "".join([chunk["text"] for chunk in chunks])
    token_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    token_labels = [0] * len(token_ids)

    # Create character-to-chunk mapping
    char_to_chunk_idx = []
    for i, chunk in enumerate(chunks):
        char_to_chunk_idx.extend([i] * len(chunk["text"]))

    # Map tokens to chunks based on character positions
    position = 0
    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id])

        # Find which chunk(s) this token overlaps with
        chunk_indices = []
        for j in range(min(len(token_text), len(char_to_chunk_idx) - position)):
            char_pos = position + j
            if char_pos < len(char_to_chunk_idx):
                chunk_indices.append(char_to_chunk_idx[char_pos])

        if chunk_indices:
            # Use the most common chunk index (majority vote)
            most_common_idx = max(set(chunk_indices), key=chunk_indices.count)
            token_labels[i] = 1 if chunks[most_common_idx]["need_translate"] else 0

        position += len(token_text)

    decode_text = tokenizer.decode(token_ids)
    assert full_text == decode_text

    assert tokenizer(full_text)["input_ids"] == token_ids

    assert len(token_ids) == len(token_labels)

    return {"text": full_text, "token_ids": token_ids, "labels": token_labels}


def convert_to_hf_dataset(data):
    return Dataset.from_list(data)


# if not 'tokenized_dataset' in dir():
data_df = load_by_ext("./data/preprocess_242k_source_target_chunks.pkl")[:1000]
data = [chunks_to_labels(chunks) for chunks in data_df["chunks"].tolist()]

max_data_len= max([len(x["token_ids"]) for x in data])
# make it devisible by 8
max_data_len = 8 * (max_data_len // 8 + 1)

hf_dataset = convert_to_hf_dataset(data)
tokenized_dataset = hf_dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_data_len),
    batched=True,
)

# Define label mappings
id2label = {0: "keep", 1: "to_translate"}
label2id = {v: k for k, v in id2label.items()}

# Load the model
import torch
import torch.nn as nn
import torch.nn.functional as F
model = AutoModelForTokenClassification.from_pretrained(
    model, num_labels=len(id2label), id2label=id2label, label2id=label2id, torch_dtype=torch.float16
)

# set model to not trainable
# for param in model.parameters():
#     param.requires_grad(False)
model.requires_grad_(False)
# model.score layer is trainable
model.score.requires_grad_(True)
# Define data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Define evaluation metric
seqeval = evaluate.load("seqeval")
import numpy as np


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./trained_model")
