import logging
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from dataset import TokenClassificationDataset
from trainer import TokenClassificationTrainer
from utils import create_token_classification_labels, format_metrics_for_logging
import os
import evaluate
import numpy as np
from peft import (
    PeftModelForSequenceClassification,
    get_peft_config,
    LoraConfig,
    PeftModelForTokenClassification,
    TaskType,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B", type=str)
    parser.add_argument("--data_path", default="../localization/data/preprocess_242k_source_target_chunks.pkl", type=str)
    parser.add_argument("--output_dir", default="./outputs/", type=str)
    parser.add_argument("--num_train_epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_labels", default=2, type=int, help="Number of token classification labels")
    parser.add_argument("--eval_steps", default=500, type=int, help="Steps between evaluations")
    parser.add_argument("--logging_steps", default=100, type=int, help="Logging steps")
    parser.add_argument("--evaluation_strategy", default="steps", type=str, help="Evaluation strategy")
    parser.add_argument("--save_strategy", default="steps", type=str, help="Save strategy")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Warmup steps")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end")
    parser.add_argument("--metric_for_best_model", default="accuracy", type=str, help="Metric for best model")
    parser.add_argument("--greater_is_better", action="store_true", help="Whether greater metric is better")
    parser.add_argument("--report_to", default="none", type=str, help="Reporting destination")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    tcd = TokenClassificationDataset(tokenizer, max_length=args.max_length)
    train_ds, test_ds = tcd.load_and_process_data(args.data_path)

    # Create label mappings
    id2label, label2id = create_token_classification_labels(args.num_labels)
    logger.info(f"Using label mapping: {id2label}")

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        id2label=id2label,
        label2id=label2id,
        # device_map="auto",
    )
    # model.score.requires_grad_(True)

    # peft_config = LoraConfig(
    #     task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1
    # )
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
    )

    peft_model = PeftModelForTokenClassification(model, peft_config)
    

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to,
        seed=args.seed,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Create data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = TokenClassificationTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    peft_model.print_trainable_parameters()
    logger.info("Starting training...")

    # Start training
    trainer.train()

    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    logger.info(format_metrics_for_logging(final_metrics))

    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
