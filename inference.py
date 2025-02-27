import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    """Load tokenizer and model from the given path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return tokenizer, model

def predict(text, tokenizer, model, device="cpu"):
    """Run token classification inference on the given text."""
    # Move model to the specified device
    model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get the predicted token classes
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert to label names
    predicted_token_classes = []
    for i, pred_seq in enumerate(predictions):
        token_labels = [model.config.id2label[t.item()] for t in pred_seq]
        predicted_token_classes.append(token_labels)
    
    # Match tokens with their predicted labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    results = []
    for token, label in zip(tokens, predicted_token_classes[0]):
        results.append((token, label))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run token classification inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")
    
    args = parser.parse_args()
    
    logger.info(f"Loading model from {args.model_path}")
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    
    logger.info(f"Running inference on: {args.text}")
    results = predict(args.text, tokenizer, model, device=args.device)
    
    print("\nToken Classification Results:")
    print("-" * 50)
    for token, label in results:
        if token.startswith("##"):
            print(f"{token[2:]:20} | {label}")
        else:
            print(f"{token:20} | {label}")
    print("-" * 50)
    
if __name__ == "__main__":
    main()
