# Token Classification

This project implements token classification using Hugging Face Transformers. It provides tools for training, evaluating, and running inference with token classification models.

## Features

- Train token classification models with proper evaluation metrics
- F1 score tracking during training/evaluation
- Easy inference with trained models

## Installation

```bash
pip install transformers datasets evaluate seqeval
```

## Training

To train a token classification model:

```bash
python train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path path/to/your/data.pkl \
  --output_dir ./output \
  --num_train_epochs 3 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --max_length 512 \
  --num_labels 2 \
  --eval_steps 500
```

## Inference

After training, you can run inference on new text:

```bash
python inference.py \
  --model_path ./output \
  --text "Your text to classify" \
  --device cpu
```

## Dataset Format

The dataset should be a pickle file containing a list of examples. Each example should have 'text' and optionally 'labels' fields.

## Metrics

The training process reports the following metrics:
- Precision
- Recall
- F1 score (used for model selection)
- Accuracy

## License

[MIT License](LICENSE)
