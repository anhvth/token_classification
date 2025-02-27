import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
import evaluate

logger = logging.getLogger(__name__)

class TokenClassificationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the seqeval metric for entity-based evaluation
        self.seqeval = evaluate.load("seqeval")
        
        # Get the labels from the model's config
        self.label_list = list(self.model.config.id2label.values())
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom compute_loss that handles potential issues with the input format.
        """
        # Log inputs for debugging (only once during training)
        if not hasattr(self, '_debug_logged'):
            logger.info(f"Input keys: {inputs.keys()}")
            if 'input_ids' in inputs:
                logger.info(f"input_ids shape: {inputs['input_ids'].shape}")
            if 'labels' in inputs:
                logger.info(f"labels shape: {inputs['labels'].shape}")
            self._debug_logged = True
        
        # Make sure all required keys are present
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Required key '{key}' not found in inputs")
        
        # Forward pass
        outputs = model(**inputs)
        
        # If we're returning outputs, do so
        if return_outputs:
            return outputs.loss, outputs
            
        return outputs.loss
    
    def __compute_metrics(self, eval_preds):
        """
        Compute metrics for token classification evaluation.
        Especially focusing on F1 score for entities.
        """
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Compute seqeval metrics
        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        
        # Log detailed metrics
        logger.info(f"Evaluation results: {results}")
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to ensure we get proper token classification metrics.
        """
        # Call the parent class evaluate method
        self.compute_metrics = self.__compute_metrics
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Log the F1 score specifically
        logger.info(f"F1 Score: {metrics.get(f'{metric_key_prefix}_f1', 'N/A')}")
        
        return metrics
    
    