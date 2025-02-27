import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def create_token_classification_labels(num_labels=2):
    """
    Create a mapping of label IDs to label names for token classification.
    
    Args:
        num_labels: Number of label classes to create
        
    Returns:
        Tuple of (id2label, label2id) dictionaries
    """
    if num_labels == 2:
        # Binary classification (e.g., 0 for regular text, 1 for special text)
        id2label = {0: "O", 1: "SPECIAL"}
        label2id = {"O": 0, "SPECIAL": 1}
    else:
        # For more complex labeling schemes, expand this function
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        label2id = {f"LABEL_{i}": i for i in range(num_labels)}
    
    return id2label, label2id

def format_metrics_for_logging(metrics):
    """Format metrics dictionary for readable logging output"""
    formatted = "\n".join([f"  {k}: {v:.4f}" for k, v in metrics.items()])
    return f"Metrics:\n{formatted}"
