from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = None  # Will be set dynamically
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    weight_decay: float = 0.01
    output_dir: str = "./results"
    data_path: str = "../localization/data/preprocess_242k_source_target_chunks.pkl"
    model_save_path: str = "./trained_model"
    sample_size: int = 10  # Set to None to use all data
    
    # Label mappings
    id2label = {
        0: "keep",
        1: "to_translate"
    }
    label2id = {
        "keep": 0,
        "to_translate": 1
    }
    
    # Evaluation settings
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
