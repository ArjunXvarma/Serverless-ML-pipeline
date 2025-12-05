from dataclasses import dataclass

@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 30000
    min_df: int = 2