"""
PyTorch model for 30-day diabetic patient readmission prediction.
Architecture: Multilayer Perceptron with BatchNorm and Dropout.

Design decisions:
- MLP over tree models: we want a PyTorch model for the serving pipeline
- BatchNorm: stabilizes training on mixed clinical features
- Dropout: regularization — clinical tabular data overfits easily
- Single output neuron with sigmoid: binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadmissionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64],
                 dropout_rate: float = 0.3):
        """
        Args:
            input_dim: number of input features (42 in our case)
            hidden_dims: list of hidden layer sizes
            dropout_rate: dropout probability for regularization
        """
        super(ReadmissionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build layers dynamically from hidden_dims list
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer — single neuron, no activation here
        # BCEWithLogitsLoss applies sigmoid internally (numerically stable)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability of positive class (readmission)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class ModelConfig:
    """
    Centralized model configuration.
    Every hyperparameter has a reason — be ready to defend each one.
    """
    INPUT_DIM = 42
    HIDDEN_DIMS = [256, 128, 64]
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    EPOCHS = 50
    POS_WEIGHT = 8.0        # ~90k negative / ~11k positive = ~8x imbalance
    EARLY_STOPPING_PATIENCE = 5
    RANDOM_SEED = 42