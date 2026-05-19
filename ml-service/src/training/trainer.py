"""
Training loop for readmission prediction model.
Handles class imbalance, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, f1_score,
                              classification_report, confusion_matrix)

from models.readmission_model import ReadmissionModel, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_processed_data(data_dir: str):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    logger.info(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size: int):
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return total_loss / len(loader), auc, f1, all_probs, all_labels


def train(data_dir: str, model_output_dir: str):
    set_seed(ModelConfig.RANDOM_SEED)
    os.makedirs(model_output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    # Dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, ModelConfig.BATCH_SIZE
    )

    # Model
    model = ReadmissionModel(
        input_dim=ModelConfig.INPUT_DIM,
        hidden_dims=ModelConfig.HIDDEN_DIMS,
        dropout_rate=ModelConfig.DROPOUT_RATE
    ).to(device)
    logger.info(f"Model architecture:\n{model}")

    # Loss with pos_weight to handle class imbalance
    # pos_weight=8 means false negatives penalized 8x more than false positives
    # Clinical rationale: missing a high-risk patient is worse than
    # over-flagging a low-risk one
    pos_weight = torch.tensor([ModelConfig.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=ModelConfig.LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    # Training loop with early stopping
    best_auc = 0
    patience_counter = 0
    best_model_path = os.path.join(model_output_dir, 'best_model.pt')

    for epoch in range(ModelConfig.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer,
                                  criterion, device)
        val_loss, val_auc, val_f1, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_f1': val_f1,
                'config': {
                    'input_dim': ModelConfig.INPUT_DIM,
                    'hidden_dims': ModelConfig.HIDDEN_DIMS,
                    'dropout_rate': ModelConfig.DROPOUT_RATE
                }
            }, best_model_path)
            logger.info(f"  Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= ModelConfig.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation on best model
    logger.info("\n=== FINAL EVALUATION ===")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, test_auc, test_f1, probs, labels = evaluate(
        model, test_loader, criterion, device
    )
    preds = (probs >= 0.5).astype(int)

    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(labels, preds)}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(labels, preds)}")

    logger.info(f"\nBest model saved to: {best_model_path}")
    return model, test_auc


if __name__ == '__main__':
    train(
        data_dir='/home/aman/clinical-ai-platform/data/processed',
        model_output_dir='/home/aman/clinical-ai-platform/ml-service/models'
    )