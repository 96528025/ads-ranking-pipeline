from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ads_ranking.data import AdsDataset, EncodedData


def train_binary_model(
    encoded_data: EncodedData,
    label_col: str,
    model: nn.Module,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    verbose: bool = True,
) -> nn.Module:
    torch.manual_seed(seed)
    dataset = AdsDataset(encoded_data.train, encoded_data.feature_spec, label_col)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for categorical, numerical, labels in loader:
            optimizer.zero_grad()
            logits = model(categorical, numerical)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        avg_loss = total_loss / len(dataset)
        if verbose:
            print(f"{label_col} epoch={epoch + 1} loss={avg_loss:.4f}")

    return model


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    encoded_data: EncodedData,
    label_col: str,
    batch_size: int,
) -> np.ndarray:
    dataset = AdsDataset(encoded_data.test, encoded_data.feature_spec, label_col)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions: list[np.ndarray] = []
    for categorical, numerical, _ in loader:
        logits = model(categorical, numerical)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions.append(probabilities)

    return np.concatenate(predictions)
