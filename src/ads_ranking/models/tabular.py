from __future__ import annotations

import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    """Linear baseline over categorical IDs and numerical features."""

    def __init__(self, category_sizes: dict[str, int], num_numerical_features: int):
        super().__init__()
        self.categorical_weights = nn.ModuleList(
            [nn.Embedding(size, 1) for size in category_sizes.values()]
        )
        self.numerical_linear = nn.Linear(num_numerical_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        cat_terms = [weight(categorical[:, index]) for index, weight in enumerate(self.categorical_weights)]
        categorical_logit = torch.cat(cat_terms, dim=1).sum(dim=1)
        numerical_logit = self.numerical_linear(numerical).squeeze(1)
        return categorical_logit + numerical_logit + self.bias


class FactorizationMachineModel(nn.Module):
    """Linear terms plus pairwise embedding interactions."""

    def __init__(
        self,
        category_sizes: dict[str, int],
        num_numerical_features: int,
        embedding_dim: int,
        interaction_scale: float = 0.1,
    ):
        super().__init__()
        self.interaction_scale = interaction_scale
        self.categorical_weights = nn.ModuleList(
            [nn.Embedding(size, 1) for size in category_sizes.values()]
        )
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in category_sizes.values()]
        )
        self.numerical_linear = nn.Linear(num_numerical_features, 1)
        self.numerical_projection = nn.Linear(num_numerical_features, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        linear_terms = [
            weight(categorical[:, index])
            for index, weight in enumerate(self.categorical_weights)
        ]
        linear_logit = torch.cat(linear_terms, dim=1).sum(dim=1)
        linear_logit = linear_logit + self.numerical_linear(numerical).squeeze(1) + self.bias

        categorical_vectors = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.categorical_embeddings)
        ]
        numerical_vector = self.numerical_projection(numerical)
        fields = torch.stack([*categorical_vectors, numerical_vector], dim=1)

        summed = fields.sum(dim=1)
        summed_square = summed * summed
        square_summed = (fields * fields).sum(dim=1)
        pairwise_interactions = 0.5 * (summed_square - square_summed).sum(dim=1)
        return linear_logit + self.interaction_scale * pairwise_interactions


class WideDeepModel(nn.Module):
    """Embedding-based MLP over sparse and dense features."""

    def __init__(
        self,
        category_sizes: dict[str, int],
        num_numerical_features: int,
        embedding_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in category_sizes.values()]
        )
        input_dim = len(category_sizes) * embedding_dim + num_numerical_features
        self.network = make_mlp(input_dim, hidden_dims)

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat([*embedded, numerical], dim=1)
        return self.network(features).squeeze(1)


class DeepFMModel(nn.Module):
    """DeepFM combines memorized pairwise interactions with nonlinear deep features."""

    def __init__(
        self,
        category_sizes: dict[str, int],
        num_numerical_features: int,
        embedding_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.fm = FactorizationMachineModel(category_sizes, num_numerical_features, embedding_dim)
        self.deep_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in category_sizes.values()]
        )
        input_dim = len(category_sizes) * embedding_dim + num_numerical_features
        self.deep_network = make_mlp(input_dim, hidden_dims)

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.deep_embeddings)
        ]
        deep_features = torch.cat([*embedded, numerical], dim=1)
        return self.fm(categorical, numerical) + self.deep_network(deep_features).squeeze(1)


def make_mlp(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.15),
            ]
        )
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, 1))
    return nn.Sequential(*layers)
