from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from torch import nn

from ads_ranking.data import EncodedData
from ads_ranking.evaluation import binary_metrics, ndcg_at_k, topk_average
from ads_ranking.models import (
    DeepFMModel,
    FactorizationMachineModel,
    LogisticRegressionModel,
    WideDeepModel,
)
from ads_ranking.ranking.auction import add_auction_scores
from ads_ranking.train import predict_probabilities, train_binary_model


ModelFactory = Callable[[EncodedData, dict], nn.Module]


def build_model_registry() -> dict[str, ModelFactory]:
    return {
        "logistic_regression": build_logistic_regression,
        "factorization_machine": build_factorization_machine,
        "wide_deep": build_wide_deep,
        "deepfm": build_deepfm,
    }


def build_logistic_regression(encoded_data: EncodedData, config: dict) -> nn.Module:
    return LogisticRegressionModel(
        category_sizes=encoded_data.category_sizes,
        num_numerical_features=len(encoded_data.feature_spec.numerical),
    )


def build_factorization_machine(encoded_data: EncodedData, config: dict) -> nn.Module:
    return FactorizationMachineModel(
        category_sizes=encoded_data.category_sizes,
        num_numerical_features=len(encoded_data.feature_spec.numerical),
        embedding_dim=config["embedding_dim"],
    )


def build_wide_deep(encoded_data: EncodedData, config: dict) -> nn.Module:
    return WideDeepModel(
        category_sizes=encoded_data.category_sizes,
        num_numerical_features=len(encoded_data.feature_spec.numerical),
        embedding_dim=config["embedding_dim"],
        hidden_dims=config["hidden_dims"],
    )


def build_deepfm(encoded_data: EncodedData, config: dict) -> nn.Module:
    return DeepFMModel(
        category_sizes=encoded_data.category_sizes,
        num_numerical_features=len(encoded_data.feature_spec.numerical),
        embedding_dim=config["embedding_dim"],
        hidden_dims=config["hidden_dims"],
    )


def run_model_experiment(
    model_name: str,
    encoded_data: EncodedData,
    training_config: dict,
    ranking_config: dict,
    seed: int,
    verbose: bool = False,
) -> dict[str, float | str]:
    registry = build_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {sorted(registry)}")

    factory = registry[model_name]
    ctr_model = factory(encoded_data, training_config)
    cvr_model = factory(encoded_data, training_config)

    ctr_model = train_binary_model(
        encoded_data=encoded_data,
        label_col="click",
        model=ctr_model,
        batch_size=training_config["batch_size"],
        epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        seed=seed,
        verbose=verbose,
    )
    cvr_model = train_binary_model(
        encoded_data=encoded_data,
        label_col="post_click_conversion",
        model=cvr_model,
        batch_size=training_config["batch_size"],
        epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        seed=seed + 1,
        verbose=verbose,
    )

    ranked = encoded_data.test.copy()
    ranked["pred_ctr"] = predict_probabilities(
        ctr_model,
        encoded_data,
        label_col="click",
        batch_size=training_config["batch_size"],
    )
    ranked["pred_cvr"] = predict_probabilities(
        cvr_model,
        encoded_data,
        label_col="post_click_conversion",
        batch_size=training_config["batch_size"],
    )
    ranked = add_auction_scores(ranked)

    ctr_metrics = binary_metrics(ranked["click"].to_numpy(), ranked["pred_ctr"].to_numpy())
    cvr_metrics = binary_metrics(
        ranked["post_click_conversion"].to_numpy(),
        ranked["pred_cvr"].to_numpy(),
    )
    ndcg = ndcg_at_k(
        ranked,
        group_col="request_id",
        label_col="conversion",
        score_col="expected_value_score",
        k=ranking_config["ndcg_k"],
    )
    top1_conversion = topk_average(
        ranked,
        group_col="request_id",
        score_col="expected_value_score",
        value_col="conversion",
        k=1,
    )

    return {
        "model": model_name,
        "ctr_auc": ctr_metrics["auc"],
        "ctr_logloss": ctr_metrics["logloss"],
        "cvr_auc": cvr_metrics["auc"],
        "cvr_logloss": cvr_metrics["logloss"],
        f"ndcg@{ranking_config['ndcg_k']}": ndcg,
        "top1_conversion_rate": top1_conversion,
    }


def run_experiment_suite(
    model_names: list[str],
    encoded_data: EncodedData,
    training_config: dict,
    ranking_config: dict,
    seed: int,
    verbose: bool = False,
) -> pd.DataFrame:
    results = []
    for model_name in model_names:
        print(f"\nRunning model: {model_name}")
        results.append(
            run_model_experiment(
                model_name=model_name,
                encoded_data=encoded_data,
                training_config=training_config,
                ranking_config=ranking_config,
                seed=seed,
                verbose=verbose,
            )
        )

    return pd.DataFrame(results)

