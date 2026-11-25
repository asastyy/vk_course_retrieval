#!/usr/bin/env python
"""Train a CPU ranker and generate predictions in the Kaggle CSV format."""

import argparse
from pathlib import Path
from timeit import default_timer as timer

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

SEED = 42
THREADS = 4
MAX_ITERATIONS = 650
PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=[0, 1, 3, 7],
    lambdarank_truncation_level=13,
    learning_rate=0.035,
    num_leaves=31,
    min_data_in_leaf=60,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=5.0,
    seed=SEED,
    num_threads=THREADS,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)
EXCLUDE = {"QueryId", "ObjectId", "Label"}


def make_features(frame, columns):
    raw = frame[columns].astype("float32")
    varying = [
        column for column in columns
        if not (column.startswith("q_") or column.endswith(("_rank", "_to_max", "_minus_mean")))
    ]
    grouped = raw[varying].groupby(frame["QueryId"], sort=False)
    ranks = grouped.rank(method="average", pct=True).astype("float32")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    zscores = ((raw[varying] - mean) / std).fillna(0).astype("float32")
    return pd.concat(
        [raw, ranks.add_suffix("__qrank"), zscores.add_suffix("__qz")], axis=1
    )


def group_sizes(frame):
    return frame.groupby("QueryId", sort=False).size().to_numpy()


def ndcg_at_10(labels, scores, sizes):
    discounts = 1 / np.log2(np.arange(2, 12))
    results = []
    for groups in zip(
        np.split(labels, np.cumsum(sizes)[:-1]),
        np.split(scores, np.cumsum(sizes)[:-1]),
    ):
        relevance, predictions = groups
        gains = np.exp2(relevance.astype(float)) - 1
        order = np.argsort(-predictions, kind="stable")[:10]
        weights = discounts[:len(order)]
        ideal = np.sort(gains)[::-1][:10] @ weights
        results.append(float(gains[order] @ weights / ideal) if ideal else 0.0)
    return float(np.mean(results))


def train(data_dir, model_file):
    data = pd.read_csv(data_dir / "train.csv").sort_values("QueryId", kind="stable")
    columns = [column for column in data.columns if column not in EXCLUDE]
    features = make_features(data, columns)

    queries = data["QueryId"].unique()
    if len(queries) < 2:
        raise ValueError("Training requires at least two distinct queries")
    validation_queries = np.random.default_rng(SEED).permutation(queries)[
        :max(1, len(queries) // 5)
    ]
    validation_mask = data["QueryId"].isin(validation_queries)
    fit = data.loc[~validation_mask]
    valid = data.loc[validation_mask]
    valid_sizes = group_sizes(valid)
    valid_labels = valid["Label"].to_numpy()

    baseline = CatBoostRanker(
        loss_function="YetiRank", iterations=200, random_seed=SEED,
        thread_count=THREADS, allow_writing_files=False, verbose=False,
    )
    baseline.fit(Pool(fit[columns], fit["Label"], group_id=fit["QueryId"]))
    baseline_score = ndcg_at_10(
        valid_labels, baseline.predict(valid[columns]), valid_sizes
    )
    print(f"CatBoost baseline: validation nDCG@10={baseline_score:.6f}", flush=True)

    experiments = [
        ("raw", columns, {}),
        ("query features", list(features.columns), {}),
        ("query features, 63 leaves", list(features.columns),
         {"num_leaves": 63, "min_data_in_leaf": 100}),
    ]
    best_score = -1.0
    for name, selected_columns, overrides in experiments:
        params = PARAMS | overrides
        fit_set = lgb.Dataset(
            features.loc[~validation_mask, selected_columns],
            label=fit["Label"], group=group_sizes(fit),
        )
        valid_set = lgb.Dataset(
            features.loc[validation_mask, selected_columns],
            label=valid["Label"], group=valid_sizes, reference=fit_set,
        )
        model = lgb.train(
            params, fit_set, num_boost_round=MAX_ITERATIONS, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        rounds = model.best_iteration
        scores = model.predict(features.loc[validation_mask, selected_columns])
        score = ndcg_at_10(valid_labels, scores, valid_sizes)
        print(f"{name}: validation nDCG@10={score:.6f}, iterations={rounds}", flush=True)
        if score > best_score:
            best_score = score
            best_name, best_columns, best_params, best_rounds = (
                name, selected_columns, params, rounds
            )

    print(f"Selected {best_name}: iterations={best_rounds}, {best_params}", flush=True)
    dataset = lgb.Dataset(
        features[best_columns], label=data["Label"], group=group_sizes(data)
    )
    lgb.train(best_params, dataset, num_boost_round=best_rounds).save_model(model_file)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(data_dir / "test.csv")
    model = lgb.Booster(model_file=model_file)
    columns = [
        column for column in model.feature_name()
        if not column.endswith(("__qrank", "__qz"))
    ]
    features = (
        test[columns].astype("float32")
        if len(columns) == model.num_feature()
        else make_features(test, columns)[model.feature_name()]
    )
    scores = model.predict(features, num_threads=THREADS, validate_features=True)
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores")
    pd.DataFrame({"ObjectId": test["ObjectId"], "Score": scores}).to_csv(
        submission_file, index=False, float_format="%.15g"
    )


def main():
    parser = argparse.ArgumentParser(description="Learning to rank homework solution")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--submission_file")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    start = timer()
    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error("--submission_file is required in prediction mode")
        predict(args.data_dir, args.model_file, args.submission_file)
    print(f"finished, elapsed = {timer() - start:.3f}")


if __name__ == "__main__":
    main()
