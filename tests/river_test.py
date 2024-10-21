from __future__ import annotations

import numbers

import polars as pl
from hypothesis import given, settings
from polars.testing.parametric import dataframes
import pytest
from river import compose, datasets, preprocessing, stream
from river.datasets.synth import AnomalySine

from river_rrcf.rrcf import RobustRandomCutForest


def rrcf_model():
    rrcf = RobustRandomCutForest(num_trees=4, tree_size=32, random_state=42)
    scaler = compose.SelectType(numbers.Number) | preprocessing.StandardScaler()  # type: ignore
    ordinal = compose.SelectType(str) | preprocessing.OrdinalEncoder()  # type: ignore
    return compose.Pipeline(
        scaler + ordinal,
        rrcf,
    )


def test_synth():
    rrcf = rrcf_model()
    dataset = AnomalySine(n_samples=100, n_anomalies=10, seed=42)

    for x, _ in dataset:
        score = rrcf.score_one(x)
        rrcf.learn_one(x)
        assert isinstance(score, float)


def test_credit_card():
    rrcf = rrcf_model()
    dataset = datasets.CreditCard().take(100)

    for x, _ in dataset:
        score = rrcf.score_one(x)
        rrcf.learn_one(x)
        assert isinstance(score, float)


@given(
    data=dataframes(
        allowed_dtypes=[pl.Int64, pl.Float32, pl.String],
        min_cols=1,
        max_cols=5,
        min_size=1,
        max_size=10,
        allow_null=False,
    )
)
@settings(deadline=None)
def test_dataframes(data: pl.DataFrame):
    rrcf = rrcf_model()

    dataset = stream.iter_polars(data)
    for x, _ in dataset:
        score = rrcf.score_one(x)
        rrcf.learn_one(x)
        assert isinstance(score, float)


def test_unseen_feature():
    rrcf = rrcf_model()
    rrcf.learn_one({"a": None, "b": None})

    assert rrcf["RobustRandomCutForest"]._keys is None

    rrcf.learn_one({"a": 0.0, "b": None})
    assert rrcf["RobustRandomCutForest"]._keys == ["a"]

    with pytest.raises(ValueError, match="Unseen features:"):
        rrcf.score_one({"a": 0.0, "b": "asdf"})
