from __future__ import annotations

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from river_rrcf import __version__
from river_rrcf.rrcf import RobustRandomCutForest


def test_version():
    assert __version__ != "unknown"


@pytest.mark.parametrize(
    "protocol",
    [*range(pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1)],
)
@given(
    num_trees=st.integers(min_value=1, max_value=8),
    tree_size=st.integers(min_value=1, max_value=32),
    shingle_size=st.integers(min_value=1, max_value=4),
    data=st.lists(
        st.fixed_dictionaries(
            {
                "a": st.one_of(st.floats(), st.integers()),
                "b": st.one_of(st.floats(), st.integers()),
                "c": st.one_of(st.floats(), st.integers()),
            },
        ),
        min_size=1,
        max_size=6,
    ),
    random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=1234567891)),
)
@settings(deadline=None)
def test_pickle(  # noqa: PLR0913 too-many-arguments
    num_trees: int,
    tree_size: int,
    shingle_size: int,
    data: list[dict],
    random_state: int | None,
    protocol: int,
):
    rrcf = RobustRandomCutForest(
        num_trees=num_trees,
        tree_size=tree_size,
        shingle_size=shingle_size,
        random_state=random_state,
    )

    for d in data[:-1]:
        rrcf.learn_one(d)

    with TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir, f"{uuid4()}.pkl")

        with tmp_file.open("wb") as f:
            pickle.dump(rrcf, f, protocol=protocol)

        with tmp_file.open("rb") as f:
            rrcf_loaded: RobustRandomCutForest
            rrcf_loaded = pickle.load(f)  # noqa: S301 suspicious-pickle-usage

        assert isinstance(rrcf_loaded, RobustRandomCutForest)
        dd = data[-1]
        assert rrcf_loaded.score_one(dd) == rrcf.score_one(dd)


@given(
    num_trees=st.integers(min_value=1, max_value=8),
    tree_size=st.integers(min_value=1, max_value=32),
    shingle_size=st.integers(min_value=1, max_value=4),
    data=st.lists(
        st.fixed_dictionaries(
            {
                "a": st.one_of(st.floats(), st.integers()),
                "b": st.one_of(st.floats(), st.integers()),
                "c": st.one_of(st.floats(), st.integers()),
            },
        ),
        min_size=1,
        max_size=6,
    ),
    random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=1234567891)),
)
@settings(deadline=None)
def test_clone(
    num_trees: int,
    tree_size: int,
    shingle_size: int,
    data: list[dict],
    random_state: int | None,
):
    rrcf = RobustRandomCutForest(
        num_trees=num_trees,
        tree_size=tree_size,
        shingle_size=shingle_size,
        random_state=random_state,
    )

    for d in data[:-1]:
        rrcf.learn_one(d)

    rrcf_clone = rrcf.clone(include_attributes=True)
    assert isinstance(rrcf_clone, RobustRandomCutForest)
    dd = data[-1]
    assert rrcf_clone.score_one(dd) == rrcf.score_one(dd)
