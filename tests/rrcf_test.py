import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from river_rrcf.rrcf import RobustRandomCutForest


@given(
    st.dictionaries(
        st.text(),
        st.one_of(st.floats(width=32), st.integers()),
        min_size=1,
    )
)
def test_preprocess_valid_input(data: dict[str, int | float]):
    rrcf = RobustRandomCutForest()

    expected_output = np.array(list(data.values()), dtype=np.float64)
    expected_output = np.nan_to_num(expected_output)
    float32_max = np.finfo(np.float32).max
    float32_min = np.finfo(np.float32).min
    expected_output = np.clip(expected_output, float32_min, float32_max)

    output = rrcf._preprocess(data)
    np.testing.assert_array_equal(output, expected_output)


@given(
    st.dictionaries(
        st.text(),
        st.one_of(st.floats(width=32), st.integers()),
        min_size=1,
    )
)
def test_preprocess_missing_keys(data: dict[str, int | float]):
    rrcf = RobustRandomCutForest()
    rrcf._preprocess(data)

    input_data = {}
    output = rrcf._preprocess(input_data)
    assert np.count_nonzero(output) == 0


@given(
    st.dictionaries(
        st.text(),
        st.one_of(st.floats(width=32), st.integers()),
        min_size=1,
    )
)
def test_preprocess_unexpected_keys(data: dict[str, int | float]):
    rrcf = RobustRandomCutForest()
    rrcf._preprocess(data)

    new_key = "".join([*data, "abc"])
    data.update({new_key: 0.0})

    with pytest.raises(ValueError, match="Unseen features:"):
        rrcf._preprocess(data)


def test_initial_keys():
    rrcf = RobustRandomCutForest(_keys=["a", "b", "c"])
    _ = rrcf._preprocess({})
    _ = rrcf._preprocess({"a": 1.0, "b": 2.0})
    _ = rrcf._preprocess({"a": 1.0, "b": 2.0, "c": 3.0})
    with pytest.raises(ValueError, match="Unseen features:"):
        rrcf._preprocess({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})


def test_preprocess_initial_keys():
    rrcf = RobustRandomCutForest()
    input_data = {"a": 1.0, "b": 2.0, "c": 3.0}
    rrcf._preprocess(input_data)
    assert rrcf._keys == ["a", "b", "c"]


def test_preprocess_nan_handling():
    rrcf = RobustRandomCutForest()
    input_data = {"a": 1.0, "b": np.nan, "c": 3.0}
    expected_output = np.array([1.0, 0.0, 3.0], dtype=np.float64)
    output = rrcf._preprocess(input_data)
    np.testing.assert_array_equal(output, expected_output)


@given(
    st.dictionaries(
        st.text(),
        st.one_of(st.floats(width=32), st.integers()),
        min_size=1,
    )
)
def test_learn_one_valid_input(data: dict[str, int | float]):
    rrcf = RobustRandomCutForest()
    initial_index = rrcf._index
    rrcf.learn_one(data)
    assert rrcf._index == initial_index + 1
    assert rrcf._shingle is not None
    assert len(rrcf._shingle) == rrcf.shingle_size
    for tree in rrcf.forest:
        assert len(tree.leaves) <= rrcf.tree_size


def test_learn_one_empty_input():
    rrcf = RobustRandomCutForest()
    initial_index = rrcf._index
    rrcf.learn_one({})
    assert rrcf._index == initial_index
    assert rrcf._shingle is None


def test_learn_one_initial_shingle():
    rrcf = RobustRandomCutForest()
    input_data = {"a": 1.0, "b": 2.0, "c": 3.0}
    rrcf.learn_one(input_data)
    assert rrcf._shingle is not None
    assert len(rrcf._shingle) == rrcf.shingle_size


def test_learn_one_tree_insertion():
    rrcf = RobustRandomCutForest()
    input_data = {"a": 1.0, "b": 2.0, "c": 3.0}
    rrcf.learn_one(input_data)
    for tree in rrcf.forest:
        assert len(tree.leaves) > 0


@given(
    st.dictionaries(
        st.one_of(st.floats(), st.integers(), st.text(), st.binary()),
        st.one_of(st.floats(width=32), st.integers()),
        min_size=1,
    )
)
def test_score_one_valid_input(data: dict[str, int | float]):
    rrcf = RobustRandomCutForest()
    rrcf.learn_one(data)
    score = rrcf.score_one(data)
    assert isinstance(score, float)
    assert score >= 0.0


def test_score_one_empty_input():
    rrcf = RobustRandomCutForest()
    score = rrcf.score_one({})
    assert score == 0.0


def test_score_one_tree_insertion():
    rrcf = RobustRandomCutForest()
    input_data = {"a": 1.0, "b": 2.0, "c": 3.0}
    rrcf.learn_one(input_data)
    initial_leaves = [len(tree.leaves) for tree in rrcf.forest]
    score = rrcf.score_one(input_data)

    assert isinstance(score, float)
    for initial, tree in zip(initial_leaves, rrcf.forest, strict=True):
        assert len(tree.leaves) == initial


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "a": st.one_of(st.floats(), st.integers()),
                "b": st.one_of(st.floats(), st.integers()),
                "c": st.one_of(st.floats(), st.integers()),
            },
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(deadline=500)
def test_learn_score_multiple(data: list[dict[str, int | float]]):
    rrcf = RobustRandomCutForest()

    try:
        for d in data:
            _ = rrcf.score_one(d)
            rrcf.learn_one(d)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"Exception raised: {e}")


@given(
    data=st.lists(
        st.fixed_dictionaries(
            {
                "a": st.one_of(st.floats(), st.integers()),
                "b": st.one_of(st.floats(), st.integers()),
                "c": st.one_of(st.floats(), st.integers()),
            },
        ),
        min_size=20,
        max_size=20,
    ),
    shingle_size=st.integers(min_value=1, max_value=4),
)
@settings(deadline=None)
def test_many_data(data: list[dict], shingle_size: int):
    rrcf = RobustRandomCutForest(num_trees=2, tree_size=4, shingle_size=shingle_size)

    try:
        for d in data:
            _ = rrcf.score_one(d)
            rrcf.learn_one(d)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"Exception raised: {e}")


@given(
    num_trees=st.integers(min_value=1, max_value=100),
    tree_size=st.integers(min_value=1, max_value=256),
    shingle_size=st.integers(min_value=1, max_value=8),
    data=st.lists(
        st.dictionaries(
            st.one_of(st.floats(), st.integers(), st.text(), st.binary()),
            st.one_of(st.floats(), st.integers()),
        ),
        min_size=1,
        max_size=5,
    ),
)
@settings(deadline=None)
def test_rrcf_init_score_learn(
    num_trees: int,
    tree_size: int,
    shingle_size: int,
    data: list[dict],
):
    assume(len(data) > 0 and len(data[0]) > 0)

    rrcf = RobustRandomCutForest(
        num_trees=num_trees,
        tree_size=tree_size,
        shingle_size=shingle_size,
    )

    assert len(rrcf.forest) == num_trees
    assert rrcf.tree_size == tree_size
    assert rrcf.shingle_size == shingle_size
    assert rrcf._index == 0
    assert rrcf._keys is None
    assert rrcf._shingle is None

    all_keys = {key for d in data for key in d}
    sample = {k: 0.0 for k in all_keys}
    rrcf._preprocess(sample)

    assert rrcf._keys is not None

    scores = []
    for d in data:
        score = rrcf.score_one(d)
        scores.append(score)
        rrcf.learn_one(d)

    assert rrcf._index == len(data)
    assert rrcf._shingle is not None
    assert len(rrcf._shingle) == shingle_size

    for tree in rrcf.forest:
        assert len(tree.leaves) <= tree_size

    for score in scores:
        assert isinstance(score, float)
        assert score >= 0.0
