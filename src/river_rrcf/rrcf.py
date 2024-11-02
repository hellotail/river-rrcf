from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
from river.anomaly.base import AnomalyDetector

from river_rrcf._vendor import rrcf

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, SupportsFloat

    from numpy.typing import NDArray


FLOAT32_MAX = 3.4028234663852886e38
FLOAT32_MIN = -3.4028234663852886e38


class RobustRandomCutForest(AnomalyDetector):
    """
    A Robust Random Cut Forest (RRCF) for anomaly detection.

    Parameters
    ----------
    num_trees : int, default=40
        The number of trees in the forest.
    tree_size : int, default=256
        The maximum size of each tree.
    shingle_size : int, default=1
        The size of the shingle (window) for time series data.
    random_state : int | None, default=None
        Random seed for each tree in the forest. (>= 0 if int)
    _keys : Sequence[Any] | None, default=None
        The keys to expect in the input data mapping. If None, the keys are inferred from the first data point.

    Attributes
    ----------
    num_trees : int
        The number of trees in the forest.
    tree_size : int
        The maximum size of each tree.
    shingle_size : int
        The size of the shingle (window) for time series data.
    forest : list
        A list of RCTree instances representing the forest.
    _index : int
        The current index of the data point being processed.
    _keys : list or None
        The keys of the input data mapping.
    _shingle : deque or None
        A deque representing the shingle for time series data.

    Methods
    -------
    _preprocess(x)
        Preprocesses the input data mapping into a numpy array.
    _init_shingle(arr)
        Initializes the shingle with zeros.
    learn_one(x)
        Learns from a single data point.
    score_one(x)
        Scores a single data point for anomaly detection.
    """

    def __init__(
        self,
        num_trees: int = 40,
        tree_size: int = 256,
        *,
        shingle_size: int = 1,
        random_state: int | None = None,
        _keys: Sequence[Any] | None = None,
    ):
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.random_state = random_state

        if random_state is not None:
            self.forest = [
                rrcf.RCTree(random_state=random_state + i) for i in range(num_trees)
            ]
        else:
            self.forest = [rrcf.RCTree(random_state=None) for _ in range(num_trees)]

        self._index = 0
        self._keys: list[Any] | None = list(_keys) if _keys is not None else None

        self._shingle: deque[NDArray[np.float64]] | None = None

    def _preprocess(self, x: Mapping[Any, SupportsFloat]) -> NDArray[np.float64]:
        """
        Preprocesses the input mapping by converting its values to a numpy array of floats.

        Parameters
        ----------
        x : Mapping[Any, SupportsFloat]
            The input mapping where keys are any hashable type and values are convertible to float.

        Returns
        -------
        NDArray[np.float64]
            An 1-d numpy array of floats corresponding to the values of the input mapping.

        Raises
        ------
        ValueError
            If there are keys in the input mapping that were not expected based on the initial keys.
        """
        if not self._keys:
            self._keys = list(x)

        xx = dict(x)
        items = [float(xx.pop(k, 0.0)) for k in self._keys]

        if xx:
            remain = ", ".join(str(r) for r in xx)
            expected = ", ".join(str(k) for k in self._keys)
            msg = f"Unseen features: {remain}\n       expected: {expected}"
            raise ValueError(msg)

        arr = np.array(items, dtype=np.float64)
        return np.nan_to_num(arr, copy=False).clip(FLOAT32_MIN, FLOAT32_MAX)

    def _init_shingle(self, arr: NDArray[np.float64]) -> deque[NDArray[np.float64]]:
        a = np.zeros_like(arr)
        return deque([a] * self.shingle_size, maxlen=self.shingle_size)

    def learn_one(self, x: Mapping[Any, SupportsFloat]) -> None:
        """
        Learn a single data point.

        Note: The input values are restricted to the float32 range.

        Parameters
        ----------
        x : Mapping[Any, SupportsFloat]
            The data point to learn.

        """
        if len(x) == 0 and not self._keys:
            return

        xx = self._preprocess(x)

        if not self._shingle:
            self._shingle = self._init_shingle(xx)

        self._shingle.append(xx)

        for tree in self.forest:
            if len(tree.leaves) >= self.tree_size:
                tree.forget_point(self._index - self.tree_size)
            arr = np.concatenate(self._shingle)
            tree.insert_point(arr, index=self._index)

        self._index += 1

    def score_one(self, x: Mapping[Any, SupportsFloat]) -> float:
        """
        Computes the anomaly score for a single data point.

        Parameters
        ----------
        x : Mapping[Any, SupportsFloat]
            The data point to be scored. It should be a mapping where keys are hashable and values are floats.

        Returns
        -------
        float
            The computed anomaly score for the given data point.

        Notes
        -----
        - If the input data point is empty and the forest is not initialized, this method returns 0.0.
        - The anomaly score is calculated as the average co-displacement (codisp) across all trees in the forest.
        """
        if len(x) == 0 and not self._keys:
            return 0.0

        xx = self._preprocess(x)
        i = -1

        if not self._shingle:
            self._shingle = self._init_shingle(xx)

        data = [*self._shingle, xx]
        if len(data) > self.shingle_size:
            data = data[-self.shingle_size :]

        for tree in self.forest:
            tree.insert_point(data, index=i)

        avg_codisp = np.mean([tree.codisp(i) for tree in self.forest], dtype=np.float64)

        for tree in self.forest:
            tree.forget_point(i)

        return avg_codisp.item()
