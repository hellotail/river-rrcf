from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from typing import TYPE_CHECKING, Any, SupportsFloat

import numpy as np
from river.anomaly.base import AnomalyDetector

from river_rrcf._vendor import rrcf

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RobustRandomCutForest(AnomalyDetector):
    def __init__(
        self,
        num_trees: int = 40,
        tree_size: int = 256,
        shingle_size: int = 1,
    ):
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.forest = [rrcf.RCTree() for _ in range(num_trees)]

        self._index = 0
        self._keys: list[Any] | None = None

        self._shingle: deque[NDArray[np.float64]] | None = None

    def _preprocess(self, x: Mapping[Any, SupportsFloat]) -> NDArray[np.float64]:
        if self._keys is None:
            self._keys = list(x)

        xx = dict(x)
        items = [float(xx.pop(k, 0.0)) for k in self._keys]

        if xx:
            remain = ", ".join(str(r) for r in xx)
            expected = ", ".join(str(k) for k in self._keys)
            msg = f"Unseen features: {remain}\n       expected: {expected}"
            raise ValueError(msg)

        arr = np.array(items, dtype=np.float64)
        return np.nan_to_num(arr, copy=False)

    def _init_shingle(self, arr: NDArray[np.float64]) -> deque[NDArray[np.float64]]:
        a = np.zeros_like(arr)
        return deque([a] * self.shingle_size, maxlen=self.shingle_size)

    def learn_one(self, x: Mapping[Any, SupportsFloat]) -> None:
        if len(x) == 0 and not self._keys:
            return

        xx = self._preprocess(x)

        if not self._shingle:
            self._shingle = self._init_shingle(xx)

        self._shingle.append(xx)

        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self._index - self.tree_size)
            arr = np.concatenate(self._shingle)
            tree.insert_point(arr, index=self._index)

        self._index += 1

    def score_one(self, x: Mapping[Any, SupportsFloat]) -> float:
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
