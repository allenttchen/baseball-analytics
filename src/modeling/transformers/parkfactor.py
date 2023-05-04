import functools
from typing import List
import json

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.modeling._abstract_bases import TransformerBase


class ParkFactor(BaseEstimator, TransformerMixin, TransformerBase):
    """Creates park factor features"""

    def __init__(self, output_cols: List[str], stats_to_compute: List[str], park_factors_file_path: str):
        self.input_cols = None
        self.output_cols = output_cols
        self.stats_to_compute = stats_to_compute
        self.park_factors_file_path = park_factors_file_path
        self.feature_names_out = output_cols
        self.park_factors_mapping = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.input_cols = X.columns.tolist()
        with open(self.park_factors_file_path, "r") as f:
            self.park_factors_mapping = json.load(f)
        for output_col, stat in zip(self.output_cols, self.stats_to_compute):
            X[output_col] = X.apply(functools.partial(self._retrieve_park_factor, stat=stat), axis=1)
        return X[self.output_cols]

    def _retrieve_park_factor(self, row, stat):
        return self.park_factors_mapping[row["home_team"]][stat][str(row["game_year"])]
