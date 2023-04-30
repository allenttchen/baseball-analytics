from typing import List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .._abstract_bases import TransformerBase


class ParkFactor(BaseEstimator, TransformerMixin, TransformerBase):
    """Creates park factor features"""
    #TODO: Adapt year to park factor features

    def __init__(self, output_cols: List[str], stats_to_compute: List[str], park_factors_file_path: str):
        self.input_cols = None
        self.output_cols = output_cols
        self.stats_to_compute = stats_to_compute
        self.park_factors_file_path = park_factors_file_path
        self.feature_names_out = output_cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.input_cols = X.columns.tolist()
        data = pd.read_csv(self.park_factors_file_path)
        park_factors_col_name_mapping = {
            "1B": "index_1b",
            "2B": "index_2b",
            "3B": "index_3b",
            "HR": "index_hr",
            "BB": "index_bb",
        }
        for output_col, stat in zip(self.output_cols, self.stats_to_compute):
            park_factors_dict = pd.Series(
                data[park_factors_col_name_mapping[stat]].values, index=data["team_abbre"]
            ).to_dict()
            X[output_col] = X["home_team"].map(park_factors_dict)
        return X[self.output_cols]
