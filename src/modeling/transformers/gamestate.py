"""
Houses game state transformers used in the feature/column transformer pipeline
"""
from datetime import datetime, date
from typing import List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from .._abstract_bases import TransformerBase


class Identity(BaseEstimator, TransformerMixin, TransformerBase):
    """Return the same outputs as the inputs"""

    def __init__(self):
        self.input_cols = None
        self.output_cols = None
        self.feature_names_out = None

    def transform(self, X: pd.DataFrame):
        self.output_cols = X.columns.tolist()
        self.feature_names_out = self.output_cols
        return X


class EncodeOnBaseOccupancy(BaseEstimator, TransformerMixin, TransformerBase):
    """Turn player ID to 1 if the base is occupied else 0"""

    def __init__(self, output_cols: List[str]):
        self.input_cols = None
        self.output_cols = output_cols
        self.feature_names_out = output_cols

    def transform(self, X: pd.DataFrame):
        self.input_cols = X.columns.tolist()
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            X[output_col] = np.where(X[input_col].notnull(), 1, 0)
        return X[self.output_cols]


class ComputeNetScore(BaseEstimator, TransformerMixin, TransformerBase):
    """Compute the difference between the team score of batter and that of pitcher"""

    def __init__(self, output_col: str):
        self.input_cols = None
        self.output_col = output_col
        self.feature_names_out = [output_col]

    def transform(self, X: pd.DataFrame):
        assert(len(X.columns.tolist()) == 2)
        self.input_cols = X.columns.tolist()
        X[self.output_col] = X[self.input_cols[0]] - X[self.input_cols[1]]
        return X[[self.output_col]]


class ComputeDaysSinceStart(BaseEstimator, TransformerMixin, TransformerBase):
    """Compute the number of days from game date to the start date of the season"""

    def __init__(self, output_col: str, season_start_date: date):
        self.season_start_date = season_start_date
        self.input_col = None
        self.output_col = output_col
        self.feature_names_out = [output_col]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert(len(X.columns.tolist()) == 1)
        self.input_col = X.columns.tolist()[0]
        X[self.output_col] = X[self.input_col].apply(lambda x: (x.date() - self.season_start_date).days)
        return X[[self.output_col]]
