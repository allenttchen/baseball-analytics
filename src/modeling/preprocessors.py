"""
Houses all transformers used in the preprocessor pipeline
"""
from typing import List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
pd.options.mode.chained_assignment = None


class ToDatetime(BaseEstimator, TransformerMixin):
    """Transform date column to datetime objects"""

    def __init__(self, input_col: str):
        self.input_col = input_col

    def fit(self, X, y):
        return self

    def transform(self, X):
        X[self.input_col] = pd.to_datetime(X[self.input_col])
        return X


# class CustomOrdinalEncoder(OrdinalEncoder):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def transform(self, X, y=None):
#         transformed_X = super().transform()
#         new_X = pd.DataFrame(transformed_X, columns=self.feature_names_in_)
#         return new_X