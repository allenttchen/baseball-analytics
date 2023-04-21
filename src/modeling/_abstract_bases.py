import abc

import numpy as np
import pandas as pd


class TransformerBase(abc.ABC):
    """Abstract Trasnformer that enforces APIs without overhead"""

    @abc.abstractmethod
    def __init__(self):
        self.feature_names_out = None
        self.generated_feature_names_out = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    @abc.abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X following the definition of Transformer"""
        pass

    def get_feature_names_out(self, input_features=None):
        """Retrieve feature names to be identified in training data"""
        return np.array(self.feature_names_out, dtype=object)
