from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from .preprocessors import ToDatetime
from .transformers import (
    ComputeDaysSinceStart,
    ComputeNetScore,
    EncodeOnBaseOccupancy,
    Identity,
)
from .constants import IDENTITY_COLS


preprocessors = Pipeline(steps=[
    ("game_date_to_datetime", ToDatetime(input_col="game_date")),
])


feature_transformers = ColumnTransformer(
    [
        (
            "identity",
            Identity(),
            IDENTITY_COLS,
        ),
        (
            "encode_on_base_occupancy",
            EncodeOnBaseOccupancy(output_cols=["on_1b", "on_2b", "on_3b"]),
            ["on_1b", "on_2b", "on_3b"]
        ),
        (
            "compute_net_score",
            ComputeNetScore(output_col="net_score"),
            ["bat_score", "fld_score"],
        ),
        (
            "compute_days_since_start",
            ComputeDaysSinceStart(output_col="days_since_start", season_start_date=datetime(2023, 3, 30)),
            ["game_date"],
        )
    ],
    remainder='drop',
)


data_pipeline = Pipeline([
    ("preprocessors", preprocessors),
    ("feature_transformers", feature_transformers),
])
