from datetime import datetime, date

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from .preprocessors import ToDatetime
from .transformers import (
    ComputeDaysSinceStart,
    ComputeNetScore,
    EncodeOnBaseOccupancy,
    Identity,
    MovingAverage,
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
            ["on_1b", "on_2b", "on_3b", ]
        ),
        (
            "compute_net_score",
            ComputeNetScore(output_col="net_score"),
            ["bat_score", "fld_score", ],
        ),
        (
            "compute_days_since_start",
            ComputeDaysSinceStart(output_col="days_since_start", season_start_date=date(2023, 3, 30)),
            ["game_date", ],
        ),
        (
            "batter_365_days_ma",
            MovingAverage(
                output_cols=["1B_perc", "2B_perc", "3B_perc", "HR_perc", "BB_perc", "SO_perc", "DP_perc", "FO_perc", "SF_perc", "SH_perc"],
                player_type="batter",
                ma_days=365,
                stats_to_compute=["1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "SF", "SH"],
                head_to_head=False,
                training_data_start_date=date(2022, 4, 7),
                training_data_end_date=date(2023, 4, 17),
                ma_start_date=date(2023, 4, 7), # datetime(2022, 4, 7) + timedelta(365)
            ),
            ["game_date", "batter", ]
        )
    ],
    remainder='drop',
)


data_pipeline = Pipeline(
    [
        ("preprocessors", preprocessors),
        ("feature_transformers", feature_transformers),\
    ]
)
