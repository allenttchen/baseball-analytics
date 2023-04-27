from datetime import date

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .preprocessors import ToDatetime
from .transformers.transformers import (
    ComputeDaysSinceStart,
    ComputeNetScore,
    EncodeOnBaseOccupancy,
    Identity,
)
from .transformers.movingaverage import MovingAverage
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
            ["on_1b", "on_2b", "on_3b", ],
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
                output_cols=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA", "mEV", "aEV", "pwOBA", "pPA"],
                player_type="batter",
                ma_days=365,
                stats_to_compute=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA", "mEV", "aEV", "pwOBA", "pPA"],
                head_to_head=False,
                training_data_start_date=date(2022, 4, 7),
                training_data_end_date=date(2023, 4, 17),
                ma_start_date=date(2023, 4, 7), # datetime(2022, 4, 7) + timedelta(365)
            ),
            ["game_date", "batter", "launch_speed", "p_throws", ],
        ),
    ],
    remainder='drop',
)


data_pipeline = Pipeline(
    [
        ("preprocessors", preprocessors),
        ("feature_transformers", feature_transformers),\
    ]
)
