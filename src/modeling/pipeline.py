from datetime import date
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .preprocessors import ToDatetime
from .transformers.gamestate import (
    ComputeDaysSinceStart,
    ComputeNetScore,
    EncodeOnBaseOccupancy,
    Identity,
)
from .transformers.movingaverage import MovingAverage
from .transformers.headtohead import HeadToHead
from .transformers.parkfactor import ParkFactor
from .constants import IDENTITY_COLS, ROOT_DIR


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
                training_data_start_date=date(2022, 4, 7),
                training_data_end_date=date(2023, 4, 17),
                ma_start_date=date(2023, 4, 7), # datetime(2022, 4, 7) + timedelta(365)
            ),
            ["game_date", "batter", "launch_speed", "p_throws", ],
        ),
        # (
        #     "pitcher_365_days_ma",
        #     MovingAverage(
        #         output_cols=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA", "mEV", "aEV", "pwOBA", "pPA"],
        #         player_type="pitcher",
        #         ma_days=365,
        #         stats_to_compute=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA", "mEV", "aEV", "pwOBA", "pPA"],
        #         training_data_start_date=date(2022, 4, 7),
        #         training_data_end_date=date(2023, 4, 17),
        #         ma_start_date=date(2023, 4, 7), # datetime(2022, 4, 7) + timedelta(365)
        #     ),
        #     ["game_date", "pitcher", "launch_speed", "stand", ],
        # ),
        # (
        #     "batter_30_days_ma",
        #     MovingAverage(
        #         output_cols=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "wOBA"],
        #         player_type="batter",
        #         ma_days=30,
        #         stats_to_compute=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "wOBA"],
        #         training_data_start_date=date(2022, 4, 7),
        #         training_data_end_date=date(2023, 4, 17),
        #         ma_start_date=date(2022, 5, 7), # datetime(2022, 4, 7) + timedelta(30)
        #     ),
        #     ["game_date", "batter", "launch_speed", "p_throws", ],
        # ),
        # (
        #     "pitcher_30_days_ma",
        #     MovingAverage(
        #         output_cols=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "wOBA"],
        #         player_type="pitcher",
        #         ma_days=30,
        #         stats_to_compute=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "wOBA"],
        #         training_data_start_date=date(2022, 4, 7),
        #         training_data_end_date=date(2023, 4, 17),
        #         ma_start_date=date(2022, 5, 7), # datetime(2022, 4, 7) + timedelta(30)
        #     ),
        #     ["game_date", "pitcher", "launch_speed", "stand", ],
        # ),
        (
            "head_to_head",
            HeadToHead(
                output_cols=["1B", "2B", "HR", "BB", "SO", "PA", "wOBA"],
                stats_to_compute=["1B", "2B", "HR", "BB", "SO", "PA", "wOBA"],
            ),
            ["batter", "pitcher", ]
        ),
        (
            "ball_park",
            ParkFactor(
                output_cols=["1B", "2B", "3B", "HR", "BB"],
                stats_to_compute=["1B", "2B", "3B", "HR", "BB"],
                park_factors_file_path=os.path.join(ROOT_DIR, "intermediate/park_factors.csv"),
            ),
            ["home_team", ],
        ),
    ],
    remainder='drop',
)


data_pipeline = Pipeline(
    [
        ("preprocessors", preprocessors),
        ("feature_transformers", feature_transformers),
    ]
)
