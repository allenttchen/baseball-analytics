"""
Houses all transformers used in the feature/column transformer pipeline
"""
from datetime import datetime, date
from typing import List
import functools
import os
import glob
import uuid

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from ._abstract_bases import TransformerBase
from .constants import STATS_TO_EVENTS, ROOT_DIR


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


class MovingAverage(BaseEstimator, TransformerMixin, TransformerBase):
    """Computes the player's moving averages for various statistics"""

    def __init__(
        self,
        output_cols: List[str],
        player_type: str,
        ma_days: int,
        stats_to_compute: List[str],
        head_to_head: bool,
        training_data_start_date: date,
        training_data_end_date: date,
        ma_start_date: date
    ):
        self.input_cols = None
        self.output_cols = output_cols # feature names corresponding to self.stats_to_compute columns
        self.player_type = player_type
        self.ma_days = ma_days
        self.stats_to_compute = stats_to_compute # col names to self.ma_stats_df
        self.head_to_head = head_to_head
        self.ma_stats_df = pd.DataFrame() # ma for all players of all stats in df
        self.training_data_start_date = training_data_start_date
        self.training_data_end_date = training_data_end_date
        self.ma_start_date = ma_start_date # ma start date
        self.feature_names_out = output_cols

    def _compute_player_ma(self, player_group):
        """Compute moving averages for all events of one player"""
        assert(player_group.columns.tolist() == ["game_date", "events"])
        # start index is the very first date in the training dataset (ma_days before the ma_start_date)
        t_index = pd.DatetimeIndex(
            pd.date_range(start=self.training_data_start_date, end=self.training_data_end_date, freq="1d")
        )

        # Calculate at bat time series
        #player_group["game_date"] = pd.to_datetime(player_group["game_date"])
        player_group["at_bat_count"] = 1.0
        #player_group.set_index("game_date", inplace=True)
        at_bat_ts = (
            player_group[["game_date", "at_bat_count"]]
            .groupby(["game_date"])
            .sum()
            .sort_index(ascending=True)
            .resample("1D")
            .mean()
            .reindex(t_index)  # most players dont have stats all the way back to the start date of the training dataset
            .fillna(0)
        )

        # Calculate each stat time series
        player_group.drop(["at_bat_count"], axis=1, inplace=True)
        stat_ma_mapping = {}
        for stat in self.stats_to_compute:
            event = STATS_TO_EVENTS[stat]
            player_event_group = player_group[player_group["events"] == event].copy()
            player_event_group["stat_count"] = 1
            stat_ts = (
                player_event_group[["game_date", "stat_count"]]
                .groupby(["game_date"])
                .sum()
                .sort_index(ascending=True)
                .resample("1D")
                .mean()
                .reindex(t_index)
                .fillna(0)
            )
            combined_ts = stat_ts.merge(at_bat_ts, left_index=True, right_index=True)
            windows = combined_ts.rolling(self.ma_days)
            stat_ma = windows.apply(
                functools.partial(self._compute_player_event_window, combined_ts=combined_ts), raw=False
            )["stat_count"].shift()
            computed_ma_start_date = stat_ma.index[self.ma_days]

            # check computed vs inputed moving average start date
            assert (computed_ma_start_date.date() == self.ma_start_date)
            stat_ma_lst = stat_ma.tolist()[365:]
            stat_ma_mapping[stat] = stat_ma_lst
            del player_event_group

        return pd.Series(stat_ma_mapping, index=self.stats_to_compute)

    @staticmethod
    def _compute_player_event_window(window, combined_ts):
        num = combined_ts.loc[window.index, "stat_count"].sum()
        den = combined_ts.loc[window.index, "at_bat_count"].sum()
        # rookie debut safety consideration (in case of extremely low or high stat)
        # TODO: come up with rules for common events vs rare events for rookies
        # if den < 10:
        #     return 0.1
        return num / (den + 1)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Computes moving average for all players' all stats starting from mv_start_date to today"""
        self.input_cols = X.columns.tolist()

        data = pd.concat(
            [X, pd.DataFrame(y, columns=["events"])],
            axis=1,
        )

        self.ma_stats_df = data.groupby([self.player_type])[["game_date", "events"]].apply(self._compute_player_ma)

        # saving as json file
        # {
        #   player: {
        #       1B%: [v0, v1, ...],
        intermediate_path = os.path.join(ROOT_DIR, "intermediate")
        os.makedirs(intermediate_path, exist_ok=True)
        unique_file_id = str(uuid.uuid4())
        intermediate_path_file = os.path.join(intermediate_path, str(date.today()) + "-" + unique_file_id + ".json")
        self.ma_stats_df.to_json(intermediate_path_file, orient="index", indent=4)
        return self

    def _retrieve_player_event_stat_ma(self, row, stat):
        player = row[self.player_type]
        date_index = (row["game_date"].date() - self.ma_start_date).days
        # guardrail for dates before ma_start_Date
        if date_index < 0:
            return 0
        stat_ma = self.ma_stats_df.loc[(self.ma_stats_df.index == player), stat].values[0][date_index]
        return stat_ma

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retrieve moving average for each requested stat"""
        if self.ma_stats_df.empty:
            intermediate_path = os.path.join(ROOT_DIR, "intermediate", "*")
            all_files = glob.glob(intermediate_path)
            latest_file = max(all_files,  key=os.path.getctime)
            self.ma_stats_df = pd.read_json(latest_file)

        for output_col, stat in zip(self.output_cols, self.stats_to_compute):
            X[output_col] = X.apply(functools.partial(self._retrieve_player_event_stat_ma, stat=stat), axis=1)

        return X[self.output_cols]
