from datetime import datetime, date
from typing import List
import functools
import os
import glob
import uuid

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .._abstract_bases import TransformerBase
from ..constants import STATS_TO_EVENTS, ROOT_DIR, WOBA_FACTORS


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
        self.pa_ts = None
        self.opp_player_type_handedness_col = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Computes moving average for all players' all stats starting from mv_start_date to today"""
        self.input_cols = X.columns.tolist()
        if self.player_type == "batter":
            self.opp_player_type_handedness_col = "p_throws"
        else:
            self.opp_player_type_handedness_col = "stand"

        data = pd.concat(
            [X, pd.DataFrame(y, columns=["events"])],
            axis=1,
        )

        self.ma_stats_df = (
            data
            .groupby([self.player_type])[["game_date", "events", "launch_speed", self.opp_player_type_handedness_col]]
            .apply(self._compute_player_ma)
        )

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

    def _compute_player_ma(self, player_group):
        """Compute moving averages for all events of one player"""
        assert(
            player_group.columns.tolist() == ["game_date", "events", "launch_speed", self.opp_player_type_handedness_col]
        )
        # start index is the very first date in the training dataset (ma_days before the ma_start_date)
        t_index = pd.DatetimeIndex(
            pd.date_range(start=self.training_data_start_date, end=self.training_data_end_date, freq="1d")
        )

        # Calculate plate appearances time series
        self.pa_ts, pa_ma = self._compute_player_pa_ma(player_group, t_index)

        # Calculate each stat time series
        stat_ma_mapping = {}
        stat_ma2 = None
        for stat in self.stats_to_compute:
            event = STATS_TO_EVENTS.get(stat)
            if stat == "PA": # number of plate appearances stat
                player_event_group = None
                stat_ma = pa_ma
            elif stat == "pPA":
                player_event_group = None
                player_event_group2 = None
                player_group1 = player_group[player_group[self.opp_player_type_handedness_col] == "L"].copy()
                player_group2 = player_group[player_group[self.opp_player_type_handedness_col] == "R"].copy()
                _, stat_ma = self._compute_player_pa_ma(player_group1, t_index)
                _, stat_ma2 = self._compute_player_pa_ma(player_group2, t_index)
            elif stat == "wOBA": # for wOBA
                player_event_group = player_group[player_group["events"].isin(event)].copy()
                player_event_group["stat_count"] = player_event_group["events"].map(WOBA_FACTORS)
                stat_ma = self._compute_player_event_ma(player_event_group, t_index, self.pa_ts)
            elif stat == "pwOBA": # for pwOBA
                player_event_group = player_group[
                    (player_group["events"].isin(event)) & (player_group[self.opp_player_type_handedness_col] == "L")
                ].copy()
                player_event_group2 = player_group[
                    (player_group["events"].isin(event)) & (player_group[self.opp_player_type_handedness_col] == "R")
                ].copy()
                player_event_group["stat_count"] = player_event_group["events"].map(WOBA_FACTORS)
                player_event_group2["stat_count"] = player_event_group2["events"].map(WOBA_FACTORS)
                player_group1 = player_group[(player_group[self.opp_player_type_handedness_col] == "L")].copy()
                player_group2 = player_group[(player_group[self.opp_player_type_handedness_col] == "R")].copy()
                pa_ts1, _ = self._compute_player_pa_ma(player_group1, t_index)
                pa_ts2, _ = self._compute_player_pa_ma(player_group2, t_index)
                stat_ma = self._compute_player_event_ma(player_event_group, t_index, pa_ts1)
                stat_ma2 = self._compute_player_event_ma(player_event_group2, t_index, pa_ts2)
            elif stat in ["mEV", "aEV"]: # for mEV and aEV
                player_event_group = player_group[player_group["launch_speed"].notnull()].copy()
                player_event_group["stat_count"] = player_event_group["launch_speed"]
                if stat == "mEV":
                    stat_ma = self._compute_player_event_mev_ma(player_event_group, t_index)
                else:
                    stat_ma = self._compute_player_event_aev_ma(player_event_group, t_index)
            else:
                player_event_group = player_group[player_group["events"] == event].copy()
                player_event_group["stat_count"] = 1
                stat_ma = self._compute_player_event_ma(player_event_group, t_index, self.pa_ts)

            # check computed vs inputed moving average start date
            computed_ma_start_date = stat_ma.index[self.ma_days]
            assert (computed_ma_start_date.date() == self.ma_start_date)
            stat_ma_lst = stat_ma.tolist()[365:]
            stat_ma_mapping[stat] = stat_ma_lst
            del player_event_group

            # extra work for platoon features
            if stat in ["pPA", "pwOBA"]:
                computed_ma_start_date2 = stat_ma2.index[self.ma_days]
                assert (computed_ma_start_date2.date() == self.ma_start_date)
                stat_ma_lst2 = stat_ma2.tolist()[365:]
                stat_ma_mapping["R"+stat] = stat_ma_lst2
                del player_event_group2
                del player_group1
                del player_group2
        return pd.Series(stat_ma_mapping, index=stat_ma_mapping.keys())

    def _compute_player_pa_ma(self, player_group, t_index):
        player_group["pa_count"] = 1.0
        pa_ts = (
            player_group[["game_date", "pa_count"]]
            .groupby(["game_date"])
            .sum()
            .sort_index(ascending=True)
            .resample("1D")
            .mean()
            .reindex(t_index)  # most players dont have stats all the way back to the start date of the training dataset
            .fillna(0)
        )
        player_group.drop(["pa_count"], axis=1, inplace=True)
        pa_ma = (
            pa_ts
            .rolling(self.ma_days)
            .sum()
            .shift()
        )["pa_count"]
        return pa_ts, pa_ma

    def _compute_player_event_ma(self, player_event_group, t_index, pa_ts):
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
        combined_ts = stat_ts.merge(pa_ts, left_index=True, right_index=True)
        stat_ma = (
            combined_ts
            .rolling(self.ma_days)
            .apply(
                functools.partial(self._compute_player_event_window, combined_ts=combined_ts), raw=False
            )
            .shift()
        )["stat_count"]
        return stat_ma

    @staticmethod
    def _compute_player_event_window(window, combined_ts):
        num = combined_ts.loc[window.index, "stat_count"].sum()
        den = combined_ts.loc[window.index, "pa_count"].sum()
        # rookie debut safety consideration (in case of extremely low or high stat)
        # TODO: come up with rules for common events vs rare events for rookies
        # if den < 10:
        #     return 0.1
        return num / (den + 0.0000001)

    def _compute_player_event_mev_ma(self, player_event_group, t_index):
        stat_ts = (
            player_event_group[["game_date", "stat_count"]]
            .groupby(["game_date"])
            .max()
            .sort_index(ascending=True)
            .resample("1D")
            .mean()
            .reindex(t_index)
            .fillna(0)
        )
        stat_ma = (
            stat_ts
            .rolling(self.ma_days)
            .max()
            .shift()
        )["stat_count"]
        return stat_ma

    def _compute_player_event_aev_ma(self, player_event_group, t_index):
        # numerator
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
        # denominator
        player_event_group["pa_count"] = 1.0
        hit_ts = (
            player_event_group[["game_date", "pa_count"]]
            .groupby(["game_date"])
            .sum()
            .sort_index(ascending=True)
            .resample("1D")
            .mean()
            .reindex(t_index)
            .fillna(0)
        )
        combined_ts = stat_ts.merge(hit_ts, left_index=True, right_index=True)
        windows = combined_ts.rolling(self.ma_days)
        stat_ma = windows.apply(
            functools.partial(self._compute_player_event_window, combined_ts=combined_ts), raw=False
        )["stat_count"].shift()
        return stat_ma

    def _retrieve_player_event_stat_ma(self, row, stat):
        player = row[self.player_type]
        date_index = (row["game_date"].date() - self.ma_start_date).days
        opp_player_type_handedness = row[self.opp_player_type_handedness_col]
        if stat in ["pwOBA", "pPA"] and opp_player_type_handedness == "R":
            stat = "R"+stat
        # guardrail for dates before ma_start_Date
        if date_index < 0:
            return 0
        stat_ma = self.ma_stats_df.loc[(self.ma_stats_df.index == player), stat].values[0][date_index]
        return stat_ma
