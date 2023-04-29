import functools
from typing import List
from collections import defaultdict
import json
import os
from datetime import date
import glob

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .._abstract_bases import TransformerBase
from ..constants import STATS_TO_EVENTS, ROOT_DIR, UNIQUE_RUN_ID, WOBA_FACTORS


class HeadToHead(BaseEstimator, TransformerMixin, TransformerBase):
    """Calculate the batter and pitcher head-to-head matchup resulting events statistics"""
    def __init__(self, output_cols: List[str], stats_to_compute: List[str]):
        self.input_cols = None
        self.output_cols = output_cols
        self.stats_to_compute = stats_to_compute
        self.matchup_stat_mapping = None
        self.saved_file_name = None
        self.feature_names_out = output_cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """compute the resulting event statistics for every past batter and pitcher matchup"""
        self.input_cols = X.columns.tolist()
        simple_stats = [stat for stat in self.stats_to_compute if stat not in ["PA", "wOBA"]]
        simple_events = [STATS_TO_EVENTS.get(stat) for stat in simple_stats]
        simple_event_to_stat_mapping = {k: v for k, v in zip(simple_events, simple_stats)}

        data = pd.concat(
            [X, pd.DataFrame(y, columns=["events"])],
            axis=1,
        )

        # Compute batter and pitcher matchup total
        matchup_count = (
            data
            .groupby(["batter", "pitcher"])["events"]
            .count()
            .reset_index()
            .rename(columns={"events": "PA"})
        )

        # Compute the batter and pitcher matchup events count
        matchup_event_count = (
            data
            .groupby(["batter", "pitcher", "events"])["events"]
            .count()
            .to_frame()
            .rename(columns={"events": "count"})
            .reset_index()
        )
        # Compute the batter and pitcher matchup simple stats
        matchup_event_count = matchup_event_count[matchup_event_count["events"].isin(simple_events)]
        matchup_df = matchup_event_count.merge(matchup_count, on=["batter", "pitcher"])
        matchup_df["rate"] = matchup_df["count"] / matchup_df["PA"]
        matchup_df.drop(["count", "PA"], axis=1, inplace=True)
        matchup_df = (
            pd.pivot_table(matchup_df, values="rate", index=["batter", "pitcher"], columns=["events"])
            .reset_index()
            .rename(columns=simple_event_to_stat_mapping)
        )

        if "PA" in self.stats_to_compute:
            matchup_df = matchup_df.merge(matchup_count, on=["batter", "pitcher"])

        if "wOBA" in self.stats_to_compute:
            woba_events = ["w", "hbp", "s", "d", "t", "hr"]
            data = data[data["events"].isin(woba_events)].copy()
            data["weight"] = data["events"].map(WOBA_FACTORS)
            matchup_woba_df = (
                data
                .groupby(["batter", "pitcher"])["weight"]
                .sum()
                .reset_index()
                .rename(columns={"weight": "count"})
            )
            matchup_woba_df = matchup_woba_df.merge(matchup_count, on=["batter", "pitcher"])
            matchup_woba_df["wOBA"] = matchup_woba_df["count"] / matchup_woba_df["PA"]
            matchup_woba_df.drop(["count", "PA"], axis=1, inplace=True)
            matchup_df = matchup_df.merge(matchup_woba_df, on=["batter", "pitcher"])

        def make_bp_index(row):
            return str(int(row["batter"])) + str(int(row["pitcher"]))

        matchup_df["bp_index"] = matchup_df.apply(make_bp_index, axis=1)
        matchup_df = (
            matchup_df
            .fillna(0.0)
            .drop(["batter", "pitcher"], axis=1)
            .set_index("bp_index")
        )
        self.matchup_stat_mapping = defaultdict(lambda: "no such matchup", **matchup_df.to_dict(orient="index"))

        intermediate_path = os.path.join(ROOT_DIR, "intermediate", str(date.today()) + "-" + UNIQUE_RUN_ID)
        os.makedirs(intermediate_path, exist_ok=True)
        self.saved_file_name = "head-to-head.json"
        intermediate_path_file = os.path.join(intermediate_path, self.saved_file_name)
        with open(intermediate_path_file, "w") as f:
            matchup_stat_json = json.dumps(self.matchup_stat_mapping, indent=4)
            f.write(matchup_stat_json)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retrieve the matchup rate for each stat-to-compute"""
        if self.matchup_stat_mapping is None:
            intermediate_path = os.path.join(ROOT_DIR, "intermediate", "*")
            all_folders = glob.glob(intermediate_path)
            latest_folder = max(all_folders, key=os.path.getctime)
            latest_file = os.path.join(latest_folder, self.saved_file_name)
            with open(latest_file, "r") as f:
                self.matchup_stat_mapping = json.load(f)

        for output_col, stat in zip(self.output_cols, self.stats_to_compute):
            X[output_col] = X.apply(functools.partial(self._retrieve_matchup_event_rate, stat=stat), axis=1)

        return X[self.output_cols]

    def _retrieve_matchup_event_rate(self, row, stat):
        bp_index = str(int(row["batter"])) + str(int(row["pitcher"]))
        matchup_stats = self.matchup_stat_mapping[bp_index]
        if matchup_stats == "no such matchup":
            return 0
        return matchup_stats[stat]
