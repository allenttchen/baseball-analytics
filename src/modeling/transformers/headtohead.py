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
from ..constants import STATS_TO_EVENTS, ROOT_DIR, UNIQUE_RUN_ID


class HeadToHead(BaseEstimator, TransformerMixin, TransformerBase):
    """Calculate the batter and pitcher head-to-head matchup resulting events statistics"""
    def __init__(self, output_cols: List[str], stats_to_compute: List[str]):
        self.input_cols = None
        self.output_cols = output_cols
        self.stats_to_compute = stats_to_compute
        self.matchup_rate_mapping = None
        self.saved_file_name = None
        self.feature_names_out = output_cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """compute the resulting event statistics for every past batter and pitcher matchup"""
        self.input_cols = X.columns.tolist()
        events_to_compute = [STATS_TO_EVENTS.get(stat) for stat in self.stats_to_compute]

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
            .rename(columns={"events": "total"})
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
        matchup_event_count = matchup_event_count[matchup_event_count["events"].isin(events_to_compute)]

        # Divide the above numbers to get the matchup event rate
        matchup_df = matchup_event_count.merge(matchup_count, on=["batter", "pitcher"])
        matchup_df["rate"] = matchup_df["count"] / matchup_df["total"]
        matchup_df.drop(["count", "total"], axis=1, inplace=True)

        # save as a dictionary mapping bpe_index to rate into json
        def make_bpe_index(row):
            return str(row["batter"]) + str(row["pitcher"]) + row["events"]

        matchup_df["bpe_index"] = matchup_df.apply(make_bpe_index, axis=1)
        matchup_df.set_index("bpe_index", inplace=True)
        self.matchup_rate_mapping = defaultdict(int, **matchup_df["rate"].to_dict())

        intermediate_path = os.path.join(ROOT_DIR, "intermediate", str(date.today()) + "-" + UNIQUE_RUN_ID)
        os.makedirs(intermediate_path, exist_ok=True)
        self.saved_file_name = "head-to-head.json"
        intermediate_path_file = os.path.join(intermediate_path, self.saved_file_name)
        with open(intermediate_path_file, "w") as f:
            matchup_rate_json = json.dumps(self.matchup_rate_mapping, indent=4)
            f.write(matchup_rate_json)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retrieve the matchup rate for each stat-to-compute"""
        if self.matchup_rate_mapping is None:
            intermediate_path = os.path.join(ROOT_DIR, "intermediate", "*")
            all_folders = glob.glob(intermediate_path)
            latest_folder = max(all_folders, key=os.path.getctime)
            latest_file = os.path.join(latest_folder, self.saved_file_name)
            with open(latest_file, "r") as f:
                self.matchup_rate_mapping = json.load(f)

        for output_col, stat in zip(self.output_cols, self.stats_to_compute):
            event = STATS_TO_EVENTS.get(stat)
            X[output_col] = X.apply(functools.partial(self._retrieve_matchup_event_rate, event=event), axis=1)

        return X[self.output_cols]

    def _retrieve_matchup_event_rate(self, row, event):
        bpe_index = str(row["batter"]) + str(row["pitcher"]) + event
        return self.matchup_rate_mapping[bpe_index]
