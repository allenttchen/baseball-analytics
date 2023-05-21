import os
from datetime import date, datetime
import functools
import glob

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, udf
from pyspark.sql.types import StructType, StructField, FloatType, StringType, ArrayType
from pyspark.sql import DataFrame
from pyspark.ml import Transformer, Estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from src.modeling.utils import time_job
from src.modeling.constants import STATS_TO_EVENTS, ROOT_DIR, WOBA_FACTORS, UNIQUE_RUN_ID
from src.modeling.spark_transformers.game_state import GetOutputColsMixin


class MovingAverageTransformer(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):
    def __init__(
        self,
        inputCols,
        outputCols,
        player_type,
        ma_days,
        stats_to_compute,
        training_data_start_date,
        training_data_end_date,
        ma_start_date,
        ma_stats_df=pd.DataFrame(),
        pa_ts=None,
        opp_player_type_handedness_col="p_throws",
        saved_file_name="batter-365.json",
    ):
        super(MovingAverageTransformer, self).__init__()
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.player_type = player_type
        self.ma_days = ma_days
        self.stats_to_compute = stats_to_compute
        self.training_data_start_date = training_data_start_date
        self.training_data_end_date = training_data_end_date
        self.ma_start_date = ma_start_date
        self.ma_stats_df = ma_stats_df
        self.pa_ts = pa_ts
        self.opp_player_type_handedness_col = opp_player_type_handedness_col
        self.saved_file_name = saved_file_name

    @time_job
    def _transform(self, dataset: DataFrame) -> DataFrame:
        if self.ma_stats_df.empty:
            intermediate_path = os.path.join(ROOT_DIR, "intermediate", "*")
            all_folders = glob.glob(intermediate_path)
            latest_folder = max(all_folders,  key=os.path.getctime)
            latest_file = os.path.join(latest_folder, self.saved_file_name)
            self.ma_stats_df = pd.read_json(latest_file, orient="index")
            self.ma_stats_df.index = self.ma_stats_df.index.map(str)

        for output_col, stat in zip(self.outputCols, self.stats_to_compute):
            dataset = dataset.withColumn(
                output_col,
                udf(functools.partial(self._retrieve_player_event_stat_ma, stat=stat), FloatType())(
                    col("game_date"),
                    col(self.player_type),
                    col(self.opp_player_type_handedness_col)
                )
            )
        return dataset

    def _retrieve_player_event_stat_ma(
        self,
        game_date,
        player,
        opp_player_type_handedness,
        stat
    ) -> float:
        date_index = (game_date.date() - self.ma_start_date).days
        if stat in ["pwOBA", "pPA"] and opp_player_type_handedness == "R":
            stat = "R"+stat
        # guardrail for dates before ma_start_Date
        if date_index < 0 or player not in self.ma_stats_df.index:
            return 0
        stat_ma = self.ma_stats_df.loc[(self.ma_stats_df.index == player), stat].values[0][date_index]
        return stat_ma


class MovingAverageEstimator(
    Estimator,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):
    def __init__(
        self,
        inputCols,
        outputCols,
        player_type,
        ma_days,
        stats_to_compute,
        training_data_start_date,
        training_data_end_date,
        ma_start_date,
    ):
        super(MovingAverageEstimator, self).__init__()
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.player_type = player_type
        self.ma_days = ma_days
        self.stats_to_compute = stats_to_compute
        self.training_data_start_date = training_data_start_date
        self.training_data_end_date = training_data_end_date
        self.ma_start_date = ma_start_date
        self.ma_stats_df = pd.DataFrame()
        self.pa_ts = None
        self.opp_player_type_handedness_col = None
        self.saved_file_name = None

    @time_job
    def _fit(self, dataset: DataFrame) -> MovingAverageTransformer:
        """Computes moving average for all players' all stats starting from mv_start_date to today"""
        if self.player_type == "batter":
            self.opp_player_type_handedness_col = "p_throws"
        else:
            self.opp_player_type_handedness_col = "stand"

        player_ma_schema_lst = (
            [StructField(self.player_type, StringType(), False)] +
            [StructField(col_name, ArrayType(FloatType()), False) for col_name in self.stats_to_compute]
        )
        # Factor in the extra 2 columns for platoon features ("pwOBA", "pPA")
        if "pwOBA" in self.stats_to_compute:
            player_ma_schema_lst.append(StructField('RpwOBA', ArrayType(FloatType()), False))
        if "pPA" in self.stats_to_compute:
            player_ma_schema_lst.append(StructField('RpPA', ArrayType(FloatType()), False))

        # Returned Player MA DF schema
        player_ma_schema = StructType(player_ma_schema_lst)

        player_ma_udf = pandas_udf(
            f=functools.partial(MovingAverageEstimator._compute_player_ma, self),
            returnType=player_ma_schema,
            functionType=PandasUDFType.GROUPED_MAP,
        )

        # Compute player MA dataset
        self.ma_stats_df = (
            dataset
            .select(*[self.player_type, "game_date", "events", "launch_speed", self.opp_player_type_handedness_col])
            .groupby(self.player_type)
            .apply(player_ma_udf)
            .toPandas()
        )

        # Saving as json file
        intermediate_path = os.path.join(ROOT_DIR, "intermediate", str(date.today()) + "-" + UNIQUE_RUN_ID)
        os.makedirs(intermediate_path, exist_ok=True)
        self.saved_file_name = self.player_type + "-" + str(self.ma_days) + ".json"
        intermediate_path_file = os.path.join(intermediate_path, self.saved_file_name)

        self.ma_stats_df.set_index(self.player_type, drop=True, inplace=True)
        self.ma_stats_df.index = self.ma_stats_df.index.map(str)
        self.ma_stats_df.to_json(intermediate_path_file, orient="index", indent=4)

        return MovingAverageTransformer(
            inputCols=self.inputCols,
            outputCols=self.outputCols,
            player_type=self.player_type,
            ma_days=self.ma_days,
            stats_to_compute=self.stats_to_compute,
            training_data_start_date=self.training_data_start_date,
            training_data_end_date=self.training_data_end_date,
            ma_start_date=self.ma_start_date,
            ma_stats_df=self.ma_stats_df,
            pa_ts=self.pa_ts,
            opp_player_type_handedness_col=self.opp_player_type_handedness_col,
            saved_file_name=self.saved_file_name,
        )

    def _compute_player_ma(self, player_group: pd.DataFrame) -> pd.DataFrame:
        """Compute moving averages for all events of one player"""
        assert(
            player_group.columns.tolist() == [self.player_type, "game_date", "events", "launch_speed", self.opp_player_type_handedness_col]
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

            # check computed vs input moving average start date
            computed_ma_start_date = stat_ma.index[self.ma_days]
            assert (computed_ma_start_date.date() == self.ma_start_date)
            stat_ma_lst = stat_ma.tolist()[self.ma_days:]
            stat_ma_mapping[stat] = stat_ma_lst
            del player_event_group

            # extra work for platoon features
            if stat in ["pPA", "pwOBA"]:
                computed_ma_start_date2 = stat_ma2.index[self.ma_days]
                assert (computed_ma_start_date2.date() == self.ma_start_date)
                stat_ma_lst2 = stat_ma2.tolist()[self.ma_days:]
                stat_ma_mapping["R"+stat] = stat_ma_lst2
                del player_event_group2
                del player_group1
                del player_group2

        # add player id as a column
        player_id = player_group[self.player_type][0]
        stat_ma_mapping[self.player_type] = player_id
        return pd.DataFrame.from_dict(
            {player_id: stat_ma_mapping},
            orient='index',
        )

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
