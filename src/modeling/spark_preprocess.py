import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from utils import read_params
from constants import NEEDED_COLS, EVENTS_CLEANING_MAP, EVENTS_CATEGORIES
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, date_format
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from src.modeling.spark_transformers.game_state import (
    ConvertToDatetime,
    EncodeOnBaseOccupancy,
    ComputeNetScore,
    ComputeDaysSinceStart,
    EncodeHandedness,
    EncodeInningTopBot,
)
from src.modeling.spark_transformers.movingaverage import MovingAverageEstimator, MovingAverageTransformer


def preprocess(config_filepath: str):
    """
    Aggregate downloaded data and prepare features.
    """
    config = read_params(config_filepath)
    preprocessing_config = config["preprocessing"]
    training_data_start_date = datetime.strptime(preprocessing_config["training_data_start_date"], "%Y-%m-%d").date()
    training_data_end_date = datetime.strptime(preprocessing_config["training_data_end_date"], "%Y-%m-%d").date()
    ma365_start_date = datetime.combine(training_data_start_date, datetime.min.time()) + timedelta(365)
    ma30_start_date = datetime.combine(training_data_start_date, datetime.min.time()) + timedelta(30)

    spark = (
        SparkSession
        .builder
        .master("local[12]")
        .appName("baseball-analytics")
        .config("spark.sql.debug.maxToStringFields", 1000)
        .getOrCreate()
    )

    # Load prepared data
    raw_data_schema = StructType(
        [
            StructField('game_date', StringType(), True),
            StructField('batter', StringType(), True),
            StructField('pitcher', StringType(), True),
            StructField('events', StringType(), True),
            StructField('description', StringType(), True),
            StructField('game_type', StringType(), True),
            StructField('stand', StringType(), True),
            StructField('p_throws', StringType(), True),
            StructField('home_team', StringType(), True),
            StructField('away_team', StringType(), True),
            StructField('game_year', ShortType(), True),
            StructField('on_3b', IntegerType(), True),
            StructField('on_2b', IntegerType(), True),
            StructField('on_1b', IntegerType(), True),
            StructField('outs_when_up', ByteType(), True),
            StructField('inning', ByteType(), True),
            StructField('inning_topbot', StringType(), True),
            StructField('launch_speed', DoubleType(), True),
            StructField('at_bat_number', ShortType(), True),
            StructField('pitch_number', ShortType(), True),
            StructField('game_pk', StringType(), True),
            StructField('bat_score', ByteType(), True),
            StructField('fld_score', ByteType(), True),
        ]
    )

    data = (
        spark
        .read
        .format("csv")
        .option("header", "true")
        .schema(raw_data_schema)
        .load(preprocessing_config["agg_data_filepath"])
    )

    # preprocessing with data pipeline
    #cols = [column for column in data.columns if column != "events"]
    #X, y = data.select(*cols), data.select(col("events"))

    data_pipeline = Pipeline(
        stages=[
            ConvertToDatetime(
                inputCol="game_date",
                outputCol="game_date",
            ),
            EncodeOnBaseOccupancy(
                inputCols=["on_3b", "on_2b", "on_1b"],
                outputCols=["on_3b", "on_2b", "on_1b"],
            ),
            ComputeNetScore(
                inputCols=["bat_score", "fld_score"],
                outputCol="net_score",
            ),
            ComputeDaysSinceStart(
                inputCol="game_date",
                outputCol="days_since_start"
            ),
            EncodeHandedness(
                inputCols=["stand", "p_throws"],
                outputCols=["batter_stand", "pitcher_throws"],
            ),
            EncodeInningTopBot(
                inputCol="inning_topbot",
                outputCol="topbot",
            ),
            # batter_365_days_ma_
            MovingAverageEstimator(
                inputCols=["events", "game_date", "batter", "launch_speed", "p_throws", ],
                outputCols=["batter_365_days_ma_"+stat for stat in ["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA", "mEV", "aEV", "pwOBA", "pPA"]],
                player_type="batter",
                ma_days=365,
                stats_to_compute=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA",
                                  "mEV", "aEV", "pwOBA", "pPA"],
                training_data_start_date=training_data_start_date,  # date(2008, 2, 27),
                training_data_end_date=training_data_end_date,  # date(2023, 5, 3),
                ma_start_date=ma365_start_date.date(),  # date(2009, 2, 26), # datetime(2022, 4, 7) + timedelta(365)
            ),
            # pitcher_365_days_ma_
            MovingAverageEstimator(
                inputCols=["events", "game_date", "pitcher", "launch_speed", "stand", ],
                outputCols=["pitcher_365_days_ma_"+stat for stat in ["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA", "mEV", "aEV", "pwOBA", "pPA"]],
                player_type="pitcher",
                ma_days=365,
                stats_to_compute=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "DP", "FO", "HBP", "SF", "SH", "wOBA",
                                  "mEV", "aEV", "pwOBA", "pPA"],
                training_data_start_date=training_data_start_date,
                training_data_end_date=training_data_end_date,
                ma_start_date=ma365_start_date.date(),
            ),
            # batter_30_days_ma_
            MovingAverageEstimator(
                inputCols=["events", "game_date", "batter", "launch_speed", "p_throws", ],
                outputCols=["batter_30_days_ma_"+stat for stat in ["PA", "1B", "2B", "3B", "HR", "BB", "SO", "wOBA"]],
                player_type="batter",
                ma_days=30,
                stats_to_compute=["PA", "1B", "2B", "3B", "HR", "BB", "SO", "wOBA"],
                training_data_start_date=training_data_start_date,
                training_data_end_date=training_data_end_date,
                ma_start_date=ma30_start_date.date(),
            ),
            # head_to_head_ma_
            # ball_park_
        ]
    )

    pipelinemodel = data_pipeline.fit(data)
    features_and_events = pipelinemodel.transform(data)

    # get all output columns
    output_cols = ['events', 'outs_when_up', 'inning', ]
    for stage in pipelinemodel.stages:
        output_cols.extend(stage.get_output_cols())

    features_and_events = features_and_events.select(*output_cols)
    features_and_events = features_and_events.withColumn(
        "game_date",
        date_format(col("game_date"), "yyyy-MM-dd HH:mm:ss")
    )
    features_and_events = features_and_events.toPandas()

    # post-cleaning
    features_and_events["game_date"] = pd.to_datetime(features_and_events["game_date"])
    features_and_events = features_and_events[features_and_events["game_date"] >= ma365_start_date]
    features_and_events = (
        features_and_events
        .reset_index(drop=True)
        .drop(["game_date"], axis=1)
    )
    features_and_events["events"] = features_and_events["events"].apply(lambda x: EVENTS_CATEGORIES.index(x))

    # save features and events
    features_and_events.to_csv(preprocessing_config["dataset_filepath"], index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config_001.yaml")
    parsed_args = args.parse_args()
    preprocess(config_filepath=parsed_args.config)
