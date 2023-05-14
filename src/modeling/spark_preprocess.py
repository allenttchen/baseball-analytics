import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from utils import read_params
from constants import NEEDED_COLS, EVENTS_CLEANING_MAP, EVENTS_CATEGORIES
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from src.modeling.spark_transformers.game_state import (
    EncodeOnBaseOccupancy,
    ComputeNetScore,
    ComputeDaysSinceStart,
    EncodeHandedness,
    EncodeInningTopBot,
)


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
    cols = [column for column in data.columns if column != "events"]
    X, y = data.select(*cols), data.select(col("events"))

    data_pipeline = PipelineModel(
        stages=[
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
        ]
    )

    Xt = data_pipeline.transform(X)
    Xt.show()
    # transformed_feature_col_names = data_pipeline["feature_transformers"].get_feature_names_out().tolist()
    # features = pd.DataFrame(Xt, columns=transformed_feature_col_names)
    # features_and_events = pd.concat([features, pd.Series(y, name="events")], axis=1)
    #
    # # post-cleaning
    # features_and_events = features_and_events[features_and_events["identity__game_date"] >= ma365_start_date]
    # features_and_events = (
    #     features_and_events
    #     .reset_index(drop=True)
    #     .drop(["identity__game_date"], axis=1)
    # )
    # features_and_events["events"] = features_and_events["events"].apply(lambda x: EVENTS_CATEGORIES.index(x))
    #
    # # save features and events
    # features_and_events.to_csv(preprocessing_config["dataset_filepath"], index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config_001.yaml")
    parsed_args = args.parse_args()
    preprocess(config_filepath=parsed_args.config)
