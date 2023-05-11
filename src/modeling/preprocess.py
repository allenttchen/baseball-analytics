"""
Run data preprocess pipeline.
"""
import sys
from collections import defaultdict
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from utils import read_params
from constants import NEEDED_COLS, EVENTS_CLEANING_MAP, EVENTS_CATEGORIES
from pipeline import data_pipeline


def preprocess(config_filepath: str):
    """
    Aggregate downloaded data and prepare features.
    """
    config = read_params(config_filepath)
    preprocessing_config = config["preprocessing"]

    # Load prepared data
    data = pd.read_csv(preprocessing_config["raw_data_filepath"])

    # preprocessing with data pipeline
    cols = [col for col in data.columns if col != "events"]
    X, y = data[cols], data["events"]
    Xt = data_pipeline.fit_transform(X, y)
    transformed_feature_col_names = data_pipeline["feature_transformers"].get_feature_names_out().tolist()
    features = pd.DataFrame(Xt, columns=transformed_feature_col_names)
    features_and_events = pd.concat([features, pd.Series(y, name="events")], axis=1)

    # post-cleaning
    ma365_start_date = datetime.strptime(preprocessing_config["raw_data_starting_date"], "%Y-%m-%d") + timedelta(365)
    features_and_events = features_and_events[features_and_events["identity__game_date"] >= ma365_start_date]
    features_and_events = (
        features_and_events
        .reset_index(drop=True)
        .drop(["identity__game_date"], axis=1)
    )
    features_and_events["events"] = features_and_events["events"].apply(lambda x: EVENTS_CATEGORIES.index(x))

    # save features and events
    features_and_events.to_csv(preprocessing_config["dataset_filepath"], index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config_001.yaml")
    parsed_args = args.parse_args()
    preprocess(config_filepath=parsed_args.config)
