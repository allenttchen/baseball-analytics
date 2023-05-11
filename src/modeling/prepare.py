import os
import concurrent.futures
from collections import defaultdict
from typing import List

from tqdm import tqdm
import pandas as pd

from constants import NEEDED_COLS, EVENTS_CLEANING_MAP


def prepare(raw_data_dir: str, seasons: List[int], teams: List[str], agg_raw_data_filepath: str):
    season_team_filepaths = [
        os.path.join(raw_data_dir, str(season), team+".csv") for season in seasons for team in teams
    ]
    dfs = []
    with tqdm(total=len(season_team_filepaths)) as progress:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_pre_clean_job, season_team_filepath) for season_team_filepath in season_team_filepaths
            }
            for future in concurrent.futures.as_completed(futures):
                dfs.append(future.result())
                progress.update(1)
    if dfs:
        aggregated_df = pd.concat(dfs, axis=0)
        aggregated_df = aggregated_df.sort_values(
            ['game_date', 'game_pk', 'at_bat_number', 'pitch_number'],
            ascending=False
        )
        aggregated_df.reset_index(drop=True, inplace=True)
    else:
        aggregated_df = pd.DataFrame()
    # save
    aggregated_df.to_csv(agg_raw_data_filepath, index=False)
    return aggregated_df


def _pre_clean_job(season_team_filepath):
    """Pre clean raw data downloaded from Statcase, returning just the necessary rows and columns for model training"""
    raw_data_df = pd.read_csv(season_team_filepath, low_memory=False)
    needed_data_df = raw_data_df[NEEDED_COLS]
    needed_data_df = needed_data_df[needed_data_df["events"].notnull()]
    events_mapping = defaultdict(lambda: "delete", EVENTS_CLEANING_MAP)
    needed_data_df["events"] = needed_data_df["events"].map(events_mapping)
    needed_data_df.drop(needed_data_df[needed_data_df["events"] == "delete"].index, inplace=True)
    needed_data_df.reset_index(drop=True, inplace=True)

    return needed_data_df


if __name__ == "__main__":
    raw_data_dir = "/Users/allenchen/projects/baseball-analytics/data/raw"
    seasons = list(range(2008, 2024))
    teams = ["LAA", "HOU", "OAK", "TOR", "ATL", "MIL", "STL",
             "CHC", "ARI", "LAD", "SF", "CLE", "SEA", "MIA",
             "NYM", "WSH", "BAL", "SD", "PHI", "PIT", "TEX",
             "TB", "BOS", "CIN", "COL", "KC", "DET", "MIN", "CWS", "NYY"]
    agg_raw_data_filepath = "/Users/allenchen/projects/baseball-analytics/data/aggregated/20230504_agg_raw_data.csv"
    _ = prepare(
        raw_data_dir=raw_data_dir,
        seasons=seasons,
        teams=teams,
        agg_raw_data_filepath=agg_raw_data_filepath
    )
