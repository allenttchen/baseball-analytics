import io
import os
import requests
import logging
from time import sleep

import pandas as pd
from tqdm.auto import tqdm
import backoff

from constants import RAW_DATA_DTYPES

NUM_TRIES = 0


def download(seasons, teams, saved_dir=None):
    """
    Download raw Pitch-level data from Statcast
    """
    global NUM_TRIES
    for season in tqdm(seasons, desc="Seasons"):
        season_dir = os.path.join(saved_dir, str(season))
        os.makedirs(season_dir, exist_ok=True)
        for team in tqdm(teams, desc="Teams"):
            data = small_request(season, team)
            NUM_TRIES = 0
            sleep(10)
            season_team_filepath = os.path.join(season_dir, team+".csv")
            data.to_csv(season_team_filepath, index=False)


class DownloadStatcastError(Exception):
    pass


@backoff.on_exception(backoff.expo, DownloadStatcastError, max_tries=5)
def small_request(season: int, team: str) -> pd.DataFrame:
    global NUM_TRIES
    url_template = (
        "https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&"
        "hfPTM=&hfPT=&hfAB=&hfGT=R%7CPO%7CS%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&"
        "hfSea={season}%7C&hfSit=&player_type=pitcher&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&"
        "hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam={team}%7C&home_road=&hfRO=&position=&"
        "hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&"
        "min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc"
    )
    url = url_template.format(season=str(season), team=team)
    try:
        response = requests.get(url, timeout=None).content.decode('utf-8')
        data = pd.read_csv(io.StringIO(response), low_memory=False)#, dtype=RAW_DATA_DTYPES)
        data = data.sort_values(
            ['game_date', 'game_pk', 'at_bat_number', 'pitch_number'],
            ascending=False,
        )
        return data
    except Exception:
        logging.error(f"Download failed for Team {team} in Season {season} on try No {NUM_TRIES}.")
        NUM_TRIES += 1
        if NUM_TRIES < 5:
            raise DownloadStatcastError()
        logging.error(f"Download failed for Team {team} in Season {season} on final try. Investigate.")


if __name__ == "__main__":
    download(
        seasons=list(range(2008, 2024)),
        teams=['LAA', 'HOU', 'OAK', 'TOR', 'ATL', 'MIL', 'STL',
               'CHC', 'ARI', 'LAD', 'SF', 'CLE', 'SEA', 'MIA',
               'NYM', 'WSH', 'BAL', 'SD', 'PHI', 'PIT', 'TEX',
               'TB', 'BOS', 'CIN', 'COL', 'KC', 'DET', 'MIN', 'CWS', 'NYY'],
        saved_dir="/Users/allenchen/projects/baseball-analytics/data/raw"
    )
