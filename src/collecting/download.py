import os
import requests


def download():

    query_url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&" \
                "hfPTM=&" \
                "hfPT=&" \
                "hfAB=&" \
                "hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&" \
                "hfPull=&hfC=&hfSea=2022%7C&hfSit=&" \
                "player_type=batter&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&" \
                "game_date_gt=&game_date_lt=&" \
                "hfMo=&hfTeam=LAA%7C&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&" \
                "metric_1=&" \
                "group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&" \
                "player_event_sort=api_p_release_speed&sort_order=desc#results"

    results = requests.get(query_url)
    print(results.content)

if __name__ == "__main__":
    download()
