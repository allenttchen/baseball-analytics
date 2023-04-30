import requests
import re
import json

from bs4 import BeautifulSoup
import pandas as pd


if __name__ == "__main__":

    year = 2023
    URL = f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?" \
        f"type=year&year={str(year)}&batSide=&stat=index_wOBA&condition=All&rolling="

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find("div", class_="article-template").find("script").text.strip()
    m = re.search(r'var data = (\[{.*}\])', results)
    data = json.loads(m.group(1))
    df = pd.DataFrame(data)
    df = df[["year_range", "venue_name", "name_display_club", "n_pa", "index_1b", "index_2b", "index_3b", "index_hr", "index_bb"]]

    club_name_to_abbre_mapping = {
        'Angels': "LAA",
        'Orioles': "BAL",
        'Red Sox': "BOS",
        'White Sox': "CWS", # can be CHW too, double check every year
        'Guardians': "CLE", # Cleveland Indians
        'Royals': "KC", # saw KAN too
        'Athletics': "OAK",
        'Rays': "TB",
        'Blue Jays': "TOR",
        'D-backs': "ARI",
        'Cubs': "CHC",
        'Rockies': "COL",
        'Dodgers': "LAD",
        'Pirates': "PIT",
        'Brewers': "MIL",
        'Reds': "CIN",
        'Cardinals': "STL",
        'Marlins': "MIA", # check for FLA for past years
        'Astros': "HOU",
        'Tigers': "DET",
        'Giants': "SF",
        'Braves': "ATL",
        'Padres': "SD",
        'Phillies': "PHI",
        'Mariners': "SEA",
        'Rangers': "TEX",
        'Mets': "NYM",
        'Nationals': "WSH",
        'Twins': "MIN",
        'Yankees': "NYY",
    }
    df["team_abbre"] = df["name_display_club"].map(club_name_to_abbre_mapping)
    df.to_csv("../intermediate/park_factors.csv", index=False)
    #print(json.dumps(data, indent=4))
