import requests
import re
import json
import os
from collections import defaultdict

from bs4 import BeautifulSoup
import pandas as pd


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
    'Expos': "WSH", # Washington Expos before 2004, changed to Nationals in 2005
}
club_name_to_abbre_mapping = defaultdict(lambda: "INVESTIGATE", **club_name_to_abbre_mapping)
all_abbre = set(club_name_to_abbre_mapping.values())


park_factors_col_names_mapping = {
    "index_1b": "1B",
    "index_2b": "2B",
    "index_3b": "3B",
    "index_hr": "HR",
    "index_bb": "BB",
}

if __name__ == "__main__":
    stats_to_compute = ["1B", "2B", "3B", "HR", "BB"]
    year_range = range(2023, 1999, -1)
    # {team: {stat: {year: ...}}}
    park_factors_mapping = {}
    for year in year_range:

        # Webscrape
        URL = f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?" \
            f"type=year&year={str(year)}&batSide=&stat=index_wOBA&condition=All&rolling="
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find("div", class_="article-template").find("script").text.strip()
        results_data = re.search(r'var data = (\[{.*}\])', results)
        data = json.loads(results_data.group(1))

        # Preprocess
        df = pd.DataFrame(data)
        df = df[["name_display_club", "venue_name", "index_1b", "index_2b", "index_3b", "index_hr", "index_bb"]]
        df.rename(columns=park_factors_col_names_mapping, inplace=True)
        df["team_abbre"] = df["name_display_club"].map(club_name_to_abbre_mapping)

        # one off fix (due to buggy source data)
        if year == 2020:
            df.loc[df["venue_name"] == "Rogers Centre", "name_display_club"] = "Blue Jays"
            df.loc[df["venue_name"] == "Rogers Centre", "team_abbre"] = "TOR"

        # fill in the park_factors_mapping
        for index, row in df.iterrows():
            team = row["team_abbre"]
            if team == "INVESTIGATE":
                print(f"The team to be investigated is {row['name_display_club']} in {year}")
            if team not in park_factors_mapping:
                park_factors_mapping[team] = {}
            for stat in stats_to_compute:
                if stat not in park_factors_mapping[team]:
                    park_factors_mapping[team][stat] = {}
                park_factors_mapping[team][stat][year] = row[stat]

        # Impute missing values (some table in some years will have < 30 rows)
        missing_teams = all_abbre - set(df["team_abbre"])
        for missing_team in missing_teams:
            if missing_team not in park_factors_mapping:
                park_factors_mapping[missing_team] = {}
            for stat in stats_to_compute:
                if stat not in park_factors_mapping[missing_team]:
                    park_factors_mapping[missing_team][stat] = {}
                park_factors_mapping[missing_team][stat][year] = 100

    # Save the dictionary into json
    with open("../intermediate/park_factors.json", "w") as f:
        park_factors_json = json.dumps(park_factors_mapping, indent=4)
        f.write(park_factors_json)

    # Validation
    print(f"total teams: {len(park_factors_mapping)}")
    for team, stats in park_factors_mapping.items():
        for stat, years in stats.items():
            if len(years) != 24:
                print(f"The {team} with the {stat} has {len(years)} records")
                print(set(year_range) - set(years.keys()))
    #df.to_csv("../intermediate/park_factors_2022.csv", index=False)
    #print(json.dumps(data, indent=4))
