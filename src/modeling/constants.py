import os
import uuid


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#UNIQUE_RUN_ID = str(uuid.uuid4())
UNIQUE_RUN_ID = str(000)

TARGET_COL = "events"

NEEDED_COLS = [
    'game_date',
    'batter',
    'pitcher',
    'events',
    'description',
    'game_type',
    'stand',
    'p_throws',
    'home_team',
    'away_team',
    'game_year',
    'on_3b',
    'on_2b',
    'on_1b',
    'outs_when_up',
    'inning',
    'inning_topbot',
    'launch_speed',
    'at_bat_number',
    'pitch_number',
    'game_pk',
    'bat_score',
    'fld_score',
]

EVENTS_CLEANING_MAP = {
    "field_out": "fo",
    "strikeout": "so",
    "single": "s",
    "double": "d",
    "triple": "t",
    "walk": "w",
    "home_run": "hr",
    "force_out": "fo",
    "grounded_into_double_play": "dp",
    "double_play": "dp",
    "hit_by_pitch": "hbp",
    "field_error": "e",
    "sac_fly": "sf",
    "sac_bunt": "sh",
    "fielders_choice": "fc",
    "caught_stealing_2b": "delete",
    "fielders_choice_out": "fo",
    "strikeout_double_play": "so",
    "catcher_interf": "e",
    "triple_play": "tp",
    "pickoff_1b": "delete",
}

IDENTITY_COLS = [
    ##'batter',
    ##'pitcher',
    'game_date',
    #'description',
    #'game_type',
    ##'stand',
    ##'p_throws',
    'outs_when_up',
    'inning',
    ##'inning_topbot',
    #'game_pk',
    ##'home_team',
]

STATS_TO_EVENTS = {
    "PA": "pa",
    "1B": "s",
    "2B": "d",
    "3B": "t",
    "HR": "hr",
    "BB": "w",
    "SO": "so",
    "DP": "dp",
    "FO": "fo",
    "HBP": "hbp",
    "SF": "sf",
    "SH": "sh",
    "wOBA": ["w", "hbp", "s", "d", "t", "hr"],
    "mEV": "not none",
    "aEV": "not none",
    "pwOBA": ["w", "hbp", "s", "d", "t", "hr"],
    "pPA": "ppa",
}

WOBA_FACTORS = {
    "w": 0.702,
    "hbp": 0.733,
    "s": 0.892,
    "d": 1.261,
    "t": 1.593,
    "hr": 2.039,
}

SEASON_START_DATES = {
    2023: "2023-03-30",
    2022: "2022-04-07",
    2021: "2021-04-01",
    2020: "2020-07-23", # COVID
    2019: "2019-03-20",
    2018: "2018-03-29",
    2017: "2017-04-02",
    2016: "2016-04-03",
    2015: "2015-04-05",
    2014: "2014-03-31",
    2013: "2013-03-31",
    2012: "2012-03-28",
    2011: "2011-03-31",
    2010: "2010-04-04",
    2009: "2009-04-05",
    2008: "2008-03-25",
    2007: "2007-04-01",
    2006: "2006-04-02",
    2005: "2005-04-03",
    2004: "2004-04-04",
    2003: "2003-03-30",
    2002: "2002-03-31",
    2001: "2001-04-01",
    2000: "2000-03-29",
    1999: "1999-04-04",
}

EVENTS_CATEGORIES = [
    's',
    'd',
    't',
    'hr',
    'w',
    'so',
    'fo',
    'dp',
    'e',
    'fc',
    'hbp',
    'sf',
    'sh',
    'tp',
]
