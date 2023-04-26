import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    'on_3b',
    'on_2b',
    'on_1b',
    'outs_when_up',
    'inning',
    'inning_topbot',
    'game_pk',
    'bat_score',
    'fld_score'
]

IDENTITY_COLS = [
    'batter',
    'pitcher',
    'game_date',
    #'description',
    #'game_type',
    #'stand',
    #'p_throws',
    'outs_when_up',
    'inning',
    'inning_topbot',
    #'game_pk',
]

STATS_TO_EVENTS = {
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
}

WOBA_FACTORS = {
    "w": 0.702,
    "hbp": 0.733,
    "s": 0.892,
    "d": 1.261,
    "t": 1.593,
    "hr": 2.039,
}
