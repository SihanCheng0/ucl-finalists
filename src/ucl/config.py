"""Constants shared by every pipeline stage."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LLM_CACHE_DIR = ROOT / "data" / "llm_cache"
OUT_DIR = ROOT / "out"

# UEFA's seasonYear is the calendar year a season ends: 2026 = 2025-26.
SEASONS = list(range(2012, 2027))
TARGET_SEASONS = [2022, 2023, 2024, 2025, 2026]


def season_label(season: int) -> str:
    return f"{season - 1}-{season % 100:02d}"


def is_league_format(season: int) -> bool:
    """2024-25 onward: one 36-team league phase instead of eight groups of four."""
    return season >= 2025


def field_size(season: int) -> int:
    return 36 if is_league_format(season) else 32


def phase_matches(season: int) -> int:
    return 8 if is_league_format(season) else 6


def ko_size(season: int) -> int:
    return 24 if is_league_format(season) else 16


# The five target finals, by UEFA team id (spec §3).
EXPECTED_FINALS = {
    2022: {"winner_id": "50051", "runner_up_id": "7889"},   # Real Madrid 1-0 Liverpool
    2023: {"winner_id": "52919", "runner_up_id": "50138"},  # Man City 1-0 Inter
    2024: {"winner_id": "50051", "runner_up_id": "52758"},  # Real Madrid 2-0 Dortmund
    2025: {"winner_id": "52747", "runner_up_id": "50138"},  # PSG 5-0 Inter
    2026: {"winner_id": "52747", "runner_up_id": "52280"},  # PSG 1-1 Arsenal, 4-3 on penalties
}

# Friendlier names than UEFA's internationalName ("Paris", "B. Dortmund").
DISPLAY_NAMES = {
    "50051": "Real Madrid",
    "7889": "Liverpool",
    "52919": "Manchester City",
    "50138": "Inter",
    "52758": "Borussia Dortmund",
    "52747": "Paris Saint-Germain",
    "52280": "Arsenal",
}

# Round names differ between seasons (spec §4); depth orders them.
ROUND_DEPTH = {
    "Group stage": 0,
    "League Phase": 0,
    "Knockout Phase Play-Offs": 1,
    "Knock-out Play-off": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Final": 5,
}
FEATURE_GROUPS = {
    "pedigree": ["coef_log"],
    "results": ["points_pg", "goal_diff_pg"],
    "style": [
        "shots_pg", "shot_accuracy", "conversion", "attacks_pg",
        "shots_against_pg", "on_target_against_pg", "save_pct",
        "possession_pct", "pass_accuracy", "passes_pg", "long_pass_share",
        "distance_km_pg", "fouls_pg",
    ],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
FEATURE_GROUP = {f: group for group, fs in FEATURE_GROUPS.items() for f in fs}

# feature -> (plain-English label, display decimals, stored as a 0-1 fraction shown as %)
FEATURE_META = {
    "coef_log": ("Pre-season UEFA club coefficient (points)", 1, False),
    "points_pg": ("Points per game", 2, False),
    "goal_diff_pg": ("Goal difference per game", 2, False),
    "shots_pg": ("Shots per game", 1, False),
    "shot_accuracy": ("Shot accuracy (% on target)", 1, True),
    "conversion": ("Shot conversion (% of shots scored)", 1, True),
    "attacks_pg": ("Attacks per game", 1, False),
    "shots_against_pg": ("Opponent shots per game", 1, False),
    "on_target_against_pg": ("Opponent shots on target per game", 1, False),
    "save_pct": ("Save rate (% of shots on target faced)", 1, True),
    "possession_pct": ("Possession (%)", 1, False),
    "pass_accuracy": ("Pass accuracy (%)", 1, True),
    "passes_pg": ("Passes per game", 0, False),
    "long_pass_share": ("Long passes (% of passes)", 1, True),
    "distance_km_pg": ("Distance covered per game (km)", 1, False),
    "fouls_pg": ("Fouls committed per game", 1, False),
}
LOWER_IS_BETTER = {"shots_against_pg", "on_target_against_pg", "fouls_pg"}

# Per-match columns each feature needs, for the coverage rule (spec §5.3).
FEATURE_SOURCES = {
    "coef_log": [],
    "points_pg": [],
    "goal_diff_pg": [],
    "shots_pg": ["attempts_on_target", "attempts_off_target"],
    "shot_accuracy": ["attempts_on_target", "attempts_off_target"],
    "conversion": ["goals", "attempts_on_target", "attempts_off_target"],
    "attacks_pg": ["attacks"],
    "shots_against_pg": ["opp_attempts_on_target", "opp_attempts_off_target"],
    "on_target_against_pg": ["opp_attempts_on_target"],
    "save_pct": ["saves", "opp_attempts_on_target"],
    "possession_pct": ["ball_possession"],
    "pass_accuracy": ["passes_completed", "passes_attempted"],
    "passes_pg": ["passes_attempted"],
    "long_pass_share": ["passes_long_attempted", "passes_attempted"],
    "distance_km_pg": ["distance_covered"],
    "fouls_pg": ["fouls_committed"],
}
OWN_STATS = [
    "goals", "attempts_on_target", "attempts_off_target", "ball_possession", "passes_attempted",
    "passes_completed", "passes_long_attempted", "attacks", "distance_covered", "fouls_committed", "saves",
]
OPP_STATS = ["attempts_on_target", "attempts_off_target"]

COVERAGE_MIN = 0.90
COEF_MATCH_MIN = 0.75
MIN_MATCHES_WITH_STATS = 4

# The match feed and the stats/coefficient feeds disagree on a few club ids; map match-feed ids to the canonical one.
TEAM_ID_ALIASES = {"2614166": "50065"}  # Steaua București / FCSB
# A match distance below this is partial tracking, not a real value.
MIN_MATCH_DISTANCE_KM = 80.0
# Plausible per-season averages; the build fails if a team-season falls outside (catches unit errors).
PLAUSIBLE_RANGES = {"possession_pct": (20.0, 80.0), "distance_km_pg": (90.0, 140.0), "pass_accuracy": (0.4, 1.0)}

GBR_PARAMS = {
    "n_estimators": 250, "learning_rate": 0.03, "max_depth": 2,
    "min_samples_leaf": 8, "subsample": 0.8, "random_state": 42,
}
LOGIT_PARAMS = {"C": 0.3, "max_iter": 2000}
TOP_DRIVERS = 6
ROBUST_SHARE = 0.8  # same sign in >= 80% of LOSO fits = 12 of 15 (spec §6)
MARGINAL_MIN = 0.05  # |Spearman| below this means a stat has no clear direction on its own
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42

MATCHES_URL = (
    "https://match.uefa.com/v5/matches?competitionId=1&seasonYear={season}"
    "&phase=TOURNAMENT&limit=500&offset=0&order=ASC"
)
MATCH_STATS_URL = "https://matchstats.uefa.com/v2/team-statistics/{match_id}"
COEF_PAGE_SIZE = 500
COEF_URL = (
    "https://comp.uefa.com/v2/coefficients?coefficientRange=OVERALL&coefficientType=MEN_CLUB"
    f"&language=EN&page={{page}}&pagesize={COEF_PAGE_SIZE}&seasonYear={{season}}"
)
STANDINGS_URL = "https://standings.uefa.com/v1/standings?competitionId=1&seasonYear={season}"
HTTP_TIMEOUT_S = 25
HTTP_RETRY_DELAYS_S = (1, 2, 4, 8)

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen/qwen3.8-27b"
LLM_PARAMS = {"temperature": 0.3, "max_tokens": 8000, "reasoning_effort": "none"}
LLM_TIMEOUT_S = 300
LLM_CONTEXT_LENGTH = 16384
LMS_BIN = Path.home() / ".lmstudio" / "bin" / "lms"
# The same model hosted on OpenRouter (its id there matches LM Studio's). It answers when UCL_LLM_PROVIDER is
# "openrouter", or when that is unset and OPENROUTER_API_KEY is.
LLM_PROVIDER_ENV = "UCL_LLM_PROVIDER"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHECK_TIMEOUT_S = 10
# Only providers that run the weights at 16 or 8 bits: none of the 4-bit or undisclosed ones (the local copy is 4-bit)
OPENROUTER_QUANTIZATIONS = ["bf16", "fp16", "fp8"]

# --- UCL Lab, the local dashboard (spec 2026-10-01-ucl-lab-dashboard-design.md) ---
WEB_PORT = 8787
WEB_DIR = ROOT / "web"
# How Explore groups the stats. save_pct is listed so a dataset that keeps it still has a home for it.
STAT_SECTIONS = {
    "Results": ["points_pg", "goal_diff_pg"],
    "Attack": ["shots_pg", "shot_accuracy", "conversion", "attacks_pg"],
    "Defence": ["shots_against_pg", "on_target_against_pg", "save_pct"],
    "Control": ["possession_pct", "pass_accuracy", "passes_pg", "long_pass_share"],
    "Intensity": ["distance_km_pg", "fouls_pg"],
    "Pedigree": ["coef_log"],
}
# Names people type that UEFA's names don't contain, matched without case or accents.
TEAM_SEARCH_ALIASES = {
    "psg": "52747",
    "man city": "52919",
    "man united": "52682",
    "manchester united": "52682",
    "barca": "50080",
    "atletico madrid": "50124",
    "spurs": "1652",
    "bvb": "52758",
    "inter milan": "50138",
    "internazionale": "50138",
    "ac milan": "50058",
    "juve": "50139",
    "bayern munich": "50037",
    "gladbach": "52757",
}

# --- UCL Lab, Plan B: the live season and players (spec §3.2, §3.3) ---
LIVE_SEASON = 2027  # 2026-27: never part of SEASONS, the dataset or the model
LIVE_MAX_AGE_S = 6 * 3600  # live data is refetched when older than this
LIVE_SETTLED_S = 24 * 3600  # a match's stats are cached for good once it finished this long ago
LIVE_RETRY_S = 300  # after a failed live refresh, the next automatic attempt waits this long
REQUEST_TIMEOUT_S = 8  # on-demand fetches made while a browser waits: one attempt only
REQUEST_FAILURE_MEMORY_S = 60
PLAYERS_DIR = RAW_DIR / "players"
PLAYER_PAGE_SIZE = 100
PLAYER_MAX_PAGES = 30
# key, label, kind, decimals, colourable on the pitch
PLAYER_STATS = [
    ("minutes_played_official", "Minutes", "minutes", 0, False),
    ("matches_appearance", "Appearances", "count", 0, False),
    ("goals", "Goals", "count", 2, True),
    ("assists", "Assists", "count", 2, True),
    ("key_passes", "Key passes", "count", 2, True),
    ("attempts", "Shots", "count", 2, True),
    ("attempts_on_target", "Shots on target", "count", 2, True),
    ("passes_attempted", "Passes attempted", "count", 1, False),
    ("passes_completed", "Passes completed", "count", 1, False),
    ("passes_accuracy", "Pass accuracy (%)", "rate", 1, True),
    ("distance_covered", "Distance covered (km)", "distance", 2, True),
    ("top_speed", "Top speed (km/h)", "speed", 1, True),
    ("tackles", "Tackles", "count", 2, False),
    ("tackles_won", "Tackles won", "count", 2, True),
    ("dribbling", "Dribbles", "count", 2, False),
    ("dribbling_successful", "Successful dribbles", "count", 2, True),
    ("fouls_committed", "Fouls committed", "count", 2, False),
    ("yellow_cards", "Yellow cards", "count", 0, False),
    ("red_cards", "Red cards", "count", 0, False),
    ("saves", "Saves", "count", 2, True),
    ("clean_sheet", "Clean sheets", "count", 0, False),
    ("goals_conceded", "Goals conceded", "count", 0, False),
]
# Plausible values, checked when a squad is parsed: per 90 minutes for distance, raw for the rest.
PLAYER_STAT_RANGES = {"distance_covered": (3.0, 16.0), "top_speed": (10.0, 40.0), "passes_accuracy": (0.0, 100.0)}
PLAYERS_URL = (
    "https://compstats.uefa.com/v1/player-ranking?competitionId=1&seasonYear={season}&phase=TOURNAMENT&order=DESC"
    "&optionalFields=PLAYER,TEAM&teamId={team_id}&limit={limit}&offset={offset}&stats="
    + ",".join(key for key, *_ in PLAYER_STATS)
)
