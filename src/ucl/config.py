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

GBR_PARAMS = {
    "n_estimators": 250, "learning_rate": 0.03, "max_depth": 2,
    "min_samples_leaf": 8, "subsample": 0.8, "random_state": 42,
}
LOGIT_PARAMS = {"C": 0.3, "max_iter": 2000}
TOP_DRIVERS = 6
ROBUST_SHARE = 0.8  # same sign in >= 80% of LOSO fits = 12 of 15 (spec §6)

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
LLM_MODEL = "qwen/qwen3.5-35b-a3b"
LLM_PARAMS = {"temperature": 0.3, "max_tokens": 8000, "reasoning_effort": "none"}
LLM_TIMEOUT_S = 300
LLM_CONTEXT_LENGTH = 16384
LMS_BIN = Path.home() / ".lmstudio" / "bin" / "lms"
