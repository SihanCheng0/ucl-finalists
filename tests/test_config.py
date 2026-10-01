from ucl import config


def test_sixteen_features_have_metadata_and_sources():
    assert len(config.FEATURES) == 16
    assert set(config.FEATURE_META) == set(config.FEATURES)
    assert set(config.FEATURE_SOURCES) == set(config.FEATURES)
    assert set(config.FEATURE_GROUP) == set(config.FEATURES)


def test_season_helpers():
    assert config.season_label(2026) == "2025-26"
    assert config.season_label(2012) == "2011-12"
    assert (config.field_size(2024), config.field_size(2025)) == (32, 36)
    assert (config.phase_matches(2024), config.phase_matches(2025)) == (6, 8)
    assert (config.ko_size(2024), config.ko_size(2025)) == (16, 24)


def test_expected_finals_cover_the_target_seasons():
    assert sorted(config.EXPECTED_FINALS) == config.TARGET_SEASONS


def test_every_round_name_has_a_depth():
    assert config.ROUND_DEPTH["League Phase"] == config.ROUND_DEPTH["Group stage"] == 0
    assert config.ROUND_DEPTH["Knockout Phase Play-Offs"] == config.ROUND_DEPTH["Knock-out Play-off"] == 1
    assert config.ROUND_DEPTH["Final"] == 5
