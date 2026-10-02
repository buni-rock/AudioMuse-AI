"""Instant Playlist limits are editable through Advanced Configurations."""

from pathlib import Path

import pytest

import app_setup
import config
from tasks.setup_manager import setup_manager


LIMITS = app_setup.INSTANT_PLAYLIST_LIMIT_MINIMUMS


def test_limits_are_visible_persistable_advanced_settings():
    setup_js = (Path(__file__).resolve().parents[2] / 'static' / 'setup.js').read_text()
    section = setup_js.split("title: 'Instant Playlist & AI Tool-Calling'", 1)[1].split(']', 1)[0]
    for name in LIMITS:
        assert app_setup.should_show_advanced(name)
        assert name not in config.SETUP_BOOTSTRAP_EXCLUDED_KEYS
        assert setup_manager.is_persistable_value(getattr(config, name))
        assert f"'{name}'" in section


@pytest.mark.parametrize('name,minimum', list(LIMITS.items()))
def test_limit_save_accepts_its_minimum_and_rejects_a_lower_value(name, minimum):
    values = {name: str(minimum)}
    assert app_setup._validate_instant_playlist_limits(values) is None
    assert values[name] == minimum
    assert app_setup._validate_instant_playlist_limits({name: str(minimum - 1)})


@pytest.mark.parametrize('invalid', ['abc', '1.5', True, ''])
def test_limit_save_rejects_non_integer_values(invalid):
    assert app_setup._validate_instant_playlist_limits({
        'COMPOSER_MAX_OUTPUT_TOKENS': invalid,
    })
