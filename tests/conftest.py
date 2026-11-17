"""The first test is itself gated on publication; no opt-out environment switch."""

from occam_rl.publication import assert_published


def pytest_sessionstart(session):
    """Meet the user's push-before-testing instruction even for direct pytest calls."""
    assert_published()
