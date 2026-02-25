"""Tests for ESPN API client."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.espn_client import ESPNClient


@pytest.fixture
def config():
    return Config.load(Path(__file__).parent.parent / "config.yaml")


@pytest.fixture
def client(config):
    return ESPNClient(config)


def test_client_init(client):
    assert client.session is not None
    assert client._last_request_time == 0.0


def test_rate_limiting(client):
    """Verify rate limiter delays between requests."""
    import time
    client.config.espn.rate_limit_seconds = 0.1
    client._last_request_time = time.time()

    start = time.time()
    client._rate_limit()
    elapsed = time.time() - start

    assert elapsed >= 0.05  # Should have waited ~0.1s


@patch("src.data.espn_client.ESPNClient._get")
def test_get_scoreboard(mock_get, client):
    mock_get.return_value = {"events": []}
    result = client.get_scoreboard("20240320")
    mock_get.assert_called_once()
    assert result == {"events": []}


@patch("src.data.espn_client.ESPNClient._get")
def test_get_team(mock_get, client):
    mock_get.return_value = {"team": {"id": "150"}}
    result = client.get_team(150)
    assert result is not None


@patch("src.data.espn_client.ESPNClient._get")
def test_get_rankings(mock_get, client):
    mock_get.return_value = {"rankings": []}
    result = client.get_rankings(2024)
    assert result is not None


def test_get_returns_none_on_404(client):
    """Test that 404 responses return None."""
    with patch.object(client.session, "get") as mock:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock.return_value = mock_resp
        result = client._get("https://example.com/bad")
        assert result is None
