import json
import stat

from lupine import _credentials


def test_credentials_share_cli_format(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("LUPINE_API_TOKEN", raising=False)

    _credentials.save_token("https://api.lupine.sh/", "lup_stored")

    path = tmp_path / "credentials.json"
    assert json.loads(path.read_text()) == {
        "tokens": {"https://api.lupine.sh": "lup_stored"}
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _credentials.token_for("https://api.lupine.sh") == "lup_stored"

    _credentials.delete_token("https://api.lupine.sh")
    assert _credentials.token_for("https://api.lupine.sh") is None


def test_environment_token_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_STATE_DIR", str(tmp_path))
    _credentials.save_token("https://api.lupine.sh", "lup_stored")
    monkeypatch.setenv("LUPINE_API_TOKEN", "lup_environment")

    assert _credentials.token_for("https://api.lupine.sh") == "lup_environment"
