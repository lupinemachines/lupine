from types import SimpleNamespace

import pytest

from lupine import __main__, _credentials


def test_login_command(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        __main__,
        "login",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(email="user@example.com", name="Example User"),
    )

    assert __main__.main(["login"]) == 0
    assert calls == [
        {
            "api_url": "https://api.lupine.sh",
            "console_url": "https://console.lupine.sh",
            "profile": None,
        }
    ]
    assert "Logged in to Lupine Cloud as user@example.com." in capsys.readouterr().out


def test_login_command_names_non_default_profile(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        __main__,
        "login",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(email="user@example.com", name=""),
    )

    assert __main__.main(["login", "--profile", "work"]) == 0
    assert calls[0]["profile"] == "work"
    assert (
        "Logged in to Lupine Cloud as user@example.com (profile work)."
        in capsys.readouterr().out
    )


def test_login_command_rejects_invalid_profile(monkeypatch, capsys):
    monkeypatch.setattr(__main__, "login", lambda **kwargs: pytest.fail("login ran"))

    assert __main__.main(["login", "--profile", "../evil"]) == 1
    assert "invalid profile name" in capsys.readouterr().err


def test_logout_command_uses_profile(monkeypatch, tmp_path):
    monkeypatch.delenv("LUPINE_API_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    _credentials.save_token("https://api.lupine.sh", "lup_default")
    _credentials.save_token("https://api.lupine.sh", "lup_work", profile="work")

    assert __main__.main(["logout", "--profile", "work"]) == 0
    assert _credentials.token_for("https://api.lupine.sh", "work") is None
    assert _credentials.token_for("https://api.lupine.sh") == "lup_default"
