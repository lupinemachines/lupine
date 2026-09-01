from types import SimpleNamespace

from lupine import __main__


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
        }
    ]
    assert "Logged in to Lupine Cloud as user@example.com." in capsys.readouterr().out
