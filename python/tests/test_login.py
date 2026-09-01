from lupine import _credentials, _login


def test_login_runs_browser_flow_and_stores_token(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("LUPINE_API_TOKEN", raising=False)
    requests = []

    def request(method, url, body=None):
        requests.append((method, url, body))
        if method == "POST":
            return 201, {}
        return 200, {
            "status": "complete",
            "token": "lup_browser",
            "user": {"email": "user@example.com", "name": "Example User"},
        }

    opened = []
    output = []
    monkeypatch.setattr(_login, "_request_json", request)

    result = _login.login(
        open_browser=lambda url: opened.append(url) or True,
        output=output.append,
    )

    assert result.email == "user@example.com"
    assert _credentials.token_for(_login.DEFAULT_API_URL) == "lup_browser"
    assert requests[0][0:2] == (
        "POST",
        "https://api.lupine.sh/v1/cli-login",
    )
    assert "/login/" in opened[0]
    assert "callback_port=" in opened[0]
    assert output[0].startswith("Open this URL")
