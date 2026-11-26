import json
import stat

import pytest

import lupine

API = "https://api.lupine.sh"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """An isolated HOME with no token override."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LUPINE_API_TOKEN", raising=False)
    return home


def test_credentials_share_cli_format(home):
    lupine._credentials.save_token("https://api.lupine.sh/", "lup_stored")

    path = home / ".lupine" / "credentials.json"
    assert json.loads(path.read_text()) == {
        "tokens": {"https://api.lupine.sh": "lup_stored"}
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert lupine._credentials.token_for("https://api.lupine.sh") == "lup_stored"

    lupine._credentials.delete_token("https://api.lupine.sh")
    assert lupine._credentials.token_for("https://api.lupine.sh") is None


def test_environment_token_takes_precedence(home, monkeypatch):
    lupine._credentials.save_token("https://api.lupine.sh", "lup_stored")
    monkeypatch.setenv("LUPINE_API_TOKEN", "lup_environment")

    assert lupine._credentials.token_for("https://api.lupine.sh") == "lup_environment"


def test_state_dir_default_profile(home):
    assert lupine._credentials.state_dir() == home / ".lupine"
    assert lupine._credentials.state_dir("default") == home / ".lupine"


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "work",
        "0",
        "team-2",
        "a" * 63,
    ],
)
def test_valid_profile_names(home, name):
    assert lupine._credentials.state_dir(name) == home / ".lupine" / "profiles" / name


@pytest.mark.parametrize(
    "name",
    [
        "..",
        ".",
        "a/b",
        "../work",
        "/abs",
        "Work",
        "-work",
        "work_1",
        "work.1",
        "a" * 64,
        "work\n",
        " work",
    ],
)
def test_invalid_profile_names(home, name):
    with pytest.raises(lupine.LupineError, match="invalid profile name"):
        lupine._credentials.state_dir(name)


def test_profiles_keep_separate_credentials(home):
    lupine._credentials.save_token(API, "lup_default")
    lupine._credentials.save_token(API, "lup_work", profile="work")

    default_file = home / ".lupine" / "credentials.json"
    work_file = home / ".lupine" / "profiles" / "work" / "credentials.json"
    assert json.loads(default_file.read_text())["tokens"] == {API: "lup_default"}
    assert json.loads(work_file.read_text())["tokens"] == {API: "lup_work"}
    assert stat.S_IMODE(work_file.parent.stat().st_mode) == 0o700

    assert lupine._credentials.token_for(API) == "lup_default"
    assert lupine._credentials.token_for(API, "work") == "lup_work"
    assert lupine._credentials.token_for(API, "default") == "lup_default"

    lupine._credentials.delete_token(API, "work")
    assert lupine._credentials.token_for(API, "work") is None
    assert json.loads(default_file.read_text())["tokens"] == {API: "lup_default"}


@pytest.mark.parametrize("name", ["../evil", "Work", ""])
def test_token_for_rejects_invalid_profile_with_env_token(home, monkeypatch, name):
    monkeypatch.setenv("LUPINE_API_TOKEN", "lup_explicit")
    with pytest.raises(lupine.LupineError, match="invalid profile name"):
        lupine._credentials.token_for(API, name)
