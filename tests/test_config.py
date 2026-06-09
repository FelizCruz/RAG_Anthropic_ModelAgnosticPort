from rag_learn.config import get_google_api_key, load_model_configs


def test_google_is_primary_then_openrouter_fallbacks(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    configs = load_model_configs()

    assert [config.model for config in configs] == [
        "gemma-4-26b-a4b-it",
        "openrouter/free",
        "moonshotai/kimi-k2.6:free",
    ]


def test_google_key_can_load_when_working_directory_is_lessons(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir("lessons")

    assert get_google_api_key()
