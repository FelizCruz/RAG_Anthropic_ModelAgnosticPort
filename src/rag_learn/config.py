import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class ModelConfig:
    name: str
    api_key: str
    base_url: str
    model: str


def load_local_env() -> None:
    env_path = find_dotenv(".env.local", usecwd=True)
    if env_path:
        load_dotenv(env_path)


def get_google_api_key() -> str:
    load_local_env()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in the project's .env.local file")
    return api_key


def load_model_configs() -> list[ModelConfig]:
    load_local_env()

    configs = []
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if google_api_key:
        configs.append(
            ModelConfig(
                name="Google AI Studio Gemma 4 26B A4B",
                api_key=google_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="gemma-4-26b-a4b-it",
            )
        )

    if openrouter_api_key:
        configs.extend(
            [
                ModelConfig(
                    name="OpenRouter free models router",
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="openrouter/free",
                ),
                ModelConfig(
                    name="OpenRouter Kimi K2.6 free",
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="moonshotai/kimi-k2.6:free",
                ),
            ]
        )

    if not configs:
        raise ValueError("Set GOOGLE_API_KEY or OPENROUTER_API_KEY in .env.local")

    return configs
