from pydantic import BaseSettings


def get_settings():
    return Settings()


class Settings(BaseSettings):
    app_name: str = "Packpacka Compute API"
    model_checkpoint: str
    cuda_device: str
    api_version: int = 1
    port: int = 5000
    host: str = "0.0.0.0"
