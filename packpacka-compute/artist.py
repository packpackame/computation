from fastapi import Depends

from model.inference import Artist
from settings import Settings, get_settings


def get_artist(settings: Settings = Depends(get_settings)) -> Artist:
    artist = Artist(
        model_checkpoint=settings.model_checkpoint,
        device=settings.cuda_device
    )
    return artist
