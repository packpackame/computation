import base64
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, File, UploadFile
from io import BytesIO
import torch

from model.inference import Artist
from models import ColorizedImage
from artist import Artist, get_artist
from settings import Settings, get_settings

route = APIRouter()


@route.get("/")
def root():
    return {"version": "0.0.1"}


@route.post("/colorize/", response_model=ColorizedImage)
def colorize(
        img_file: UploadFile = File(...),
        # artist: Artist = Depends(get_artist)
):
    artist = get_artist(get_settings())

    try:
        contents = img_file.file.read()
        img_filename = f"{int(datetime.now(timezone.utc).timestamp() * 10 ** 6)}.jpg"
        with open(img_filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"error": "There was an error uploading the file"}
    finally:
        img_file.file.close()

    image, prediction, prediction_step = artist.colorize(img_filename, 5)
    result_image = artist.visualize(image, prediction, original_size=False, inline=3)

    buffered = BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return ColorizedImage(
        image=img_str
    )
