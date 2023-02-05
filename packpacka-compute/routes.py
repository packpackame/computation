import base64
from datetime import datetime, timezone
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile
from io import BytesIO
from PIL import Image
import torch

from model.inference import Artist
from models import ColorizedImage
from artist import Artist, get_artist
from settings import Settings, get_settings

route = APIRouter()


@route.get("/")
def root():
    return {"version": "0.0.1"}


def __img_array_to_b64string(image: np.ndarray) -> str:
    image = Image.fromarray(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


@route.post("/colorize/", response_model=ColorizedImage)
def colorize(
        img_file: UploadFile = File(...),
        # artist: Artist = Depends(get_artist)
):
    artist = get_artist(get_settings())

    try:
        contents = img_file.file.read()
        current_timestamp = int(datetime.now(timezone.utc).timestamp() * 10 ** 6)

        img_filename = f"{current_timestamp}.jpg"
        with open(img_filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"error": "There was an error uploading the file"}
    finally:
        img_file.file.close()

    original_image, colorized_images = artist.colorize(img_filename, 3)
    original_image_str = __img_array_to_b64string(original_image)

    image_strings = []

    for img_array in colorized_images:
        image_strings.append(__img_array_to_b64string(img_array))

    return ColorizedImage(
        original_image=original_image_str,
        images=image_strings
    )

