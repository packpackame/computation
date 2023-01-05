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
        current_timestamp = int(datetime.now(timezone.utc).timestamp() * 10 ** 6)

        img_filename = f"{current_timestamp}.jpg"
        with open(img_filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"error": "There was an error uploading the file"}
    finally:
        img_file.file.close()

    _, colorized_images, _ = artist.colorize(img_filename, 3)

    # result_image = artist.visualize(image, prediction, original_size=False, inline=3)

    image_strings = []

    for img_array in colorized_images:
        colorized_image = Image.fromarray(img_array)

        buffered = BytesIO()
        colorized_image.save(buffered, format="JPEG")
        colorized_image_string = base64.b64encode(buffered.getvalue())
        image_strings.append(colorized_image_string)

    return ColorizedImage(
        images=image_strings
    )
