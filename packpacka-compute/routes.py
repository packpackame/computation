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

    image, prediction = artist.colorize(img_filename, 3)
    results = artist.prepare_results(image, prediction)
    # result_image = artist.visualize(image, prediction, original_size=False, inline=3)

    for name, image in results.items():
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        results[f'{name}_str'] = img_str

    return ColorizedImage(
        image_original=results['image_original_str'],
        image_colorized_v1=results['image_colorized_v1_str'],
        image_colorized_v2=results['image_colorized_v2_str'],
        image_colorized_v3=results['image_colorized_v3_str']
    )
