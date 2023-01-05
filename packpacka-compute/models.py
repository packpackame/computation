from pydantic import BaseModel


class ColorizedImage(BaseModel):
    image_original: str
    image_colorized_v1: str
    image_colorized_v2: str
    image_colorized_v3: str
