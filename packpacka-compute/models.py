from pydantic import BaseModel
from typing import List


class ColorizedImage(BaseModel):
    original_image: str
    images: List[str]