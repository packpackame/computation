from pydantic import BaseModel
from typing import List


class ColorizedImage(BaseModel):
    images: List[str]
