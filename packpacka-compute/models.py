from pydantic import BaseModel


class ColorizedImage(BaseModel):
    image: str
