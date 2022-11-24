from fastapi import FastAPI
from settings import get_settings
import uvicorn

from routes import route

app = FastAPI()
app_settings = get_settings()
app.include_router(route, prefix=f"/api/v{app_settings.api_version}")


if __name__ == "__main__":
    uvicorn.run("main:app", host=app_settings.host, port=app_settings.port)
