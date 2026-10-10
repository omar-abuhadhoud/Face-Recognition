from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from config import ENROLL_ON_STARTUP
from recognition import enroll_people_from_folder


@asynccontextmanager
async def lifespan(app):
    if ENROLL_ON_STARTUP:
        enroll_people_from_folder()   # sync PEOPLE_DIR into the database before serving requests
    yield


app = FastAPI(title="Face Recognition", lifespan=lifespan)
app.include_router(router)
