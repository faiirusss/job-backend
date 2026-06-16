from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.embeddings import embeddings_service
from app.api import auth, conversations, cv, health, history, jobs, search
from app.config import settings
from app.utils.logger import configure_logging
from app.ws import search as ws_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=settings.log_level, app_env=settings.app_env)
    embeddings_service.load()
    yield


app = FastAPI(title="JHAI Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(cv.router, prefix="/api/v1")
app.include_router(conversations.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(ws_search.router)  # /ws/search (no prefix)
