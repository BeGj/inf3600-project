"""Satellite generative-fill backend.

Loads the SD1.5-inpainting pipeline (plus registry LoRA adapters) once at startup,
then serves catalogue search and inpainting over HTTP.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .inference.pipeline import engine
from .routers import api

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the inpainting pipeline before serving requests.
    engine.load()
    yield


app = FastAPI(title="Satellite Generative Fill", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "satellite-generative-fill"}
