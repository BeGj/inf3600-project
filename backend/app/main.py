"""Satellite generative-fill backend.

Loads the SD1.5-inpainting pipeline (plus registry LoRA adapters) once at startup,
then serves catalogue search and inpainting over HTTP.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .inference.pipeline import engine
from .routers import api

# Comma-separated origins; defaults to the Vite dev server.
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the inpainting pipeline before serving requests.
    engine.load()
    yield


app = FastAPI(title="Satellite Generative Fill", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "satellite-generative-fill"}
