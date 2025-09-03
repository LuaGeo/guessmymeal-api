from fastapi import FastAPI
from src.api_openai import router

app = FastAPI(title="Food Detection API", version="1.0.0")
app.include_router(router, prefix="/api")