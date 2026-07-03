import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001


app = FastAPI(
    title="Data Mining Visualization API",
    description="Classification algorithm visualization backend for teaching demos.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def read_root():
    return {"name": "Data Mining Visualization API", "docs": "/docs"}


def main():
    host = os.getenv("API_HOST", DEFAULT_HOST)
    port = int(os.getenv("API_PORT", str(DEFAULT_PORT)))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    uvicorn.run("main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
