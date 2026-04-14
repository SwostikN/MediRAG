from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

# app = FastAPI()

def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow your React app's URL
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )