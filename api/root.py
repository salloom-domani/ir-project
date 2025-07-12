from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.engine import create_db_and_tables

from .routers import documents, query


app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}
