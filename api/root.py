from fastapi import FastAPI

from db.engine import create_db_and_tables

from .routers import documents, query


app = FastAPI()

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}
