from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from services.custom_model import search_images, train_model


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search-images")
async def search(request: Request):
    terms = await request.json()

    return StreamingResponse(search_images(terms), media_type="text/plain")  # noqa: E501


@app.post('/train-model/{id}')
async def train(id: str):

    return StreamingResponse(
        train_model(id),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=model.pkl"
        }
    )
