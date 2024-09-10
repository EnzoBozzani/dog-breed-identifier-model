from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from services.model import train_model, predict_image
from services.images import search_images


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


@app.post('/predict')
async def predict(model: UploadFile = File(...), image: UploadFile = File(...)):  # noqa: E501
    categories = predict_image(model, image)

    return categories
