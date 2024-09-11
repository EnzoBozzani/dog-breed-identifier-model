from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from services.model import train_model, predict_image
from services.images import search_images
import traceback
import os
from uuid import uuid4
import shutil


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

    return StreamingResponse(search_images(terms), media_type="text/plain")


@app.post('/train-model/{id}')
async def train(id: str):
    try:
        return StreamingResponse(
            train_model(id),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=model.pkl"
            }
        )
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=400,
            content={'error': 'Something went wrong while training the model!'}
        )


@app.post('/predict')
async def predict(model: UploadFile = File(...), images: list[UploadFile] = File(...)):
    if not os.path.exists('./predictions'):
        os.mkdir('./predictions')

    id = str(uuid4())

    os.mkdir(f'./predictions/{id}')

    with open(f'./predictions/{id}/model.pkl', 'wb') as buffer:
        shutil.copyfileobj(model.file, buffer)

    try:
        categories = []

        for image in images:
            category = predict_image(id, image)
            categories.append(category)

        shutil.rmtree(f'./predictions/{id}')

        return categories
    except Exception as e:
        print(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={'error': 'Something went wrong while predicting your images!'}
        )
