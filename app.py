from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from services.model import train_model, predict_image, predict_dog_breed
from services.images import search_images
from pydantic import BaseModel
import traceback
import os
from uuid import uuid4
import shutil

MB = 1000000


class SearchImagesBody(BaseModel):
    terms: list[str]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search-images")
async def search(terms: list[str]):
    if len(terms) > 50 or len(terms) < 2:
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid fields'}
        )

    for term in terms:
        if len(term) > 20 or len(term) < 3:
            return JSONResponse(
                status_code=400,
                content={'error': 'Invalid fields'}
            )

    try:
        return StreamingResponse(search_images(terms), media_type="text/plain")
    except Exception as e:
        print(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={'error': 'Something went wrong while searching images!'}
        )


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
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={'error': 'Something went wrong while training the model!'}
        )


@app.post('/predict')
async def predict(model: UploadFile = File(...), images: list[UploadFile] = File(...)):
    if model.size is None or model.size > 200 * MB or model.content_type != 'application/octet-stream':
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid fields'}
        )

    if len(images) > 10 or len(images) < 1:
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid fields'}
        )

    for image in images:
        if image.size is None or image.size > 2 * MB or image.content_type != 'image/jpeg':
            return JSONResponse(
                status_code=400,
                content={'error': 'Invalid fields'}
            )

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
            status_code=500,
            content={'error': 'Something went wrong while predicting your images!'}
        )


@app.post('/dog-breed-identifier')
async def predict_dog_image(images: list[UploadFile] = File(...)):
    if len(images) > 10 or len(images) < 1:
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid fields'}
        )

    for image in images:
        if image.size is None or image.size > 2 * MB or image.content_type != 'image/jpeg':
            return JSONResponse(
                status_code=400,
                content={'error': 'Invalid fields'}
            )

    if not os.path.exists('./dog-images'):
        os.mkdir('./dog-images')

    id = uuid4()

    os.mkdir(f'./dog-images/{id}')

    try:
        breeds = []

        for image in images:
            breed = predict_dog_breed(f'./dog-images/{id}', image)
            breeds.append(breed)

        shutil.rmtree(f'./dog-images/{id}')

        return breeds
    except Exception as e:
        print(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={'error': 'Something went wrong while predicting your images!'}
        )
