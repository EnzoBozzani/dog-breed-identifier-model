from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def operation_stream(strings: list):
    for string in strings:
        yield f"Buscando imagens para: {string}...\n"
        time.sleep(2)

    yield "Iniciando outro processo...\n"
    time.sleep(2)


@app.post("/process")
async def process_strings(request: Request):
    strings = await request.json()

    return StreamingResponse(operation_stream(strings), media_type="text/plain")  # noqa: E501
