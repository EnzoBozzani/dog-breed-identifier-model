from duckduckgo_search import DDGS
from fastcore.all import L  # type: ignore
from fastai.vision.all import Path, download_images, resize_images, get_image_files, verify_images  # type: ignore
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import os

ddgs = DDGS()


def search_unique(term: str, path):
    dest = (path/term)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=L(ddgs.images(f'{term} photos', max_results=50)).itemgot('image'))
    sleep(.1)
    resize_images(path/term, max_size=400, dest=path/term)
    failed = verify_images(get_image_files(dest))
    failed.map(Path.unlink)


def search_images(terms):
    executor = ThreadPoolExecutor(max_workers=10)

    id = str(uuid4())

    if not os.path.exists('./images'):
        os.mkdir('./images')

    path = Path(f'./images/{id}')

    for term in terms:
        yield f'Searching for {term} photos...'
        sleep(.1)
        executor.submit(search_unique, term, path)

    executor.shutdown(wait=True)

    yield 'Training the model with the images...'

    sleep(.5)

    yield id
