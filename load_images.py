from duckduckgo_search import DDGS
from fastcore.all import L
from fastai.vision.all import Path, download_images, resize_images, verify_images, get_image_files  # noqa: E501
from time import sleep
import json

ddgs = DDGS()


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddgs.images(term, max_results=max_images)).itemgot('image')


breeds: list[str] = []


with open('./test.json') as file:
    data = json.load(file)['message']

    for key in data:
        breedList: list[str] = data[key]

        if len(breedList) != 0:
            for breed in breedList:
                breeds.append(f"{key} {breed}")
        else:
            breeds.append(key)

path = Path('./dogs')

for breed in breeds:
    dest = (path/breed)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{breed} photos'))
    sleep(10)
    resize_images(path/breed, max_size=400, dest=path/breed)
    failed = verify_images(get_image_files(dest))
    failed.map(Path.unlink)
    print((f"{len(failed)} failed"))
