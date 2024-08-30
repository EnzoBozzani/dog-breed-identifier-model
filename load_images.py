from duckduckgo_search import DDGS
from fastcore.all import L  # type: ignore
from fastai.vision.all import Path, download_images, resize_images, verify_images, get_image_files  # type: ignore # noqa: E501
from time import sleep
import json
from concurrent.futures import ThreadPoolExecutor
from threading import local

ddgs = DDGS()
thread_local = local()

path = Path('./dogs')

if not path.exists():
    path.mkdir()


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddgs.images(term, max_results=max_images)).itemgot('image')


def search_breed(breed):
    dest = (path/breed)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{breed} dog photos', 50))
    sleep(10)
    resize_images(path/breed, max_size=400, dest=path/breed)
    failed = verify_images(get_image_files(dest))
    failed.map(Path.unlink)
    print((f"failed images for {breed}: {len(failed)}"))


def search_all(breeds):
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(search_breed, breeds)


def main() -> None:
    breeds: list[str] = []

    with open('./breeds.json') as file:
        data = json.load(file)['message']

        for key in data:
            breedList: list[str] = data[key]

            if len(breedList) != 0:
                for breed in breedList:
                    breeds.append(f"{key} {breed}")
            else:
                breeds.append(key)

    search_all(breeds)


if __name__ == '__main__':
    main()
