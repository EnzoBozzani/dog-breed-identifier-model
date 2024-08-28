from duckduckgo_search import DDGS
from fastcore.all import L
from fastai.vision.all import Path, download_images, resize_images, verify_images, get_image_files, DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize
import requests
from time import sleep

ddgs = DDGS()


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddgs.images(term, max_results=max_images)).itemgot('image')


# urls = search_images('kuvasz photos', max_images=1)

res = requests.get('https://dog.ceo/api/breeds/list/all')

json = res.json()

data = json['message']

breeds: list[str] = []

for key in data:
    breeds.append(key)

path = Path('./dogs')

for breed in breeds:
    dest = (path/breed)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{breed} photos'))
    sleep(10)
    resize_images(path/breed, max_size=400, dest=path/breed)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

# shows a batch example (in this case, 6 images)
dls.show_batch(max_n=6)

