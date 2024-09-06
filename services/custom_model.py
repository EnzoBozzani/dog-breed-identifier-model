from duckduckgo_search import DDGS
from fastcore.all import L  # type: ignore
from fastai.vision.all import Path, download_images, resize_images, get_image_files, DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize, vision_learner, error_rate, RandomResizedCrop, resnet18, verify_images  # type: ignore # noqa: E501
from time import sleep
from threading import local

ddgs = DDGS()
thread_local = local()

path = Path('./test')


def search_images(terms):
    for term in terms:
        dest = (path/term)
        dest.mkdir(exist_ok=True, parents=True)
        yield f'Searching for {term} photos...'
        download_images(dest, urls=L(ddgs.images(f'{term} photos', max_results=50)).itemgot('image'))  # noqa: E501
        sleep(10)
        resize_images(path/term, max_size=400, dest=path/term)
        failed = verify_images(get_image_files(dest))
        failed.map(Path.unlink)

    yield 'Training the model with the images...'


def train_model():
    data_blocks = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    )

    data_blocks = data_blocks.new(
        item_tfms=RandomResizedCrop(128, min_scale=0.3)
    )

    data_loaders = data_blocks.dataloaders(path)

    learn = vision_learner(data_loaders, resnet18, metrics=error_rate)  # noqa: E501
    learn.fine_tune(3)

    learn.export('./teste.pkl')

    with open('./teste.pkl', 'rb') as file:
        while chunk := file.read(8192):
            yield chunk
