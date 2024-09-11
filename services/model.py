from fastapi import UploadFile, File
from fastai.vision.all import Path, resize_images, get_image_files, DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize, vision_learner, error_rate, RandomResizedCrop, resnet18, load_learner, PILImage  # type: ignore # noqa: E501
from uuid import uuid4
import os
import shutil
from torch import Tensor


def train_model(pathId: str):
    path = Path(f'./images/{pathId}')

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

    learn = vision_learner(data_loaders, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    shutil.rmtree(f'./images/{pathId}')

    if not os.path.exists('./models'):
        os.mkdir('./models')

    learn.export(f'./models/{pathId}.pkl')

    with open(f'./models/{pathId}.pkl', 'rb') as file:
        yield from file

    os.remove(f'./models/{pathId}.pkl')


def predict_image(model: UploadFile = File(...), image: UploadFile = File(...)):  # noqa: E501
    if not os.path.exists('./predictions'):
        os.mkdir('./predictions')

    id = str(uuid4())

    os.mkdir(f'./predictions/{id}')

    path = Path(f'./predictions/{id}')

    with open(f'./predictions/{id}/model.pkl', 'wb') as buffer:
        shutil.copyfileobj(model.file, buffer)

    with open(f'./predictions/{id}/image.jpg', 'wb') as buffer:
        shutil.copyfileobj(image.file, buffer)

    resize_images(path, max_size=400, dest=path)

    learn_inf = load_learner(f'./predictions/{id}/model.pkl')

    _,  most_likely_index,  probs = learn_inf.predict(PILImage.create(f'./predictions/{id}/image.jpg'))  # noqa: E501

    available_categories = list(learn_inf.dls.vocab)

    probabilities: list[Tensor] = []

    count = 0
    for prob in probs:
        if count != most_likely_index:
            probabilities.append(prob)
        count += 1

    categories: list[tuple[str, Tensor]] = [(available_categories[most_likely_index], probs[most_likely_index])]  # noqa: E501

    available_categories.pop(most_likely_index)

    for i in range(len(available_categories)):
        category: str = available_categories[i]

        categories.append((category, probabilities[i]))

    categories.sort(reverse=True, key=(lambda element: element[1].item()))

    shutil.rmtree(f'./predictions/{id}')

    return [[cat, prob.item()] for cat, prob in categories]
