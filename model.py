# from fastai.vision.widgets import ImageClassifierCleaner  # type: ignore
from fastai.vision.all import Path, get_image_files, DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize, vision_learner, error_rate, resnet18, PILImage, RandomResizedCrop, ClassificationInterpretation  # type: ignore  # noqa: E501
import os


def list_breeds():
    dirs = os.listdir('./dogs')
    dirs.sort()
    return dirs


def main() -> None:
    if not os.path.exists('./dogs'):
        print("Run 'python3 load_images.py' first!")
        return

    path = Path('./dogs')

    dest = input('Image: ')
    # resize_image(path2/dest, max_size=400, dest=path2/dest)

    data_blocks = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    )

    data_blocks = data_blocks.new(
        item_tfms=RandomResizedCrop(128, min_scale=0.3)
        # batch_tfms=aug_transforms()
    )

    data_loaders = data_blocks.dataloaders(path)

    learn = vision_learner(data_loaders, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    interp = ClassificationInterpretation.from_learner(learn)
    print(interp.plot_confusion_matrix())

    print(interp.plot_top_losses(5, nrows=1))

    # cleaner = ImageClassifierCleaner(learn)
    # print(cleaner)

    available_breeds = list_breeds()

    most_likely_breed,  _,  probs = learn.predict(PILImage.create(dest))
    print(f"This is probably a: {most_likely_breed}.")

    probabilities: list[float] = []

    for prob in probs:
        probabilities.append(prob)

    ten_most_likely: list[tuple[str, float]] = []

    for i in range(10):
        prob = max(probabilities)
        index = probabilities.index(prob)
        probabilities.pop(index)
        breed = available_breeds[index]

        ten_most_likely.append((breed, prob))

    for breed in ten_most_likely:
        print(f'Probability of being a {breed[0]}: {breed[1] * 100:.2f} %')


if __name__ == '__main__':
    main()
