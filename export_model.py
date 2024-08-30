# from fastai.vision.widgets import ImageClassifierCleaner  # type: ignore
from fastai.vision.all import Path, get_image_files, DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize, vision_learner, error_rate, resnet18, RandomResizedCrop, ClassificationInterpretation  # type: ignore  # noqa: E501
import os


def main() -> None:
    if not os.path.exists('./dogs'):
        raise Exception("Run 'python3 load_images.py' first!")

    path = Path('./dogs')

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
    interp.plot_confusion_matrix()

    interp.plot_top_losses(5, nrows=1)

    # cleaner = ImageClassifierCleaner(learn)
    # print(cleaner)

    learn.export('model.pkl')


if __name__ == '__main__':
    main()
