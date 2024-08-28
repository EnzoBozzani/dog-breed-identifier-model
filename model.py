from fastai.vision.all import Path, get_image_files, DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize, vision_learner, error_rate, resnet18, PILImage, Image, resize_image  # noqa: E501

path = Path('./dogs')

path2 = Path('./')

dest = 'dog.jpg'
resize_image(path2/dest, max_size=400, dest=path2/dest)


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

breed,  _,  probs = learn.predict(PILImage.create(dest))
print(f"This is a: {breed}.")
print(f"Probability this dog belongs to this breed: {probs[1]:.4f}")
