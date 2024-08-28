## Download images:

```
python3 load_images.py
```

## Add a image (example: dog.jpg) to the root folder

## Change the path in model.py (the path to the image you previously added):

```
breed,_,probs = learn.predict(PILImage.create('dog.jpg'))
```

## Run the model:

```
python3 model.py
```
