import os
from fastai.vision.all import load_learner, PILImage, resize_images, Path  # type: ignore


def main() -> None:
    dest = input("Image: ")

    if not os.path.exists(dest):
        raise Exception('Image not found!')

    path = Path('./')

    resize_images(path, max_size=400, dest=path)

    learn_inf = load_learner('./model/model.pkl')

    most_likely_breed,  most_likely_index,  probs = learn_inf.predict(PILImage.create(dest))
    print(f"This is probably a: {most_likely_breed}.")

    available_breeds = learn_inf.dls.vocab

    probabilities: list[float] = []

    count = 0
    for prob in probs:
        if count != most_likely_index:
            probabilities.append(prob)
        count += 1

    ten_most_likely: list[tuple[str, float]] = [(available_breeds[most_likely_index], probs[most_likely_index])]

    for i in range(9):
        prob = max(probabilities)
        index = probabilities.index(prob)
        probabilities.pop(index)
        breed = available_breeds[index]

        ten_most_likely.append((breed, prob))

    for breed in ten_most_likely:
        print(f'Probability of being a {breed[0]}: {breed[1] * 100:.2f} %')


if __name__ == "__main__":
    main()
