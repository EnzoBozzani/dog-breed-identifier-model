import os
from fastai.vision.all import load_learner, PILImage  # type: ignore


def main() -> None:
    filename = input("Model (.pkl): ")

    file = filename if filename.endswith('.pkl') else f'{filename}.pkl'

    if not os.path.exists(file):
        raise Exception('Model not found!')

    dest = input("Image: ")

    learn_inf = load_learner(file)

    most_likely_breed,  _,  probs = learn_inf.predict(PILImage.create(dest))
    print(f"This is probably a: {most_likely_breed}.")

    available_breeds = learn_inf.dls.vocab

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


if __name__ == "__main__":
    main()
