## create a dataset of several thousand examples of the following format:
## current categories: fruit, animal, color, number, shape

# Type: fruit
# List: [dog apple cherry bus cat grape bowl]
# Answer: (3)

import random
import json

ANIMAL = ["dog", "cat", "bird", "fish", "horse"]
FRUIT = ["apple", "banana", "cherry", "orange", "pear"]
COLOR = ["red", "blue", "green", "yellow", "purple"]
NUMBER = ["one", "two", "three", "four", "five"]
SHAPE = ["circle", "square", "triangle", "rectangle", "pentagon"]

def generate_list(length:int):
    """
    generates a random list of length len including all categorites
    output random list and dictionary of counts for each category
    """
    lst = []
    for _ in range(length):
        lst.append(random.choice(ANIMAL + FRUIT + COLOR + NUMBER + SHAPE))
    # list is a list of strings in random order
    counts = {
        "animal": len([item for item in lst if item in ANIMAL]),
        "fruit": len([item for item in lst if item in FRUIT]),
        "color": len([item for item in lst if item in COLOR]),
        "number": len([item for item in lst if item in NUMBER]),
        "shape": len([item for item in lst if item in SHAPE])
    }
    return lst, counts

def generate_dataset(num_examples:int):
    """
    generates a dataset of num_examples examples
    in a format that can be used to benchmark a language model

    count is the answer key for the count of each category in the list
    """
    dataset = []
    for _ in range(num_examples):
        lst, count = generate_list(random.randint(10, 20))
        for key, value in count.items():
            dataset.append({"type": key, "list": lst, "answer": value})
    with open("dataset.json", "w") as f:
        json.dump(dataset, f)
    return dataset

if __name__ == "__main__":
    generate_dataset(1000)
