import json
import os
from itertools import chain
from dotenv import load_dotenv
import pandas as pd
import numpy as np


def to_relative_focus(stack):
    best_index = stack["best_index"]
    images = stack["images"]

    best_value = images[best_index]["focus_height"]
    for i in range(len(images)):
        images[i]["focus_height"] = images[i]["focus_height"] - best_value
    return stack


def flatten_stack(stack):
    images = stack["images"]

    def f(image):
        del image["neighbours"]
        image["stack_id"] = stack["stack_id"]
        image["obj_name"] = stack["obj_name"]
        return image

    images = list(map(f, images))
    return images


if __name__ == "__main__":
    load_dotenv()

    train_split = float(os.getenv("TRAIN_SPLIT"))
    val_split = float(os.getenv("VALIDATION_SPLIT"))
    test_split = float(os.getenv("TEST_SPLIT"))

    assert abs(1 - (train_split + val_split + test_split)) < 0.01

    in_folder = os.getenv("IN_FOLDER")
    out_folder = os.getenv("OUT_FOLDER")

    train_list = []
    val_list = []
    test_list = []

    for file in os.listdir(in_folder):
        if file.endswith(".json"):
            with open(os.path.join(in_folder, file)) as f:
                content = json.load(f)

            annotated = filter(lambda x: x["best_index"], content)
            relative_focus = map(to_relative_focus, annotated)
            flattened = list(chain(*map(flatten_stack, relative_focus)))

            # https://stackoverflow.com/a/49556954
            train, validate, test = np.split(
                flattened,
                [
                    int(len(flattened) * train_split),
                    int(len(flattened) * (train_split + val_split)),
                ],
            )

            train_list.extend(train)
            val_list.extend(validate)
            test_list.extend(test)

    np.random.shuffle(train_list)
    np.random.shuffle(val_list)
    np.random.shuffle(test_list)

    dataframe = pd.DataFrame(train_list)
    dataframe.to_csv(os.path.join(out_folder, "train_metadata.csv"))

    dataframe = pd.DataFrame(val_list)
    dataframe.to_csv(os.path.join(out_folder, "validation_metadata.csv"))

    dataframe = pd.DataFrame(test_list)
    dataframe.to_csv(os.path.join(out_folder, "test_metadata.csv"))
