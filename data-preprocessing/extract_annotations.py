import json
import os
from itertools import chain
from dotenv import load_dotenv
import pandas as pd


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
    data_file = os.getenv("DATA_FILE")
    out_file = os.getenv("OUT_FILE")

    with open(data_file) as f:
        content = json.load(f)

    annotated = filter(lambda x: x["best_index"], content)
    relative_focus = map(to_relative_focus, annotated)
    flattened = chain(*map(flatten_stack, relative_focus))

    dataframe = pd.DataFrame(flattened)
    dataframe.to_csv(out_file)
