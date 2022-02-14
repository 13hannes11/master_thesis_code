import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import math
import json
import random

class StackEntry:
    def __init__(self):
        self.images = []
        self.objects = []
    def add_image(self, image):
        self.images.append(image)
    def add_object(self, object):
        self.objects.append(object)
    def sort(self):
        self.images.sort(key=lambda x: x.focus_value)

def get_neighbours(img, x, y, dimensions):
    neighbour_candidates = [(-1,-1), (0, -1), (1, -1), (-1, 0), (1,0), (-1,1), (0,1), (1,1)]

    width, height = img.size

    neighbours = []
    for x_offset, y_offset in neighbour_candidates:
        neighbour_x = x + x_offset * dimensions
        neighbour_y = y + y_offset * dimensions

        if neighbour_x >= 0 and neighbour_x + dimensions <= width and      neighbour_y >= 0 and neighbour_y + dimensions <= height:
            box = [neighbour_x, neighbour_y, neighbour_x + dimensions, neighbour_y + dimensions]
            neighbours.append((neighbour_x, neighbour_y, img.crop(box)))
        else:
            neighbours.append(None)
    return neighbours

def extract_object_tiles(obj, stack_images):
    x_start = int(obj.x_min / size) * size
    x_end = int(math.ceil(obj.x_max / size)) * size
    y_start = int(obj.y_min / size) * size
    y_end = int(math.ceil(obj.y_max / size)) * size

    tiles = []

    focus_stack_images = list(map(lambda x: (x, Image.open(x.file_path)), stack_images))

    # Get tiles of the image that contain bounding box of object
    for y in range(y_start, y_end, size):
        for x in range(x_start, x_end, size):
            stack = []
            for row, img in focus_stack_images:
                box = [x, y, x + size, y + size]
                crop = img.crop(box)
            
                neighbours = get_neighbours(img, x, y, size)
                stack.append((row, box[:2], crop, neighbours))
            tiles.append(stack)
    return tiles


def save_tile(original_file_path, out_dir, x : int, y : int, img, overwrite = False):
    path, file_name = os.path.split(original_file_path)
    name, ext = os.path.splitext(file_name)

    out_path = os.path.join(out_dir, path)
    save_to = os.path.join(out_path, f'{name}_{x}_{y}{ext}')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if overwrite or not os.path.exists(save_to):
        img.save(save_to)
    return save_to

def save_obj_tiles(obj, out_folder, stack_images):
    extracted = extract_object_tiles(obj, stack_images)
    z_stacks = []
    for z_stack in extracted:
        z_stack_images = []
        for row, box, img, neigbours in z_stack:

            neighbours = []

            image_path = save_tile(row.file_path, out_folder, box[0], box[1], img)
            for neighbour in neigbours:
                n_path = None
                if neighbour:
                    x, y, n_img = neighbour
                    n_path = save_tile(row.file_path, out_folder, x, y, n_img)
                neighbours.append(n_path)

            z_stack_images.append({
                "focus_value": row.focus_value,
                "image_path": image_path,
                "neighbours": neighbours,
                "original_filename": row.file_name,
                "scan_uuid": row.uuid,
                "study_id": row.study_id,
            })
        z_stacks.append({ 
            "best_index": None,
            "images" : z_stack_images,
            "obj_name": obj.name,
            "stack_id": obj.stack_id,
        })

    return z_stacks

def save_stack(stack, out_folder):
    z_stacks = []
    for obj in stack.objects:
        z_stacks.extend(save_obj_tiles(obj, out_folder, stack.images))
    return z_stacks


if __name__ == "__main__":
    load_dotenv()
    print("Geting environment variables...")
    size = int(os.getenv('IMG_SIZE'))

    print("Loading data from csv files...")
    objects = pd.read_csv("test_objects.csv", index_col=0)
    stacks = pd.read_csv("test_stacks.csv", index_col=0)


    stacks_dict = defaultdict(lambda: StackEntry())
    
    print("Building internal datastructure...")
    # adding images to dict
    for (index, row) in stacks.iterrows():
        stacks_dict[row.stack_id].add_image(row)

    for values in stacks_dict.values():
        values.sort()

    # adding objects
    for (index, row) in objects.iterrows():
        stacks_dict[row.stack_id].add_object(row)

    out_folder = "out"
    z_stacks = []

    print("Generating image tiles and writing them to file...")
    for stack in stacks_dict.values():
        z_stacks.extend(save_stack(stack,""))

    # randomize z_stacks
    print("Shuffling data...")
    random.shuffle(z_stacks)

    print("Writing meta-data for annotation to file...")
    with open(os.path.join(out_folder, "data.json"), 'w') as file:
        file.write(json.dumps(z_stacks))
 