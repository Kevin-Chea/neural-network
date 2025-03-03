from PIL import Image
import os
import re
import pickle
import random

image_base_path = "data/by_class/"

NB_PREPROCESSED_IMAGES_PER_NUMBER = 1000

def pre_process_image(path):
    img = Image.open(path).convert('L')
    img_resized = img.resize((28, 28))
    # img_resized.show()
    pixels = [pixel / 255.0 for pixel in img_resized.getdata()]
    return pixels


def inverse_colors(image_data):
    return [1.0 - pixel for pixel in image_data]

def files_of_folder(path: str):
    """
    Returns all the files names in a folder.
    Params:
    - path: folder's path (string)
    """
    return os.listdir(path)

def png_files_of_folder(path):
    pattern = re.compile(r"^.*\d{5}\.png$")
    files = files_of_folder(path)
    return [os.path.join(path, file) for file in files if pattern.match(file)]

def first_images_of_folder(path, limit=None):
    limit = limit if limit is not None else 1000
    png_files = png_files_of_folder(path)
    png_files.sort()
    return png_files[:limit]

def folder_of_number(n):
    """Take a number in parameter and return the name of the folder\n
    The folder name is a number between 30 (0) and 39 (9)"""
    return "3" + str(n)

def first_images_of_n(n, limit=None):
    folder = folder_of_number(n)
    folder_path = image_base_path + folder + "/train_" + folder
    return first_images_of_folder(folder_path, limit)

def pre_process_and_save_images_of_n_to_pickle(n: int, output_file: str, limit: int|None = None):
    """
    Find files concerning number n, pre-process them and save them into a binary\n
    Params:\n
    - n: number
    - output_file: destination, where to save
    - limit (fac.): number of file to load
    """
    all_data = []
    for image_file in first_images_of_n(n, limit):
        pixels = pre_process_image(image_file)
        inverted_pixels = inverse_colors(pixels)
        all_data.append(inverted_pixels)
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)


def pre_process_all_numbers():
    for i in range(0, 10):
        pre_process_and_save_images_of_n_to_pickle(i, "train_" + str(i), NB_PREPROCESSED_IMAGES_PER_NUMBER)

def load_batch_from_pickle(file_path: str, images_per_batch: int, batch_index: int):
    """
    Load batch data.\n
    Params:\n
    - file_path: Path of the binary file where the batch is stored (string)
    - batch_size: Size of a batch (number)
    - batch_index: Index of a batch, first batch is 0 (number)
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    start = batch_index * images_per_batch
    end = start + images_per_batch
    return data[start:end]

def load_batch_for_all_numbers(images_per_batch: int=100):
    data = []
    expected_output = []
    
    label_counts = {i: 0 for i in range(10)}
    batch_to_load_max_index = (NB_PREPROCESSED_IMAGES_PER_NUMBER // images_per_batch) - 1
    index_to_load = random.randint(0, batch_to_load_max_index)
    for i in range(10):
        # 28 * 28 * 100 : image size * nb of images in a batch
        data.extend(load_batch_from_pickle("train_" + str(i), images_per_batch, index_to_load))
        expected_output.extend([i] * images_per_batch)
        label_counts[i] += images_per_batch
    # Shuffle
    combined = list(zip(data, expected_output))
    random.shuffle(combined)
    data, expected_output = zip(*combined)
    return list(data), list(expected_output)


def show_image_from_batch(image_data):
    """
    Show an image from extracted data (batch)
    
    Params:
    - image_data (list): list of 784 values (28x28) normalized between 0 and 1.
    """
    # Convertir les valeurs en niveaux de gris (0-255)
    pixel_values = [int(p * 255) for p in image_data]

    # Créer une image en niveaux de gris (L) avec PIL
    img = Image.new('L', (28, 28))
    img.putdata(pixel_values)

    # Afficher l'image
    img.show()