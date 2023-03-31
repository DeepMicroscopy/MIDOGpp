from SlideRunner.dataAccess.database import Database
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from slide.data_loader import SlideContainer
from random import *

def create_indices(files, patches):
    indices = []
    patches_per_slide = patches//len(files)
    for i, file in enumerate(files):
        indices += patches_per_slide * [i]
    return indices

def sample_function(y, classes, size, level_dimensions, level):
    width, height = level_dimensions[level]
    if len(y[0]) == 0 or choice([True, False]):
        xmin, ymin = randint(0, width - size[0]), randint(0, height - size[1])
    else:
        class_id = np.random.choice(classes, 1)[0]
        ids = np.array(y[1]) == class_id
        xmin, ymin, _, _ = np.array(y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
        xmin -= randint(0,size[0])
        ymin -= randint(0,size[1])
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmin, ymin = min(xmin, width - size[0]), min(ymin, height - size[1])
    return xmin, ymin


def load_images(slide_folder, annotation_file, level, patch_size, categories, tumortypes):
    train_files = []
    valid_files = []
    test_files = []
    anno_dict = {1: 'mitotic figure', 2: 'hard negative'}

    database = Database()
    database.open(annotation_file)
    slides = pd.read_csv('datasets_xvalidation.csv', delimiter=";")

    getslides = """SELECT uid, filename, width, height FROM Slides"""
    for currslide, file_name, width, height in tqdm(database.execute(getslides).fetchall()):
        bboxes, labels = [], []
        database.loadIntoMemory(currslide)
        row = slides[slides["Slide"] == currslide]
        tumortype = row['Tumor'].values
        if tumortypes.__contains__(tumortype):
            image_file = Path(glob.glob("{}/**/{}".format(str(slide_folder), file_name.replace("png", "tiff")), recursive=True)[0])
            for id, annotation in database.annotations.items():
                if len(annotation.labels) != 0 and annotation.deleted != 1:
                    label = annotation.agreedClass
                    if categories.__contains__(label):
                        labels.append(anno_dict[label])
                        bboxes.append(annotation.coordinates.reshape(-1))
            if row["Dataset"].values[0] == "train":
                train_files.append( SlideContainer(image_file, y=[bboxes, labels], tumortype=tumortype[0], level=level,width=patch_size, height=patch_size, sample_func=sample_function))
            elif row["Dataset"].values[0] == "valid":
                valid_files.append(SlideContainer(image_file, y=[bboxes, labels], tumortype=tumortype[0], level=level,width=patch_size, height=patch_size, sample_func=sample_function))
            elif row["Dataset"].values[0] == "test":
                test_files.append(SlideContainer(image_file, y=[bboxes, labels], tumortype=tumortype[0], level=level,width=patch_size, height=patch_size, sample_func=sample_function))

    return train_files, valid_files, test_files