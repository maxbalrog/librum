import random
import os, os.path
from os import listdir
import sys
import glob

from pathlib import Path
from shutil import copyfile

import numpy as np


RANDOM_SEED = 48534

def split_dataset(path, path_splitted, test_percentage, validation_percentage):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    #creating a dictionary of dataset splitted into train/test
    dataset = {'train': {}, 'validation': {}, 'test': {}}
    only_png = glob.glob(path + '*/*.png')
    classes = listdir(path)

    print('Ready to start splitting.')
    for cls in classes:
        if cls not in dataset['train'].keys():
            dataset['train'][cls] = []
            dataset['validation'][cls] = []
            dataset['test'][cls] = []

        path_local = os.path.join(path, cls)
        png_local = glob.glob(os.path.join(path_local, '*.png'))
        for png in png_local:
            prob = np.random.rand() * 100.

            if prob < validation_percentage:
                dataset['validation'][cls].append(png)
            elif (prob > validation_percentage) and (prob < validation_percentage + test_percentage):
                dataset['test'][cls].append(png)
            else:
                dataset['train'][cls].append(png)

    #creating needed directories
    print('Creating directories ...')
    Path(path_splitted).mkdir(parents=True, exist_ok=True)
    for group in ['train', 'validation', 'test']:
        path_group = os.path.join(path_splitted, group)
        Path(path_group).mkdir(parents=True, exist_ok=True)
        for cls in classes:
            Path(os.path.join(path_group, cls)).mkdir(parents=True, exist_ok=True)

    #copying files
    print('Copying files ...')
    for group in ['train', 'validation', 'test']:
        print(group)
        for cls in classes:
            print(cls)
            for png in dataset[group][cls]:
                dest = os.path.join(path_splitted, group, cls, png.split('/')[-1])
                copyfile(png, dest)
        print('====================')

    print("Dataset split is complete. Check {} folder ...".format(path_splitted))
