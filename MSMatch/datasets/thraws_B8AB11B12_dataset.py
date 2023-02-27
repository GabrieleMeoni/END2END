import imageio
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import pandas as pd 

from .utils import upsample_ds

class ThrawsB8AB11B12Dataset(torch.utils.data.Dataset):
    """Thraws dataset"""

    def __init__(self, train, root_dir="/data/PyDeepLearning/END2END/MSMatch/DATA/warmup_events_dataset/", transform=None, seed=42, upsample_ratio=[1,7]):
        """
        Args:
            train (bool): If true returns training set, else test
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            seed (int): seed used for train/test split
            upsample_ratio (list): ratio of event, notevent upsampling
        """
        self.seed = seed
        self.upsample_ratio = upsample_ratio
        self.size = [256, 256]
        self.num_channels = 3
        self.num_classes = 2
        self.root_dir = root_dir
        self.transform = transform
        self.test_ratio = 0.1
        self.train = train
        self.N = 4528  # Modified to be inferred from data.
        self._load_data()

    def _normalize_to_0_to_1(self, img):
        """Normalizes the passed image to 0 to 1

        Args:
            img (np.array): image to normalize

        Returns:
            np.array: normalized image
        """
        # img = img + np.minimum(0, np.min(img))  # move min to 0
        # img = img / np.max(img)  # scale to 0 to 1
        img = img / 4095  # scale to 0 to 1
        return img


    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed. By default resized to 256,256
        """
        images = np.zeros([self.N, self.size[0], self.size[1], 3], dtype="uint8")
        labels = []
        filenames = []

        i = 0
        # read all the files from the image folder
        for item in tqdm(os.listdir(self.root_dir)):
            f = os.path.join(self.root_dir, item)
            if os.path.isfile(f):
                continue
            for subitem in os.listdir(f):
                sub_f = os.path.join(f, subitem)
                filenames.append(sub_f)
                # a few images are a few pixels off, we will resize them
                images[i] = img_as_ubyte(self._normalize_to_0_to_1(pd.read_pickle(sub_f)))
                i += 1
                labels.append(item)

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # sort by filenames
        images = images[filenames.argsort()]
        labels = labels[filenames.argsort()]

        # convert to integer labels
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)
        self.label_encoding = list(le.classes_)  # remember label encoding

        # split into a train and test set as provided data is not presplit
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=labels,
        )

        # TODO: Add Upsample here.
        UpsR = self.upsample_ratio
        X_train, y_train = upsample_ds(ds=X_train, lb=y_train, N=UpsR[0], M=UpsR[1], tol=100, recursive=False)
        X_test, y_test = upsample_ds(ds=X_train, lb=y_train, N=UpsR[0], M=UpsR[1], tol=100, recursive=False)


        if self.train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test
        
        self.N = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]
