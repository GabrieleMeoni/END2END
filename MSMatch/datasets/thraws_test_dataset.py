import numpy as np
import os
from PIL import Image
from skimage import img_as_ubyte
from sklearn import preprocessing
import torch
from tqdm import tqdm
import pandas as pd 
import numpy as np



class THRAWS_test_dataset(torch.utils.data.Dataset):
    """Thraws test dataset"""

    def __init__(self, root_dir="./DATA/THRAWS/Test", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            seed (int): Used for compatibility. Will be ignored.
        """
        self.size = [256, 256]
        self.num_channels = 3
        self.num_classes = 2
        self.root_dir = root_dir
        self.transform = transform
        self.N = 531  # Modified to be inferred from data.
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

        self.data = images
        self.targets = labels

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
