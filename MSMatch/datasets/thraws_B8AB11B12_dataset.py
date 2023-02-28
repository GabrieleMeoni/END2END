import imageio
import numpy as np
import os
from PIL import Image
from skimage import img_as_ubyte
from sklearn import preprocessing
import torch
from tqdm import tqdm
import pandas as pd 
import random

from copy import deepcopy
import numpy as np

def geographical_splitter(images, labels, filenames, test_size_percentage, seed=42, split_percentage_error_tolerance=0.01):
    """Splits the events according to a geographical position. In this way, patches related to a specific area can be only in train or validation. 

    Args:
        images (np.array): list of images to split.
        labels (np.array): list of labels to split.
        filenames (list): file names of the different images.
        test_size_percentage (float): split perecentage.
        seed (int, optional): seed for reproducibility. Defaults to 42.
        split_percentage_error_tolerance (float, optional): tolerance on the split percentage error. Defaults to 0.01.

    Returns:
        np.array: train data 
        np.array: test data
        np.array: train labels
        np.array: test labels
    """
    #Not events dictionaries 
    ##Fire not_event: locations - n_events dictionary
    fire_not_events_locations_dict={'Bolivia': 174, 'France': 28, 'Greece': 288, 'Greenland': 14, 'Italy': 54, 'Kenya': 19, 'Latvia': 155, 'Mexico': 121, 'Spain': 99, 'Sweden': 97, 'Ukraine': 49}

    ## Volcanoes not_event: locations - n_events dictionary
    volcano_not_events_locations_dict={'Barren_Island': 146, 'Chillan_Nevados_de': 76, 'Copahue': 77, 'Etna': 232, 'Fuego': 172, 'Karangetang': 40, 'Krysuvik-Trolladyngja': 209, 'La_Palma': 70, 'Mayon': 67, 'Nyamulagira': 489, 'Piton_de_la_Fournaise': 310, 'Poas': 96, 'Raung': 248, 'San_Miguel': 271, 'Sangay': 218, 'Santa_Maria': 96, 'Stromboli': 75, 'Telica': 73, 'Tinakula': 109}

    #Events dictionaries
    ## Fire event: locations - n_events dictionary
    fire_events_locations_dict={'Bolivia': 16, 'France': 14, 'Greece': 36, 'Italy': 9, 'Kenya': 12, 'Latvia': 2, 'Mexico': 21, 'Spain': 10, 'Sweden': 10, 'Ukraine': 12}

    ## Volcanoes event: locations - n_events dictionary
    volcano_events_locations_dict={'Barren_Island': 8, 'Chillan_Nevados_de': 4, 'Copahue': 2, 'Etna': 15, 'Fuego': 14, 'Karangetang': 1, 'Krysuvik-Trolladyngja': 14, 'La_Palma': 8, 'Mayon': 7, 'Nyamulagira': 25, 'Piton_de_la_Fournaise': 17, 'Poas': 3, 'Raung': 28, 'San_Miguel': 42, 'Sangay': 10, 'Santa_Maria': 7, 'Stromboli': 2, 'Telica': 4, 'Tinakula': 3}
    
    while(1):
        volcanoes_events_keys_shuffled=deepcopy(list(volcano_events_locations_dict.keys()))
        fires_events_keys_shuffled=deepcopy(list(fire_events_locations_dict.keys()))
        volcanoes_not_events_keys_shuffled=deepcopy(list(volcano_not_events_locations_dict.keys()))
        fires_not_events_keys_shuffled=deepcopy(list(fire_not_events_locations_dict.keys()))
        # Fixing seed to ensure that dataset train and test will be splitted in the same way into different iterations
        random.seed(seed)

        random.shuffle(volcanoes_events_keys_shuffled)
        random.shuffle(fires_events_keys_shuffled)
        random.shuffle(volcanoes_not_events_keys_shuffled)
        random.shuffle(fires_not_events_keys_shuffled)

        # Number of total events
        n_events=np.sum(np.array([n for n in fire_events_locations_dict.values()]+[n for n in volcano_events_locations_dict.values()]))
        # Number of total nonevents
        n_not_events=np.sum(np.array([n for n in fire_not_events_locations_dict.values()]+[n for n in volcano_not_events_locations_dict.values()]))

        #Maximum number of events in tests
        n_events_test_max=int(test_size_percentage * n_events)

        #Number of events in tests
        n_events_test=0

        #Currernt labels for volcano e fires
        volcano_label_idx=0
        fire_label_idx=0

        #List of events location in test
        events_locations_tests_list=[]
        n_events_list=[]
        while(n_events_test < n_events_test_max):
            n_events_test+=volcano_events_locations_dict[volcanoes_events_keys_shuffled[volcano_label_idx]]
            n_events_list.append(volcano_events_locations_dict[volcanoes_events_keys_shuffled[volcano_label_idx]])
            events_locations_tests_list.append(volcanoes_events_keys_shuffled[volcano_label_idx])
            volcano_label_idx+=1

            if(n_events_test < n_events_test_max):
                n_events_test+=fire_events_locations_dict[fires_events_keys_shuffled[fire_label_idx]]
                n_events_list.append(fire_events_locations_dict[fires_events_keys_shuffled[fire_label_idx]])
                events_locations_tests_list.append(fires_events_keys_shuffled[fire_label_idx])
                fire_label_idx+=1    

        #Maximum number of nonevents in tests
        n_not_events_test_max=int(test_size_percentage * n_not_events)

        #List of nonevents location in test
        n_not_events_test=0
        volcano_label_idx=0
        fire_label_idx=0
        n_non_events_list=[]
        not_events_locations_tests_list=[]
        while(n_not_events_test < n_not_events_test_max):
            n_not_events_test+=volcano_not_events_locations_dict[volcanoes_not_events_keys_shuffled[volcano_label_idx]]
            n_non_events_list.append(volcano_not_events_locations_dict[volcanoes_not_events_keys_shuffled[volcano_label_idx]])
            not_events_locations_tests_list.append(volcanoes_not_events_keys_shuffled[volcano_label_idx])
            volcano_label_idx+=1

            if(n_not_events_test < n_not_events_test_max):
                n_not_events_test+=fire_not_events_locations_dict[fires_not_events_keys_shuffled[fire_label_idx]]
                not_events_locations_tests_list.append(fires_not_events_keys_shuffled[fire_label_idx])
                n_non_events_list.append(fire_not_events_locations_dict[fires_not_events_keys_shuffled[fire_label_idx]])
                fire_label_idx+=1 
        
        real_percentage=(n_events_test + n_not_events_test)/len(labels)
        
        if real_percentage - test_size_percentage <= split_percentage_error_tolerance:
            break

    # Images and labels for trains
    X_train=np.zeros([len(labels) - (n_events_test + n_not_events_test), images[0].shape[0], images[0].shape[1],  images[0].shape[2]], dtype=np.uint8)
    y_train=np.zeros([len(labels) - (n_events_test + n_not_events_test)], dtype='int64')
    # Images and labels for tests
    X_test=np.zeros([n_events_test + n_not_events_test , images[0].shape[0], images[0].shape[1],  images[0].shape[2]], dtype=np.uint8)
    y_test=np.zeros([n_events_test + n_not_events_test],dtype='int64')
    
    #Number of selected images for train and tests
    n_train_selected=0
    n_test_selected=0

    #Perform splitting 
    for n, filename in enumerate(filenames):
        filename_dropped=filename.split(os.sep)[-1].split("_G")[0]
        if filename_dropped[-2] == "_":
            location=filename_dropped[:-2]
        else:
            location=filename_dropped[:-3]
        
        

        if ((location in events_locations_tests_list) and labels[n] == 0) or ((location in not_events_locations_tests_list) and (labels[n] == 1)):
            X_test[n_test_selected]=images[n]
            y_test[n_test_selected]=labels[n]
            n_test_selected+=1
        else:
            X_train[n_train_selected]=images[n]
            y_train[n_train_selected]=labels[n]
            n_train_selected+=1
    return X_train, X_test, y_train, y_test 

from .utils import upsample_ds

class ThrawsB8AB11B12Dataset(torch.utils.data.Dataset):
    """Thraws dataset"""

    def __init__(self, train, root_dir="/data/PyDeepLearning/END2END/MSMatch/DATA/warmup_events_dataset/", transform=None, seed=42, upsample_ratio=None):
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
        self.upsample_ratio = upsample_ratio
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
        #X_train, X_test, y_train, y_test = train_test_split(
        #    images,
        #    labels,
        #    test_size=self.test_ratio,
        #    random_state=self.seed,
        #    stratify=labels,
        #)
        # Add geographical splitting.
        filenames_sorted=sorted(filenames)
        X_train, X_test, y_train, y_test = geographical_splitter(images, labels, filenames_sorted, self.test_ratio, self.seed)

        # Add upsampling.
        if self.upsample_ratio is not None:
            NE, EV = self.upsample_ratio
            X_train, y_train = upsample_ds(ds=X_train, lb=y_train, NotEvent=NE, Event=EV)
            X_test, y_test = upsample_ds(ds=X_test, lb=y_test, NotEvent=NE, Event=EV)

        if self.train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test
        
        self.N = int(X_train.shape[0]+X_test.shape[0])

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
