import torch
from .data_utils import split_ssl_data
from .dataset import BasicDataset
from .thraws_train_dataset import THRAWS_train_dataset
from .thraws_test_dataset import THRAWS_test_dataset


import torchvision
from torchvision import transforms

mean, std = {}, {}
mean["cifar10"] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean["cifar100"] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean["thraws_swir_train"] = [0, 0, 0]  # zero mean
mean["thraws_swir_test"] = [0, 0, 0]  # zero mean
# std['thraws_swir']=[(2**8)-1,(2**8)-1,(2**8)-1] # 8 bit sampling
std["thraws_swir_train"] = [1, 1, 1]  # 8 bit sampling
std["thraws_swir_test"] = [1, 1, 1]  # 8 bit sampling

std["cifar10"] = [x / 255 for x in [63.0, 62.1, 66.7]]
std["cifar100"] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_transform(mean, std, train=True):
    if train:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0, 0.125)),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )


def get_inverse_transform(mean, std):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean_inv, std_inv)]
    )


class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(
        self,
        name="cifar10",
        train=True,
        data_dir=None,
        seed=42,
        eval_split_ratio=0.3,
        upsample_event=7,
        upsample_notevent=1,
    ):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            data_dir: path of directory, where data is downloaed or stored.
            seed: seed to use for the train / test split. Not available for cifar which is presplit
            eval_split_ratio: percentage of the eval split over the entire dataset.
            upsample_ratio: ratio of notevent, event upsampling
        """

        self.name = name
        self.seed = seed
        self.train = train
        self.data_dir = data_dir

        self.transform = get_transform(mean[name], std[name], train)
        self.inv_transform = get_inverse_transform(mean[name], std[name])
        self.upsample_event = upsample_event
        self.upsample_notevent = upsample_notevent
        self.eval_split_ratio = eval_split_ratio
        self.use_ms_augmentations = False

    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.name in ["cifar10", "cifar100"]:
            dset = getattr(torchvision.datasets, self.name.upper())
            dset = dset(self.data_dir, train=self.train, download=True)
        elif self.name == "thraws_swir_train":
            dset = THRAWS_train_dataset(
                train=self.train,
                root_dir=self.data_dir,
                seed=self.seed,
                eval_split_ratio=self.eval_split_ratio,
                upsample_ratio=[self.upsample_notevent, self.upsample_event],
            )

        elif self.name == "thraws_swir_test":
            dset = THRAWS_test_dataset(root_dir=self.data_dir)
            self.data_dir = dset.root_dir

        if self.name == "cifar10":
            self.label_encoding = None
            self.num_classes = 10
            self.num_channels = 3
        elif self.name == "cifar100":
            self.label_encoding = None
            self.num_classes = 100
            self.num_channels = 3
        else:
            self.label_encoding = dset.label_encoding
            self.num_classes = dset.num_classes
            self.num_channels = dset.num_channels

        if self.data_dir is None:
            self.data_dir = dset.root_dir

        data, targets = dset.data, dset.targets
        self.size = dset.size
        return data, targets

    def get_dset(self, use_strong_transform=False, strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.

        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """

        data, targets = self.get_data()

        return BasicDataset(
            data,
            targets,
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
            onehot,
            self.use_ms_augmentations,
        )

    def get_ssl_dset(
        self,
        num_labels,
        index=None,
        include_lb_to_ulb=True,
        use_strong_transform=True,
        strong_transform=None,
        onehot=False,
    ):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.

        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair.
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.

        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """

        data, targets = self.get_data()

        _ = self.num_classes
        _ = self.transform

        lb_data, lb_targets, _, _ = split_ssl_data(
            data, targets, num_labels, self.num_classes, index, include_lb_to_ulb
        )

        lb_dset = BasicDataset(
            lb_data,
            lb_targets,
            self.num_classes,
            self.transform,
            False,
            None,
            onehot,
            self.use_ms_augmentations,
        )

        ulb_dset = BasicDataset(
            data,
            targets,
            self.num_classes,
            self.transform,
            use_strong_transform,
            strong_transform,
            onehot,
            self.use_ms_augmentations,
        )

        return lb_dset, ulb_dset
