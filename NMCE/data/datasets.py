import os
import numpy as np
import torchvision
from torch.utils.data import ConcatDataset
from .aug import load_transforms, ContrastiveLearningViewGenerator


def load_dataset(data_name, transform_name=None, use_baseline=False, train=True, contrastive=False, n_views=2, path="../../data/"):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    
    Parameters:
        data_name (str): name of the dataset
        transform_name (torchvision.transform): name of transform to be applied (see aug.py)
        use_baseline (bool): use baseline transform or augmentation transform
        train (bool): load training set or not
        contrastive (bool): whether to convert transform to multiview augmentation for contrastive learning.
        n_views (bool): number of views for contrastive learning
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    aug_transform, baseline_transform = load_transforms(transform_name)
    transform = baseline_transform if use_baseline else aug_transform 
    if contrastive:
        transform = ContrastiveLearningViewGenerator(transform,n_views=n_views)
        
    _name = data_name.lower()
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "CIFAR10"), train=train,
                                                download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train,
                                                 download=True, transform=transform)
        trainset.num_classes = 100
    elif _name == "cifar100coarse":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train,
                                                 download=True, transform=transform)
        trainset.targets = sparse2coarse(trainset.targets) 
        trainset.num_classes = 20
    elif _name == "mnist":
        trainset = torchvision.datasets.MNIST(root=os.path.join(path, "MNIST"), train=train, 
                                              download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "imagenet-dogs":
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(path, "imagenet/Imagenet-dogs"),transform=transform)
        trainset.num_classes = 15
    elif _name == "imagenet-10":
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(path, "imagenet/Imagenet-10"),transform=transform)
        trainset.num_classes = 10
    elif _name == "fashionmnist":
        trainset = torchvision.datasets.FashionMNIST(root=os.path.join(path, "FashionMNIST"), train=True, 
                                              download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=os.path.join(path, "FashionMNIST"), train=False, 
                                              download=True, transform=transform)
        
        trainset = ConcatDataset([trainset,testset])
        
        trainset.num_classes = 10
        
    elif _name == "stl10":
        #combine stl10 train and test data, total 13k images...
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "STL-10"), split='train', 
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "STL-10"), split='test', 
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset
        else:
            trainset.data = np.concatenate([trainset.data, testset.data])
            trainset.labels = trainset.labels.tolist() + testset.labels.tolist()
            trainset.targets = trainset.labels
            return trainset
    elif _name == "stl10sup":
        #separate stl10 train and test set
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "STL-10"), split='train', 
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "STL-10"), split='test', 
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset
        else:
            trainset.targets = trainset.labels
            return trainset
    
    elif _name == "stl10unsup":
        #100k unlabled images of stl10
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "STL-10"), split='unlabeled', 
                                              transform=transform, download=True)
        trainset.num_classes = 10
    else:
        raise NameError("{} not found in trainset loader".format(_name))
    return trainset

def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                       9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                      12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                       4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                       9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                      19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    return np.array(coarse_targets)[targets]