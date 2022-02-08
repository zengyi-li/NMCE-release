import os
import logging
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def init_pipeline(model_dir, headers=None):
    """Initialize folder and .csv logger."""
    # project folder
    os.makedirs(model_dir,exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'),exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'cluster_imgs'),exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'pca_figures'),exist_ok=True)
    if headers is None:
        headers = ["epoch", "step", "loss_c", "loss_d","z_sim"]
    create_csv(model_dir, 'losses.csv', headers)
    print("project dir: {}".format(model_dir))

def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def save_params(model_dir, params):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def update_params(model_dir, pretrain_dir):
    """Updates architecture and feature dimension from pretrain directory 
    to new directoy. """
    params = load_params(model_dir)
    old_params = load_params(pretrain_dir)
    params['arch'] = old_params["arch"]
    params['fd'] = old_params['fd']
    save_params(model_dir, params)

def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict

def save_state(model_dir, *entries, filename='losses.csv'):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))

def save_ckpt(model_dir, net, optimizer, scheduler, epoch):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    save_dict = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_dict, os.path.join(model_dir, 'checkpoints', 
        'model-epoch{}.pt'.format(epoch)))

def load_ckpt(model_dir, net, optimizer, scheduler, epoch):
    save_dict = torch.load(os.path.join(model_dir, 'checkpoints', 
        'model-epoch{}.pt'.format(epoch)))
    net.load_state_dict(save_dict['net'])
    optimizer.load_state_dict(save_dict['optimizer'])
    scheduler.load_state_dict(save_dict['scheduler'])
    print('checkpoint loaded')
    
def save_labels(model_dir, labels, epoch):
    """Save labels of a certain epoch to directory. """
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def plot_on_sphere(x,y=None):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter3D(x[:,0],x[:,1],x[:,2],'.',c=y)
    ax.set_zlim(zmin=-1,zmax=1)
    ax.set_xlim(xmin=-1,xmax=1)
    ax.set_ylim(ymin=-1,ymax=1)
    
def plot_on_2d(x,y=None):
    plt.scatter(x[:,0],x[:,1],c=y)