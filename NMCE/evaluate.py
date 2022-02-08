import argparse
import os
from tqdm import tqdm

import torch
import sys
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from architectures.models import SelfSupervisedNetwork, SubspaceClusterNetwork
from evals.supervised import svm, knn, nearsub
from evals.cluster import kmeans, ensc
from evals.linear import linear
from data.datasets import load_dataset
from func import chunk_avg

parser = argparse.ArgumentParser(description='Unsupervised Learning')

parser.add_argument('--linear', help='evaluate using linear prob', action='store_true')
parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
parser.add_argument('--nearsub', help='evaluate using Nearest Subspace', action='store_true')
parser.add_argument('--kmeans', help='evaluate using KMeans', action='store_true')
parser.add_argument('--ensc', help='evaluate using Elastic Net Subspace Clustering', action='store_true')

#evaluation arguments
parser.add_argument('--k', type=int, default=11, help='top k components for kNN')
parser.add_argument('--n', type=int, default=10, help='number of clusters for cluster (default: 10)')
parser.add_argument('--algo', type=str, default='lasso_lars',help='algorithm for ENSC')
parser.add_argument('--gam', type=int, default=100, help='gamma paramter for subspace clustering (default: 100)')
parser.add_argument('--tau', type=float, default=1.0,help='tau paramter for subspace clustering (default: 1.0)')
parser.add_argument('--n_comp', type=int, default=30, help='number of components for PCA (default: 30)')
#arguments for linear prob
parser.add_argument('--epo', type=int, default=100,help='number of epochs for training (default: 100)')
parser.add_argument('--bs', type=int, default=128,help='input batch size for training (default: 1000)')
parser.add_argument('--lr', type=float, default=0.1,help='learning rate (default: 0.001)')
parser.add_argument('--momo', type=float, default=0.9,help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-5,help='weight decay (default: 5e-4)')


parser.add_argument('--arch', type=str, default='resnet18cifar',help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--feature_type', type=str, default='pool',help='type of feature to use, can be pool, pre_feature, subspace, proj, clustering (default: pool)')
#pool: feature after global avg pooling in resnet 50
#pre_feature: after passing avg pooling result through pre-feature layer
#subspace: subspace projection
#proj: after second linear layer of simclr projection head.
parser.add_argument('--n_classes', type=int, default=10,help='number of classes for supervised evaluation (default: 10)')
parser.add_argument('--n_clusters', type=int, default=10,help='number of clusters, for loading subspace clustering network (default: 10)')
parser.add_argument('--z_dim', type=int, default=128,help='dimension of subspace or simclr projection output (default: 10)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--aug_name', type=str, default='cifar10_sup',
                    help='name of augmentation to use')
parser.add_argument('--aug_avg', type=int, default=1,
                    help='number of samples to averge with different augmentations, <2 disable augmentation.')
parser.add_argument('--load_epo', type=int, default=600,
                    help='epo to load pre-trained checkpoint from')
parser.add_argument('--doc', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_samples', action='store_true',
                    help='whether to save samples into a file')
parser.add_argument('--save_images', action='store_true',
                    help='whether to save images along with samples')
parser.add_argument('--save_dir', type=str, default='./exps/',
                    help='base directory for saving PyTorch model. (default: ./exps/)')
parser.add_argument('--data_dir', type=str, default='../../data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
parser.add_argument('--gpu_ids', default=[0], type=eval, 
                    help='IDs of GPUs to use')
args = parser.parse_args()

#GPU setup
device = 'cuda:'+str(args.gpu_ids[0])
torch.backends.cudnn.benchmark = True
## get model directory
model_dir = os.path.join(args.save_dir,
               'selfsup_{}_{}_{}'.format(
                    args.arch, args.data, args.doc))

#data
use_baseline = True if args.aug_avg<1 else False
# contrastive = False if args.aug_avg<1 else True
contrastive = False

train_data = 'stl10sup' if args.data in ['stl10unsup','stl10'] else args.data
train_dataset = load_dataset(train_data,args.aug_name,use_baseline=use_baseline,train=True,contrastive=contrastive,path=args.data_dir)
# test_data = 'stl10' if args.data=='stl10unsup' else args.data
test_dataset = load_dataset(train_data,args.aug_name,use_baseline=use_baseline,train=False,contrastive=contrastive,path=args.data_dir)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=8)

#model
if args.feature_type in ['pool', 'pre_feature', 'subspace', 'clustering']:
    net = SubspaceClusterNetwork(args.arch,args.z_dim,args.n_clusters,normalize=True).to(device)
elif args.feature_type == 'proj':
    net = SelfSupervisedNetwork(args.arch,args.z_dim,classifier=False,n_classes=args.n_classes,normalize=True).to(device)

net = torch.nn.DataParallel(net,args.gpu_ids)
net.eval()

#load checkpoint
save_dict = torch.load(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(args.load_epo)))
log = net.load_state_dict(save_dict['net'],strict=False)
print('evaluating folder' + model_dir)
print(log)

train_z_list, train_y_list, test_z_list, test_y_list = [], [], [], []
train_x_list, train_logits_list, test_x_list, test_logits_list = [], [], [], []
with torch.no_grad():
    n_iter = max(args.aug_avg,1)
    for i in range(n_iter):
        print('collect train features and labels')
        for x, y in train_loader:
            x = x.to(device)
            if not args.feature_type=='clustering':
                z = net(x,return_feature=args.feature_type).detach().cpu()
            else:
                z, logits = net(x,return_feature=args.feature_type)
                z, logits = z.detach().cpu(), logits.detach().cpu()
                train_logits_list.append(logits)
            train_z_list.append(z)
            if i==0:
                train_y_list.append(y)
                train_x_list.append(x.detach().cpu())
        print('collect test features and labels')
        for x, y in test_loader:
            x = x.to(device)
            if not args.feature_type=='clustering':
                z = net(x,return_feature=args.feature_type).detach().cpu()
            else:
                z, logits = net(x,return_feature=args.feature_type)
                z, logits = z.detach().cpu(), logits.detach().cpu()
                test_logits_list.append(logits)
            test_z_list.append(z)
            if i==0:
                test_y_list.append(y)
                test_x_list.append(x.detach().cpu())
        
train_features, train_labels, test_features, test_labels = torch.cat(train_z_list,dim=0), torch.cat(train_y_list,dim=0), torch.cat(test_z_list,dim=0), torch.cat(test_y_list,dim=0)
if args.aug_avg>1:
    normalize = args.feature_type == 'proj'
    train_features, test_features = chunk_avg(train_features,n_chunks=args.aug_avg,normalize=normalize), chunk_avg(test_features,n_chunks=args.aug_avg,normalize=normalize)

if args.save_samples:
    train_x, test_x = torch.cat(train_x_list,dim=0), torch.cat(test_x_list,dim=0)
    if args.feature_type=='clustering':
        train_logits, test_logits = torch.cat(train_logits_list,dim=0), torch.cat(test_logits_list,dim=0)
    else:
        train_logits, test_logits = None, None
    if not args.save_images:
        train_x, test_x = None, None
    save_dict = {
        'train_x': train_x,
        'train_logits': train_logits,
        'train_y': train_labels,
        'train_z': train_features,
        'test_x': test_x,
        'test_logits': test_logits,
        'test_y': test_labels,
        'test_z': test_features
    }
    torch.save(save_dict,os.path.join(model_dir,'samples_' + args.feature_type + '.pth'))
    print('samples saved')
    
if args.linear:
    linear(args, train_features, train_labels, test_features, test_labels)
if args.svm:
    svm(args, train_features, train_labels, test_features, test_labels)
if args.knn:
    knn(args, train_features, train_labels, test_features, test_labels)
if args.nearsub:
    nearsub(args, train_features, train_labels, test_features, test_labels)
if args.kmeans:
    kmeans(args, train_features, train_labels)
if args.ensc:
    ensc(args, train_features, train_labels)