import argparse
import os
from tqdm import tqdm

import torch
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from architectures.models import SubspaceClusterNetwork, Gumble_Softmax
from data.datasets import load_dataset
from loss import MaximalCodingRateReduction
from func import chunk_avg, cluster_acc
from lars import LARS, LARSWrapper

import utils

parser = argparse.ArgumentParser(description='Unsupervised Learning')
parser.add_argument('--arch', type=str, default='resnet18cifar',
                    help='architecture for deep neural network (default: resnet18cifar)')
parser.add_argument('--z_dim', type=int, default=64,
                    help='dimension of subspace feature dimension (default: 64)')
parser.add_argument('--n_clusters', type=int, default=10,
                    help='number of subspace clusters to use (default: 10)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset used for training (default: cifar10)')
parser.add_argument('--aug_name', type=str, default='cifar10_sup',
                    help='name of augmentation to use')
parser.add_argument('--epo', type=int, default=100,
                    help='number of epochs for training (default: 100)')
parser.add_argument('--load_epo', type=int, default=600,
                    help='epo to load pre-trained checkpoint from')
parser.add_argument('--train_backbone', action='store_true',
                    help='whether to also train parameters in backbone')
parser.add_argument('--validate_every', type=int, default=10,
                    help='validate clustering accuracy every this epochs and save results (default: 10)')
parser.add_argument('--bs', type=int, default=512,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--n_views', type=int, default=2,
                    help='number of augmentations per sample')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momo', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd1', type=float, default=1e-4,
                    help='weight decay for all other parameters except clustering head(default: 1e-4)')
parser.add_argument('--wd2', type=float, default=5e-3,
                    help='weight decay for clustering head (default: 5e-3)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared for MCR2 objective (default: 0.1)')
parser.add_argument('--tau', type=float, default=1,
                    help='temperature for gumble softmax (default: 1)')
parser.add_argument('--z_weight', type=float, default=100.,
                    help='weight for z_sim loss (default: 100)')
parser.add_argument('--doc', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./exps/',
                    help='base directory for saving PyTorch model. (default: ./exps/)')
parser.add_argument('--data_dir', type=str, default='../../data/',
                    help='path to dataset folder')
parser.add_argument('--gpu_ids', default=[0], type=eval, 
                    help='IDs of GPUs to use')
parser.add_argument('--fp16', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
args = parser.parse_args()

#GPU setup
device = 'cuda:'+str(args.gpu_ids[0])
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True
## get model directory
model_dir = os.path.join(args.save_dir,
               'selfsup_{}_{}_{}'.format(
                    args.arch, args.data, args.doc))

#model
net = SubspaceClusterNetwork(args.arch,args.z_dim,args.n_clusters).to(device)
net = torch.nn.DataParallel(net,args.gpu_ids)
G_Softmax = Gumble_Softmax(args.tau)

#data
train_dataset = load_dataset(args.data,args.aug_name,use_baseline=False,train=True,contrastive=True if args.n_views>1 else False,n_views=args.n_views,path=args.data_dir)
test_data = 'stl10' if args.data in ['stl10unsup','stl10'] else args.data
test_dataset = load_dataset(test_data,args.aug_name,use_baseline=True,train=True,contrastive=False,path=args.data_dir)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=8)

#loss
criterion = MaximalCodingRateReduction(eps=args.eps,gamma=1.0)

#load model checkpoint
save_dict = torch.load(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(args.load_epo)))
log = net.load_state_dict(save_dict['net'],strict=False)
print(log)

#only optimize cluster and subspace module
print(args.train_backbone)
if args.train_backbone:
    para_list = [p for p in net.module.backbone.parameters()] + [p for p in net.module.subspace.parameters()]
    para_list_c = [p for p in net.module.cluster.parameters()]
else:
    para_list = [p for p in net.module.subspace.parameters()]
    para_list_c = [p for p in net.module.cluster.parameters()]

optimizer = optim.SGD(para_list, lr=args.lr, momentum=args.momo, weight_decay=args.wd1,nesterov=False)
optimizer = LARSWrapper(optimizer,eta=0.02,clip=True,exclude_bias_n_norm=True,) #

optimizerc = optim.SGD(para_list_c, lr=0.5, momentum=args.momo, weight_decay=args.wd2,nesterov=False)
scaler = GradScaler()
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epo, eta_min=0,last_epoch=-1)

## Training
for epoch in range(args.epo):
    with tqdm(total=len(train_loader)) as progress_bar:
        for step, (x,y) in enumerate(train_loader):
            x = torch.cat(x,dim=0)
            x, y = x.float().to(device), y.to(device)
            
            with autocast(enabled=args.fp16):
                z, logits = net(x,detach_feature=True if not args.train_backbone else False)

                prob = G_Softmax(logits)

                if args.n_views>1:
                    z_avg, prob = chunk_avg(z,n_chunks=args.n_views,normalize=True), chunk_avg(prob,n_chunks=args.n_views)
                loss, loss_list= criterion(z_avg,prob,num_classes=args.n_clusters)
            
            z_list = z.chunk(args.n_views,dim=0)
            z_sim = (z_list[0]*z_list[1]).sum(1).mean()
            
            loss = loss - args.z_weight*z_sim

            loss_list += [z_sim.item()]

            optimizer.zero_grad()
            optimizerc.zero_grad()
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizerc.step()
            else:
                loss.backward()
                optimizer.step()
                optimizerc.step()
            
            progress_bar.set_description(str(epoch))
            progress_bar.set_postfix(loss= -loss_list[0] + loss_list[1],
                                     loss_d=loss_list[0],
                                     loss_c=loss_list[1],
                                     z_sim=z_sim.item()
                                    )
            progress_bar.update(1)
    scheduler.step()
    if (epoch+1)%args.validate_every==0:
        save_name_img = model_dir + '/cluster_imgs/cluster_imgs_ep' + str(epoch+1)
        save_name_fig = model_dir + '/pca_figures/z_space_pca' + str(epoch+1)
        cluster_acc(test_loader,net,device,print_result=True,save_name_img=save_name_img,save_name_fig=save_name_fig)
        utils.save_ckpt(model_dir, net, optimizer, scheduler, epoch + 1 + args.load_epo)

print("training complete.")
