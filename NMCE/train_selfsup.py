import argparse
import os
from tqdm import tqdm

import torch
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from architectures.models import SelfSupervisedNetwork
from data.datasets import load_dataset
from loss import TotalCodingRate, SimCLR, Z_loss
from func import marginal_H, chunk_avg, cluster_acc, set_gamma, warmup_lr
from lars import LARS, LARSWrapper
import utils

parser = argparse.ArgumentParser(description='Unsupervised Learning')
parser.add_argument('--arch', type=str, default='resnet18cifar',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--z_dim', type=int, default=64,
                    help='dimension of subspace feature dimension (default: 32)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--aug_name', type=str, default='cifar10_sup',
                    help='name of augmentation to use')
parser.add_argument('--epo', type=int, default=500,
                    help='number of epochs for training (default: 500)')
parser.add_argument('--save_every', type=int, default=50,
                    help='save checkpoint every certain epoch during training (default: 50)')
parser.add_argument('--start_epo', type=int, default=0,
                    help='number of epochs for starting training (default: 0) >0 will load checkpoint')
parser.add_argument('--bs', type=int, default=512,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--n_views', type=int, default=2,
                    help='number of augmentations per sample')
parser.add_argument('--optim', type=str, default='LARS_SGD',
                    help='select optimizer [LARS, AdamW]')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momo', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared for coding rate objective (default: 0.1)')
parser.add_argument('--temperature', type=float, default=0.5,
                    help='temperature for simclr loss (default: 0.5)')
parser.add_argument('--z_weight', type=float, default=100.,
                    help='weight for z_sim loss (default: 100)')
parser.add_argument('--z_hinge', type=float, default=1.2,
                    help='value above which z sim is not optimized')
parser.add_argument('--doc', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--loss', type=str, default='simclr',
                    help='loss used to train self-supervised model. [simclr, vicreg, totalcodingrate]')
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

model_dir = os.path.join(args.save_dir,
               'selfsup_{}_{}_{}'.format(
                    args.arch, args.data, args.doc))

utils.init_pipeline(model_dir)

if args.loss=='simclr':
    criterion = SimCLR(args.temperature,args.n_views)
    criterion_z = Z_loss()
elif args.loss =='simclr_contrastive':
    criterion = SimCLR(args.temperature,args.n_views,contrastive=True)
    criterion_z = Z_loss()
elif args.loss=='totalcodingrate':
    criterion = TotalCodingRate(eps=args.eps)
    criterion_z = Z_loss()

net = SelfSupervisedNetwork(args.arch,args.z_dim,normalize=True)
net = torch.nn.DataParallel(net.to(device),args.gpu_ids)

train_dataset = load_dataset(args.data,args.aug_name,use_baseline=False,train=True,contrastive=True,n_views=args.n_views,path=args.data_dir)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=16)

if args.optim == 'LARS_SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momo, weight_decay=args.wd,nesterov=False)
    optimizer = LARSWrapper(optimizer,eta=0.02,clip=True,exclude_bias_n_norm=True,)
    
scaler = GradScaler()
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epo, eta_min=0,last_epoch=-1)
utils.save_params(model_dir, vars(args))

if args.start_epo > 0:
    utils.load_ckpt(model_dir, net, optimizer, scheduler, args.start_epo)

if args.fp16:
    print('using fp16 precision')
    
## Training
for epoch in range(args.start_epo, args.epo):
    warmup_lr(optimizer,epoch,args.lr,warmup_epoch=10)
    with tqdm(total=len(train_loader)) as progress_bar:
        for step, (x,y) in enumerate(train_loader):
            x = torch.cat(x,dim=0)
            x, y = x.float().to(device), y.to(device)
            
            with autocast(enabled=args.fp16):
                z = net(x)
                #calculate cosine similarity between z vectors for reference
                loss_z, z_sim = criterion_z(z)
                
                if args.loss in ['simclr','simclr_contrastive']:
                    loss_t = criterion(z)
                    loss = loss_t
                elif args.loss in ['totalcodingrate']:
                    z_list = z.chunk(2,dim=0)
                    loss = (criterion(z_list[0]) + criterion(z_list[1]))/2
                    loss_t = loss + args.z_weight*loss_z
                
            loss_list = [z_sim.item()]

            optimizer.zero_grad()
            if args.fp16:
                scaler.scale(loss_t).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_t.backward()
                optimizer.step()
            
            #show loss_t for some methods
            if args.loss in ['simclr','vicreg']:
                loss = loss_t
            utils.save_state(model_dir, epoch, step, loss.item(), *loss_list)
            progress_bar.set_description(str(epoch))
            progress_bar.set_postfix(loss = loss.item(),
                                     z_sim=z_sim.item(),
                                     lr=optimizer.param_groups[0]['lr']
                                    )
            progress_bar.update(1)
            
    scheduler.step()
    if (epoch+1)%args.save_every==0:
        utils.save_ckpt(model_dir, net, optimizer, scheduler, epoch + 1)
    
print("training complete.")
