import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import sklearn.datasets as datasets

from loss import MaximalCodingRateReduction, Z_loss, TotalCodingRate
from func import chunk_avg, cluster_match, analyze_latent, save_cluster_imgs
from architectures.models import Gumble_Softmax, SubspaceClusterNetwork
from data.aug import GaussianBlur
from evals.cluster import kmeans, ensc

import torchvision.transforms as tfs

data_dict = torch.load('COIL20.pth')
# data_dict = torch.load('EYB.pth')

img_mtx = data_dict['img_mtx']
label_mtx = data_dict['label_mtx'] - 1

def down_sample(img_mtx,factor=2):
    in_h, in_w = img_mtx.shape[2:4]
    out_h, out_w = in_h//factor, in_w//factor
    down_sample = tfs.Resize((out_h,out_w))
    img_list = []
    for img in img_mtx:
        img_list.append(down_sample(img))
    return torch.stack(img_list,dim=0)

img_mtx = down_sample(img_mtx)
print(img_mtx.shape)

#COIL minimum, 
transforms = tfs.Compose([tfs.ToPILImage(),
                          tfs.RandomHorizontalFlip(p=0.5),
                          tfs.RandomPerspective(distortion_scale=0.3, p=0.6),
                          tfs.ColorJitter(0.8, 0.8, 0.8, 0.2),
                          tfs.ToTensor()])

def generate_views(img_mtx,transforms,n_views=2):
    out_list = []
    for i in range(n_views):
        img_list = []
        for img in img_mtx:
            img_list.append(transforms(img))
        out_list.append(torch.cat(img_list,dim=0).unsqueeze(1)) #add axis since somehow it gets squeezed.
    return out_list

def validate():
    net.eval()
    with torch.no_grad():
        z, logits = net(img_mtx.float().cuda()/255)
        preds = logits.max(dim=1)[1]
    preds = preds.cpu()
    args = nn.Identity()
    args.n_clusters = 20
    args.gam = 400
    args.tau = 1
    args.algo = 'lasso_lars'
    return ensc(args, z.cpu(), label_mtx)

n_steps = 2000
print_every = 50
validate_every = 200
z_dim = 40
n_clusters = int(label_mtx.max())

net = SubspaceClusterNetwork('resnet18imgs',z_dim,n_clusters,norm_p=2).cuda()

optimizer = optim.Adam(net.parameters(),lr=0.001,betas=(0.9,0.99),weight_decay=0.00001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,n_steps)
G_Softmax = Gumble_Softmax(1,straight_through=False)

criterion = TotalCodingRate(eps=0.01)
criterion_z = Z_loss()


for i in range(n_steps):
    x = generate_views(img_mtx,transforms)
    x = torch.cat(x,dim=0).float().cuda()
#     x = img_mtx.float().cuda()

    z, logits = net(x)
    loss_z, z_sim = criterion_z(z)
    z_sim = z_sim.mean()
    prob = G_Softmax(logits)
    z, prob = chunk_avg(z,n_chunks=2,normalize=True), chunk_avg(prob,n_chunks=2)
    
    loss = criterion(z)
    loss_list = [loss.item(),loss.item()]
    
    loss += 20*loss_z
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i%print_every == 0:
        print('{} steps done, loss c {}, loss d {}, z sim {}'.format(i+1,loss_list[0],loss_list[1],z_sim.item()))
    if i%validate_every == 0:
        acc, preds = validate()
        net.train()