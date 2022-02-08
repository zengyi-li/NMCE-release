import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import torchvision
# import torch.nn
def set_gamma(loss_fn,epoch,total_epoch=500,warmup_epoch=100,gamma_min=0.,gamma_max=1.0):
    warmup_start = total_epoch - warmup_epoch
    warmup_end = total_epoch
    
    if warmup_start < epoch<=warmup_end:
        loss_fn.gamma = ((epoch - warmup_start)/(warmup_end - warmup_start))*(gamma_max - gamma_min) + gamma_min
    else:
        loss_fn.gamma = gamma_min

def warmup_lr(optimizer,epoch,base_lr,warmup_epoch=10):
    if epoch<warmup_epoch:
        optimizer.param_groups[0]['lr'] = base_lr*min(1.,(epoch+1)/warmup_epoch)
        
        
def marginal_H(logits):
    bs = torch.tensor(logits.shape[0]).float()
    logps = torch.log_softmax(logits,dim=1)
    marginal_p = torch.logsumexp(logps - bs.log(),dim=0)
    H = (marginal_p.exp()*(-marginal_p)).sum()*(1.4426950)
    return H

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)

def cluster_match(cluster_mtx,label_mtx,n_classes=10,print_result=True):
    #verified to be consistent to optimimal assignment problem based algorithm
    cluster_indx = list(cluster_mtx.unique())
    assigned_label_list = []
    assigned_count = []
    while (len(assigned_label_list)<=n_classes) and len(cluster_indx)>0:
        max_label_list = []
        max_count_list = []
        for indx in cluster_indx:
            #calculate highest number of matchs
            mask = cluster_mtx==indx
            label_elements, counts = label_mtx[mask].unique(return_counts=True)
            for assigned_label in assigned_label_list:
                counts[label_elements==assigned_label] = 0
            max_count_list.append(counts.max())
            max_label_list.append(label_elements[counts.argmax()])

        max_label = torch.stack(max_label_list)
        max_count = torch.stack(max_count_list)
        assigned_label_list.append(max_label[max_count.argmax()])
        assigned_count.append(max_count.max())
        cluster_indx.pop(max_count.argmax())
    total_correct = torch.tensor(assigned_count).sum().item()
    total_sample = cluster_mtx.shape[0]
    acc = total_correct/total_sample
    if print_result:
        print('{}/{} ({}%) correct'.format(total_correct,total_sample,acc*100))
    else:
        return total_correct, total_sample, acc

def cluster_merge_match(cluster_mtx,label_mtx,print_result=True):
    cluster_indx = list(cluster_mtx.unique())
    n_correct = 0
    for cluster_id in cluster_indx:
        label_elements, counts = label_mtx[cluster_mtx==cluster_id].unique(return_counts=True)
        n_correct += counts.max()
    total_sample = len(cluster_mtx)
    acc = n_correct.item()/total_sample
    if print_result:
        print('{}/{} ({}%) correct'.format(n_correct,total_sample,acc*100))
    else:
        return n_correct, total_sample, acc

    
def cluster_acc(test_loader,net,device,print_result=False,save_name_img='cluster_img',save_name_fig='pca_figure'):
    cluster_list = []
    label_list = []
    x_list = []
    z_list = []
    net.eval()
    for x, y in test_loader:
        with torch.no_grad():
            x, y = x.float().to(device), y.to(device)
            z, logit = net(x)
            if logit.sum() == 0:
                logit += torch.randn_like(logit)
            cluster_list.append(logit.max(dim=1)[1].cpu())
            label_list.append(y.cpu())
            x_list.append(x.cpu())
            z_list.append(z.cpu())
    net.train()
    cluster_mtx = torch.cat(cluster_list,dim=0)
    label_mtx = torch.cat(label_list,dim=0)
    x_mtx = torch.cat(x_list,dim=0)
    z_mtx = torch.cat(z_list,dim=0)
    _, _, acc_single = cluster_match(cluster_mtx,label_mtx,n_classes=label_mtx.max()+1,print_result=False)
    _, _, acc_merge = cluster_merge_match(cluster_mtx,label_mtx,print_result=False)
    NMI = normalized_mutual_info_score(label_mtx.numpy(),cluster_mtx.numpy())
    ARI = adjusted_rand_score(label_mtx.numpy(),cluster_mtx.numpy())
    if print_result:
        print('cluster match acc {}, cluster merge match acc {}, NMI {}, ARI {}'.format(acc_single,acc_merge,NMI,ARI))
    
    save_name_img += '_acc'+ str(acc_single)[2:5]
    save_cluster_imgs(cluster_mtx,x_mtx,save_name_img)
    save_latent_pca_figure(z_mtx,cluster_mtx,save_name_fig)
    
    return acc_single, acc_merge
    
def save_cluster_imgs(cluster_mtx,x_mtx,save_name,npercluster=100):
    cluster_indexs, counts = cluster_mtx.unique(return_counts=True)
    x_list = []
    counts_list = []
    for i, c_indx in enumerate(cluster_indexs):
        if counts[i]>npercluster:
            x_list.append(x_mtx[cluster_mtx==c_indx,:,:,:])
            counts_list.append(counts[i])

    n_clusters = len(counts_list)
    fig, ax = plt.subplots(n_clusters,1,dpi=80,figsize=(1.2*n_clusters, 3*n_clusters))
    for i, ax in enumerate(ax):
        img = torchvision.utils.make_grid(x_list[i][:npercluster],nrow=npercluster//5,normalize=True)
        ax.imshow(img.permute(1,2,0))
        ax.set_axis_off()

        ax.set_title('Cluster with {} images'.format(counts_list[i]))
    
    fig.savefig(save_name+'.pdf')
    plt.close(fig)
    
def save_latent_pca_figure(z_mtx,cluster_mtx,save_name):
    _, s_z_all, _ = z_mtx.svd()
    cluster_n = []
    cluster_s = []
    for cluster_indx in cluster_mtx.unique():
        _, s_cluster, _ = z_mtx[cluster_mtx==cluster_indx,:].svd()
        cluster_n.append((cluster_mtx==cluster_indx).sum().item())
        cluster_s.append(s_cluster/s_cluster.max())

    #make plot
    fig, ax = plt.subplots(1,2,figsize=(9, 3))
    ax[0].plot(s_z_all)
    for i, s_curve in enumerate(cluster_s):
        ax[1].plot(s_curve,label=cluster_n[i])
    ax[1].set_xlim(xmin=0,xmax=20)
    ax[1].legend()
    fig.savefig(save_name +'.pdf')
    plt.close(fig)
    
def analyze_latent(z_mtx,cluster_mtx):
    _, s_z_all, _ = z_mtx.svd()
    cluster_n = []
    cluster_s = []
    cluster_d = []
    for cluster_indx in cluster_mtx.unique():
        _, s_cluster, _ = z_mtx[cluster_mtx==cluster_indx,:].svd()
        s_cluster = s_cluster/s_cluster.max()
        cluster_n.append((cluster_mtx==cluster_indx).sum().item())
        cluster_s.append(s_cluster)
#         print(list(cluster_s))
        print(s_cluster)
#         s_diff = s_cluster[:-1] - s_cluster[1:]
#         cluster_d.append(s_diff.max(0)[1])
        cluster_d.append((s_cluster>0.01).sum())
    for i in range(len(cluster_n)):
        print('subspace {}, dimension {}, samples {}'.format(i,cluster_d[i],cluster_n[i]))