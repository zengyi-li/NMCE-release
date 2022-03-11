import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Resnet10MNIST, Resnet10imgs, Resnet18imgs, Resnet18CIFAR, Resnet18STL10, Resnet34CIFAR, Resnet34STL10

def get_backbone(backbone_name):
    if backbone_name=='resnet10mnist':
        feature_dim, backbone = 512, Resnet10MNIST()
    elif backbone_name=='resnet10imgs':
        feature_dim, backbone = 256, Resnet10imgs()
    elif backbone_name=='resnet18imgs':
        feature_dim, backbone = 256, Resnet18imgs()
    elif backbone_name=='resnet18cifar':
        feature_dim, backbone = 512, Resnet18CIFAR()
    elif backbone_name=='resnet18stl10':
        feature_dim, backbone = 512, Resnet18STL10()
    elif backbone_name=='resnet34cifar':
        feature_dim, backbone = 512, Resnet34CIFAR()
    elif backbone_name=='resnet34stl10':
        feature_dim, backbone = 512, Resnet34STL10()
    else:
        raise NameError("{} not found in transform loader".format(backbone_name))
    
    return feature_dim, backbone

class Gumble_Softmax(nn.Module):
    def __init__(self,tau, straight_through=False):
        super().__init__()
        self.tau = tau
        self.straight_through = straight_through
    
    def forward(self,logits):
        logps = torch.log_softmax(logits,dim=1)
        gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
        logits = logps + gumble
        out = (logits/self.tau).softmax(dim=1)
        if not self.straight_through:
            return out
        else:
            out_binary = (logits*1e8).softmax(dim=1).detach()
            out_diff = (out_binary - out).detach()
            return out_diff + out

class SubspaceClusterNetwork(nn.Module):
    def __init__(self,backbone_name,z_dim,n_clusters,norm_p=2,normalize=True):
        super().__init__()
        feature_dim, backbone = get_backbone(backbone_name)
        self.backbone = backbone
        self.norm_p = norm_p
        self.normalize = normalize
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim,4096),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU()
                                        )
        
        self.cluster = nn.Linear(4096,n_clusters)
#         self.cluster.weight = nn.Parameter(torch.zeros_like(self.cluster.weight))
        self.subspace = nn.Linear(4096,z_dim)
#         self.subspace.weight = nn.Parameter(torch.zeros_like(self.subspace.weight))
        
    def forward(self,x, detach_feature=False, return_feature='clustering'):
        if detach_feature:
            with torch.no_grad():
                feature = self.backbone(x)
                pre_feature = self.pre_feature(feature)
        else:
            feature = self.backbone(x)
            pre_feature = self.pre_feature(feature)
        
        logits = self.cluster(pre_feature)
        z = self.subspace(pre_feature)
        if self.normalize:
            z = F.normalize(z,p=self.norm_p)
        
        if return_feature=='clustering':
            return z, logits
        elif return_feature=='pool':
            return feature
        elif return_feature=='pre_feature':
            return pre_feature
        elif return_feature=='subspace':
            return z
    

class SelfSupervisedNetwork(nn.Module):
    def __init__(self,backbone_name,z_dim,classifier=False,n_classes=10,normalize=True,norm_p=2):
        super().__init__()
        feature_dim, backbone = get_backbone(backbone_name)
        self.backbone = backbone
        self.norm_p = norm_p
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim,4096),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU()
                                        )
        self.classifier = classifier
        self.normalize = normalize
        if classifier:
            self.classifier = nn.Linear(feature_dim,n_classes)
        else:
            self.projection = nn.Linear(4096,z_dim)
    
    def forward(self,x,detach_feature=False,return_feature=None):
        #return_feature argument here to make it the same as SubspaceClusterNetwork
        if detach_feature:
            with torch.no_grad():
                feature = self.backbone(x)
        else:
            feature = self.backbone(x)
                
        if self.classifier:
            logits = self.classifier(feature)
            return logits
        else:
            if self.normalize:
                z = F.normalize(self.projection(self.pre_feature(feature)),p=self.norm_p)
                return z
            else:
                return self.projection(self.drop(self.pre_feature(feature)))
        
        
    