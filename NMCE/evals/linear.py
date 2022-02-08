import torch
import torch.nn as nn

from torch.utils import data
from torch.utils.data import DataLoader


def linear(args, train_features, train_labels, test_features, test_labels):
    train_data = tensor_dataset(train_features,train_labels)
    test_data = tensor_dataset(test_features,test_labels)
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=2)
    
    LL = nn.Linear(train_features.shape[1],args.n_classes)
    optimizer = torch.optim.SGD(LL.parameters(), lr=args.lr, momentum=args.momo, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    
    test_acc_list = []
    for epoch in range(args.epo):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch
            y_batch = y_batch
            
            logits = LL(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)

        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch
            y_batch = y_batch

            logits = LL(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        
        test_acc_list.append(top1_accuracy)
        
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    acc_vect = torch.tensor(test_acc_list)
    print('best linear test acc {}, last acc {}'.format(acc_vect.max().item(),acc_vect[-1].item()))
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
            
class tensor_dataset(data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.length = x.shape[0]
    
    def __getitem__(self,indx):
        return self.x[indx], self.y[indx]
    
    def __len__(self):
        return self.length

