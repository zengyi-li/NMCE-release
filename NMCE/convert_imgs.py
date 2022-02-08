import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse

from PIL import Image
    
parser = argparse.ArgumentParser(description='Unsupervised Learning')
parser.add_argument('--COIL_20_path', type=str, default=None,
                    help='path to COIL-20')
parser.add_argument('--COIL_100_path', type=str, default=None,
                    help='path to COIL-100')
args = parser.parse_args()


def find_num(in_str):
    #find integer number from a string and return it
    out_str = ''
    num_list = [str(i) for i in range(10)]
    for char in in_str:
        if char in num_list:
            out_str += char
    return int(out_str)

if args.COIL_20_path:
    img_data_list = []
    label_list = []
    img_list = os.listdir(args.COIL_20_path)
    for img_file in img_list:
        if img_file.endswith('.png'):
            img = plt.imread(os.path.join(args.COIL_20_path,img_file))
            img_data_list.append(255*img)
            str_list = img_file.split('__')
            obj_id = find_num(str_list[0])
            label_list.append(obj_id)
            
    img_mtx = torch.tensor(img_data_list,dtype=torch.uint8)
    img_mtx = img_mtx.unsqueeze(1)
    label_mtx = torch.tensor(label_list)
    data_dict = {
    'name': 'COIL20',
    'img_mtx': img_mtx,
    'label_mtx': label_mtx
    }
    torch.save(data_dict,'COIL20.pth')
    
if args.COIL_100_path:
    img_data_list = []
    label_list = []
    img_list = os.listdir(args.COIL_100_path)
    for img_file in img_list:
        if img_file.endswith('.png'):
            img = plt.imread(os.path.join(args.COIL_100_path,img_file))
            img_data_list.append(255*img.mean(2))
            str_list = img_file.split('__')
            obj_id = find_num(str_list[0])
            label_list.append(obj_id)
            
    img_mtx = torch.tensor(img_data_list,dtype=torch.uint8)
    img_mtx = img_mtx.unsqueeze(1)
    label_mtx = torch.tensor(label_list)
    data_dict = {
    'name': 'COIL100',
    'img_mtx': img_mtx,
    'label_mtx': label_mtx
    }
    torch.save(data_dict,'COIL100.pth')