import sys
sys.path.append('./')

import torch

import copy
import numpy as np
import pandas as pd
from pathlib import Path

from pamap2.utils import PAMAP2, POSITIONS, AXES, PERSONS
from utils.train import training2

from models.sample import SimpleCNN
from models.vgg import *
from models.resnet import *

def get_dataset(train_persons, test_persons, frame_size, activities, attributes, positions, axes):
    pamap2 = PAMAP2('D:/datasets/PAMAP2/PAMAP2_Dataset/Protocol/', cache_dir='data_cache/org/')
    ret = pamap2.framing(frame_size, train_persons, activities, attributes, positions, axes)
    x_train, y_train, sub_labels, cid2act, pid2name = ret
    x_train = np.transpose(x_train, [0, 2, 1])
    print('[Train Dataset]')
    print(cid2act)
    flg = False
    for lid in range(len(activities)):
        act = cid2act[lid]
        print('{}({}): {}'.format(act, lid, np.sum(y_train == lid)))
        if lid not in y_train:
            flg = True 
            print(' >>> [Warning] Activity(label id {}) not found in train dataset'.format(lid))
    if flg:
        raise RuntimeError('Activity classes are not enough.')

    ret = pamap2.framing(frame_size, test_persons, activities, attributes, positions, axes)
    x_test, y_test, sub_labels, cid2act, pid2name = ret
    x_test= np.transpose(x_test, [0, 2, 1])
    print('[Test Dataset]')
    print(cid2act)
    flg = False
    for lid in range(len(activities)):
        act = cid2act[lid]
        print('{}({}): {}'.format(act, lid, np.sum(y_test == lid)))
        if lid not in y_train:
            flg = True 
            print(' >>> [Warning] Activity(label id {}) not found in train dataset'.format(lid))
    if flg:
        raise RuntimeError('Activity classes are not enough.')

    train_ds = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader= torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print('x_train: {}'.format(x_train.shape))
    print('y_train: {}'.format(y_train.shape))
    print('x_test: {}'.format(x_test.shape))
    print('y_test: {}'.format(y_test.shape))

    return train_loader, test_loader



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Hyper Parameters
n_classes = 5
batch_size = 256
n_epochs = 300
frame_size = 256
activities = [1, 2, 3, 4, 5]
attributes = ['acc1']
positions = copy.deepcopy(POSITIONS)
axes = copy.deepcopy(AXES)
in_channels = len(positions) * len(axes)
all_persons = np.array(PERSONS)


model_list = {
    # 'simplecnn': (SimpleCNN, {'in_shape': (in_channels, frame_size), 'n_classes': n_classes}),
    'vgg11': vgg11, #'vgg16': vgg16, 'vgg19': vgg19,
    'resnet18': resnet18, #'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
}


for model_name in model_list:
    print('='*100)
    print('[{}]'.format(model_name))
    model = model_list[model_name]
    if type(model) is tuple:
        model, args = model
    else:
        args = {'in_channels' :in_channels, 'num_classes': n_classes}
    
    for person in all_persons:
        print('<{}>'.format(person))
        test_persons = [person]
        train_persons = all_persons[all_persons != person]
        train_loader, test_loader = get_dataset(train_persons, test_persons, frame_size, activities, attributes, positions, axes)

        net = model(**args).to(device)

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
        # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 50, 100], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
        scheduler = None
        hist = training2(net, train_loader, test_loader, n_epochs, criterion, optimizer, scheduler, device, best_param_name=None)

        pd.DataFrame(hist).to_csv('{}_{}.csv'.format(model_name, person))
