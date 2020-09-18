# LPFやHPFを通したとき最終的なsliding-windowがどうなっているのかを調査

import torch

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fftpack import fft, fftfreq

from pamap2.utils import PAMAP2, POSITIONS, AXES, PERSONS, ACTIVITIES
from pamap2.preprocessing import lpf


def get_dataset(train_activities, test_activities, frame_size, attributes, positions, axes, preprocesses=None):
    if preprocesses is None: preprocesses = list()
    subjects = [
        'subject101', 'subject102',
        'subject105', 'subject106',
        'subject108',
        # 'subject103', 'subject104', 'subject107',
    ]

    pamap2 = PAMAP2('D:/datasets/PAMAP2/PAMAP2_Dataset/Protocol/', cache_dir='data_cache/org/')
    ret = pamap2.framing(frame_size, subjects, train_activities, attributes, positions, axes, preprocesses)
    x_train, _, y_train, train_cid2act, train_pid2name = ret
    x_train = np.transpose(x_train, [0, 2, 1])
    print('[Train Dataset]')
    print(train_pid2name)
    flg = False
    for sub_id in range(len(subjects)):
        sub = train_pid2name[sub_id]
        print('{}({}): {}'.format(sub, sub_id, np.sum(y_train == sub_id)))
        if sub_id not in y_train:
            flg = True 
            print(' >>> [Warning] Subject(label id {}, name {}) not found in train dataset'.format(sub_id, sub))
    if flg:
        raise RuntimeError('Activity classes are not enough.')

    ret = pamap2.framing(frame_size, subjects, test_activities, attributes, positions, axes, preprocesses)
    x_test, _, y_test, test_cid2act, test_pid2name = ret
    x_test= np.transpose(x_test, [0, 2, 1])
    print('[Test Dataset]')
    print(test_pid2name)
    flg = False
    for sub_id in range(len(subjects)):
        sub = test_pid2name[sub_id]
        print('{}({}): {}'.format(sub, sub_id, np.sum(y_test == sub_id)))
        if sub_id not in y_test:
            flg = True 
            print(' >>> [Warning] Subject(label id {}, name {}) not found in train dataset'.format(sub_id, sub))
    if flg:
        raise RuntimeError('Activity classes are not enough.')

    train_ds = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader= torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print('x_train: {}'.format(x_train.shape))
    print('y_train: {}'.format(y_train.shape))
    print('x_test: {}'.format(x_test.shape))
    print('y_test: {}'.format(y_test.shape))

    return train_loader, test_loader

# Hyper Parameters
n_classes = 8
batch_size = 256
n_epochs = 300
frame_size = 256
activities = np.array([1, 2, 3, 4, 5])
attributes = ['acc1']
positions = copy.deepcopy(POSITIONS)
axes = copy.deepcopy(AXES)
in_channels = len(positions) * len(axes)
all_persons = np.array(PERSONS)

for preprocessed in [False, True]:
    for act in activities:
        print('<act_id {}>'.format(act))
        if preprocessed:
            pp = [lambda x: lpf(x, 50, 100)]
        else:
            pp = None
        test_activity = [act]
        train_activities = activities[activities != act]
        train_loader, test_loader = get_dataset(train_activities, test_activity, frame_size, attributes, positions, axes, pp)
        print('Train activities: {}'.format(train_activities))
        print('Test activities: {}'.format(test_activity))

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.numpy(), labels.numpy()
            for l, data in zip(labels, inputs):
                y = data[0]
                if l == 3 or l == 4: break
            x = fftfreq(len(y), 1./100.)
            yf = fft(y.copy())
            yf /= (len(y) / 2.)
            yf[0] /= 2.
            x = x[1:int(len(y)/2)]
            yf = np.abs(yf)[1:int(len(y)/2)]
            y = yf

            # x = list(range(len(y)))

            plt.clf()
            plt.figure(figsize=(7, 7))
            plt.plot(x, y)
            plt.savefig('{}_{}_{}.png'.format(act, preprocessed, i))
            break

        del train_loader, test_loader

        print('='*100)
