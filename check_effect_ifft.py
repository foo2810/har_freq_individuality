import torch

import copy
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal
from pathlib import Path

import pickle
import matplotlib.pyplot as plt

from pamap2.utils import PAMAP2, POSITIONS, AXES, PERSONS
from utils.train import training2

from models.vgg import *

def get_dataset(train_persons, test_persons, frame_size, activities, attributes, positions, axes, preprocesses):
    pamap2 = PAMAP2('D:/datasets/PAMAP2/PAMAP2_Dataset/Protocol/', cache_dir='data_cache/org/')
    ret = pamap2.framing(frame_size, train_persons, activities, attributes, positions, axes, preprocesses)
    x_train, y_train, sub_labels, cid2act, pid2name = ret
    x_train = np.transpose(x_train, [0, 2, 1])
    print(cid2act)
    flg = False
    for lid in range(len(activities)):
        if lid not in y_train:
            flg = True 
            print(' >>> [Warning] Activity(label id {}) not found in train dataset'.format(lid))
    if flg:
        raise RuntimeError('Activity classes are not enough.')

    ret = pamap2.framing(frame_size, test_persons, activities, attributes, positions, axes, preprocesses)
    x_test, y_test, sub_labels, cid2act, pid2name = ret
    x_test= np.transpose(x_test, [0, 2, 1])
    print(cid2act)
    flg = False
    for lid in range(len(activities)):
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

def fft_ifft_np(y):
    yf = np.fft.fft(y)
    yf /= (len(y)/2.)
    freq = np.fft.fftfreq(len(y), 1./100.)
    # yf[0] = yf[0] / 2.
    yf = yf.copy()
    # yf[freq < 0] = 0

    yd = np.fft.ifft(yf)
    yd = np.real(yd * len(y) / 2)

    return yd


def fft_ifft(y):
    yf = fft(y)
    # yf /= (len(y)/2.)
    freq = fftfreq(len(y), 1./100.)
    # yf[0] = yf[0] / 2.
    yf = yf.copy()
    # yf[freq < 0] = 0

    yd = ifft(yf)
    # yd = np.real(yd * len(y) / 2)
    yd = np.real(yd)

    return yd


# 姿勢情報はのせていない
def analysis_spectrum(y, fs=100, fname='spectrum.png'):
    yf = fft(y).copy()
    freq = fftfreq(len(y), 1./fs)
    yf /= (len(y) / 2.)
    yf[0] /= 2.

    x_freq = freq[1:int(len(y)/2)]
    spectrum = np.abs(yf)[1:int(len(y)/2)]
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.plot(x_freq, spectrum)
    plt.savefig(fname)
    plt.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Hyper Parameters
n_classes = 5
batch_size = 256
n_epochs = 200
frame_size = 256
activities = [1, 2, 3, 4, 5]
attributes = ['acc1']
positions = copy.deepcopy(POSITIONS)
axes = copy.deepcopy(AXES)
in_channels = len(positions) * len(axes)
all_persons = np.array(PERSONS)


# with open('data_cache/org/segments_subject101.pkl', 'rb') as fp:
#     segments = pickle.load(fp)

# # columns = ['IMU_ankle_acc1_x', 'IMU_ankle_acc1_y', 'IMU_ankle_acc1_z']
# for i, seg in enumerate(segments):
#     seg = np.array(seg['IMU_ankle_acc1_x'])
#     plt.clf()
#     plt.figure(figsize=(20, 10))
#     # X = list(range(len(seg)))
#     N = 1000
#     X = list(range(N))
#     plt.plot(X, seg[:N])
#     plt.savefig('ankle_acc1_x_normal_{}.png'.format(i))
#     plt.close()

#     analysis_spectrum(seg.copy(), fs=100, fname='spectrum_{}.png'.format(i))

#     seg_fft = lpf(seg, 10, 100)
#     plt.clf()
#     plt.figure(figsize=(20, 10))
#     plt.plot(X, seg_fft[:N])
#     plt.savefig('ankle_acc1_x_lpf_{}.png'.format(i))
#     plt.close()

#     analysis_spectrum(seg_fft.copy(), fs=100, fname='spectrum_lpf_{}.png'.format(i))

#     seg_fft = hpf(seg, 10, 100)
#     plt.clf()
#     plt.figure(figsize=(20, 10))
#     plt.plot(X, seg_fft[:N])
#     plt.savefig('ankle_acc1_x_hpf_{}.png'.format(i))
#     plt.close()

#     analysis_spectrum(seg_fft.copy(), fs=100, fname='spectrum_hpf_{}.png'.format(i))

#     seg_fft = bpf(seg, [10, 20], 100)
#     plt.clf()
#     plt.figure(figsize=(20, 10))
#     plt.plot(X, seg_fft[:N])
#     plt.savefig('ankle_acc1_x_bpf_{}.png'.format(i))
#     plt.close()

#     analysis_spectrum(seg_fft.copy(), fs=100, fname='spectrum_bpf_{}.png'.format(i))


#     # print(seg)
#     # print(seg_fft)
#     # # print(np.where(np.abs(seg-seg_fft) != 0)[0])
#     # diff = np.abs(seg-seg_fft)
#     # print(len(np.where(diff != 0)[0]))
#     # print('Max error: {}'.format(diff.max()))
#     # print('Min error: {}'.format(diff.min()))
#     # print('='*50)


args = {'in_channels' :in_channels, 'num_classes': n_classes}

n_samples = 10

for person in all_persons:
    print('<{}>'.format(person))
    for cnt in range(n_samples):
        print('[{}]'.format(cnt))
        for use_fft in [True, False]:
            print('(use fft: {})'.format(use_fft))
            test_persons = [person]
            train_persons = all_persons[all_persons != person]
            if use_fft:
                preprocess = [lambda x: fft_ifft(x)]
            else:
                preprocess = []
            train_loader, test_loader = get_dataset(train_persons, test_persons, frame_size, activities, attributes, positions, axes, preprocess)

            net = vgg11(**args).to(device)

            # optimizer = torch.optim.SGD(net.parameters(), lr=1e-7, momentum=0.8)#, weight_decay=1e-3)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch=-1)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 50, 100], gamma=0.1)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
            scheduler = None

            hist = training2(net, train_loader, test_loader, n_epochs, optimizer, scheduler, device, best_param_name=None)

            fname = '{}_{}_{}_{}.csv'.format('vgg11', person, 'preprocessed' if use_fft else 'normal', cnt)
            pd.DataFrame(hist).to_csv(fname)
