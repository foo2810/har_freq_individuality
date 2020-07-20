import sys
sys.path.append('./')

import torch

import optuna
import numpy as np
from pathlib import Path

from pamap2.utils import PAMAP2
from utils.train import training

from models.sample import SimpleCNN
from models.vgg import *
from models.resnet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Hyper Parameters
n_classes = 5
batch_size = 256
n_epochs = 200
frame_size = 256
activities = [1, 2, 3, 4, 5]
attributes = ['acc1']
positions = ['chest']

# dataset
print('[Dataset]')
pamap2 = PAMAP2('D:/datasets/PAMAP2/PAMAP2_Dataset/Protocol/', cache_dir='data_cache/org/')
ret = pamap2.framing(frame_size, None, activities, attributes, positions, axes)
frames, act_labels, sub_labels, cid2act, pid2name = ret
frames = np.transpose(frames, [0, 2, 1])
p = np.random.permutation(len(frames))
n_train = int(len(frames) * 0.5)
x_train, x_test = frames[p][:n_train], frames[p][n_train:]
y_train, y_test = act_labels[p][:n_train], act_labels[p][n_train:]
train_ds = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
test_ds = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader= torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))



model_list = {
    'simplecnn': SimpleCNN,
    'vgg11': vgg11, 'vgg16': vgg16, 'vgg19': vgg19,
    'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
}


for model_name in model_list:
    print('='*100)
    print('[{}]'.format(model_name))
    model = model_list[model_name]

    def get_lr(trial):
        lr = trial.suggest_loguniform('adam_lr', 1e-6, 1e-4)
        return lr

    def objective(trial):
        model = vgg11(in_channels=3, classes=n_classes)
        lr = get_lr(trial)

        hist = training(model, train_loader, test_loader, n_epochs, lr, batch_size, device, best_param_name=None)

        test_acc = np.array(hist['test_acc'])

        return 1 - test_acc.max()

    trial_size = 10
    study = optuna.create_study()
    study.optimize(objective, n_trials=trial_size)

    print('[Best params]')
    print(study.best_params)

    print('[Best values (error rate)]')
    print(study.best_value)

    print('[Search process]')
    fig = optuna.visualization.plot_optimization_history(study)
    img = fig.to_image('jpg')
    with open('search_process_{}.jpg'.format(model_name), 'wb') as fp:
        fp.write(img)

    fig2 = optuna.visualization.plot_intermediate_values(study)
    img2 = fig.to_image('jpg')
    with open('intermediate_values_{}.jpg'.format(model_name), 'wb') as fp:
        fp.write(img2)

    df = study.trials_dataframe()
    df = df.to_csv('history_of_searching_hparams_{}.csv'.format(model_name))
