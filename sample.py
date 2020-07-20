import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from pamap2.utils import PAMAP2
from models.sample import SimpleCNN
from models.vgg import vgg11

frame_size = 256
activities = [1, 2, 3, 4, 5]
attributes = ['acc1']
positions = ['chest']
axes = ['x', 'y', 'z']

n_epochs = 100
batch_size = 256
lr = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Datasets
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
print('in_shape: {}'.format(x_train.shape[1:]))
print('n_train: {}'.format(n_train))
print('n_test: {}'.format(len(frames)-n_train))

# Models
# model = SimpleCNN(x_train.shape[1:], n_classes=5).to(device)
model = vgg11(in_channels=3, num_classes=5).to(device)

# Training
# weights = torch.tensor([np.sum(act_labels == i) for i in range(5)], dtype=torch.float).to(device)
# weights /= weights.max(0)[0]
# weights = 1 / weights
weights = None
print('weights: {}'.format(weights))
criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch in range(1, n_epochs+1):
    train_loss = train_acc = 0
    test_loss = test_acc = 0

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item() * inputs.size(0)
        train_acc += torch.sum(torch.max(outputs, dim=1)[1] == labels).item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += torch.sum(torch.max(outputs, dim=1)[1] == labels).item()

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)

    train_loss_list += [train_loss]
    train_acc_list += [train_acc]
    test_loss_list += [test_loss]
    test_acc_list += [test_acc]

    template = 'Epoch({}/{}) loss: {:.3f}, acc: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}'
    print(template.format(
        epoch, n_epochs,
        train_loss, train_acc,
        test_loss, test_acc
    ))


