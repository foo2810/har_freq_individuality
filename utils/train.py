import torch
import torch.nn as nn

def training(model, train_loader, test_loader, n_epochs, lr, batch_size, device, best_param_name='best_param.pt'):
    # weights = torch.tensor([np.sum(act_labels == i) for i in range(5)], dtype=torch.float).to(device)
    # weights /= weights.max(0)[0]
    # weights = 1 / weights
    weights = None
    criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_acc = 0

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

        if best_param_name is not None and best_acc < test_acc:
            torch.save(model.state_dict(), best_param_name)

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

        hist = {
            'train_loss': train_loss_list, 'train_acc': train_acc_list,
            'test_loss': test_loss_list, 'test_acc': test_acc_list,
        }

    return hist
