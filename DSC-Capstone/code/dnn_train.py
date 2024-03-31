import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
import configs
import pickle
import model_util
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = configs.dnn_epoch
learning_rate= configs.dnn_learning_rate
train_size = configs.training_size_scale

with open ('feature', 'rb') as fp:
    data_feature = pickle.load(fp)
fp.close()

with open ('target', 'rb') as fp:
    data_target = pickle.load(fp)
fp.close()

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target

cap = int(len(data_feature)*train_size - 15000*train_size)
train_set = data_feature[:cap]
train_target = data_target[:cap]
valid_set = data_feature[-15000:-10000]
valid_target = data_target[-15000:-10000]
test_set = data_feature[-10000:]
test_target = data_target[-10000:]

valid_set = torch.Tensor(valid_set).to(device)
valid_target = torch.Tensor(valid_target).to(device)
test_set = torch.Tensor(test_set).to(device)
test_target = torch.Tensor(test_target).to(device)

train_dataset = CustomDataset(train_set, train_target)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = model_util.DNN(300, 52, 25).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epochs):
    model.train() 
    running_loss = []
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss =  criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
    
    loss = np.mean(running_loss)

    model.eval()
    pred = model(valid_set)
    vloss =  criterion(pred, valid_target)

    pred_index = torch.max(pred, 1)[1]
    correct = 0
    for j in range (len(valid_target)):
        if valid_target[j][pred_index[j]] == 1:
            correct +=1
    v_acc = correct/len(valid_target)
    model.train()

    print("Finish epoch {}, loss: {}, vloss: {}, vAcc: {}".format(i, loss, vloss, v_acc))
    

model.eval()
pred = model(test_set)
pred_index = torch.max(pred, 1)[1]
correct = 0
for i in range (len(test_target)):
    if test_target[i][pred_index[i]] == 1:
        correct +=1

print("accruracy: ", correct/len(test_target))

torch.save(model.state_dict(), "./models/dnn_model.pth")