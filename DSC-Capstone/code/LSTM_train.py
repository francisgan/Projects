import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
import pickle
import model_util
import configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = configs.LSTM_epoch
learning_rate= configs.LSTM_learning_rate
train_size = configs.LSTM_training_size_scale

with open ('feature', 'rb') as fp:
    data_feature = pickle.load(fp)
fp.close()

with open ('target', 'rb') as fp:
    data_target = pickle.load(fp)
fp.close()
train_size=1
cap = int(len(data_feature)*train_size - 15000*train_size)
train_set = data_feature[:cap]
train_target = data_target[:cap]
valid_set = data_feature[-15000:-10000]
valid_target = data_target[-15000:-10000]
test_set = data_feature[-10000:]
test_target = data_target[-10000:]
train_set = torch.Tensor(train_set).to(device)
train_target = torch.Tensor(train_target).to(device)
valid_set = torch.Tensor(valid_set).to(device)
valid_target = torch.Tensor(valid_target).to(device)
test_set = torch.Tensor(test_set).to(device)
test_target = torch.Tensor(test_target).to(device)

model = model_util.LSTMClassifier(52, 250, 25, 2).to(device)
criterion = nn.CrossEntropyLoss()

batch_size = configs.LSTM_batch_size
def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(epochs):
    model.train()
    running_loss = []
    for inputs, labels in get_batches(train_set, train_target, batch_size):
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # Ensure labels are long type for CrossEntropyLoss
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
    

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
    path = "./models/LSTM_"+str(i)+".pth"
    torch.save(model.state_dict(), path)
    print("Finish epoch {}, loss: {}, vloss: {}, vAcc: {}".format(i, np.mean(running_loss), vloss, v_acc))

model.eval()
pred = model(test_set)
pred_index = torch.max(pred, 1)[1]
correct = 0
for i in range (len(test_target)):
    if test_target[i][pred_index[i]] == 1:
        correct +=1

print("accruracy: ", correct/len(test_target))

torch.save(model.state_dict(), "./models/LSTM.pth")