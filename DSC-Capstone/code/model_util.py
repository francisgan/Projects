import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
import pickle




class CNN(nn.Module):

    def __init__(self, n_class, n_seq, n_feature):
        super().__init__()

        self.layer1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(n_feature)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_feature)
        self.act2 = nn.ReLU()

        self.fc = nn.Linear(n_feature*n_seq, n_class)

    def forward(self, x):
        b,t,c = x.size()

        out = self.layer1(x.permute(0,2,1))
        
        out = self.act1(self.bn1(out))

        out = self.layer2(out)
        out = self.act2(self.bn2(out))

        logits = self.fc(out.reshape(b,-1))

        return logits
    
class DNN(nn.Module):

    def __init__(self, n_seq, n_feature, n_class):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(n_seq*n_feature, 2048)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout()

        self.fc2 = nn.Linear(2048, 1024)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(1024, 512)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(512, 256)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(256, 128)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(128, n_class)
        

    def forward(self, x):
        b, t, c = x.size()
        x = x.reshape(b, -1)
        out = self.dropout(self.act1(self.fc1(x)))
        out = self.act2(self.fc2(out))
        out = self.act3(self.fc3(out))
        out = self.act4(self.fc4(out))
        out = self.act5(self.fc5(out))
        out = self.fc6(out)

        return out
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.05)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_classes, num_heads, num_layers, max_seq_len, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))
        
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, x):
        x = self.feature_embedding(x) * self.scale.to(x.device) 
        pe = self.positional_encoding[:,:x.size(1),:].to(x.device)
        x = x + pe
        
        x = x.permute(1, 0, 2)
        
        x = self.transformer_encoder(x)
        
        x = x[0, :, :]
    
        out = self.classifier(x)
        return out