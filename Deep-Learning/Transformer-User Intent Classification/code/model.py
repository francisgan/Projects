import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from utils import layer_wise_learning_rate
from torchcontrib.optim import SWA

class IntentModel(nn.Module):
    def __init__(self, args, tokenizer, target_size, datasets=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        if datasets:
            num_training_steps = len(datasets) * args.n_epochs
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=100,
                                                num_training_steps=num_training_steps)
        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args, self.target_size)
    
    def model_setup(self, args):
        print(f"Setting up {args.model} model")

        # task1: get a pretrained model of 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    def forward(self, inputs, targets):
        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the output of the dropout layer to the Classifier which is provided for you.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        pooled_output = self.dropout(cls_token)
        return self.classify(pooled_output)

class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit


class CustomModel(IntentModel):

    def __init__(self, args, tokenizer, target_size,datasets=None):
        super().__init__(args, tokenizer, target_size,datasets=None)

        # Check if layer-wise learning rate (0) is specified
        if 0 in args.advanced:
            # Assuming layer_wise_learning_rate is a function you define that returns parameter groups with custom learning rates
            print("layer wise weight is applied")
            parameter_groups = layer_wise_learning_rate(args, self.encoder)  # This needs to be defined
            self.optimizer = optim.Adam(parameter_groups)
        else:
            print("original optimizer is created here")
            # Default optimizer setup
            self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
            
        if datasets:
            num_training_steps = len(datasets) * args.n_epochs
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=100,
                                                num_training_steps=num_training_steps)
class SupConModel(IntentModel):
    def __init__(self, args, tokenizer, target_size, feat_dim=768):
        super().__init__(args, tokenizer, target_size)
        
        # task1: initialize a linear head layer
        self.head = nn.Sequential(
                nn.Linear(feat_dim, args.CL_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(args.CL_hidden_size, args.embeded_size)
            )
        self.mask = True
        if args.CL_Mask ==0:
            self.mask = False

        
    def forward(self, inputs, targets):

        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the normalized output of the dropout layer to the linear head layer; return the embedding
        """
        # Task1: Feed the input to the encoder
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        if self.mask:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids)

        last_hidden_state = outputs.last_hidden_state
        
        # Task2: Extract <CLS> token and apply dropout
        cls_token = last_hidden_state[:, 0, :]  # Assuming the first token is the <CLS> token
        
        
        # TODO: add a dropout layer to the [CLS] token before the projection head 
        # in order to generate different embeddings for the same text input.
        pooled_output = self.dropout(cls_token)
        pooled_output1 = self.dropout(cls_token)
        
        # Task3: Normalize and feed to the linear head layer

#         normalized_output = F.normalize(pooled_output, p=2, dim=1)  # L2 normalization
#         normalized_output1 = F.normalize(pooled_output1, p=2, dim=1)  # L2 normalization

        # Without normalization, model performs better with lower loss.
        embeddings = self.head(pooled_output)
        embeddings1 = self.head(pooled_output1)

        return torch.stack((embeddings, embeddings1), dim=1), cls_token
