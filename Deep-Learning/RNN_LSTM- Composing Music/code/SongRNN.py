import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *


class SongRNN(nn.Module):
    def __init__(self, input_size, output_size, config, device):
        """
        Initialize the SongRNN model.
        """
        super(SongRNN, self).__init__()

        HIDDEN_SIZE = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        MODEL_TYPE = config["model_type"]
        DROPOUT_P = config["dropout"]

        self.model_type = MODEL_TYPE
        self.input_size = input_size
        self.hidden_size = HIDDEN_SIZE
        self.output_size = output_size
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT_P
        self.device = device
        
        """
        Complete the code

        TODO: 
        (i) Initialize embedding layer with input_size and hidden_size
        (ii) Initialize the recurrent layer based on model type (i.e., LSTM or RNN) using hidden size and num_layers
        (iii) Initialize linear output layer using hidden size and output size
        (iv) Initialize dropout layer with dropout probability
        """

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers)
        elif self.model_type == 'rnn':
            self.rnn = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers)
        else:
            raise ValueError("Unsupported RNN model type")

        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.hidden = None

    def init_hidden(self):
        """
        Initializes the hidden state for the recurrent neural network.

        TODO:
        Check the model type to determine the initialization method:
        (i) If model_type is LSTM, initialize both cell state and hidden state.
        (ii) If model_type is RNN, initialize the hidden state only.

        Initialise with zeros.
        """
        if self.model_type == 'lstm':
            hidden_state = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
            cell_state = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
            self.hidden = (hidden_state, cell_state)
        elif self.model_type == 'rnn':
            hidden_state = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
            self.hidden = hidden_state
        else:
            raise ValueError("Unsupported RNN model type")
        
    def forward(self, seq):
        """
        Forward pass of the SongRNN model.
        (Hint: In teacher forcing, for each run of the model, input will be a single character
        and output will be pred-probability vector for the next character.)

        Parameters:
        - seq (Tensor): Input sequence tensor of shape (seq_length)

        Returns:
        - output (Tensor): Output tensor of shape (output_size)
        - activations (Tensor): Hidden layer activations to plot heatmap values


        TODOs:
        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        (iii) Apply dropout (if needed)
        (iv) Pass through the linear output layer
        """
        embedded = self.embedding(seq)
        rnn_out, self.hidden = self.rnn(embedded, self.hidden)
        out = self.dropout_layer(rnn_out)
        output = self.fc(out)
        return output, rnn_out
        
        
        