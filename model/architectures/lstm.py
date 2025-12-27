import torch
import torch.nn as nn
from .base_model import BaseModel


class LSTMModel(BaseModel):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()

        # Store parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        #LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=dropout
        )

        #Dropout layer
        self.dropout = nn.Dropout(dropout)

        #Output layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        '''
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            out: Output tensor of shape (batch_size, output_size)
        ''' 
        # Pass through LSTM
        lstm_out, hidden_states = self.lstm(x)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_output)

        # Pass through linear layer
        out = self.fc(dropped)

        return out