import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        model = []
        model.append(nn.Linear(input_size + hidden_size, hidden_size))
        model.append(nn.ReLU())
        for i in range(num_layers):
            model.append(nn.Linear(hidden_size, hidden_size))
            model.append(nn.ReLU())
            model.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*model)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = self.model(combined)
        output = self.output_layer(hidden)
        output = F.relu(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(torch.empty(1, self.hidden_size))


class LSTM(nn.Module):
    def __init__(
        self, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout=0
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        # self.embedding = nn.Linear(input_size, embedding_dim)
        self.num_layers = num_layers
        self.model = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, x_len):
        # x = self.embedding(x)
        packed_input = pack_padded_sequence(
            x, x_len, batch_first=True, enforce_sorted=False
        )
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        hidden, _ = self.model(packed_input, (h0, c0))
        output, _ = pad_packed_sequence(hidden, batch_first=True)
        output = self.output_layer(output[:, -1, :])
        output = F.relu(output)
        return output
