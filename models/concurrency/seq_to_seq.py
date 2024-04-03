import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def xavier_init(m):
    if isinstance(m, nn.Module):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)


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
        self,
        input_size,
        embedding_dim,
        hidden_size,
        output_size,
        num_layers,
        dropout=0.1,
        last_output=True,
        use_seperation=False,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.bn = nn.BatchNorm1d(input_size)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )
        xavier_init(self.embedding)
        self.num_layers = num_layers
        self.model = nn.LSTM(
            embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        xavier_init(self.model)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.last_output = last_output
        self.use_seperation = use_seperation
        xavier_init(self.output_layer)

    def model_forward(self, x, x_len):
        x = torch.transpose(x, 1, 2)
        x = self.bn(x)
        x = torch.transpose(x, 1, 2)
        x = self.embedding(x)
        packed_input = pack_padded_sequence(
            x, x_len, batch_first=True, enforce_sorted=False
        )
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        hidden, _ = self.model(packed_input, (h0, c0))
        output, _ = pad_packed_sequence(hidden, batch_first=True)
        if self.last_output:
            output = output[torch.arange(len(x_len)), x_len - 1]
        else:
            output = torch.mean(output, dim=1)
        output = self.output_layer(output)
        return output

    def model_forward_with_seperation(self, x, x_len, pre_info_length):
        y_prime = self.model_forward(x, x_len)
        return y_prime

    def forward(self, x, x_len, pre_info_length=None):
        if pre_info_length is not None and self.use_seperation:
            return self.model_forward_with_seperation(x, x_len, pre_info_length)
        else:
            return self.model_forward(x, x_len)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model), nn.Linear(d_model, d_model)
        )
        self.d_model = d_model
        self.linear = nn.Linear(d_model, output_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(0, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(0, initrange)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(
            src, src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.linear(output)
        return output
