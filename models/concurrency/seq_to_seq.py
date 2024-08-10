import torch
import math
from typing import Optional, Union, List
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence


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
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, output_size),
        )

        self.last_output = last_output
        self.use_seperation = use_seperation
        xavier_init(self.output_layer)

    def model_forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        x_len: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
        is_padded: bool = True,
    ):
        if not is_padded or x_len is None:
            seq_lengths = [len(seq) for seq in x]
            x = pad_sequence(x, batch_first=True, padding_value=0)
            x_len = torch.tensor(seq_lengths, dtype=torch.long)
        if x.shape[1] > 1:
            x = torch.transpose(x, 1, 2)
            x = self.bn(x)
            x = torch.transpose(x, 1, 2)
        x = self.embedding(x)
        packed_input = pack_padded_sequence(
            x, x_len, batch_first=True, enforce_sorted=False
        )
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, (hn, cn) = self.model(packed_input, (h0, c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        if self.last_output:
            output = output[torch.arange(len(x_len)), x_len - 1]
        else:
            y = torch.zeros((len(output), output.shape[-1]), requires_grad=False)
            for i in range(len(output)):
                y[i] = torch.mean(output[i, : int(x_len[i]), :], dim=0)
            output = y
        output = self.output_layer(output)
        return output, (hn, cn)

    def model_forward_with_seperation(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        x_len: Optional[torch.Tensor] = None,
        pre_info_length: Optional[torch.Tensor] = None,
        is_padded: bool = True,
    ):
        """
        Seperate queries before and after
        :param x: input feature sequence
        :param x_len: can ignore
        :param pre_info_length:
        :return:
        """
        if not is_padded or x_len is None:
            seq_lengths = [len(seq) for seq in x]
            x = pad_sequence(x, batch_first=True, padding_value=0)
            x_len = torch.tensor(seq_lengths, dtype=torch.long)

        pre_info_x = []
        avg_rt = []
        post_info_x = []
        post_info_len = []
        zero_idx = []
        non_zero_idx = []
        for i in range(len(x)):
            pre_info_l = max(int(pre_info_length[i]), 1)
            pre_info_x.append(x[i, :pre_info_l, :])
            avg_rt.append(float(x[i][0][0]))
            post_info_l = int(x_len[i]) - pre_info_l
            if post_info_l <= 0:
                zero_idx.append(i)
            else:
                non_zero_idx.append(i)
                post_info_x.append(x[i, pre_info_l:, :])
                post_info_len.append(post_info_l)
        # pad pre and post info
        avg_rt = torch.tensor(avg_rt, requires_grad=False).reshape(-1, 1)
        pre_seq_lengths = torch.tensor([len(x) for x in pre_info_x], dtype=torch.long)
        padded_pre_info_x = pad_sequence(pre_info_x, batch_first=True, padding_value=0)
        y_prime, (hn, cn) = self.model_forward(padded_pre_info_x, pre_seq_lengths)
        y_prime = y_prime * avg_rt / 3
        if len(non_zero_idx) == 0:
            # very unlikely that a whole batch has no post info
            return y_prime

        hn = hn[:, non_zero_idx, :]
        cn = cn[:, non_zero_idx, :]

        new_post_info_x = []
        for i in range(len(non_zero_idx)):
            curr_post_info_x = post_info_x[i].clone()
            for j in range(len(curr_post_info_x)):
                curr_post_info_x[j, 0] = y_prime[non_zero_idx[i]]
            new_post_info_x.append(curr_post_info_x)

        post_seq_lengths = torch.tensor(post_info_len, dtype=torch.long)
        padded_post_info_x = pad_sequence(
            new_post_info_x, batch_first=True, padding_value=0
        )
        y, _ = self.model_forward(padded_post_info_x, post_seq_lengths, hn, cn)
        y = y * y_prime[non_zero_idx]
        output = torch.zeros((len(y_prime), 1), requires_grad=False)
        if len(zero_idx) != 0:
            output[zero_idx] = y_prime[zero_idx]
        output[non_zero_idx] = y
        return output

    def model_forward_pre_info(self, x, pre_info_length):
        pre_info_x = []
        for i in range(len(x)):
            pre_info_l = max(int(pre_info_length[i]), 1)
            pre_info_x.append(x[i, :pre_info_l, :])
        pre_seq_lengths = torch.tensor([len(x) for x in pre_info_x], dtype=torch.long)
        padded_pre_info_x = pad_sequence(pre_info_x, batch_first=True, padding_value=0)
        y_prime, _ = self.model_forward(padded_pre_info_x, pre_seq_lengths)
        return y_prime

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        x_len: Optional[torch.Tensor] = None,
        pre_info_length: Optional[torch.Tensor] = None,
        is_padded: bool = True,
    ):
        if pre_info_length is not None and self.use_seperation:
            return self.model_forward_with_seperation(
                x, x_len, pre_info_length, is_padded
            )
        else:
            output, _ = self.model_forward(x, x_len, is_padded=is_padded)
            return output


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
        last_output: bool = False,
        use_seperation: bool = False,
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
        self.output_layer = nn.Linear(d_model, output_size)
        self.last_output = last_output
        self.use_seperation = use_seperation
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.bias.data.zero_()
        self.embedding.weight.data.uniform_(0, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(0, initrange)

    def model_forward(
        self,
        src: torch.Tensor,
        x_len: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ):
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
        # Todo: implement this
        src_key_padding_mask = x_len
        output = self.transformer_encoder(
            src, src_mask, src_key_padding_mask=src_key_padding_mask
        )
        if self.last_output:
            output = output[torch.arange(len(x_len)), x_len - 1]
        else:
            output = torch.mean(output, dim=1)

        output = self.output_layer(output)

        return output

    def forward(
        self,
        src: torch.Tensor,
        x_len: torch.Tensor = None,
        pre_info_length: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return
