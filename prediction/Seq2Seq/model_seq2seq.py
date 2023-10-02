import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_rnn_layers, drop_prob):
        super(EncoderRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, self.hidden_dim, nb_rnn_layers, batch_first=True, bidirectional=True, dropout=drop_prob)

    def forward(self, input):
        out, rnn_states = self.rnn(input)
        return rnn_states

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, hidden_dim, nb_rnn_layers, drop_prob):
        super(DecoderRNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.rnn_regression = nn.LSTM(output_dim, hidden_dim, nb_rnn_layers, batch_first=True, bidirectional=True,
                                      dropout=drop_prob)
        self.fc_regression = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, input_regression, init_rnn_state_regression):
        out_regression, rnn_states_regression = self.rnn_regression(input_regression, init_rnn_state_regression)

        logits_regression_seq = self.fc_regression(out_regression)

        return logits_regression_seq, rnn_states_regression

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, in_seq, out_seq, teacher_forcing_ratio=0.5):

        batch_size = out_seq.shape[0]
        out_seq_len = out_seq.shape[1]
        out_seq_dim = out_seq.shape[2]

        prev_rnn_states_regression = self.encoder(in_seq)

        logits_regression_seq = []
        next_input_regression = (-1) * torch.ones(batch_size, 1, out_seq_dim).to(self.device)

        for t in range(0, out_seq_len):
            out_regression, prev_rnn_states_regression = self.decoder(next_input_regression, prev_rnn_states_regression)
            logits_regression_seq.append(out_regression)
            use_teacher_force = random.random() < teacher_forcing_ratio
            if use_teacher_force:
                next_input_regression = out_seq[:, t, :].unsqueeze(dim=1)
            else:
                next_input_regression = out_regression

        logits_regression_seq = torch.cat(logits_regression_seq, dim=1)
        return logits_regression_seq

if __name__ == '__main__':
    # test
    device = torch.device('cuda', 0)
    encoder = EncoderRNN(8, 128, 2, 0.1).to(device)
    decoder = DecoderRNN(1, 128, 2, 0.1).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    input = torch.randn(256, 36, 8).to(device)
    output = torch.randn(256, 24, 1).to(device)
    output_pred = model.forward(input, output, teacher_forcing_ratio=0.0)

    print(output_pred.shape)