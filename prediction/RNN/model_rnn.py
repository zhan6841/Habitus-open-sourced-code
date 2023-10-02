import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nb_rnn_layers, drop_prob, tput_step, device):
        super(RNNNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, nb_rnn_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.bins = torch.zeros([output_dim, 1], dtype=torch.float)
        for i in range(1, output_dim - 1):
            self.bins[i][0] = tput_step*(2.0*i + 1.0)/2.0
        self.bins[0][0] = 0.0
        self.bins[output_dim-1][0] = tput_step*(output_dim - 1)
        self.bins = self.bins.to(device)

    def forward(self, input):
        out, rnn_states = self.rnn(input)
        out = self.fc(out)
        out_min = torch.min(out, dim=1, keepdim=True)[0]
        out = out - out_min
        out_sum = torch.sum(out, dim=1, keepdim=True)
        out = torch.div(out, out_sum)
        out = torch.matmul(out, self.bins)
        out = out[:, 0, :]
        return out.reshape(out.shape[0], -1, out.shape[-1])

if __name__ == '__main__':
    # test RNNNet
    # input_dim = 1+6+1
    # hidden_dim = 20
    # output_dim = 41
    # rnn layer = 3
    # drop_prob = 0.1
    # tput_step = 50.0
    myNet = RNNNet(8, 20, 41, 3, 0.1, 50.0, torch.device('cpu'))
    input = torch.randn(20, 20, 8)
    output = myNet(input)
    print(output.shape)
    # print(output)