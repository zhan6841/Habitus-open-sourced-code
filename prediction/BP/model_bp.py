import torch
import torch.nn as nn
import torch.nn.functional as F

class BPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tput_step, device):
        super(BPNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.bins = torch.zeros([output_dim, 1], dtype=torch.float)
        for i in range(1, output_dim - 1):
            self.bins[i][0] = tput_step*(2.0*i + 1.0)/2.0
        self.bins[0][0] = 0.0
        self.bins[output_dim-1][0] = tput_step*(output_dim - 1)
        self.bins = self.bins.to(device)

    def forward(self, input):
        out = input.view(-1, self.input_dim)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out_min = torch.min(out, dim=1, keepdim=True)[0]
        out = out - out_min
        out_sum = torch.sum(out, dim=1, keepdim=True)
        out = torch.div(out, out_sum)
        out = torch.matmul(out, self.bins)
        return out.reshape(out.shape[0], -1, out.shape[-1])

if __name__== '__main__':
    # test BPNet
    # input_dim = 8+6+1
    # hidden_dim = 40
    # output_dim = 41
    # tput_step = 50.0
    myNet = BPNet(15, 40, 41, 50.0, torch.device('cpu'))
    input = torch.randn(20, 15)
    output = myNet(input)
    print(output.shape)
    # print(output)