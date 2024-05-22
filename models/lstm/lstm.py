
import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bias=bias, batch_first=True)
        self.bn = nn.BatchNorm3d(hidden_dim)

        self.conv_output = nn.Conv2d(hidden_dim, input_dim, (1,1), padding=(0,0))

    def forward(self, input_tensor):
        """
        @param input_tensor: 5-D Tensor either of shape (b, t, c, h, w)
        """

        b, seq_len, c, h, w = input_tensor.size()

        x = input_tensor.permute((0,3,4,1,2)).reshape(b*h*w, seq_len, c)
        x, _ = self.lstm(x)
        x = x.reshape(b, h, w, seq_len, self.hidden_dim).permute((0,3,4,1,2))
        x = self.bn(x.swapaxes(1,2)).swapaxes(1,2)

        x = self.conv_output(x[:,-1])
        x = torch.tanh(x)
        return x
