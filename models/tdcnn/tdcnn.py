import torch.nn as nn
import torch

# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/3
class TimeDistributed(nn.Module):
    def __init__(self, module, ndim_input=1, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.ndim_input = ndim_input
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *x.shape[-self.ndim_input:])  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, *y.shape[-self.ndim_input:])  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), *y.shape[-self.ndim_input:])  # (timesteps, samples, output_size)

        return y

class TDCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=4,
                 bias=True, return_all_layers=False):
        super(TDCNN, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        bnorm_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            cell_list.append(TimeDistributed(nn.Conv2d(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], bias=self.bias, padding=(1,1)),ndim_input=3,batch_first=True))
            bnorm_list.append(TimeDistributed(nn.BatchNorm2d(self.hidden_dim[i]),ndim_input=3,batch_first=True))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.bnorm_list = nn.ModuleList(bnorm_list)

        self.convlstm = ConvLSTMCell(input_dim=self.hidden_dim[-1],
                                        hidden_dim=self.hidden_dim[-1],
                                        kernel_size=self.kernel_size[-1],
                                        bias=self.bias)

        self.conv_output = nn.Conv2d(self.hidden_dim[-1], self.input_dim, (3,3), padding=(1,1))

    def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (b, t, c, h, w)

        Returns
        -------
        last_state_list, layer_output
        """

        b, seq_len, _, h, w = input_tensor.size()

        # Since the init is done in forward. Can send image size here
        hidden_state_convlstm = self._init_hidden(batch_size=b, image_size=(h, w))

        x = input_tensor

        for layer_idx in range(self.num_layers):
            x = self.cell_list[layer_idx](x)
            x = self.bnorm_list[layer_idx](x)

        h, c = hidden_state_convlstm[layer_idx]
        output_inner = []
        for t in range(seq_len):
            h, c = self.convlstm(input_tensor=x[:, t, :, :, :],
                                                cur_state=[h, c])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)
        
        x = self.conv_output(layer_output[:,-1])
        x = torch.tanh(x)
        return x

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.convlstm.init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
