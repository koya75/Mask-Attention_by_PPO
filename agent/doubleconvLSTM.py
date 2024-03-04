import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv.weight.data.mul_(relu_gain)

    def forward(self, input_tensor, cur_state):

        if cur_state is None:
            cur_state = self.init_hidden(input_tensor.size(0))

        h_cur1, c_cur1, h_cur2, c_cur2 = cur_state

        combined1 = torch.cat(
            [input_tensor, h_cur1], dim=1
        )  # concatenate along channel axis

        combined2 = torch.cat(
            [input_tensor, h_cur2], dim=1
        )  # concatenate along channel axis

        combined_conv1 = self.conv(combined1)
        cc_i1, cc_f1, cc_o1, cc_g1 = torch.split(combined_conv1, self.hidden_dim, dim=1)
        i1 = torch.sigmoid(cc_i1)
        f1 = torch.sigmoid(cc_f1)
        o1 = torch.sigmoid(cc_o1)
        g1 = torch.tanh(cc_g1)

        combined_conv2 = self.conv(combined2)
        cc_i2, cc_f2, cc_o2, cc_g2 = torch.split(combined_conv2, self.hidden_dim, dim=1)
        i2 = torch.sigmoid(cc_i2)
        f2 = torch.sigmoid(cc_f2)
        o2 = torch.sigmoid(cc_o2)
        g2 = torch.tanh(cc_g2)

        c_next1 = f1 * c_cur1 + i1 * g1
        h_next1 = o1 * torch.tanh(c_next1)

        c_next2 = f2 * c_cur2 + i2 * g2
        h_next2 = o2* torch.tanh(c_next2)

        return h_next1, c_next1, h_next2, c_next2

    def init_hidden(self, batch_size):
        return (
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).cuda(),
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).cuda(),
        )


def weights_init(m):
    classname = m.__class__.__name__
    np.random.seed(0)
    if classname.find("Conv2d") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        # m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
