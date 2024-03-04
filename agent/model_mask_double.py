# Implemented with reference to the following repository
# https://github.com/mahyaret/kuka_rl.git
import torch
import torch.nn as nn
import torch.nn.functional as F
import repos.pfrl.pfrl as pfrl
from repos.pfrl.pfrl.utils.recurrent import (
    get_packed_sequence_info,
    unwrap_packed_sequences_recursive,
    wrap_packed_sequences_recursive,
)

from agent.convLSTM import ConvLSTMCell
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def build_hidden_layer(input_dim, hidden_layers):
    """Build hidden layer.
    Params
    ======
        input_dim (int): Dimension of hidden layer input
        hidden_layers (list(int)): Dimension of hidden layers
    """
    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
    if len(hidden_layers) > 1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    return hidden


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        critic_hidden_layers=[],
        actor_hidden_layers=[],
        seed=0,
        init_type=None,
        use_lstm=False,
    ):
        """Initialize parameters and build policy.
        Params
        ======
            state_size (int,int,int): Dimension of each state
            action_size (int): Dimension of each action
            critic_hidden_layers (list(int)): Dimension of the critic's hidden layers
            actor_hidden_layers (list(int)): Dimension of the actor's hidden layers
            seed (int): Random seed
            init_type (str): Initialization type
            use_lstm (bool): use LSTM flag
        """
        super(ActorCritic, self).__init__()
        self.init_type = init_type
        self.seed = torch.manual_seed(seed)
        self.use_lstm = use_lstm
        self.sigma = nn.Parameter(torch.zeros(action_size))

        # Add shared hidden layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=5, stride=2, padding=2):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[0])))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[1])))

        # add LSTM
        if self.use_lstm:
            self.convlstm1 = ConvLSTMCell(
                input_size=(convh, convw),
                input_dim=64,
                hidden_dim=64,
                kernel_size=(3, 3),
                bias=True,
            )

        # Add critic layers
        # mask_attention
        self.critic_mask_conv = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_c = nn.Sigmoid()
        self.critic_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        if critic_hidden_layers:
            # Add hidden layers for critic net if critic_hidden_layers is not empty
            self.critic_hidden = build_hidden_layer(
                input_dim=32 * convw * convh, hidden_layers=critic_hidden_layers
            )
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(32 * convw * convh, 1)

        # Add actor layers
        # mask_attention
        self.actor_mask_conv = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_a = nn.Sigmoid()

        self.actor_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        if actor_hidden_layers:
            # Add hidden layers for actor net if actor_hidden_layers is not empty
            self.actor_hidden = build_hidden_layer(
                input_dim=32 * convw * convh, hidden_layers=actor_hidden_layers
            )
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(32 * convw * convh, action_size)

        # Apply Tanh() to bound the actions
        self.tanh = nn.Tanh()

        self.prob = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

        # Initialize hidden and actor-critic layers
        if self.init_type is not None:
            self.critic_mask_conv.apply(self._initialize)
            self.critic_conv.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor_mask_conv.apply(self._initialize)
            self.actor_conv.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

    def _initialize(self, n):
        """Initialize network weights."""
        if isinstance(n, nn.Linear):
            if self.init_type == "xavier-uniform":
                nn.init.xavier_uniform_(n.weight.data)
            elif self.init_type == "xavier-normal":
                nn.init.xavier_normal_(n.weight.data)
            elif self.init_type == "kaiming-uniform":
                nn.init.kaiming_uniform_(n.weight.data)
            elif self.init_type == "kaiming-normal":
                nn.init.kaiming_normal_(n.weight.data)
            elif self.init_type == "orthogonal":
                nn.init.orthogonal_(n.weight.data)
            elif self.init_type == "uniform":
                nn.init.uniform_(n.weight.data)
            elif self.init_type == "normal":
                nn.init.normal_(n.weight.data)
            else:
                raise KeyError(
                    "initialization type is not found in the set of existing types"
                )

    def forward(self, x, recurrent_state=None):
        """Build a network that maps state -> (action, value)."""

        def apply_multi_layer(layers, x, f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        if self.use_lstm:
            batch_sizes, sorted_indices = get_packed_sequence_info(x)
            x = unwrap_packed_sequences_recursive(x)

        self.input_image = x * 1.0

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if self.use_lstm:
            if len(batch_sizes) == 1:
                if recurrent_state is not None:
                    hx, cx = recurrent_state
                    hx = hx.squeeze(0)
                    cx = cx.squeeze(0)
                    recurrent_state = (hx, cx)
                hx, cx = self.convlstm1(input_tensor=x, cur_state=recurrent_state)
                x = hx
                hx = hx.unsqueeze(0)
                cx = cx.unsqueeze(0)
            else:
                if recurrent_state is not None:
                    hx, cx = recurrent_state
                    hx = hx.squeeze(0)
                    cx = cx.squeeze(0)
                    recurrent_state = (hx, cx)
                if not all(batch_sizes[0] == batch_sizes):
                    raise RuntimeError('not all(batch_sizes[0] == batch_sizes).')
                xs = torch.split(x, batch_sizes[0], dim=0)
                hxs = []
                cxs = []
                for i in range(len(batch_sizes)):
                    hx, cx = self.convlstm1(input_tensor=xs[i], cur_state=recurrent_state)
                    hxs.append(hx)
                    cxs.append(cx)
                    recurrent_state = (hx, cx)
                
                hx = torch.stack(hxs)
                cx = torch.stack(cxs)
                
                x = hx.reshape(-1, *hx.size()[2:])

        # critic
        v_hid = x
        v_hid = F.relu(self.critic_conv(v_hid))

        att_v_feature = self.critic_mask_conv(x)
        self.att_c = self.sigmoid_c(att_v_feature)  # mask-attention
        self.att_v_sig5 = self.sigmoid_c(att_v_feature * 5.0)
        v_mask_hid = v_hid * self.att_c  # mask processing

        v_hid = v_mask_hid
        v_hid = v_hid.reshape(v_hid.size(0), -1)
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden, v_hid)
        value = self.critic(v_hid)

        # actor
        a_hid = x
        a_hid = F.relu(self.actor_conv(a_hid))

        att_a_feature = self.actor_mask_conv(x)
        self.att_a = self.sigmoid_a(att_a_feature)  # mask-attention
        self.att_a_sig5 = self.sigmoid_a(att_a_feature * 5.0)
        a_mask_hid = a_hid * self.att_a  # mask processing

        a_hid = a_mask_hid
        a_hid = a_hid.reshape(a_hid.size(0), -1)
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden, a_hid)
        a = self.tanh(self.actor(a_hid))
        a = self.prob(a)

        if self.use_lstm:
            x = wrap_packed_sequences_recursive((a, value), batch_sizes, sorted_indices)
            return x, (hx, cx)
        else:
            return a, value
