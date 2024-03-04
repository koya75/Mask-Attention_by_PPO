# Implemented with reference to the following repository
# https://github.com/mahyaret/kuka_rl.git
import torch
import torch.nn as nn
import torch.nn.functional as F
import pfrl as pfrl
from pfrl.utils.recurrent import (
    get_packed_sequence_info,
    unwrap_packed_sequences_recursive,
    wrap_packed_sequences_recursive,
)

from agent.convLSTM import ConvLSTMCell
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from agent.omt.transformer import Transformer
from agent.omt.buffer import Buffer
import torchvision
from torchvision import transforms
from torchvision.datasets.utils import download_url

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
        demo,
        critic_hidden_layers=[],
        actor_hidden_layers=[],
        seed=0,
        init_type=None,
        use_lstm=False,
        target='item21',
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
        self.demo = demo

        max_len = 12
        batch_size = 100
        epoch_num = 500
        embedding_dim = 1000
        num_encoder_layers = 6
        num_decoder_layers = 6
        nheads = 8
        hidden_dim = 256


        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        del self.resnet50.fc
        self.conv_1 = nn.Conv2d(2048, hidden_dim, 1)
        self.conv_2 = nn.Conv2d(2048, hidden_dim, 1)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        

        """self.Visual_Features = nn.Sequential(nn.Linear(128, 512), nn.ReLU,
                                                            nn.Linear(512, 1024), nn.ReLU,
                                                            nn.Linear(1024, 128))

        self.buffer = Buffer(max_len)"""


        filename = 'target_item/' + target + '_light.png'

        img = cv2.imread(filename)
        #img = img[0:960, 160:1120]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.resize(img , state_size)
        #img = img.reshape(128, *img.shape)
        #cv2.imwrite('item21.png',img)
        img = torch.from_numpy(img.astype(np.float32)).clone().to(torch.float)
        img = img.transpose(1,2)
        self.img = img.transpose(0,1)

        """self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)"""

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        assert embedding_dim % nheads == 0
        self.transformer = Transformer(hidden_dim, num_encoder_layers, nheads)
        
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Add critic layers
        if critic_hidden_layers:
            # Add hidden layers for critic net if critic_hidden_layers is not empty
            self.critic_hidden = build_hidden_layer(
                input_dim=16384, hidden_layers=critic_hidden_layers
            )
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(16384, 1)

        # Add actor layers
        if actor_hidden_layers:
            # Add hidden layers for actor net if actor_hidden_layers is not empty
            self.actor_hidden = build_hidden_layer(
                input_dim=16384, hidden_layers=actor_hidden_layers
            )
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(16384, action_size)

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
            self.critic.apply(self._initialize)
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

    def forward(self, x):
        """Build a network that maps state -> (action, value)."""

        def apply_multi_layer(layers, x, f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x
        
        self.input_image = x
        num,_,_,_ = x.size()
        img = torch.cat([self.img.unsqueeze(0) for g in range(num)]).to('cuda')

        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        # hidden_dim次元に削減
        x = self.conv_1(x)

        img_x = self.resnet50.conv1(img)
        img_x = self.resnet50.bn1(img_x)
        img_x = self.resnet50.relu(img_x)
        img_x = self.resnet50.maxpool(img_x)
        img_x = self.resnet50.layer1(img_x)
        img_x = self.resnet50.layer2(img_x)
        img_x = self.resnet50.layer3(img_x)
        img_x = self.resnet50.layer4(img_x)
        # hidden_dim次元に削減
        img_x = self.conv_2(img_x)

        """x = self.resnet50(x)
        img_x = self.resnet50(img)"""

        H, W = x.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        img_x=torch.permute(img_x,(2, 3, 0, 1)).flatten(0, 1)

        x = self.transformer(pos + 0.1 * x.flatten(2).permute(2, 0, 1),
                             img_x)
        
        x = torch.permute(x,(1,0,2))

        #x = self.Visual_Features(x)

        #self.buffer.append(x)

        # critic
        v_hid = x
        v_hid = v_hid.reshape(v_hid.size(0), -1)
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden, v_hid)
        value = self.critic(v_hid)

        # actor
        a_hid = x
        a_hid = a_hid.reshape(a_hid.size(0), -1)
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden, a_hid)
        a = self.tanh(self.actor(a_hid))
        a = self.prob(a)

        return a, value
