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
import math


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
        step,
        critic_hidden_layers=[],
        actor_hidden_layers=[],
        seed=0,
        init_type=None,
        use_lstm=False,
        target='item21',
        outdir="resluts",
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

        # CNN
        self.conv2_1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.conv2_3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2_3 = nn.BatchNorm2d(64)

        # attention
        self.conv_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)

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

        self.kldiv = nn.KLDivLoss(reduction="sum")

        self.out_dir = outdir
        filename = 'target_item/' + target + '_light.png'
        self.step = step
        self.steps = 0
        self.episod = 0

        img = cv2.imread(filename)
        #img = img[0:960, 160:1120]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.resize(img , state_size)
        #img = img.reshape(128, *img.shape)
        #cv2.imwrite('item21.png',img)
        img = torch.from_numpy(img.astype(np.float32)).clone().to(torch.float).to('cuda')
        img = img.transpose(1,2)
        self.img = img.transpose(0,1)

        # Add critic layers
        # mask_attention
        self.critic_mask_conv = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_c = nn.Sigmoid()
        self.critic_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        if critic_hidden_layers:
            # Add hidden layers for critic net if critic_hidden_layers is not empty
            self.critic_hidden = build_hidden_layer(
                input_dim=64 * convw * convh, hidden_layers=critic_hidden_layers
            )
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(64 * convw * convh, 1)

        # Add actor layers
        # mask_attention
        self.actor_mask_conv = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_a = nn.Sigmoid()

        self.actor_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        if actor_hidden_layers:
            # Add hidden layers for actor net if actor_hidden_layers is not empty
            self.actor_hidden = build_hidden_layer(
                input_dim=64 * convw * convh, hidden_layers=actor_hidden_layers
            )
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(64 * convw * convh, action_size)

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

    def forward(self, x, recurrent_state=None, ok=False):
        """Build a network that maps state -> (action, value)."""
        if self.steps%24 == 0:
            self.episod += 1
        if self.episod > self.step*3/4:
            half_step = True
        else:
            half_step = False

        def apply_multi_layer(layers, x, f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        if self.use_lstm:
            batch_sizes, sorted_indices = get_packed_sequence_info(x)
            x = unwrap_packed_sequences_recursive(x)

        self.input_image = x * 1.0

        mask_blue = torch.ones(1,16,16, requires_grad=True).to(torch.float).to('cuda')
        mask_red = torch.ones(1,16,16, requires_grad=True).to(torch.float).to('cuda')
        num,_,_,_ = x.size()
        img = torch.cat([self.img.unsqueeze(0) for g in range(num)])
        mask_blue = torch.cat([mask_blue.unsqueeze(0) for g in range(num)])
        mask_red = torch.cat([mask_red.unsqueeze(0) for g in range(num)])
        zero = torch.zeros_like(mask_blue, requires_grad=True).to(torch.float).to('cuda')

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        img = F.relu(self.bn2_1(self.conv2_1(img)))
        img = F.relu(self.bn2_2(self.conv2_2(img)))
        img = F.relu(self.bn2_3(self.conv2_3(img)))

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
        
        img_hid = img
        img_hid = F.relu(self.conv_conv(img_hid))

        # critic
        v_hid = x
        v_hid_v = F.relu(self.critic_conv(v_hid))

        att_v_feature = self.critic_mask_conv(x)
        self.att_c = self.sigmoid_c(att_v_feature)  # mask-attention
        self.att_v_sig5 = self.sigmoid_c(att_v_feature * 5.0)

        # actor
        a_hid = x
        a_hid_a = F.relu(self.actor_conv(a_hid))

        att_a_feature = self.actor_mask_conv(x)
        self.att_a = self.sigmoid_a(att_a_feature)  # mask-attention
        self.att_a_sig5 = self.sigmoid_a(att_a_feature * 5.0)

        v_mask_hid = v_hid_v * self.att_c  # mask processing
        v_hid = v_mask_hid
        v_hid = torch.cat((v_hid,img_hid),dim=1)
        v_hid = v_hid.reshape(v_hid.size(0), -1)
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden, v_hid)
        value = self.critic(v_hid)

        a_mask_hid = a_hid_a * self.att_a  # mask processing
        a_hid = a_mask_hid
        a_hid = torch.cat((a_hid,img_hid),dim=1)
        a_hid = a_hid.reshape(a_hid.size(0), -1)
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden, a_hid)
        a = self.tanh(self.actor(a_hid))
        ac = self.prob(a)

        ld_v = False
        ld_a = False
        if ok and half_step:
            with torch.no_grad():
                #v_mse = torch.zeros(128,256)
                #a_mse = torch.zeros(128,256)
                loss_v = []
                loss_a = []
                mask_v1 = torch.zeros_like(mask_blue, requires_grad=True).to(torch.float).to('cuda')
                mask_a1 = torch.zeros_like(mask_blue, requires_grad=True).to(torch.float).to('cuda')
                #for k in range(1,2):
                for h in range(0,16):
                    for w in range(0,16):
                        mask_blue[:,:,h,w] = 0
                        mask_red = mask_red*mask_blue
                        mask_blue[:,:,h,w] = 1

                        v_mask_hid_v1 = v_hid_v * mask_red  # mask processing
                        v_hid_v1 = v_mask_hid_v1
                        v_hid_v1 = torch.cat((v_hid_v1,img_hid),dim=1)
                        v_hid_v1 = v_hid_v1.reshape(v_hid_v1.size(0), -1)
                        if self.critic_hidden is not None:
                            v_hid_v1 = apply_multi_layer(self.critic_hidden, v_hid_v1)
                        value_v1 = self.critic(v_hid_v1)
                        v_mse = torch.sqrt((value-value_v1)*(value-value_v1))

                        a_mask_hid_v1 = a_hid_a * mask_red  # mask processing
                        a_hid_v1 = a_mask_hid_v1
                        a_hid_v1 = torch.cat((a_hid_v1,img_hid),dim=1)
                        a_hid_v1 = a_hid_v1.reshape(a_hid_v1.size(0), -1)
                        if self.actor_hidden is not None:
                            a_hid_v1 = apply_multi_layer(self.actor_hidden, a_hid_v1)
                        a_v1 = self.tanh(self.actor(a_hid_v1))
                        a_mse = torch.sqrt((a-a_v1)*(a-a_v1))
                        #a_v1 = self.prob(a_v1)
                        #a_mse = self.kldiv(a_v1.log_prob(),ac)

                        v_v3 = v_mse > 0.05
                        a_v3 = a_mse > 0.05
                        v_v4 = v_v3.sum(dim=1)
                        a_v4 = a_v3.sum(dim=1)
                        for n in range(num):
                            if v_v4[n] == True:
                                mask_v1[n,:,:,:] = mask_v1[n,:,:,:] + (mask_blue[n,:,:,:] - mask_red[n,:,:,:])
                                ld_v = True
                            elif a_v4[n] == True:
                                mask_a1[n,:,:,:] = mask_a1[n,:,:,:] + (mask_blue[n,:,:,:] - mask_red[n,:,:,:])
                                ld_a = True
                        mask_red[:,:,h,w] = 1
            #print(mask_v1[0,:,:,:].tolist())
            if ld_v and ld_a:
                pack = (self.att_c, self.att_a, mask_v1,mask_a1,ld_v,ld_a)
            elif ld_v:
                pack = (self.att_c, _, mask_v1,_,ld_v,ld_a)
            elif ld_a:
                pack = (_, self.att_a, _,mask_a1,ld_v,ld_a)
            else:
                pack = (_,_,_,_,ld_v,ld_a)
            ok = False
        else:
            pack = (_,_,_,_,ld_v,ld_a)
        self.steps += 1

        if self.use_lstm:
            x = wrap_packed_sequences_recursive((ac, value), batch_sizes, sorted_indices)
            return x, (hx, cx), pack
        else:
            return ac, value
