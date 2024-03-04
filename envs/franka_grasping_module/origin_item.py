from pathlib import Path
import random

import torch

from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np


from isaacgym.torch_utils import *


class Item:
    def create(self, gym, sim, device, num_actors, asset_root, item_names):
        self.gym = gym
        self.sim = sim
        self.device = device

        self.t = 0
        self.y = 0
        
        # load bin asset
        # http://mprg.jp/en/research/arc_dataset_2017_e
        self.item_names = item_names

        self.initial_pose = gymapi.Transform()
        self.initial_pose.p = gymapi.Vec3(0, 0.4, 0.6)

        self.scale = 0.5
        self.num_actors = num_actors

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.02 # 0.002
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.vhacd_enabled = True

        self.assets = []
        self.__num_bodies = []
        self.__num_dofs = []
        self.__num_shapes = []
        for item_name in self.item_names:
            asset_file = "{}.urdf".format(item_name)
            print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
            asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
            self.assets.append(asset)

            self.__num_bodies.append(self.gym.get_asset_rigid_body_count(asset))
            self.__num_dofs.append(self.gym.get_asset_dof_count(asset))
            self.__num_shapes.append(self.gym.get_asset_rigid_shape_count(asset))

        self.actor_idxs = []

        self.start_handle = None

    def select_item(self):
        self._num_bodies = 0
        self._num_dofs = 0
        self._num_shapes = 0
        self.item_nums = []
        for i in range(self.num_actors):
            #item_num = np.random.randint(len(self.item_names))
            item_num = i
            self.item_nums.append(item_num)

            self._num_bodies += self.__num_bodies[item_num]
            self._num_dofs += self.__num_dofs[item_num]
            self._num_shapes += self.__num_shapes[item_num]

    def add(self, env, collisionGroup):

        for i, item_num in enumerate(self.item_nums):
            actor_name = "item_{}".format(i)

            handle = self.gym.create_actor(
                env, self.assets[item_num], self._random_pose(), actor_name, collisionGroup, 0
            )

            if self.start_handle is None:
                self.start_handle = handle

            self.actor_idxs.append(
                self.gym.find_actor_index(env, actor_name, gymapi.DOMAIN_SIM)
            )
            self.gym.set_actor_scale(env, handle, self.scale)

    def set_tensors(self, root_state_tensor, num_envs, rigid_body_states):
        self.root_state_tensor = root_state_tensor
        self.num_envs = num_envs

        self.actor_idxs = torch.tensor(
            self.actor_idxs, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.states = self.root_state_tensor[
            :, self.start_handle : self.start_handle + self.num_actors
        ]

        self.rigid_body_states = rigid_body_states
        self.num_envs = num_envs

    def reset(self, env_ids):
        indices = self.actor_idxs[env_ids, :].flatten()
        random_states = []
        for _ in range(len(env_ids) * self.num_actors):
            state = self._random_state()
            random_states.append(state)
        random_states = to_torch(
            random_states, device=self.device, dtype=torch.float
        ).view(len(env_ids), self.num_actors, 13)
        self.states[env_ids] = random_states
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(indices),
            len(indices),
        )

    def reset_one(self, env_ids, actor_idx):
        self.y += 1
        self.y %= 3
        indices = self.actor_idxs[env_ids, actor_idx].flatten()
        random_states = []
        for _ in range(len(env_ids)):
            state = self._random_state()
            random_states.append(state)
        random_states = to_torch(
            random_states, device=self.device, dtype=torch.float
        ).view(len(env_ids), 1, 13)
        self.states[env_ids] = random_states
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(indices),
            len(indices),
        )
        if self.y == 0: 
            self.t += 1 
            print(self.t)

    def _random_pose(self):
        pose = gymapi.Transform()
        pose.p.x = self.initial_pose.p.x - 0.1 + 0.2 * np.random.rand()
        pose.p.y = self.initial_pose.p.y - 0.1 + 0.2 * np.random.rand()
        pose.p.z = self.initial_pose.p.z + 0.1 * np.random.rand()
        pose.r.x = 0.5 - np.random.rand()
        pose.r.y = 0.5 - np.random.rand()
        pose.r.z = 0.5 - np.random.rand()
        pose.r.w = 0.5 - np.random.rand()
        pose.r = pose.r.normalize()
        return pose

    def _random_state(self):
        pose = self._random_pose()
        return [
            pose.p.x,
            pose.p.y,
            pose.p.z,
            pose.r.x,
            pose.r.y,
            pose.r.z,
            pose.r.w,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

    @property
    def num_bodies(self):
        return self._num_bodies

    @property
    def num_shapes(self):
        return self._num_shapes

    @property
    def num_dofs(self):
        return self._num_dofs

    @property
    def pos(self):
        return self.root_state_tensor[
            :, self.start_handle : self.start_handle + self.num_actors, 0:3
        ]

    @property
    def rot(self):
        return self.root_state_tensor[
            :, self.start_handle : self.start_handle + self.num_actors, 3:7
        ]