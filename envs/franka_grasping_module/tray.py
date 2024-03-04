from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymtorch
import math
import numpy as np

from isaacgym.torch_utils import *

class Tray:
    def create(self, gym, sim, device, asset_root):
        self.gym = gym
        self.sim = sim
        self.device = device

        # load bin asset
        asset_file = "urdf/tray/traybox.urdf"
        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))

        self.actor_name = "tray"

        self.pose = gymapi.Transform()
        self.pose.p = gymapi.Vec3(0, 0.4, 0.5)

        self.color = gymapi.Vec3(0.24, 0.35, 0.8)
        self.scale = 0.9

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        self.asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        self._num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self._num_dofs = self.gym.get_asset_dof_count(self.asset)
        self._num_shapes = self.gym.get_asset_rigid_shape_count(self.asset)

    def add(self, env, collisionGroup):
        self.handle = self.gym.create_actor(
            env, self.asset, self.pose, self.actor_name, collisionGroup, 0
        )
        self.gym.set_rigid_body_color(
            env, self.handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.color
        )
        self.gym.set_actor_scale(env, self.handle, self.scale)

    @property
    def num_bodies(self):
        return self._num_bodies

    @property
    def num_shapes(self):
        return self._num_shapes

    @property
    def num_dofs(self):
        return self._num_dofs
