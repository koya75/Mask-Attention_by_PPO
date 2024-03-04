import sys

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgym.torch_utils import *
from matplotlib.pyplot import figure

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class Franka:
    def create(self, gym, sim, device, discrete, actionRepeat, asset_root, max_episode_length, num_env):
        self.gym = gym
        self.sim = sim
        self.device = device
        self.discrete = discrete
        self.actionRepeat = actionRepeat
        self.max_episode_length = max_episode_length

        asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        self.actor_name = "franka"

        self.t = 0
        self.r = 0
        self.p = torch.zeros(num_env)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        self.asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        # 剛性パラメータ
        franka_dof_stiffness = to_torch(
            [400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6],
            dtype=torch.float,
            device=self.device,
        )
        # 減衰パラメータ
        franka_dof_damping = to_torch(
            [80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2],
            dtype=torch.float,
            device=self.device,
        )
        # デフォルトのdof
        self.default_dof_pos = to_torch(
            [0, 0.0, 0, -2.0, 0, 2.5, 0.0, 0.035, 0.035], device=self.device
        )

        self._num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self._num_dofs = self.gym.get_asset_dof_count(self.asset)
        self._num_shapes = self.gym.get_asset_rigid_shape_count(self.asset)

        # set franka dof properties
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)  # frankaの仕様を読み込み
        self._dof_lower_limits = []
        self._dof_upper_limits = []
        for i in range(self._num_dofs):
            self.dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            self.dof_props["stiffness"][i] = franka_dof_stiffness[i]
            self.dof_props["damping"][i] = franka_dof_damping[i]

            self._dof_lower_limits.append(self.dof_props["lower"][i])
            self._dof_upper_limits.append(self.dof_props["upper"][i])
        print(f"{self._dof_lower_limits = }")
        print(f"{self._dof_upper_limits = }")
        # self._dof_lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0]
        # self._dof_upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04]

        self._dof_lower_limits = to_torch(self._dof_lower_limits, device=self.device)
        self._dof_upper_limits = to_torch(self._dof_upper_limits, device=self.device)
        self.dof_speed_scales = torch.ones_like(self._dof_lower_limits)
        self.dof_speed_scales[[7, 8]] = 0.1
        self.dof_props["effort"][7] = 200
        self.dof_props["effort"][8] = 200

        self.start_pose = gymapi.Transform()
        self.start_pose.p = gymapi.Vec3(0.0, -0.1, 0.5)
        self.start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 1.0).normalize()

        self.actor_idxs = []

        self.target_hand_pos_lower_limits = to_torch(
            [-0.15, -0.15, 0], device=self.device
        )
        self.target_hand_pos_upper_limits = to_torch(
            [0.15, 0.15, 1.0], device=self.device
        )
        self.target_hand_rot_z_lower_limits = to_torch([-2.0], device=self.device)
        self.target_hand_rot_z_upper_limits = to_torch([2.0], device=self.device)

    def add(self, env, i):
        actor = self.gym.create_actor(
            env, self.asset, self.start_pose, self.actor_name, i, 1, 0
            )

        self.gym.set_actor_dof_properties(env, actor, self.dof_props)
        self.hand_handle = self.gym.find_actor_rigid_body_handle(
            env, actor, "panda_hand"
        )
        # print(f"{self.gym.get_actor_rigid_body_dict(env, actor)}")
        self.right_finger_handle = self.gym.find_actor_rigid_body_handle(
            env, actor, "panda_rightfinger"
        )
        self.left_finger_handle = self.gym.find_actor_rigid_body_handle(
            env, actor, "panda_leftfinger"
        )

        # props = self.gym.get_actor_rigid_shape_properties(env, actor)
        # props[self.left_finger_handle].friction = 2.0
        # props[self.right_finger_handle].friction = 2.0
        # self.gym.set_actor_rigid_shape_properties(env, actor, props)

        self.actor_idxs.append(
            self.gym.find_actor_index(env, self.actor_name, gymapi.DOMAIN_SIM)
        )

    def set_tensors(self, rigid_body_states, num_envs, dof_state, sim_params, viewer):
        self.rigid_body_states = rigid_body_states
        self.num_envs = num_envs
        self.viewer = viewer

        self.dof_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.dof_state = dof_state.view(self.num_envs, -1, 2)[:, : self.num_dofs]
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]

        self.actor_idxs = torch.tensor(
            self.actor_idxs, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        self.sim_params = sim_params
        #####################################下降ステップ
        self._dv = 0.025 #0.3/self.max_episode_length

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_handle - 1, :]

        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_dir = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.z_dir_ones = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.ones = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ones_r = torch.ones(1, dtype=torch.float, device=self.device)
        self.zeros_r = torch.zeros(1, dtype=torch.float, device=self.device)

        self.target_hand_pos_offset = torch.zeros_like(self.hand_pos)
        self.target_hand_pos_offset[:, 1] = 0.4
        self.target_hand_pos_offset[:, 2] = 0.57

        self.target_hand_rot_offset = torch.zeros_like(self.hand_rot)
        self.target_hand_rot_offset[:, 1] = 1

        self.target_hand_pos = torch.zeros_like(self.hand_pos)
        self.target_hand_rot_z = torch.zeros_like(self.hand_rot[:, 0])

    def render(self):
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # check for keyboard events
        self.manual_action = 0
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

            elif self.use_manual_action:
                if evt.action == "left":
                    self.manual_action = 1
                elif evt.action == "right":
                    self.manual_action = 2
                elif evt.action == "down":
                    self.manual_action = 3
                elif evt.action == "up":
                    self.manual_action = 4
                elif evt.action == "clockwise":
                    self.manual_action = 5
                elif evt.action == "counter_clockwise":
                    self.manual_action = 6

        self.gym.draw_viewer(self.viewer, self.sim, True)

    def reset(self, env_ids):
        # frankaの初期位置を再設定する
        # 初期位置を若干ランダムに設定する。
        pos = tensor_clamp(
            self.default_dof_pos.unsqueeze(0)
            + 0.25
            * (torch.rand((len(env_ids), self.num_dofs), device=self.device) - 0.5),
            self._dof_lower_limits,
            self._dof_upper_limits,
        )
        self.dof_pos[env_ids, :] = pos
        self.dof_targets[env_ids, :] = pos

        # 初期速度はゼロとする
        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # 環境を指定する
        multi_env_ids_int32 = self.actor_idxs[env_ids].flatten()

        # ターゲットのdofを初期位置と一致させる。
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_targets),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        # 位置と速度を設定する。
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

        self.target_hand_pos[env_ids, 0] = 0.0
        self.target_hand_pos[env_ids, 1] = 0.0
        self.target_hand_pos[env_ids, 2] = 0.3
        self.target_hand_rot_z[env_ids] = 0.0

    def pre_physics_step(self, actions, item_pos):
        dv = self._dv  # velocity per physics step.
        if self.discrete:
            # compute position and orientation error
            one_hot = torch.nn.functional.one_hot(actions, num_classes=7).to(
                torch.float
            )
            dx = to_torch([0, -dv, dv, 0, 0, 0, 0]).inner(one_hot)
            dy = to_torch([0, 0, 0, -dv, dv, 0, 0]).inner(one_hot)
            dz = -dv * torch.ones_like(actions[:, 1])
            da = to_torch([0, 0, 0, 0, 0, -10 * dv, 10 * dv]).inner(one_hot)
        else:
            dx = dv * actions[:, 0]
            dy = dv * actions[:, 1]
            dz = -dv * torch.ones_like(actions[:, 1])
            da = 10 * dv * actions[:, 2]

        dtarget_hand_pos = torch.stack(
            (
                dx,
                dy,
                dz,
            ),
            dim=1,
        )
        dtarget_hand_rot_z = da

        done = self._step_continuous(dtarget_hand_pos, dtarget_hand_rot_z)
        return done

    def _step_continuous(self, dtarget_hand_pos, dtarget_hand_rot_z):
        # Perform commanded action.
        #finger = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.t += 1
        if self.t <=12 and self.r != 2:
            finger = 0.04 * self.ones
            """for i in range(self.num_envs):
                    if self.p[i] == 1:
                        finger[i] = torch.zeros(1, dtype=torch.float, device=self.device)
                    else:
                        finger[i] = 0.04 * torch.ones(1, dtype=torch.float, device=self.device)"""
            self._apply_action(dtarget_hand_pos, dtarget_hand_rot_z, finger)
            for _ in range(self.actionRepeat):
                self._step_simulation()
                if self.viewer is not None:
                    self.render()
            done = False
            # If we are close to the bin, attempt grasp.
        elif self.t > 12 and self.r != 2:
            actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
                self.num_envs, -1, 13
            )
            item_pos = self.root_state_tensor[
                :, 0:3, 0:3
            ]
            finger = self.zeros
            #finger = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            dtarget_hand_pos[:, :2] = 0
            dtarget_hand_pos[:, 2] = self._dv
            self._apply_action(dtarget_hand_pos, dtarget_hand_rot_z, finger)
            for _ in range(self.actionRepeat):
                self._step_simulation()
                if self.viewer is not None:
                    self.render()
            done = False
            if self.target_hand_pos[0][2] > 0.3:
                for i in range(self.num_envs):
                    if item_pos[i,2,2] > 0.7:
                        self.p[i] = 1
                self.t = 0
                self.r += 1
        elif self.r == 2:
            finger = 0.04 * self.ones
            """for i in range(self.num_envs):
                    if self.p[i] == 1:
                        finger[i] = torch.zeros(1, dtype=torch.float, device=self.device)
                    else:
                        finger[i] = 0.04 * torch.ones(1, dtype=torch.float, device=self.device)"""
            self._apply_action(dtarget_hand_pos, dtarget_hand_rot_z, finger)
            for _ in range(self.actionRepeat):
                self._step_simulation()
                if self.viewer is not None:
                    self.render()

            done = False
            # If we are close to the bin, attempt grasp.
            if self.target_hand_pos[0][2] <= 0.001:
                finger = self.zeros
                dtarget_hand_pos[:, :2] = 0
                dtarget_hand_pos[:, 2] = 0.1 * self._dv
                self._apply_action(dtarget_hand_pos, dtarget_hand_rot_z, finger)
                for _ in range(self.actionRepeat):
                    self._step_simulation()
                    if self.viewer is not None:
                        self.render()
                dtarget_hand_pos[:, :2] = 0
                dtarget_hand_pos[:, 2] = self._dv
                while True:
                    self._apply_action(dtarget_hand_pos, dtarget_hand_rot_z, finger)
                    for _ in range(self.actionRepeat):
                        self._step_simulation()
                        if self.viewer is not None:
                            self.render()
                    if self.target_hand_pos[0][2] > 0.4:
                        self.t = 0
                        self.r = 0
                        #self.p[i] = self.zeros_r
                        break

                done = True
        return done

    def _apply_action(self, dtarget_hand_pos, dtarget_hand_rot_z, finger):

        self.target_hand_pos = self.target_hand_pos + dtarget_hand_pos
        # self.target_hand_pos[..., 2] = 0.2
        self.target_hand_pos = tensor_clamp(
            self.target_hand_pos,
            self.target_hand_pos_lower_limits,
            self.target_hand_pos_upper_limits,
        )
        self.target_hand_rot_z = self.target_hand_rot_z + dtarget_hand_rot_z
        self.target_hand_rot_z = tensor_clamp(
            self.target_hand_rot_z,
            self.target_hand_rot_z_lower_limits,
            self.target_hand_rot_z_upper_limits,
        )

        self.apply_target_pose(finger=finger)

    def apply_target_pose(self, finger=None):
        q = quat_from_angle_axis(self.target_hand_rot_z, self.z_unit_tensor)
        self.target_hand_rot = quat_mul(q, self.target_hand_rot_offset)

        pos_err = self.target_hand_pos + self.target_hand_pos_offset - self.hand_pos
        orn_err = orientation_error(self.target_hand_rot, self.hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to(self.device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(
            self.num_envs, 9
        )
        dof_pos = self.dof_pos

        # update position targets
        self.dof_targets = dof_pos + u
        if finger is not None:
            self.dof_targets[:, 7] = finger
            self.dof_targets[:, 8] = finger
        """if finger is not None:
            for i in range(self.num_envs):
                self.dof_targets[i, 7] = finger[i]
                self.dof_targets[i, 8] = finger[i]"""
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_targets)
        )

    def _step_simulation(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

    @property
    def dof_lower_limits(self):
        return self._dof_lower_limits

    @property
    def dof_upper_limits(self):
        return self._dof_upper_limits

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
    def hand_pos(self):
        return self.rigid_body_states[:, self.hand_handle][:, 0:3]

    @property
    def hand_rot(self):
        return self.rigid_body_states[:, self.hand_handle][:, 3:7]

    @property
    def right_finger_pos(self):
        return self.rigid_body_states[:, self.right_finger_handle][:, 0:3]

    @property
    def left_finger_pos(self):
        return self.rigid_body_states[:, self.left_finger_handle][:, 0:3]
