import os
import sys
from pathlib import Path

import numpy as np
import imageio

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
import torch

from gym import spaces

from .franka_grasping_module.table import Table
from .franka_grasping_module.tray import Tray
from .franka_grasping_module.very_hard_franka import Franka
from .franka_grasping_module.ur5e import ur5e
from .franka_grasping_module.new_item import Item
from .franka_grasping_module.ur5e_camera import Camera

from pfrl.env import VectorEnv


class FrankaGraspingEnv(VectorEnv):
    def __init__(
        self,
        num_envs,
        width,
        height,
        discrete,
        image_type,
        output_debug_images_dir,
        item_asset_root,
        isaacgym_asset_root,
        num_items,
        item_names,
        use_viewer,
        action_repeat,
        device_id,
        interval,
        steps,
        descentstep,
        target,
        usedrop=False,
        hand=False,
    ):

        self.steps = steps
        self.target = target
        self.hand = hand
        self.interval = interval

        self.max_episode_length = descentstep
        self.dt = 1 / 60.0
        self.substep = 4
        # self.device_type = "cuda"
        self.device_id = device_id
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.output_debug_images_dir = output_debug_images_dir
        self.discrete = discrete
        self.item_asset_root = item_asset_root
        self.isaacgym_asset_root = isaacgym_asset_root
        if image_type == "color":
            self.image_type = gymapi.IMAGE_COLOR
        elif image_type == "depth":
            self.image_type = gymapi.IMAGE_DEPTH
        self.actionRepeat = action_repeat  # 15
        self.num_items = num_items
        self.item_names = item_names
        self.use_viewer = use_viewer
        self.usedrop = usedrop

        if self.image_type == gymapi.IMAGE_COLOR:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32
            )
        elif self.image_type == gymapi.IMAGE_DEPTH:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(1, height, width), dtype=np.float32
            )

        if self.discrete:
            self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(9, 1))

        self.device = "cuda:{}".format(self.device_id)

        self.use_manual_action = False
        self.manual_action = 0

        self.gym = gymapi.acquire_gym()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.observation_space.shape[2], self.height, self.width),
            device=self.device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.done_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.infos = [{} for _ in range(self.num_envs)]
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # create envs, sim and viewer
        self._create_sim()
        self.gym.prepare_sim(self.sim)

        if self.use_viewer:
            # todo: read from config
            self.enable_viewer_sync = False

            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT, "left"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT, "right"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_DOWN, "down"
            )
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "counter_clockwise"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "clockwize"
            )

            # set the camera position based on up axis
            cam_pos = gymapi.Vec3(10.0, 10.0, 3.0)
            cam_target = gymapi.Vec3(5.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        else:
            self.viewer = None

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(
            self.sim
        )  # シミュレーション全体の状態を取得
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )

        self.num_bodies = self.rigid_body_states.shape[1]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.franka.set_tensors(
            self.rigid_body_states,
            self.num_envs,
            self.dof_state,
            self.sim_params,
            self.viewer,
        )
        self.item.set_tensors(
            self.root_state_tensor, self.num_envs, self.rigid_body_states
        )

        self.axes_geom = gymutil.AxesGeometry(0.3)

        self.global_indices = torch.arange(
            self.num_envs * (3), dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

    def _create_sim(self):
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.substeps = self.substep
        sim_params.use_gpu_pipeline = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        # sim_params.physx.solver_type = 1
        # sim_params.physx.num_position_iterations = 4
        # sim_params.physx.num_velocity_iterations = 1
        # sim_params.physx.num_threads = 4
        # sim_params.physx.use_gpu = True
        # sim_params.physx.num_subscenes = 0
        # sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = 8

        self.sim_params = sim_params

        self.sim = self.gym.create_sim(
            self.device_id,
            self.device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )

        self._create_ground_plane()

        envSpacing = 0.5
        self._create_envs(envSpacing, int(np.sqrt(self.num_envs)))

        l_color = gymapi.Vec3(0.4, 0.4, 0.4)
        l_ambient = gymapi.Vec3(0.5, 0.5, 0.5)
        l_direction = gymapi.Vec3(1, 1, 3)
        self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
        l_direction = gymapi.Vec3(-1, 1, 3)
        self.gym.set_light_parameters(self.sim, 1, l_color, l_ambient, l_direction)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, spacing, num_per_row):
        #########################################################
        # 一つの環境の大きさ
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        ########## franka ##########
        self.franka = Franka()
        self.franka.create(
            self.gym,
            self.sim,
            self.device,
            self.discrete,
            self.actionRepeat,
            self.isaacgym_asset_root,
            self.max_episode_length,
            self.interval,
        )
        ########## table ##########
        self.table = Table()
        self.table.create(self.gym, self.sim)
        ########## tray ##########
        self.tray = Tray()
        self.tray.create(self.gym, self.sim, self.device, self.isaacgym_asset_root)
        ########## item ##########
        self.item = Item()
        self.item.create(
            self.gym,
            self.sim,
            self.device,
            self.num_items,
            self.item_asset_root,
            self.item_names,
            self.steps,
            self.usedrop,
            self.item_names.index(self.target),
        )
        ########## camera ##########
        self.camera = Camera()
        self.camera.create(self.gym, self.sim, self.width, self.height, self.image_type, self.hand)

        #########################################################

        print("num envs: ", self.num_envs)
        print("num franka bodies: ", self.franka.num_bodies)
        print("num franka dofs: ", self.franka.num_dofs)
        print("num table bodies: ", self.table.num_bodies)
        print("num table dofs: ", self.table.num_dofs)
        print("num tray bodies: ", self.tray.num_bodies)
        print("num tray dofs: ", self.tray.num_dofs)

        print("num item name:", self.item.item_names)
        print("target item:", self.target)

        self.target_number = self.item.item_names.index(self.target) + 1

        # compute aggregate size

        self.envs = []
        self.names = []
        self.cam_tensors = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            self.item.select_item()
            # print("num item bodies: ", self.item.num_bodies)
            # print("num item dofs: ", self.item.num_dofs)

            max_agg_bodies = (
                self.franka.num_bodies
                + self.table.num_bodies
                + self.tray.num_bodies
                + self.item.num_bodies
            )
            max_agg_shapes = (
                self.franka.num_shapes
                + self.table.num_shapes
                + self.tray.num_shapes
                + self.item.num_shapes
            )

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self.franka.add(env_ptr, i)
            self.table.add(env_ptr, i)
            self.tray.add(env_ptr, i)
            self.item.add(env_ptr, i)
            self.cam_tensors.append(self.camera.add(env_ptr, self.franka.hand_handle))
            self.gym.end_aggregate(env_ptr)
            #names = self.gym.get_actor_name(env_ptr, self.handle)

        self.names = torch.Tensor(self.item.itemnames).to("cuda")
        self._init_data()

    def step(self, actions):
        done = self._pre_physics_step(actions)
        self.done_buf[:] = done
        self._post_physics_step()

        return (
            tuple(self.obs_buf),
            tuple(self.rew_buf.cpu().numpy()),
            tuple(self.done_buf.cpu().numpy()),
            tuple(self.infos),
        )

    def _reset(self, env_ids):
        self.franka.reset(env_ids)
        self.progress_buf[env_ids] = 0
        self.done_buf[env_ids] = 0
        for actor_idx in range(self.item.num_actors):
            self.item.reset_one(env_ids, actor_idx)
            for _ in range(20):
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_jacobian_tensors(self.sim)
                self.franka.apply_target_pose()
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                if self.use_viewer:
                    self.gym.draw_viewer(self.viewer, self.sim, True)
        self._compute_observations()
        return self.obs_buf

    def reset(self, mask=None):
        if mask is not None:
            if np.any(~mask): # can not partial envs reset.
                self._reset(torch.arange(self.num_envs, device=self.device))
        else:
            self._reset(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf

    def _init_data(self):

        franka_local_grasp_pose = gymapi.Transform()
        franka_local_grasp_pose.p = gymapi.Vec3(0, 0, 0.1)
        self.franka_local_grasp_pos = to_torch(
            [
                franka_local_grasp_pose.p.x,
                franka_local_grasp_pose.p.y,
                franka_local_grasp_pose.p.z,
            ],
            device=self.device,
        ).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch(
            [
                franka_local_grasp_pose.r.x,
                franka_local_grasp_pose.r.y,
                franka_local_grasp_pose.r.z,
                franka_local_grasp_pose.r.w,
            ],
            device=self.device,
        ).repeat((self.num_envs, 1))

    def _compute_reward(self, actions):
        self.rew_buf[:], self.done_buf[:] = compute_franka_reward(
            self.done_buf,
            self.progress_buf,
            self.actions,
            self.global_franka_pos,
            self.item_pos,
            self.names,
            self.max_episode_length,
            self.target_number,
            self.franka.target_hand_pos[:,2],
        )

    def _compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.hand_pos = self.franka.hand_pos
        self.hand_rot = self.franka.hand_rot
        self.right_finger_pos = self.franka.right_finger_pos
        self.left_finger_pos = self.franka.left_finger_pos
        self.global_franka_rot, self.global_franka_pos = tf_combine(
            self.hand_rot,
            self.hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
        )
        self.item_pos = self.item.pos
        #self.item_name = self.gym.get_actor_name(self.envs, self.handle)
        #print(self.item_name)
        self.item_rot = self.item.rot
        if False:
            self.gym.clear_lines(self.viewer)
            for i, env in enumerate(self.envs):
                pose = gymapi.Transform()
                pose.p.x = self.item_pos[i, 0, 0]
                pose.p.y = self.item_pos[i, 0, 1]
                pose.p.z = self.item_pos[i, 0, 2]
                # pose.r.x = self.item_rot[i, 0, 0]
                # pose.r.y = self.item_rot[i, 0, 1]
                # pose.r.z = self.item_rot[i, 0, 2]
                # pose.r.w = self.item_rot[i, 0, 3]
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)

                pose = gymapi.Transform()
                pose.p.x = self.global_franka_pos[i, 0]
                pose.p.y = self.global_franka_pos[i, 1]
                pose.p.z = self.global_franka_pos[i, 2]
                pose.r.x = self.global_franka_rot[i, 0]
                pose.r.y = self.global_franka_rot[i, 1]
                pose.r.z = self.global_franka_rot[i, 2]
                pose.r.w = self.global_franka_rot[i, 3]
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)

        # camera buf
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        """os.makedirs("camera",exist_ok=True)
        self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera.cam_handle, gymapi.IMAGE_COLOR, "camera/ur5e_hand.png")"""

        if self.output_debug_images_dir is not None:
            for i, _ in enumerate(self.envs):
                os.makedirs(self.output_debug_images_dir, exist_ok=True)

                cam_img = self.cam_tensors[i].cpu().numpy()
                imageio.imwrite(Path(self.output_debug_images_dir).joinpath("image_env_{}.png".format(i)), cam_img)

        if self.image_type == gymapi.IMAGE_COLOR:
            self.obs_buf = torch.stack(self.cam_tensors)[:, :, :, :3]
            self.obs_buf = self.obs_buf.permute(0, 3, 1, 2) / 255.0
        if self.image_type == gymapi.IMAGE_DEPTH:
            self.obs_buf = torch.stack(self.cam_tensors).unsqueeze(-1)
            self.obs_buf = self.obs_buf.permute(0, 3, 1, 2)

        self.gym.end_access_image_tensors(self.sim)

        return self.obs_buf

    def _pre_physics_step(self, actions):
        self.actions = (
            torch.from_numpy(actions.astype(np.float32)).clone().to(self.device)
        )
        if self.use_manual_action:
            self.actions[0] = self.manual_action
        done = self.franka.pre_physics_step(self.actions)

        return done

    def _post_physics_step(self):
        self._compute_observations()
        self._compute_reward(self.actions)
        self.progress_buf += 1

    def get_screen(self):
        return self.obs_buf

    def get_step(self):
        return self.progress_buf

    def seed(self, seeds):
        raise NotImplementedError()

    def close(self):
        pass
        # print("--- env close---")
        # if self.viewer is not None:
        #     self.gym.destroy_viewer(self.viewer)
        # self.gym.destroy_sim(self.sim)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    done_buf,
    progress_buf,
    actions,
    global_franka_pos,
    item_pos,
    names,
    max_episode_length,
    target_number,
    franka_height,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, int, Tensor) -> Tuple[Tensor, Tensor]

    # regularization on the actions (summed for each environment)

    dpos = (item_pos - global_franka_pos.unsqueeze(1))
    distance = torch.norm(dpos,dim=-1)
    item_height = item_pos[:, :, 2]
    
    distance_condition = distance < 0.5
    height_condition = item_height >= 0.7
    height_distance_condition = (height_condition * distance_condition).sum(dim=1) > 0

    names = (names * height_condition).sum(dim=1) == target_number

    height_rewards = torch.where(franka_height == 1, -0.1, 0.0)
    
    target_rewards = torch.where(torch.bitwise_and(names, progress_buf > 10), 1.0, 0.0)
    rewards = height_rewards + target_rewards
    done_buf = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(done_buf), done_buf
    )
    # print("rewards mean: ", rewards.mean())
    return rewards, done_buf


    """franka_height = 0 + franka_height < 0.4
    franka_height = 1 + franka_height < 0.5
    franka_height = 2 + franka_height < 0.6
    franka_height = 3 + franka_height < 0.7
    height_rewards = torch.where(franka_height == 1, 0.0,
                                    torch.where(franka_height == 2, -0.01, 
                                                torch.where(franka_height == 3, -0.05,
                                                            torch.where(franka_height == 4, -0.1, -0.5))))"""