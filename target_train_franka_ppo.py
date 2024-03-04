import argparse
import logging
from pathlib import Path
import os

from isaacgym.torch_utils import to_torch

import torch.optim as optim
import torch
from pfrl import experiments
from pfrl.agents import ppo

import numpy as np
import cv2
from PIL import Image

import requests
import datetime
import pytz
from make_urdf import URDF

##############################################################
#                           env
##############################################################
camera_ani = []
bard_ani = []
model_att_v_ani = []
model_att_a_ani = []
def write_mask(env, agent: ppo, step, model, gif_dir, image_dir):
    global camera_ani
    global bard_ani
    global model_att_v_ani
    global model_att_a_ani
    batch_size = env.num_envs
    episode = step // env.max_episode_length
    
    if step % env.max_episode_length == 0:
        if step > 0:
            for i in range(batch_size):
                camera_ani[i][0].save(Path(gif_dir).joinpath(f'gif_input/camera_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = camera_ani[i][1:] , duration = 400 , loop = 0)
                bard_ani[i][0].save(Path(gif_dir).joinpath(f'gif_bard/bard_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = bard_ani[i][1:] , duration = 400 , loop = 0)
                if model in ["mask_single_value", "mask_double", "new_mask_double", "target_mask_double"]:
                    model_att_v_ani[i][0].save(Path(gif_dir).joinpath(f'gif_value/model_att_v_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = model_att_v_ani[i][1:] , duration = 400 , loop = 0)
                if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double"]:
                    model_att_a_ani[i][0].save(Path(gif_dir).joinpath(f'gif_action/model_att_a_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = model_att_a_ani[i][1:] , duration = 400 , loop = 0)
        camera_ani = [ [] for i in range(batch_size)]
        bard_ani = [ [] for i in range(batch_size)]
        model_att_v_ani = [ [] for i in range(batch_size)]
        model_att_a_ani = [ [] for i in range(batch_size)]
    camera_all = agent.model.input_image.cpu()
    camera_all = camera_all.detach().numpy()
    bard_all = env.bard_cam_tensors
    if model in ["mask_single_value", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
        model_att_v_all = agent.model.att_v_sig5.cpu()
        model_att_v_all = model_att_v_all.detach().numpy()
    if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
        model_att_a_all = agent.model.att_a_sig5.cpu()
        model_att_a_all = model_att_a_all.detach().numpy()
        """model_att_a_all_inv = agent.model.att_a_sig5_inv.cpu()
        model_att_a_all_inv = model_att_a_all_inv.detach().numpy()"""

    for i in range(batch_size):
        camera = camera_all[i]
        camera = camera.transpose(1, 2, 0) * 255
        camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR).astype(np.uint8)
        gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
        gray = np.repeat(gray[..., np.newaxis], 3, axis=-1)
        camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        camera = Image.fromarray(camera)
        camera.save(Path(image_dir).joinpath(f'image_input/image_{i:02}_input/camera_{episode:04}_{step:04}.png'))
        camera_ani[i].append(camera)

        bard = bard_all[i].cpu().numpy()
        bard = Image.fromarray(bard)
        bard.save(Path(image_dir).joinpath(f'bard_input/image_{i:02}_bard/bard_{episode:04}_{step:04}.png'))
        bard_ani[i].append(bard)

        if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
            model_att_v = model_att_v_all[i]
            model_att_v = min_max_image(model_att_v)
            model_att_v = model_att_v.transpose(1, 2, 0) * 255
            model_att_v = cv2.resize(
                model_att_v,
                dsize=gray.shape[:2],
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            model_att_v_color = cv2.applyColorMap(model_att_v, cv2.COLORMAP_JET)
            model_att_v_color = cv2.addWeighted(
                gray, 0.7, model_att_v_color, 0.3, 0
            )
            model_att_v_color = cv2.cvtColor(model_att_v_color, cv2.COLOR_BGR2RGB)
            model_att_v_color = Image.fromarray(model_att_v_color)
            model_att_v_color.save(Path(image_dir).joinpath(f'image_value/image_{i:02}_value/value_{episode:04}_{step:04}.png'))
            model_att_v_ani[i].append(model_att_v_color)

        if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
            model_att_a = model_att_a_all[i]
            model_att_a = min_max_image(model_att_a)
            model_att_a = model_att_a.transpose(1, 2, 0) * 255
            model_att_a = cv2.resize(
                model_att_a,
                dsize=gray.shape[:2],
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            model_att_a_color = cv2.applyColorMap(model_att_a, cv2.COLORMAP_JET)
            model_att_a_color = cv2.addWeighted(
                gray, 0.7, model_att_a_color, 0.3, 0
            )
            model_att_a_color = cv2.cvtColor(model_att_a_color, cv2.COLOR_BGR2RGB)
            model_att_a_color = Image.fromarray(model_att_a_color)
            model_att_a_color.save(Path(image_dir).joinpath(f'image_action/image_{i:02}_action/action_{episode:04}_{step:04}.png'))
            model_att_a_ani[i].append(model_att_a_color)

def to_tensor(x):
    if isinstance(x, list):
        if isinstance(x[0], list):
            x_tensor = []
            for y in x:
                x_tensor.append(to_tensor(y))
        else:
            x_tensor = x
        return torch.stack(x_tensor, dim=0)
    else:
        return x


# def batch_states(states, device, phi):
#     return to_tensor(states)

def min_max_image(x):
    _min = x.min(keepdims=True)
    _max = x.max(keepdims=True)
    result = (x-_min)/(_max-_min)
    return result

def main():
    urdf = URDF()
    urdf.create_urdf()

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--isaacgym-assets-dir", type=str, default="../isaacgym/assets")
    parser.add_argument("--item-urdf-dir", type=str, default="./urdf")
    parser.add_argument(
        "--model",
        help="choose model",
        type=str,
        choices=[
            "proto1",
            "proto2",
            "mask_single_value",
            "mask_single_policy",
            "mask_double",
            "new_mask_double",
            "newnew_mask_double",
            "target_mask_double",
            "stripe_mask_double",
            "noattention",
        ],
    )
    parser.add_argument(
        "--use-lstm", action="store_true", default=False, help="use lstm"
    )
    parser.add_argument("--steps", type=int, default=10 ** 7)
    parser.add_argument("--step_offset", type=int, default=0)
    parser.add_argument("--update-batch-interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--lambd", type=float, default=0.95, help="Lambda-return factor [0, 1]"
    )
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-batch-interval", type=int, default=10 ** 4)
    parser.add_argument("--eval-n-runs", type=int, default=128)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument(
        "--max-grad-norm", type=float, default=40, help="value loss coefficient"
    )
    """parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')"""
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=0,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--output-sensor-images-dir",
        type=str,
        default=None,
        help="Output sensor images directory. Image files are updated per step",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=1,
        help="Number of items on tray.",
    )
    parser.add_argument(
        "--item-names", nargs="+", default=None, help=["List of item names."]
    )
    parser.add_argument("--descentstep", type=int, default=12)
    
    parser.add_argument(
        "--target",
        type=str,
        default="item21",
        help="target-item",
    )
    parser.add_argument("--drop", action="store_true", default=False, help="use drop")
    parser.add_argument("--hand", action="store_true", default=False, help="use handcamera")
    parser.add_argument(
        "--mode",
        help="choose mode",
        type=str,
        default="normal",
        choices=[
            "normal",
            "hard",
            "veryhard",
        ],
    )

    

    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()
    
    """args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')"""
    
    logging.basicConfig(level=args.log_level)
    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))
    outdir = args.outdir

    height = 128
    width = 128
    action_repeat = 30
    item_asset_root = args.item_urdf_dir
    isaacgym_asset_root = args.isaacgym_assets_dir

    num_items = args.num_items
    item_names = args.item_names
    if item_names is None:
        item_names = sorted(list(Path(item_asset_root).glob("*.urdf")))
        item_names = [path.stem for path in item_names]
    output_debug_images_dir = args.output_sensor_images_dir

    from envs.target_franka_grasping_env import FrankaGraspingEnv

    def make_batch_env(num_envs):
        env = FrankaGraspingEnv(
            num_envs=num_envs,
            height=height,
            width=width,
            discrete=False,
            image_type="color",
            item_asset_root=item_asset_root,
            isaacgym_asset_root=isaacgym_asset_root,
            num_items=num_items,
            item_names=item_names,
            use_viewer=args.render,
            action_repeat=action_repeat,
            output_debug_images_dir=output_debug_images_dir,
            device_id=args.gpu,
            steps = args.steps/(args.descentstep * args.num_envs),
            descentstep = args.descentstep,
            target = args.target,
            hand = args.hand,
            demo = args.demo,
        )
        return env

    envs = make_batch_env(num_envs=args.num_envs)

    if args.model == "proto1":
        from agent.model_proto1 import ActorCritic

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            shared_layers=[128, 64],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
        )
    elif args.model == "proto2":
        from agent.model_proto2 import ActorCritic

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
        )
    elif args.model == "mask_single_value":
        from agent.model_mask_single_value import ActorCritic

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
        )
    elif args.model == "mask_single_policy":
        from agent.model_mask_single_policy import ActorCritic

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
        )
    elif args.model == "mask_double":
        from agent.model_mask_double import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
        )
    elif args.model == "target_mask_double":
        from agent.target_mask_double import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
            target = args.target,
        )
    elif args.model == "stripe_mask_double":
        from agent.stripe_mask_double import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            step = args.steps/(args.descentstep * args.num_envs),
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
            target = args.target,
            outdir = outdir,
        )
    elif args.model == "new_mask_double":
        from agent.new_model_mask_double import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            demo = args.demo,
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
            target = args.target,
        )
    elif args.model == "newnew_mask_double":
        from agent.newnew_model_mask_double import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
            target = args.target,
        )
    elif args.model == "noattention":
        from agent.noattention import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=args.use_lstm,
            target = args.target,
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    update_step_interval = args.num_envs * envs.max_episode_length * args.update_batch_interval

    agent = ppo.PPO(
        model,
        optimizer,
        gamma=args.gamma,
        lambd=args.lambd,
        gpu=args.gpu,
        update_interval=update_step_interval,
        max_grad_norm=args.max_grad_norm,
        minibatch_size=(args.descentstep * args.num_envs),
        epochs=args.epochs,
        recurrent=args.use_lstm,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:

        gif_dir = Path(args.outdir).joinpath("gif_images")
        gif_input = Path(gif_dir).joinpath("gif_input")
        gif_bard = Path(gif_dir).joinpath("gif_bard")
        gif_action = Path(gif_dir).joinpath("gif_action")
        gif_value = Path(gif_dir).joinpath("gif_value")
        image_dir = Path(args.outdir).joinpath("images")
        image_input = Path(image_dir).joinpath("image_input")
        bard_input = Path(image_dir).joinpath("bard_input")
        image_action = Path(image_dir).joinpath("image_action")
        image_value = Path(image_dir).joinpath("image_value")
        os.makedirs(gif_dir)
        os.makedirs(gif_input)
        os.makedirs(gif_bard)
        os.makedirs(gif_action)
        os.makedirs(gif_value)
        os.makedirs(image_dir)
        os.makedirs(image_input)
        os.makedirs(bard_input)
        os.makedirs(image_action)
        os.makedirs(image_value)
        for i in range(envs.num_envs):
            in_ep = "image_%02d_input" % (i)
            bd_ep = "image_%02d_bard" % (i)
            ac_ep = "image_%02d_action" % (i)
            va_ep = "image_%02d_value" % (i)
            image_ep_input = Path(image_input).joinpath(in_ep)
            image_ep_bard = Path(bard_input).joinpath(bd_ep)
            image_ep_action = Path(image_action).joinpath(ac_ep)
            image_ep_value = Path(image_value).joinpath(va_ep)
            os.makedirs(image_ep_input)
            os.makedirs(image_ep_bard)
            os.makedirs(image_ep_action)
            os.makedirs(image_ep_value)

        with agent.eval_mode():
            obss = envs.reset()
            step = 0
            while True:
                actions = agent.batch_act(obss, args.demo)
                obss, rs, dones, infos = envs.step(actions)
                
                if args.model != "noattention":
                    write_mask(envs, agent, step, args.model, gif_dir, image_dir)

                step +=1
                if step % 12 == 0:
                    print(rs)
                    rs = list(rs)
                    now = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%m%d%H%M%S")
                    """score = "rsj/" + args.model +"_"+ str(now) + '_score.txt'
                    np.savetxt(score, rs, delimiter=",")"""
                    sam = sum(rs)
                    age = np.mean(rs)
                    print("合計：", sam, "平均値：",age)
                
                resets = np.zeros(args.num_envs, dtype=bool)

                # Agent observes the consequences
                agent.batch_observe(obss, rs, dones, resets)

                # Make mask. 0 if done/reset, 1 if pass
                end = np.logical_or(resets, dones)
                not_end = np.logical_not(end)

                obss = envs.reset(not_end)

    else:

        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        )

        eval_interval =  envs.max_episode_length * envs.num_envs * args.eval_batch_interval
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=envs,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=eval_interval,
            outdir=args.outdir,
            step_offset=args.step_offset,
            save_best_so_far_agent=True,
            log_interval=update_step_interval,
            step_hooks=step_hooks,
            checkpoint_freq=1000000,
            use_tensorboard=True,
        )
        


if __name__ == "__main__":
    main()