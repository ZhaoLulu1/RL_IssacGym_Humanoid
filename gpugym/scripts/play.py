# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch

# 检查文件夹是否存在，不存在则创建
folder_path = '../analysis/data/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)  # 获取环境和训练配置
    # 覆盖一些参数以进行测试
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)  # 最小化环境数目
    env_cfg.terrain.num_rows = 5  # 地形行数
    env_cfg.terrain.num_cols = 5  # 地形列数
    env_cfg.terrain.curriculum = False  # 是否使用课程设置
    env_cfg.noise.add_noise = True  # 添加噪音
    env_cfg.domain_rand.randomize_friction = False  # 随机化摩擦力
    env_cfg.domain_rand.push_robots = False  # 是否推动机器人（True/False）
    env_cfg.domain_rand.push_interval_s = 2  # 推动间隔时间（秒）
    env_cfg.domain_rand.max_push_vel_xy = 1.0  # 最大推动速度（水平方向）
    env_cfg.init_state.reset_ratio = 0.8  # 重置比率

    # 准备环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)  # 创建环境
    obs = env.get_observations()  # 获取观测数据

    # 加载策略
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)  # 创建算法运行器
    policy = ppo_runner.get_inference_policy(device=env.device)  # 获取推断策略

    # 将策略导出为 jit 模块（用于从 C++ 运行）
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')  # 路径设置
        export_policy(ppo_runner.alg.actor_critic, path)  # 导出策略
        print('已导出策略模型至：', path)

    # 将评论家导出为 jit 模块（用于从 C++ 运行）
    if EXPORT_CRITIC:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'critics')  # 路径设置
        export_critic(ppo_runner.alg.actor_critic, path)  # 导出评论家
        print('已导出评论家模型至：', path)

    logger = Logger(env.dt)  # 创建日志记录器
    robot_index = 0  # 用于记录的机器人索引
    joint_index = 2  # 用于记录的关节索引
    stop_state_log = 1000  # 绘制状态前的步数
    stop_rew_log = env.max_episode_length + 1  # 打印平均回报前的步数
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)  # 摄像机位置
    camera_vel = np.array([1., 1., 0.])  # 摄像机速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)  # 摄像机朝向
    img_idx = 0  # 图像索引

    play_log = []  # 用于记录播放数据的列表
    #env.max_episode_length = 1000./env.dt
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())  # 根据观测数据获取动作
        obs, _, rews, dones, infos = env.step(actions.detach())  # 在环境中执行动作
        if RECORD_FRAMES:
            if i % 2:
                文件名 = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")  # 文件名设置
                env.gym.write_viewer_image_to_file(env.viewer, 文件名)  # 将视图图像写入文件
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt  # 更新摄像机位置
            env.set_camera(camera_position, camera_position + camera_direction)  # 设置摄像机位置和朝向

        if i < stop_state_log:
            ### 人型机器人 PBRS 记录 ###
            # [ 1]  时间步
            # [38]  代理观察
            # [10]  代理动作（关节设定点）
            # [13]  世界坐标系中的浮动基座状态
            # [ 6]  脚的接触力
            # [10]  关节力矩
            play_log.append(
                [i*env.dt]
                + obs[robot_index, :].cpu().numpy().tolist()
                + actions[robot_index, :].detach().cpu().numpy().tolist()
                + env.root_states[0, :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env.torques[robot_index, :].detach().cpu().numpy().tolist()
            )
        elif i == stop_state_log:
            np.savetxt('../analysis/data/play_log.csv', play_log, delimiter=',')  # 将播放数据保存为 CSV 文件
            # logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                # if num_episodes>0:
                    # logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()  # 打印奖励信息

if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_CRITIC = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()  # 获取命令行参数
    play(args)  # 调用 play 函数进行实验

