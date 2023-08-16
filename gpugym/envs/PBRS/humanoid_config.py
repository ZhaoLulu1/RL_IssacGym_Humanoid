"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch  # 导入PyTorch库
from gpugym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotCfgPPO  # 导入自定义模块 LeggedRobotCfg 和 LeggedRobotCfgPPO


class HumanoidCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096  # 环境数量
        num_observations = 38  # 观测维度数
        num_actions = 10  # 动作维度数
        episode_length_s = 20  # 每个回合的时长（秒）

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False  # 是否使用课程学习
        mesh_type = 'plane'  # 地形网格类型
        measure_heights = False  # 是否测量高度

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # 是否使用课程学习
        max_curriculum = 1.  # 最大课程
        num_commands = 4  # 命令维度数
        resampling_time = 5.  # 重新采样时间间隔
        heading_command = False  # 是否使用朝向命令
        ang_vel_command = True  # 是否使用角速度命令

        class ranges:
            # 训练命令范围 #
            lin_vel_x = [2., 3.]        # 线速度 x 范围 [最小值, 最大值]（米/秒）
            lin_vel_y = [-0.75, 0.75]   # 线速度 y 范围 [最小值, 最大值]（米/秒）
            ang_vel_yaw = [-2., 2.]     # 角速度 yaw 范围 [最小值, 最大值]（弧度/秒）
            heading = [0., 0.]  # 朝向范围 [最小值, 最大值]（弧度）

            # 游玩命令范围 #
            # lin_vel_x = [3., 3.]    # 线速度 x 范围 [最小值, 最大值]（米/秒）
            # lin_vel_y = [-0., 0.]     # 线速度 y 范围 [最小值, 最大值]（米/秒）
            # ang_vel_yaw = [0, 0]      # 角速度 yaw 范围 [最小值, 最大值]（弧度/秒）
            # heading = [0, 0]  # 朝向范围 [最小值, 最大值]（弧度）


    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_range'  # 重置模式为将状态重置到指定范围内
        penetration_check = False  # 是否进行穿透检查，这里设置为不检查

        # 初始位置 [x, y, z]，单位：米
        pos = [0.0, 0.0, 2]

        # 初始旋转 [x, y, z, w]，四元数表示
        rot = [0.0, 0.0, 0.0, 1.0]

        # 初始线性速度 [x, y, z]，单位：米/秒
        lin_vel = [0.0, 0.0, 0.0]

        # 初始角速度 [x, y, z]，单位：弧度/秒
        ang_vel = [0.0, 0.0, 0.0]

        # 各个状态维度的范围
        root_pos_range = [
            [0., 0.],  # x 范围 [最小值, 最大值]
            [0., 0.],  # y 范围 [最小值, 最大值]
            [0.72, 0.72],  # z 范围 [最小值, 最大值]
            [-torch.pi / 10, torch.pi / 10],  # roll 范围 [最小值, 最大值]
            [-torch.pi / 10, torch.pi / 10],  # pitch 范围 [最小值, 最大值]
            [-torch.pi / 10, torch.pi / 10]  # yaw 范围 [最小值, 最大值]
        ]

        # 各个状态维度的速度范围
        root_vel_range = [
            [-.5, .5],  # v_x 范围 [最小值, 最大值]
            [-.5, .5],  # v_y 范围 [最小值, 最大值]
            [-.5, .5],  # v_z 范围 [最小值, 最大值]
            [-.5, .5],  # w_x 范围 [最小值, 最大值]
            [-.5, .5],  # w_y 范围 [最小值, 最大值]
            [-.5, .5]  # w_z 范围 [最小值, 最大值]
        ]

        # 默认关节角度
        default_joint_angles = {
            'left_hip_yaw': 0.,  # 左髋偏航角
            'left_hip_abad': 0.,  # 左髋俯仰角
            'left_hip_pitch': -0.2,  # 左髋偏航角
            'left_knee': 0.25,  # 左膝关节
            'left_ankle': 0.0,  # 左踝关节
            'right_hip_yaw': 0.,  # 右髋偏航角
            'right_hip_abad': 0.,  # 右髋俯仰角
            'right_hip_pitch': -0.2,  # 右髋偏航角
            'right_knee': 0.25,  # 右膝关节
            'right_ankle': 0.0,  # 右踝关节
        }

        # 各关节角度范围
        dof_pos_range = {
            'left_hip_yaw': [-0.1, 0.1],  # 左髋偏航角范围
            'left_hip_abad': [-0.2, 0.2],  # 左髋俯仰角范围
            'left_hip_pitch': [-0.2, 0.2],  # 左髋偏航角范围
            'left_knee': [0.6, 0.7],  # 左膝关节范围
            'left_ankle': [-0.3, 0.3],  # 左踝关节范围
            'right_hip_yaw': [-0.1, 0.1],  # 右髋偏航角范围
            'right_hip_abad': [-0.2, 0.2],  # 右髋俯仰角范围
            'right_hip_pitch': [-0.2, 0.2],  # 右髋偏航角范围
            'right_knee': [0.6, 0.7],  # 右膝关节范围
            'right_ankle': [-0.3, 0.3],  # 右踝关节范围
        }

        # 各关节角速度范围
        dof_vel_range = {
            'left_hip_yaw': [-0.1, 0.1],  # 左髋偏航角速度范围
            'left_hip_abad': [-0.1, 0.1],  # 左髋俯仰角速度范围
            'left_hip_pitch': [-0.1, 0.1],  # 左髋偏航角速度范围
            'left_knee': [-0.1, 0.1],  # 左膝关节速度范围
            'left_ankle': [-0.1, 0.1],  # 左踝关节速度范围
            'right_hip_yaw': [-0.1, 0.1],  # 右髋偏航角速度范围
            'right_hip_abad': [-0.1, 0.1],  # 右髋俯仰角速度范围
            'right_hip_pitch': [-0.1, 0.1],  # 右髋偏航角速度范围
            'right_knee': [-0.1, 0.1],  # 右膝关节速度范围
            'right_ankle': [-0.1, 0.1],  # 右踝关节速度范围
        }

    class control(LeggedRobotCfg.control):
        # 关节刚度和阻尼
        stiffness = {
            'left_hip_yaw': 30.,  # 左髋偏航角刚度
            'left_hip_abad': 30.,  # 左髋俯仰角刚度
            'left_hip_pitch': 30.,  # 左髋偏航角刚度
            'left_knee': 20.,  # 左膝关节刚度
            'left_ankle': 30.,  # 左踝关节刚度
            'right_hip_yaw': 30.,  # 右髋偏航角刚度
            'right_hip_abad': 30.,  # 右髋俯仰角刚度
            'right_hip_pitch': 30.,  # 右髋偏航角刚度
            'right_knee': 20.,  # 右膝关节刚度
            'right_ankle': 30.,  # 右踝关节刚度
        }

        damping = {
            'left_hip_yaw': 5.,  # 左髋偏航角阻尼
            'left_hip_abad': 5.,  # 左髋俯仰角阻尼
            'left_hip_pitch': 5.,  # 左髋偏航角阻尼
            'left_knee': 3.,  # 左膝关节阻尼
            'left_ankle': 5.,  # 左踝关节阻尼
            'right_hip_yaw': 5.,  # 右髋偏航角阻尼
            'right_hip_abad': 5.,  # 右髋俯仰角阻尼
            'right_hip_pitch': 5.,  # 右髋偏航角阻尼
            'right_knee': 3.,  # 右膝关节阻尼
            'right_ankle': 5.  # 右踝关节阻尼
        }

        action_scale = 1.0  # 动作缩放
        exp_avg_decay = None  # 指数移动平均衰减（未设置）
        decimation = 10  # 降采样

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False  # 是否随机化摩擦
        friction_range = [0.5, 1.25]  # 摩擦范围 [最小值, 最大值]

        randomize_base_mass = False  # 是否随机化基础质量
        added_mass_range = [-1., 1.]  # 额外质量范围 [最小值, 最大值]

        push_robots = True  # 是否推动机器人
        push_interval_s = 2.5  # 推动间隔时间，单位：秒
        max_push_vel_xy = 0.5  # 最大推动速度（x和y方向），单位：米/秒

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}' \
               '/resources/robots/ncstate_humanoid/big_chungus.urdf'
        keypoints = ["base"]  # 关键点，这里只包含 "base"
        end_effectors = ['left_foot', 'right_foot']  # 末端执行器，左右脚
        foot_name = 'foot'  # 脚的名称
        terminate_after_contacts_on = [
            'base',  # 在这些接触点发生后终止
            'left_upper_leg',  # 左上腿
            'left_lower_leg',  # 左小腿
            'right_upper_leg',  # 右上腿
            'right_lower_leg',  # 右小腿

        ]

        disable_gravity = False  # 是否禁用重力
        disable_actions = False  # 是否禁用动作
        disable_motors = False  # 是否禁用马达

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0  # 自身碰撞，默认为禁用
        collapse_fixed_joints = False  # 是否折叠固定关节
        flip_visual_attachments = False  # 是否翻转视觉附件

        # 检查 GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3  # 默认关节驱动模式为 effort（力）

    class rewards(LeggedRobotCfg.rewards):
        # ! "Incorrect" specification of height
        # 高度的错误规定
        # base_height_target = 0.7
        base_height_target = 0.62  # 基准高度目标
        soft_dof_pos_limit = 0.9  # 关节位置限制软约束
        soft_dof_vel_limit = 0.9  # 关节速度限制软约束
        soft_torque_limit = 0.8  # 关节扭矩限制软约束

        # negative total rewards clipped at zero (avoids early termination)
        # 将负总奖励剪切为零（避免过早终止）
        only_positive_rewards = False  # 仅使用正奖励
        tracking_sigma = 0.5  # 跟踪标准差

        class scales(LeggedRobotCfg.rewards.scales):
            # * "True" rewards * #
            action_rate = -1.e-3
            action_rate2 = -1.e-4
            tracking_lin_vel = 10.
            tracking_ang_vel = -4.
            torques = -1e-4
            dof_pos_limits = -10
            torque_limits = -1e-2
            termination = -100
            feet_air_time = 4

            # * Shaping rewards * #
            # Sweep values: [0.5, 2.5, 10, 25., 50.]
            # Default: 5.0
            # orientation = 5.0

            # Sweep values: [0.2, 1.0, 4.0, 10., 20.]
            # Default: 2.0
            # base_height = 2.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # joint_regularization = 1.0

            # * PBRS rewards * #
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            ori_pb = 1.5

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            baseHeight_pb = 1.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            jointReg_pb = 4.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            base_z = 1. / 0.6565  # 基准高度的比例尺

        clip_observations = 100.  # 观测值剪切范围
        clip_actions = 10.  # 动作剪切范围

    class noise(LeggedRobotCfg.noise):
        add_noise = True  # 是否添加噪声
        noise_level = 1.0  # 噪声的比例尺（用于缩放其他值）

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            base_z = 0.05  # 基准高度的噪声比例
            dof_pos = 0.005  # 关节位置的噪声比例
            dof_vel = 0.01  # 关节速度的噪声比例
            lin_vel = 0.1  # 线性速度的噪声比例
            ang_vel = 0.05  # 角速度的噪声比例
            gravity = 0.05  # 重力的噪声比例
            in_contact = 0.1  # 接触状态的噪声比例
            height_measurements = 0.1  # 高度测量的噪声比例

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 时间步长
        substeps = 1  # 子步数
        gravity = [0., 0., -9.81]  # 重力矢量

        class physx:
            max_depenetration_velocity = 10.0  # 最大去穿透速度

class HumanoidCfgPPO(LeggedRobotCfgPPO):
    do_wandb = False  # 是否使用WandB进行记录
    seed = -1  # 随机种子

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # 算法训练超参数
        value_loss_coef = 1.0  # 值损失系数
        use_clipped_value_loss = True  # 是否使用剪切的值损失
        clip_param = 0.2  # 剪切参数
        entropy_coef = 0.01  # 熵系数
        num_learning_epochs = 5  # 训练轮数
        num_mini_batches = 4  # 小批量数 = 环境数 * 步数 / 小批量数
        # learning_rate = 1.e-5  # 学习率
        learning_rate = 5.e-4  # 学习率
        schedule = 'adaptive'  # 调度方式（可选：自适应，固定）
        gamma = 0.99  # 折扣因子
        lam = 0.95  # λ参数
        # desired_kl = 0.01  # 期望的KL散度
        desired_kl = 0.008  # 期望的KL散度
        max_grad_norm = 1.  # 梯度裁剪阈值

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 24  # 每个环境的步数
        max_iterations = 8000  # 最大迭代次数
        run_name = 'chungus'  # 运行名称
        experiment_name = 'PBRS_HumanoidLocomotion'  # 实验名称
        save_interval = 50  # 保存间隔
        plot_input_gradients = False  # 是否绘制输入梯度
        plot_parameter_gradients = False  # 是否绘制参数梯度

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256, 256]  # 演员网络的隐藏层维度
        critic_hidden_dims = [256, 256, 256]  # 评论家网络的隐藏层维度
        # （elu，relu，selu，crelu，lrelu，tanh，sigmoid）
        activation = 'elu'  # 激活函数
