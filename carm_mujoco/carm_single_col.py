from typing import List, Tuple, Dict, Optional   # 新增 Tuple, Dict
import mujoco
import mujoco.viewer
import numpy as np
import os
import json

class CArmSingleCol:
    def __init__(self, xml_path: str, render: bool = True):
        """
        Args:
            xml_path: MJCF/XML模型路径
            render: 是否打开MuJoCo Viewer
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render = render
        self.viewer = None

        # 获取末端site的ID（假设XML中已添加 name="ee_flange" 的site）
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_flange"
        )

        # 夹爪关节索引（0‑based，joint7=6, joint8=7）
        self.gripper_joint1_id = 6
        self.gripper_joint2_id = 7

        self.arm_body_ids = set()
        for i in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith('link'):   # 改成你模型中的实际前缀
                self.arm_body_ids.add(i)


        self.tip_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_right")
        self.tip_left_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_left")
        
        self.control_mode = 0                     # 0: IDLE
        # # 机械臂位置环 PI 增益
        # self.kp_arm_pos = 70.0
        # self.ki_arm_pos = 5.0
        # # 机械臂速度环 PI 增益
        # self.kp_arm_vel = 30.0
        # self.ki_arm_vel = 2.0
        # 机械臂位置环 PI 增益 #修复了抓取时，机械臂会抖动的bug
        self.kp_arm_pos = 30.0
        self.ki_arm_pos = 1.0
        # 机械臂速度环 PI 增益
        self.kp_arm_vel = 10.0
        self.ki_arm_vel = 1.0
        self.kd_arm_vel = 5.0

        # 夹爪位置 PD 增益（简化处理）
        # 注意：夹爪已改为 motor 力矩执行器，ctrl[6] 直接是力矩，
        # 因此这里按力矩控制的量级调参（有效惯量约 0.042 kg，临界阻尼 kd≈6@kp=200）
        self.kp_grip = 200.0
        self.kd_grip = 10.0

        self.gripper_pos_pre = 0.0

        # set_gripper 待生效力矩：在 step() 内、_update_traj_control() 之后应用，
        # 保证不被轨迹/保持逻辑每步重新下发的夹爪力矩覆盖
        self._gripper_pending = None

        # 夹爪力限 PI 控制参数 (set_gripper)
        self.kp_grip_pi = 100.0            # 夹爪 PI 比例增益 (N/m)
        self.ki_grip_pi = 20.0             # 夹爪 PI 积分增益 (N/(m·s))
        self.gripper_err_int = 0.0         # 夹爪 PI 积分状态
        self.gripper_err_int_limit = 0.5   # 积分限幅（防 windup）

        self.kp_mit =np.diag([300,300,250,250,250,250])  # MIT 位置环增益（对每个关节单独设置）
        self.kd_mit =np.diag([5,5,5,5,5,5])       # MIT 速度环增益（对每个关节单独设置）

        # 积分状态（机械臂 6 关节）
        self.pos_error_int = np.zeros(6)   # 位置环积分
        self.vel_error_int = np.zeros(6)   # 速度环积分

        # 积分限幅（防止 windup）
        self.pos_int_limit = 10.0
        self.vel_int_limit = 10.0

        if self.render:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

        self.joint_tau_sensor_ids = [self.model.sensor(f"joint{i}_f").id for i in range(1, 7)]

        self.damping = self.model.dof_damping[:6]
        self.frictionloss = self.model.dof_frictionloss[:6]

        # ===== 轨迹跟踪状态 =====
        self._traj_active = False
        self._traj_start_time = 0.0
        self._traj_duration = 0.0
        self._traj_q0 = None
        self._traj_qf = None

        # ===== 关节运动约束 (用于 desire_time = -1 的 double S 规划) =====
        self.joint_vel_limits = np.array([4.2, 4.2, 3.14, 3.14, 3.14, 3.14])  # rad/s
        self.joint_acc_limits = np.array([8.4, 8.4, 6.28, 6.28, 6.28, 6.28])  # rad/s²
        self.joint_jerk_limits = np.array([37.8, 37.8, 28.26, 28.26, 28.26, 28.26])  # rad/s³

        # Double S 轨迹相关变量
        self._ds_active = False
        self._ds_start_time = 0.0
        self._ds_duration = 0.0
        self._ds_target = None  # 目标关节位置 (6,)
        self._ds_params = []    # 各关节的双 S 参数列表

        self._hold_target = None 
        self._traj_queue = []         # 待执行轨迹队列

        self._line_ds_active = False
        self._line_ds_start_time = 0.0
        self._line_ds_duration = 0.0
        self._line_ds_q0 = None          # 起点
        self._line_ds_dq = None          # 总位移向量
        self._line_ds_params = None      # λ 的双 S 参数字典

        self._plan_q = self.data.qpos[:6].copy()   # 初始化为当前关节角
        self._plan_v = np.zeros(6)
        self._plan_tau = np.zeros(6)
        
        # 示教录制相关
        self._teach_data = {}           # name -> {'joint_pos': list, 'time_stamps': list}
        self._recording = False         # 是否正在录制
        self._rec_name = ""             # 当前录制名称
        self._rec_joint_pos = []        # 临时存储关节角
        self._rec_time_stamps = []      # 临时存储相对时间
        self._rec_start_time = 0.0      # 录制开始时刻
        self._prev_control_mode = 0

        self._export_dir = "./traj_data"
        os.makedirs(self._export_dir, exist_ok=True)

    def step(self, nstep: int = 1):
        """
        推进仿真
        """
        for _ in range(nstep):
            if self._recording:
                # 记录当前6个关节角和相对时间
                self._rec_joint_pos.append(self.data.qpos[:6].copy())
                self._rec_time_stamps.append(self.data.time - self._rec_start_time)
            self._update_traj_control()
            if self._gripper_pending is not None:
                # set_gripper 的力矩最后应用（锁存），避免被 _update_traj_control 里的
                # track_joint/保持逻辑覆盖；直到下次 set_gripper 更新或
                # track_joint(gripper_pos>=0) 切回位置式控制才取消
                self.data.ctrl[6] = self._gripper_pending
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        """
        关闭viewer
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_joint_pos(self) -> List[float]:
        """
        获取6个关节角度(rad)
        """
        return self.data.qpos[:6].tolist()

    def get_joint_vel(self) -> List[float]:
        """
        获取6个关节角速度(rad/s)
        """
        return self.data.qvel[:6].tolist()

    def get_joint_tau(self) -> List[float]:
        """
        获取6个关节力矩(Nm)
        """
        return self.data.qfrc_actuator[:6].tolist()
    # def get_joint_tau(self) -> List[float]:
    #     """
    #     获取6个关节力矩(Nm)，通过MuJoCo底层传感器数组读取
    #     """
    #     # 直接去底层大数组 sensordata 里按 ID 取值，绝对不会报错
    #     return [float(self.data.sensordata[idx]) for idx in self.joint_tau_sensor_ids]

    def get_cart_pose(self) -> List[float]:
        """
        获取末端法兰盘中心位姿
        Returns:
            [x, y, z, qx, qy, qz, qw]  (位置 + 四元数)
        """
        # 确保运动学数据是最新的（若紧跟step调用，此步可省略，但保留也无妨）
        mujoco.mj_kinematics(self.model, self.data)

        pos = self.data.site_xpos[self.ee_site_id].copy()
        rot_mat = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)

        # 旋转矩阵 -> 四元数 (MuJoCo顺序: w, x, y, z)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot_mat.ravel())

        # 返回 [x, y, z, qx, qy, qz, qw]
        return [
            pos[0], pos[1], pos[2],
            quat[1], quat[2], quat[3], quat[0]
        ]

    def get_plan_cart_pose(self) -> List[float]:
        """
        获取控制法兰相对基座的期望位姿（基于当前规划/期望关节角 _plan_q）

        Returns:
            [x, y, z, qx, qy, qz, qw]  (位置 + 四元数)
            求解失败时返回空列表
        """
        ret, pose = self.forward_kine(self._plan_q.tolist(), tool_index=0)
        if ret != 1:
            return []
        return pose

    def get_joint_external_tau(self) -> List[float]:
        """
        获取关节空间的所有外部力矩 (Nm)，仅取前 6 个机械臂关节。
        包含：环境接触力 + 用户施加的笛卡尔力(xfrc_applied) + 直接关节力矩(qfrc_applied)
        排除：重力、关节限位力、自碰撞
        """
        nv = self.model.nv
        tau = np.zeros(nv)

        # ---------- 1. 接触力 (机械臂 ↔ world) ----------
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]

            in_arm1 = body1 in self.arm_body_ids
            in_arm2 = body2 in self.arm_body_ids

            # 跳过的情况：
            # 1. 两个都是机械臂连杆（自碰撞）
            # 2. 两个都不是机械臂（障碍物之间碰撞等，无关）
            if (in_arm1 and in_arm2) or (not in_arm1 and not in_arm2):
                continue

            # 到此，必然是一个是机械臂，另一个是外部物体
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            # mj_contactForce 返回作用在 geom2 上的力（世界系）
            # 我们需要得到环境对机械臂的作用力

            if in_arm2:
                # geom2 是机械臂，力已经是环境→机械臂
                arm_body = body2
                # 力方向无需调整
            else:
                # geom1 是机械臂，geom2 是外部物体，需要反向得到环境→机械臂
                arm_body = body1
                force = -force

            # 接触点在 arm_body 局部坐标
            arm_pos = self.data.xpos[arm_body]
            arm_mat = self.data.xmat[arm_body].reshape(3, 3)
            global_point = con.pos
            local_point = arm_mat.T @ (global_point - arm_pos)  # 替代 mju_global2body

            # arm_body 在接触点的雅可比
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, local_point, arm_body)

            tau += jacp.T @ force[:3] + jacr.T @ force[3:]

        # ---------- 2. 用户施加的笛卡尔力 (xfrc_applied) ----------
        for body_id in range(1, self.model.nbody):
            xfrc = self.data.xfrc_applied[body_id]
            if np.all(xfrc == 0):
                continue
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, body_id)
            tau += jacp.T @ xfrc[:3] + jacr.T @ xfrc[3:]

        # ---------- 3. 直接关节力矩 (qfrc_applied) ----------
        tau += self.data.qfrc_applied

        return tau[:6].tolist()
    def get_cart_external_tau(self) -> List[float]:
        """
        获取末端 site 坐标系下的总外力旋量 [Fx,Fy,Fz,Mx,My,Mz] (N, Nm)。
        包含：环境接触力 + 用户施加的笛卡尔力(xfrc_applied) + 直接关节力矩(qfrc_applied)的等效末端力。
        排除：重力、关节限位力、自碰撞
        """
        ee_pos = self.data.site_xpos[self.ee_site_id]
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        wrench_local = np.zeros(6)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]

            # 跳过两个都是世界（极少发生）
            if body1 == 0 and body2 == 0:
                continue
            # 核心修复：只跳过机械臂自身的连杆碰撞
            if body1 in self.arm_body_ids and body2 in self.arm_body_ids:
                continue

            # 确定哪个是机械臂本体（可能同时有多个机械臂连杆？此处按接触力方向处理）
            if body1 == 0:
                arm_body = body2
            else:
                arm_body = body1   # 这里假设至少有一个属于机械臂，或者更严谨地判断

            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if body1 == 0:
                force = -force

            # 力系平移、旋转到末端局部坐标系 (原代码)
            contact_pos = con.pos
            arm = contact_pos - ee_pos
            f_world = force[:3]
            m_world = force[3:] + np.cross(arm, f_world)

            # f_local = ee_mat.T @ f_world
            # m_local = ee_mat.T @ m_world
            f_local = f_world
            m_local = m_world
            wrench_local[:3] += f_local
            wrench_local[3:] += m_local

        # ---------- 2. 用户施加的笛卡尔力 (xfrc_applied) ----------
        for body_id in range(1, self.model.nbody):
            xfrc = self.data.xfrc_applied[body_id]
            if np.all(xfrc == 0):
                continue
            body_pos = self.data.xpos[body_id]
            arm = body_pos - ee_pos
            f_world = xfrc[:3]
            m_world = xfrc[3:] + np.cross(arm, f_world)

            # f_local = ee_mat.T @ f_world
            # m_local = ee_mat.T @ m_world
            f_local = f_world
            m_local = m_world
            wrench_local[:3] += f_local
            wrench_local[3:] += m_local

        # ---------- 3. 关节力矩 (qfrc_applied) 等效到末端笛卡尔力 ----------
        # 计算末端 site 的世界雅可比 (6×nv)
        site_pos_local = self.model.site_pos[self.ee_site_id]
        parent_body = self.model.site_bodyid[self.ee_site_id]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, site_pos_local, parent_body)
        J = np.vstack((jacp, jacr))

        # 取前6个关节的雅可比（假设机械臂为6自由度固定基座）
        J6 = J[:, :6]
        tau_applied = self.data.qfrc_applied[:6]
        if np.any(tau_applied):
            # 使用伪逆，避免奇异
            try:
                J_inv_T = np.linalg.inv(J6).T
            except np.linalg.LinAlgError:
                J_inv_T = np.linalg.pinv(J6).T
            wrench_from_qfrc = J_inv_T @ tau_applied   # 世界坐标系末端力旋量
            # 转到末端局部坐标系
            # f_local = ee_mat.T @ wrench_from_qfrc[:3]
            # m_local = ee_mat.T @ wrench_from_qfrc[3:]
            f_local =  wrench_from_qfrc[:3]
            m_local =  wrench_from_qfrc[3:]
            wrench_local[:3] += f_local
            wrench_local[3:] += m_local

        return wrench_local.tolist()
    
        # ---------- 夹爪接口 ----------
    def get_gripper_pos(self) -> float:
        """夹爪两指尖间距（米）"""
        mujoco.mj_kinematics(self.model, self.data)
        p_right = self.data.site_xpos[self.tip_right_id]
        p_left  = self.data.site_xpos[self.tip_left_id]
        return float(np.linalg.norm(p_right - p_left))

    def get_gripper_vel(self) -> float:
        """
        获取夹爪两指相对运动速度 (米/秒)，正值表示张开，负值表示闭合。
        开合程度 ≈ 2*joint7（两指各滑 joint7），故张开速度 = 2*v_joint7。
        """
        v7 = self.data.qvel[self.gripper_joint1_id]
        return float(2.0 * v7)

    def get_gripper_tau(self) -> float:
        """
        获取夹爪驱动力/力矩总和 (N)。
        模型只有 joint7 有 actuator，joint8 无力执行器，因此返回 joint7 的执行器力。
        """
        return float(self.data.qfrc_actuator[self.gripper_joint1_id])

    def set_gripper(self, pos: float, tau: float = 10.0) -> int:
        """
        夹爪力限位置控制（PI + 力矩限幅），每步下发。

        以误差 e = pos - 当前开合程度 做 PI 控制，输出夹爪力矩并限幅到 [-tau, tau]。
        模型夹爪为 motor（力矩型）执行器，ctrl[6] 直接是力矩，因此限幅真实生效。

        Args:
            pos: 目标开合程度（指尖间距，米），范围约 [0, 0.074]
            tau: 夹持力矩限幅 (N)，默认 10，控制输出被限制在 [-tau, tau]

        Returns:
            1: 成功
            0: 参数错误 (pos < 0)
        """
        if pos < 0:
            return 0

        e = pos - self.get_gripper_pos()

        # PI 输出（限幅前）
        tau_raw = self.kp_grip_pi * e + self.ki_grip_pi * self.gripper_err_int

        # 条件积分（防 windup）: 输出未饱和时才积分。
        # 若被 tau 限幅（饱和）则冻结积分，避免抓取期间积分积满、
        # 释放/反向动作时积分难以及时反向回绕而拖慢甚至卡住动作。
        if abs(tau_raw) < tau:
            self.gripper_err_int += e * self.model.opt.timestep
            self.gripper_err_int = float(np.clip(
                self.gripper_err_int,
                -self.gripper_err_int_limit,
                self.gripper_err_int_limit,
            ))

        # 力矩限幅
        tau_cmd = float(np.clip(tau_raw, -tau, tau))

        # 同步夹爪目标开合，便于后续 track_joint/保持逻辑保持同一目标
        self.gripper_pos_pre = pos

        # 立即写入 + 登记待生效（step() 内最后应用，保证不被轨迹/保持逻辑覆盖）
        self.data.ctrl[6] = tau_cmd
        self._gripper_pending = tau_cmd
        return 1
    
    def set_control_mode(self, mode: int) -> int:
        """
        设置控制模式
        0 - IDLE 空闲
        1 - 点位控制
        2 - MIT 控制（PD + 重力补偿）
        3 - 关节拖动（预留）
        4 - PF 力位混合（预留）
        """
        if mode not in (0, 1, 2, 3, 4):
            return 0            # 无效模式
        if mode != self.control_mode:
            self.pos_error_int.fill(0.0)
            self.vel_error_int.fill(0.0)
        self.control_mode = mode
        print(f"控制模式切换到: {mode}")
        return 1 
    
    def _compute_tau(
        self,
        q_des: np.ndarray,
        v_des: np.ndarray = None,
    ):
        """
        根据当前 control_mode 计算控制力矩

        Args:
            q_des: 目标关节位置 (6,)
            v_des: 目标关节速度 (6,)

        Returns:
            tau_arm: (6,)
        """

        if v_des is None:
            v_des = np.zeros(6)

        q_arm = self.data.qpos[:6].copy()
        v_arm = self.data.qvel[:6].copy()

        tau_arm = np.zeros(6)

        # -------------------------
        # 摩擦补偿
        # -------------------------
        v_sign = np.tanh(v_arm)

        friction_est = -(
            self.damping * v_arm
            + self.frictionloss * v_sign
        )

        # ==================================================
        # MODE 0
        # ==================================================
        if self.control_mode == 0:

            return tau_arm

        # ==================================================
        # MODE 1
        # 位置PI -> 速度PI
        # ==================================================
        elif self.control_mode == 1:

            pos_error = q_des - q_arm

            self.pos_error_int += (
                pos_error * self.model.opt.timestep
            )

            np.clip(
                self.pos_error_int,
                -self.pos_int_limit,
                self.pos_int_limit,
                out=self.pos_error_int,
            )

            v_cmd = (
                self.kp_arm_pos * pos_error
                + self.ki_arm_pos * self.pos_error_int
                + v_des
            )

            vel_error = v_cmd - v_arm

            self.vel_error_int += (
                vel_error * self.model.opt.timestep
            )

            np.clip(
                self.vel_error_int,
                -self.vel_int_limit,
                self.vel_int_limit,
                out=self.vel_error_int,
            )

            tau_arm = (
                self.kp_arm_vel * vel_error
                + self.ki_arm_vel * self.vel_error_int
            )

        # ==================================================
        # MODE 2
        # MIT
        # ==================================================
        elif self.control_mode == 2:

            pos_error = q_des - q_arm
            vel_error = v_des - v_arm

            bias = self.data.qfrc_bias[:6]

            tau_arm = (
                self.kp_mit @ pos_error
                + self.kd_mit @ vel_error
                + bias
                + friction_est
            )

        # ==================================================
        # MODE 3
        # 预留
        # ==================================================
        elif self.control_mode == 3:

            # 拖动模式：输出重力+摩擦力补偿，使机械臂处于零重力状态
            # 注意：v_des 在此处被忽略，因为用户手动拖动，目标速度不是控制目标
            tau_arm = self.data.qfrc_bias[:6] + friction_est

        # ==================================================
        # MODE 4
        # 力矩限制模式
        # ==================================================
        elif self.control_mode == 4:

            pos_error = q_des - q_arm

            self.pos_error_int += (
                pos_error * self.model.opt.timestep
            )

            np.clip(
                self.pos_error_int,
                -self.pos_int_limit,
                self.pos_int_limit,
                out=self.pos_error_int,
            )

            v_cmd = (
                self.kp_arm_pos * pos_error
                + self.ki_arm_pos * self.pos_error_int
                + v_des
            )

            vel_error = v_cmd - v_arm

            self.vel_error_int += (
                vel_error * self.model.opt.timestep
            )

            np.clip(
                self.vel_error_int,
                -self.vel_int_limit,
                self.vel_int_limit,
                out=self.vel_error_int,
            )

            tau_arm = (
                self.kp_arm_vel * vel_error
                + self.ki_arm_vel * self.vel_error_int
            )

            mujoco.mj_rne(
                self.model,
                self.data,
                1,
                self.data.qfrc_inverse,
            )

            for i in range(6):

                limit = abs(self.data.qfrc_inverse[i])

                if abs(tau_arm[i]) > limit:

                    tau_arm[i] = (
                        limit * np.sign(tau_arm[i])
                        + 4 * np.sign(tau_arm[i])
                    )

        return tau_arm

    def track_joint(
        self,
        targets: List[float],
        gripper_pos: float = -1.0,
    ) -> int:

        if self.control_mode not in (1, 2, 3, 4):
            return 0

        if len(targets) != 6:
            return 0

        # 用户主动下发关节指令 = 接管控制，取消之前 move_* 留下的保持目标，
        # 否则 _update_traj_control 末尾的保持分支会在每次 step 时覆盖本指令
        self._hold_target = None

        q_des = np.array(targets)

        tau_arm = self._compute_tau(
            q_des=q_des,
            v_des=np.zeros(6),
        )

        self.data.ctrl[:6] = tau_arm

        # ----------夹爪----------
        # 模型夹爪为 motor（力矩型）执行器，夹爪目标开合用 PD 控制折算为力矩下发

        if gripper_pos >= 0:
            self.gripper_pos_pre = gripper_pos
            self._gripper_pending = None   # 显式位置式夹爪指令，取消 set_gripper 力控锁存

        err_grip = self.gripper_pos_pre - self.get_gripper_pos()
        tau_grip = (
            self.kp_grip * err_grip
            - self.kd_grip * self.get_gripper_vel()
            + self.data.qfrc_bias[self.gripper_joint1_id]
        )
        self.data.ctrl[6] = tau_grip

        self._plan_q = q_des.copy()
        self._plan_v = np.zeros(6)          # 当前 track_joint 总是速度为零的停靠
        self._plan_tau = tau_arm.copy()

        return 1
    def _quintic_interp(self, q0, qf, T, t):
        """
        五次多项式插值：
        边界条件：起止速度、加速度都为0
        返回：q, v, a
        """
        if T <= 0:
            raise ValueError("T must be positive")

        s = np.clip(t / T, 0.0, 1.0)

        h = 10*s**3 - 15*s**4 + 6*s**5
        dh = (30*s**2 - 60*s**3 + 30*s**4) / T
        ddh = (60*s - 180*s**2 + 120*s**3) / (T**2)

        dq = qf - q0
        q = q0 + dq * h
        v = dq * dh
        a = dq * ddh
        return q, v, a
    
    def _try_start_next_traj(self):
        """若队列非空，取出下一段并调用 move_joint"""
        if not self._traj_queue:
            return
        next_target, next_time, grip = self._traj_queue.pop(0)
        current_dist = self.get_gripper_pos()
        current_vel = self.get_gripper_vel()

        if grip >= 0:
            self._gripper_pending = None   # 轨迹段显式带夹爪指令，取消 set_gripper 力控锁存

        error = grip - current_dist

        tau_grip = (
            self.kp_grip * error
            - self.kd_grip * current_vel + self.data.qfrc_bias[self.gripper_joint1_id]
        )
        self.data.ctrl[6] = tau_grip
        self.move_joint(next_target, desire_time=next_time, is_sync=False)
    
    def _update_traj_control(self):
        if self.control_mode == 3:
            # 计算当前状态下的补偿力矩（无需目标位置）
            q_arm = self.data.qpos[:6].copy()
            v_arm = self.data.qvel[:6].copy()
            friction_est = -(self.damping * v_arm + self.frictionloss * np.tanh(v_arm))
            tau_arm = self.data.qfrc_bias[:6] + friction_est
            self.data.ctrl[:6] = tau_arm
            return  # 跳过后续所有轨迹处理
        # ========== 0. 处理关节空间直线运动（标量 λ 双 S） ==========
        if self._line_ds_active:
            t = self.data.time - self._line_ds_start_time
            if t < 0:
                return

            # 运动结束
            if t >= self._line_ds_duration:
                self._line_ds_active = False
                hold_target = (self._line_ds_q0 + self._line_ds_dq).copy()
                self.track_joint(hold_target.tolist(), gripper_pos=-1.0)
                self._hold_target = hold_target   # track_joint 会清空 _hold_target，这里恢复保持
                self._try_start_next_traj()
                return

            # 插值 λ
            s, ds, _ = self._double_s_interp(t, self._line_ds_params)  # s 即为 λ (0→1)
            lam = s
            dlam = ds

            q_des = self._line_ds_q0 + lam * self._line_ds_dq
            v_des = dlam * self._line_ds_dq

            self.data.ctrl[:6] = self._compute_tau(q_des, v_des)
            self._plan_q = q_des.copy()
            self._plan_v = v_des.copy()
            self._plan_tau = self._compute_tau(q_des, v_des).copy()
            return
        # ========== 1. 处理双 S 轨迹 ==========
        if self._ds_active:
            t = self.data.time - self._ds_start_time
            if t < 0:
                return
            # 轨迹结束
            if t >= self._ds_duration:
                self._ds_active = False
                hold_target = self._ds_target.copy()   # ★ 记录保持目标
                self.track_joint(hold_target.tolist(), gripper_pos=-1.0)
                self._hold_target = hold_target   # track_joint 会清空 _hold_target，这里恢复保持
                self._try_start_next_traj()          # ★ 新增
                return

            # 正常插值
            q_des = np.zeros(6)
            v_des = np.zeros(6)
            for j in range(6):
                par = self._ds_params[j]
                if par.get('is_static', False):
                    q_des[j] = par['q0']
                    v_des[j] = 0.0
                else:
                    q_rel, v_rel, _ = self._double_s_interp_7phase(t, par)
                    q_des[j] = par['q0'] + q_rel
                    v_des[j] = v_rel
            self.data.ctrl[:6] = self._compute_tau(q_des, v_des)
            self._plan_q = q_des.copy()
            self._plan_v = v_des.copy()
            self._plan_tau = self._compute_tau(q_des, v_des).copy()
            return

        # ========== 2. 处理五次多项式轨迹 ==========
        if self._traj_active:
            t = self.data.time - self._traj_start_time
            if t >= self._traj_duration:
                # 轨迹结束
                q_des = self._traj_qf.copy()
                v_des = np.zeros(6)
                self._traj_active = False
                self._hold_target = self._traj_qf.copy()    # ★ 记录保持目标
                self._try_start_next_traj()          # ★ 新增
            else:
                q_des = np.zeros(6)
                v_des = np.zeros(6)
                for i in range(6):
                    q_des[i], v_des[i], _ = self._quintic_interp(
                        self._traj_q0[i], self._traj_qf[i],
                        self._traj_duration, t
                    )
            self.data.ctrl[:6] = self._compute_tau(q_des, v_des)
            self._plan_q = q_des.copy()
            self._plan_v = v_des.copy()
            self._plan_tau = self._compute_tau(q_des, v_des).copy()
            # if self.model.nu > 6:
            #     self.data.ctrl[6] = 0.0
            return

        # ========== 3. 无任何轨迹，但存在保持目标 ==========
        if self._hold_target is not None:
            hold_target = self._hold_target
            self.track_joint(hold_target.tolist(), gripper_pos=-1.0)
            self._hold_target = hold_target   # track_joint 会清空 _hold_target，这里恢复保持
        

    # ============== 双 S 曲线规划与插值 ==============
    def _plan_double_s(self, D: float, v_max: float, a_max: float, j_max: float):
        """
        对称双 S 曲线规划 (纯时间最优)
        D: 位移量 (可正可负)
        v_max, a_max, j_max: 正标量约束
        返回 (总时间 T, 参数字典)
        参数字典中包含用于插值的所有数据 (均为绝对值)
        """
        if abs(D) < 1e-12:
            return 0.0, {
                'q0': 0.0, 'qf': 0.0, 'sign': 1, 'T': 0.0,
                't_j1': 0.0, 't_j2': 0.0, 't_v': 0.0,
                'v_lim': 0.0, 'a_lim': 0.0, 'j_max': j_max
            }

        sign = 1.0 if D > 0 else -1.0
        D_abs = abs(D)

        # 1. 计算若达到 a_max 且无匀加速段时的速度
        v_a = a_max**2 / j_max

        if v_max <= v_a:
            # 加速度受限，无法达到 a_max
            t_j1 = np.sqrt(v_max / j_max)
            t_j2 = 0.0
            a_lim = np.sqrt(v_max * j_max)  # 实际加速度峰值
            v_lim = v_max
            # 加速段位移 (无匀加速)
            s_a = t_j1 * v_lim  # = (2*t_j1) * (v_lim/2)
            D_req = 2.0 * s_a
            if D_abs >= D_req:
                # 存在匀速段
                t_v = (D_abs - D_req) / v_lim
            else:
                # 无匀速段，且 v_peak < v_max
                # D = 2 * v_peak^{3/2} / sqrt(j_max)
                v_lim = (D_abs * np.sqrt(j_max) / 2.0) ** (2.0 / 3.0)
                t_j1 = np.sqrt(v_lim / j_max)
                a_lim = np.sqrt(v_lim * j_max)
                t_v = 0.0
                s_a = t_j1 * v_lim  # 更新
        else:
            # 能到达 a_max
            t_j1 = a_max / j_max
            a_lim = a_max
            # 检查能否达到 v_max
            # 加速段含匀加速时，达到 v_max 所需的 t_j2
            t_j2_vmax = (v_max - v_a) / a_max
            if t_j2_vmax < 0:
                t_j2_vmax = 0.0  # 理论上不会出现
            # 加速到 v_max 的位移
            s_a_vmax = (2 * t_j1 + t_j2_vmax) * v_max / 2.0
            D_req = 2.0 * s_a_vmax
            if D_abs >= D_req:
                # 达到 v_max，有匀速段
                t_j2 = t_j2_vmax
                v_lim = v_max
                t_v = (D_abs - D_req) / v_max
            else:
                # 达不到 v_max，无匀速段，但有匀加速段 (因为 D >= D_a_amax)
                # 解二次方程：D = (2*t_j1 + t_j2) * (v_a + a_max*t_j2)
                # 系数：A=a_max, B=3*a_max**2/j_max, C=2*a_max**3/j_max**2 - D_abs
                A = a_max
                B = 3.0 * a_max**2 / j_max
                C = 2.0 * a_max**3 / j_max**2 - D_abs
                disc = B**2 - 4.0 * A * C
                if disc < 0:
                    disc = 0.0
                t_j2 = (-B + np.sqrt(disc)) / (2.0 * A)
                if t_j2 < 0:
                    t_j2 = 0.0
                v_lim = v_a + a_max * t_j2
                t_v = 0.0

        T_acc = 2.0 * t_j1 + t_j2
        T_total = 2.0 * T_acc + t_v

        return T_total, {
            'q0': 0.0,          # 占位，外部会填入实际值
            'qf': D,            # 位移量
            'sign': sign,
            'T': T_total,
            't_j1': t_j1,
            't_j2': t_j2,
            't_v': t_v,
            'v_lim': v_lim,
            'a_lim': a_lim,
            'j_max': j_max,
        }

    def _double_s_interp(self, t: float, params: dict):
        """
        根据参数字典，返回时刻 t 的 (q, v, a) (绝对坐标系，考虑符号)
        t: 从轨迹开始计算的绝对时间 (应 <= params['T'])
        """
        T = params['T']
        if T < 1e-12:
            return params['qf'], 0.0, 0.0

        sign = params['sign']
        t_j1 = params['t_j1']
        t_j2 = params['t_j2']
        t_v = params['t_v']
        v_lim = params['v_lim']
        a_lim = params['a_lim']
        j_max = params['j_max']

        # 加速段总时间
        T_acc = 2.0 * t_j1 + t_j2
        # 加速段位移
        s_acc = (T_acc * v_lim) / 2.0  # 平均速度 v_lim/2

        if t <= T_acc:
            # ---------- 加速段 ----------
            if t <= t_j1:
                # 加加速段
                a = j_max * t
                v = 0.5 * j_max * t**2
                s = j_max * t**3 / 6.0
            elif t <= t_j1 + t_j2:
                # 匀加速段
                dt = t - t_j1
                v1 = 0.5 * j_max * t_j1**2
                s1 = j_max * t_j1**3 / 6.0
                a = a_lim
                v = v1 + a_lim * dt
                s = s1 + v1 * dt + 0.5 * a_lim * dt**2
            else:
                # 减加速段
                dt = t - (t_j1 + t_j2)
                v2 = 0.5 * j_max * t_j1**2 + a_lim * t_j2
                s2 = (j_max * t_j1**3 / 6.0) + \
                    (0.5 * j_max * t_j1**2) * t_j2 + \
                    0.5 * a_lim * t_j2**2
                a = a_lim - j_max * dt
                v = v2 + a_lim * dt - 0.5 * j_max * dt**2
                s = s2 + v2 * dt + 0.5 * a_lim * dt**2 - j_max * dt**3 / 6.0
        elif t <= T_acc + t_v:
            # ---------- 匀速段 ----------
            dt = t - T_acc
            s = s_acc + v_lim * dt
            v = v_lim
            a = 0.0
        else:
            # ---------- 减速段 ----------
            td = t - (T_acc + t_v)  # 减速段局部时间 0 ~ T_acc
            # 利用对称性：v_dec(td) = v_acc(T_acc - td)
            t_rev = T_acc - td
            # 调用自身计算加速段该时刻的绝对速度和位移 (不需要符号)
            # 为避免递归，复制加速段逻辑或调用内部函数
            # 直接内联计算
            if t_rev <= t_j1:
                a_acc = j_max * t_rev
                v_acc = 0.5 * j_max * t_rev**2
                s_acc_rev = j_max * t_rev**3 / 6.0
            elif t_rev <= t_j1 + t_j2:
                dt = t_rev - t_j1
                v1 = 0.5 * j_max * t_j1**2
                s1 = j_max * t_j1**3 / 6.0
                a_acc = a_lim
                v_acc = v1 + a_lim * dt
                s_acc_rev = s1 + v1 * dt + 0.5 * a_lim * dt**2
            else:
                dt = t_rev - (t_j1 + t_j2)
                v2 = 0.5 * j_max * t_j1**2 + a_lim * t_j2
                s2 = (j_max * t_j1**3 / 6.0) + \
                    (0.5 * j_max * t_j1**2) * t_j2 + \
                    0.5 * a_lim * t_j2**2
                a_acc = a_lim - j_max * dt
                v_acc = v2 + a_lim * dt - 0.5 * j_max * dt**2
                s_acc_rev = s2 + v2 * dt + 0.5 * a_lim * dt**2 - j_max * dt**3 / 6.0

            v = v_acc
            # 减速段位移 = s_acc - s_acc_rev
            s = s_acc + v_lim * t_v + (s_acc - s_acc_rev)
            a = -a_acc

        # 加上符号和初位移 (q0 在外部通过 params 传入，此处 s 是相对位移)
        q_rel = s * sign
        v_rel = v * sign
        a_rel = a * sign
        return q_rel, v_rel, a_rel
    
    def _sync_double_s_plan(self, q0: np.ndarray, qf: np.ndarray,
                        vmax: np.ndarray, amax: np.ndarray, jmax: np.ndarray) -> Tuple[float, List[Dict]]:
        """
        多轴同步 Double-S 轨迹规划
        返回：(同步总时间 Tsync, 各关节参数字典列表)
        """
        n = len(q0)
        D = qf - q0
        zero_mask = np.abs(D) < 1e-12

        T_indiv = np.zeros(n)
        for i in range(n):
            if zero_mask[i]:
                T_indiv[i] = 0.0
            else:
                T_indiv[i] = self._min_time_double_s(abs(D[i]), vmax[i], amax[i], jmax[i])
        Tsync = np.max(T_indiv)

        params_list = []
        for i in range(n):
            if zero_mask[i] or Tsync < 1e-12:
                params_list.append({
                    'q0': q0[i],
                    'qf': qf[i],
                    'sign': 1.0,
                    'v_peak': 0.0,
                    'a_peak': 0.0,
                    'j_used': jmax[i],
                    't1': 0.0, 't2': 0.0, 't3': 0.0,
                    't4': Tsync, 't5': Tsync, 't6': Tsync, 't7': Tsync,
                    'Tsync': Tsync,
                    'is_static': True
                })
            else:
                par = self._sync_joint_double_s(abs(D[i]), vmax[i], amax[i], jmax[i], Tsync, D[i])
                if par is None:
                    # 同步规划失败，使用保守的纯 jerk 参数并拉伸到 Tsync
                    # 计算纯 jerk 的时间比例
                    t_j_indiv = np.cbrt(abs(D[i]) / (2.0 * jmax[i]))
                    T_indiv_i = 4.0 * t_j_indiv
                    if T_indiv_i < Tsync:
                        # 拉伸：保持 jerk 形状，时间比例缩放
                        scale = Tsync / T_indiv_i
                        t_j = t_j_indiv * scale
                        v_peak = jmax[i] * t_j**2
                        a_peak = jmax[i] * t_j
                        t1 = t_j
                        t2 = t_j          # 纯 jerk 无匀加速段，但为了统一，匀加速时间为0
                        t3 = 2 * t_j
                        t4 = 2 * t_j
                        t5 = 3 * t_j
                        t6 = 3 * t_j
                        t7 = 4 * t_j
                    else:
                        # 不应该发生，直接使用独立最短时间（放弃同步）
                        t_j = t_j_indiv
                        v_peak = jmax[i] * t_j**2
                        a_peak = jmax[i] * t_j
                        t1 = t_j; t2 = t_j; t3 = 2*t_j
                        t4 = 2*t_j; t5 = 3*t_j; t6 = 3*t_j; t7 = 4*t_j
                        Tsync = max(Tsync, t7)   # 延长全局同步时间（会影响其他关节，慎用，此处仅作保底）

                    par = {
                        'sign': 1.0 if D[i] > 0 else -1.0,
                        'v_peak': v_peak,
                        'a_peak': a_peak,
                        'j_used': jmax[i],
                        't1': t1, 't2': t2, 't3': t3,
                        't4': t4, 't5': t5, 't6': t6, 't7': t7,
                        'Tsync': t7,
                        'is_static': False
                    }
                par['q0'] = q0[i]
                par['qf'] = qf[i]
                par['Tsync'] = Tsync        # 强制使用全局同步时间作为结束判断
                par['is_static'] = False
                params_list.append(par)

        # 如果回退导致 Tsync 变大，需要更新所有关节的 Tsync 和时长
        self._ds_duration = Tsync
        for par in params_list:
            par['Tsync'] = Tsync

        return Tsync, params_list


    @staticmethod
    def _min_time_double_s(D_abs: float, v_max: float, a_max: float, j_max: float) -> float:
        # 临界位移
        D_a_amax = 2.0 * a_max**3 / j_max**2          # 刚好达到 a_max 的位移
        D_v_vmax = v_max * (v_max / a_max + a_max / j_max)  # 刚好达到 v_max 的位移

        if D_abs >= D_v_vmax:
            # 能达到 v_max，有匀速段
            return D_abs / v_max + v_max / a_max + a_max / j_max

        elif D_abs >= D_a_amax:
            # 能达到 a_max，无匀速段
            # 解 v_peak：v_peak²/a_max + v_peak·a_max/j_max - D_abs = 0
            coeff = [1.0/a_max, a_max/j_max, -D_abs]
            roots = np.roots(coeff)
            v_peak = float(np.max(roots[np.isreal(roots)]).real)
            # 总时间 T = 2 × (v_peak/a_max + a_max/j_max)
            return 2.0 * (v_peak / a_max + a_max / j_max)

        else:
            # 达不到 a_max，纯 jerk 受限
            t_j = np.cbrt(D_abs / (2.0 * j_max))
            return 4.0 * t_j

    @staticmethod
    def _sync_joint_double_s(D_abs: float, v_max: float, a_max: float, j_max: float,
                            Tsync: float, D_total: float) -> Optional[Dict]:
        """
        给定同步时间 Tsync，求解该关节的 v_peak, a_peak 以满足位移。
        返回参数字典，若失败则返回 None。
        """
        if D_abs < 1e-12 or Tsync < 1e-12:
            return None

        sign = 1.0 if D_total > 0 else -1.0

        def solve_apeak(vp):
            """给定 vp，用牛顿法解 a_peak"""
            ap = a_max
            if ap <= 0:
                ap = 0.1 * j_max * (Tsync / 2.0)
            for _ in range(50):
                if ap <= 0:
                    return None
                term = vp**2/ap + ap*vp/j_max - ap**3/(3*j_max**2)
                f = vp * Tsync - term - D_abs
                if abs(f) < 1e-12:
                    break
                df = vp**2/ap**2 - vp/j_max + ap**2/j_max**2
                if abs(df) < 1e-12:
                    return None
                delta = f / df
                if abs(delta) > 0.5 * a_max:
                    delta = 0.5 * a_max * np.sign(delta)
                ap -= delta
                if ap <= 0:
                    ap = 1e-6
                if ap > a_max:
                    ap = a_max
            T_min = 2.0 * (vp/ap + ap/j_max)  # 无匀速段最短时间
            if T_min > Tsync + 1e-12 or ap > a_max + 1e-12:
                return None
            return ap

        # 二分搜索 v_peak (0, v_max]
        v_low, v_high = 0.0, v_max
        v_peak = 0.0
        a_peak = None
        for _ in range(60):
            v_mid = (v_low + v_high) / 2.0
            ap = solve_apeak(v_mid)
            if ap is not None:
                v_peak = v_mid
                a_peak = ap
                v_low = v_mid
            else:
                v_high = v_mid
            if v_high - v_low < 1e-8 * v_max:
                break

        if a_peak is None:
            return None

        # 计算七段切换时间点
        T_j1 = a_peak / j_max
        T_j2 = (v_peak - a_peak**2 / j_max) / a_peak if a_peak > 0 else 0.0
        if T_j2 < 0:
            T_j2 = 0.0
        T_acc = T_j1 + T_j2 + T_j1
        s_acc = v_peak**2/(2*a_peak) + (v_peak*a_peak)/(2*j_max) - a_peak**3/(6*j_max**2)
        T_const = (D_abs - 2*s_acc) / v_peak if v_peak > 0 else 0.0
        if T_const < 0:
            T_const = 0.0

        t1 = T_j1
        t2 = t1 + T_j2
        t3 = t2 + T_j1
        t4 = t3 + T_const
        t5 = t4 + T_j1
        t6 = t5 + T_j2
        t7 = t6 + T_j1

        return {
            'sign': sign,
            'v_peak': v_peak,
            'a_peak': a_peak,
            'j_used': j_max,
            't1': t1, 't2': t2, 't3': t3,
            't4': t4, 't5': t5, 't6': t6, 't7': t7
        }

    @staticmethod
    def _double_s_interp_7phase(t: float, par: Dict) -> Tuple[float, float, float]:
        """七段式 Double-S 插值，返回 (q_rel, v, a) 带符号"""
        if par.get('is_static', False) or par['Tsync'] < 1e-12:
            return 0.0, 0.0, 0.0

        # 超出总时间则返回终点
        if t >= par['Tsync'] - 1e-12:
            return par['qf'] - par['q0'], 0.0, 0.0

        sign = par['sign']
        v_peak = par['v_peak']
        a_peak = par['a_peak']
        j_used = par['j_used']
        t1, t2, t3, t4, t5, t6, t7 = [par[k] for k in ('t1','t2','t3','t4','t5','t6','t7')]

        def phase1(tau): return (j_used*tau**3/6, 0.5*j_used*tau**2, j_used*tau)
        def phase2(tau):
            dt = tau - t1
            s1, v1, _ = phase1(t1)
            return (s1 + v1*dt + 0.5*a_peak*dt**2, v1 + a_peak*dt, a_peak)
        def phase3(tau):
            dt = tau - t2
            s2, v2, _ = phase2(t2)
            a_ = a_peak - j_used*dt
            return (s2 + v2*dt + 0.5*a_peak*dt**2 - j_used*dt**3/6,
                    v2 + a_peak*dt - 0.5*j_used*dt**2, a_)
        def phase4(tau):
            dt = tau - t3
            s3, v3, _ = phase3(t3)
            return (s3 + v_peak*dt, v_peak, 0.0)
        def phase5(tau):
            dt = tau - t4
            s4, v4, _ = phase4(t4)
            a_ = -j_used*dt
            return (s4 + v_peak*dt - j_used*dt**3/6,
                    v_peak - 0.5*j_used*dt**2, a_)
        def phase6(tau):
            dt = tau - t5
            s5, v5, _ = phase5(t5)
            return (s5 + v5*dt - 0.5*a_peak*dt**2,
                    v5 - a_peak*dt, -a_peak)
        def phase7(tau):
            dt = tau - t6
            s6, v6, _ = phase6(t6)
            a_ = -a_peak + j_used*dt
            return (s6 + v6*dt - 0.5*a_peak*dt**2 + j_used*dt**3/6,
                    v6 - a_peak*dt + 0.5*j_used*dt**2, a_)

        if t <= t1:          s, v, a = phase1(t)
        elif t <= t2:        s, v, a = phase2(t)
        elif t <= t3:        s, v, a = phase3(t)
        elif t <= t4:        s, v, a = phase4(t)
        elif t <= t5:        s, v, a = phase5(t)
        elif t <= t6:        s, v, a = phase6(t)
        else:                s, v, a = phase7(t)

        return s * sign, v * sign, a * sign


    # ---------- 修改 move_joint 方法 ----------
    def move_joint(
        self,
        target_pos: List[float],
        desire_time=-1,
        is_sync=True,
    ):
        """
        关节空间点到点运动
        desire_time >  0 : 五次多项式 (已有实现)
        desire_time == -1: 时间最优双 S 曲线 (新实现)
        """
        self._hold_target = None
        if len(target_pos) != 6:
            return 0

        q_cur = self.data.qpos[:6].copy()
        q_tar = np.array(target_pos, dtype=float)

        # ----- 原有分支：五次多项式 (desire_time > 0) -----
        if desire_time > 0:
            # 保持原逻辑不变
            self._traj_q0 = q_cur
            self._traj_qf = q_tar
            self._traj_start_time = self.data.time
            self._traj_duration = float(desire_time)
            self._traj_active = True
            self._ds_active = False  # 关闭 double S

            if is_sync:
                while self._traj_active:
                    self.step(5)
            return 1

        # ----- 新分支：desire_time == -1 双 S 时间最优同步 -----
        if desire_time == -1:
            self._traj_active = False
            self._ds_active = True
            self._ds_start_time = self.data.time
            self._ds_target = q_tar.copy()

            vlim = self.joint_vel_limits
            alim = self.joint_acc_limits
            jlim = self.joint_jerk_limits
            Tsync, params_list = self._sync_double_s_plan(q_cur, q_tar, vlim, alim, jlim)
            self._ds_duration = Tsync
            self._ds_params = params_list

            if is_sync:
                while self._ds_active:
                    self.step(5)
            return 1
        
    def _ik_solve(
        self,
        pos_des: np.ndarray,
        quat_des: np.ndarray,
        ref_joint: Optional[np.ndarray] = None,   # 新增
        max_iter: int = 2000,
        tol: float = 1e-3,
        damp: float = 0.1
    ) -> Tuple[np.ndarray, bool]:
        """
        解析逆运动学求解器。

        求解逻辑与 sw_six_kinematics.cpp 一致：先将参考关节角转换为
        控制器坐标，生成 8 组解析解，处理腕部奇异情况和 2*pi 等价角，
        按控制器坐标的关节范围过滤后选择距离参考解最近的结果。

        max_iter、tol、damp 保留在函数签名中，以兼容原有调用方；解析
        求解不使用这些数值迭代参数。
        """
        q_backup = self.data.qpos[:6].copy()

        try:
            pos_des = np.asarray(pos_des, dtype=float).reshape(3)
            quat_des = np.asarray(quat_des, dtype=float).reshape(4)
        except (TypeError, ValueError):
            return q_backup, False

        if not np.all(np.isfinite(pos_des)) or not np.all(np.isfinite(quat_des)):
            return q_backup, False

        if ref_joint is None:
            ref_joint = q_backup
        else:
            try:
                ref_joint = np.asarray(ref_joint, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return q_backup, False
            if ref_joint.size < 6 or not np.all(np.isfinite(ref_joint[:6])):
                return q_backup, False
            ref_joint = ref_joint[:6]

        # C++ 的 toControllerJointPos：解析求解使用控制器坐标。
        joint_offset = np.array([0.0, -3.1416, 1.5708, 0.0, 0.0, 0.0])
        ref_joint_controller = ref_joint + joint_offset

        quat_norm = np.linalg.norm(quat_des)
        if quat_norm <= 1e-12:
            return q_backup, False
        quat_wxyz = np.array([
            quat_des[3], quat_des[0], quat_des[1], quat_des[2]
        ]) / quat_norm
        R_0_6_flat = np.zeros(9)
        mujoco.mju_quat2Mat(R_0_6_flat, quat_wxyz)
        R_0_6 = R_0_6_flat.reshape(3, 3)

        #            a         alpha        d       theta
        dh_param = [[0,         0,        0.157,      0],
                    [0,      -np.pi/2,       0,   -np.pi],
                    [0.35,       0,          0,      np.pi/2],
                    [0.079,   -np.pi/2,   0.242,      0],
                    [0,      np.pi/2,      0,          0],
                    [0,      -np.pi/2,    0.108,      0]]

        d6 = dh_param[5][2]
        Pwc = pos_des - R_0_6 @ np.array([0.0, 0.0, d6])
        x, y, z = Pwc
        nx, ny, nz = R_0_6[:, 0]
        ox, oy, oz = R_0_6[:, 1]
        ax, ay, az = R_0_6[:, 2]

        d1 = dh_param[0][2]
        d3 = dh_param[2][2]
        d4 = dh_param[3][2]
        a2 = dh_param[2][0]
        a3 = dh_param[3][0]
        tiny_err_ = 1e-6

        def checked_sqrt_factor(value):
            if value < 0.0 and abs(value) < tiny_err_:
                return 0.0
            if value < 0.0:
                return None
            return np.sqrt(value)

        # 求解关节1的两个候选值。
        xy_square = x * x + y * y
        if xy_square <= 0.0:
            return q_backup, False
        tmp = 1.0 - d3 * d3 / xy_square
        sqrt_tmp = checked_sqrt_factor(tmp)
        if sqrt_tmp is None:
            return q_backup, False
        xy_norm = np.sqrt(xy_square)
        theta1_1 = np.arctan2(y, x) - np.arctan2(
            d3 / xy_norm, sqrt_tmp
        )
        theta1_2 = np.arctan2(y, x) - np.arctan2(
            d3 / xy_norm, -sqrt_tmp
        )

        def normalize_theta1(theta1):
            if theta1 > np.pi:
                theta1 -= 2.0 * np.pi
            elif theta1 < -np.pi:
                theta1 += 2.0 * np.pi
            if ref_joint_controller[0] > np.pi / 2.0 and theta1 < 0.0:
                theta1 += 2.0 * np.pi
            elif ref_joint_controller[0] < -np.pi / 2.0 and theta1 > 0.0:
                theta1 -= 2.0 * np.pi
            return theta1

        theta1_values = (
            normalize_theta1(theta1_1),
            normalize_theta1(theta1_2),
        )

        # 求解关节3的两个候选值。
        k1 = 2.0 * a2 * a3
        k2 = 2.0 * a2 * d4
        k3 = (x * x + y * y + (z - d1) * (z - d1)
              - d3 * d3 - a2 * a2 - a3 * a3 - d4 * d4)
        tmp = 1.0 - k3 * k3 / (k1 * k1 + k2 * k2)
        sqrt_tmp = checked_sqrt_factor(tmp)
        if sqrt_tmp is None:
            return q_backup, False
        k_norm = np.sqrt(k1 * k1 + k2 * k2)
        theta3_1 = np.arctan2(k1, k2) - np.arctan2(
            k3 / k_norm, sqrt_tmp
        )
        theta3_2 = np.arctan2(k1, k2) - np.arctan2(
            k3 / k_norm, -sqrt_tmp
        )

        theta3_values = []
        angle_wrap_tolerance = 1e-4
        for theta3 in (theta3_1, theta3_2):
            # Fixed link quaternions in the MJCF are rounded to about 1e-6.
            # At the home pose this can move -3*pi/2 a few micro-radians
            # inside the unwrapped interval, preventing the equivalent angle
            # from being selected before joint-range filtering.
            if theta3 > np.pi / 2.0:
                if theta3 <= np.pi / 2.0 + angle_wrap_tolerance:
                    theta3 = np.pi / 2.0
                else:
                    theta3 -= 2.0 * np.pi
            # C++ 中的 -3 / 2 * M_PI 按其几何意图为 -3*pi/2。
            elif theta3 < -3.0 * np.pi / 2.0 + angle_wrap_tolerance:
                theta3 += 2.0 * np.pi
            theta3_values.append(theta3)

        # 每个关节3候选对应两个关节2候选，共四组上游关节解。
        theta2_values = []
        for theta3 in theta3_values:
            k1 = -a3 * np.sin(theta3) - d4 * np.cos(theta3)
            k2 = a2 + a3 * np.cos(theta3) - d4 * np.sin(theta3)
            k3 = z - d1
            k_norm = np.sqrt(k1 * k1 + k2 * k2)
            if k_norm <= tiny_err_:
                return q_backup, False
            tmp = 1.0 - k3 * k3 / (k_norm * k_norm)
            sqrt_tmp = checked_sqrt_factor(tmp)
            if sqrt_tmp is None:
                return q_backup, False

            theta2_1 = np.arctan2(k1, k2) - np.arctan2(
                k3 / k_norm, sqrt_tmp
            )
            theta2_2 = np.arctan2(k1, k2) - np.arctan2(
                k3 / k_norm, -sqrt_tmp
            )
            for theta2 in (theta2_1, theta2_2):
                if theta2 > np.pi / 2.0:
                    theta2 -= 2.0 * np.pi
                elif theta2 < -3.0 * np.pi / 2.0:
                    theta2 += 2.0 * np.pi
                theta2_values.append(theta2)

        tmp_vector_joints = np.zeros((8, 6), dtype=float)
        for i in range(4):
            tmp_vector_joints[2 * i, 0] = theta1_values[i % 2]
            tmp_vector_joints[2 * i + 1, 0] = theta1_values[i % 2]
            tmp_vector_joints[2 * i, 1] = theta2_values[i]
            tmp_vector_joints[2 * i + 1, 1] = theta2_values[i]
            tmp_vector_joints[2 * i:2 * i + 2, 2] = theta3_values[
                0 if i <= 1 else 1
            ]

        # 对每组上游解求腕部三个关节。
        for i in range(4):
            t1, t2, t3 = tmp_vector_joints[2 * i, :3]
            c1, s1 = np.cos(t1), np.sin(t1)
            c23, s23 = np.cos(t2 + t3), np.sin(t2 + t3)

            r11 = nx * c1 * c23 + ny * s1 * c23 - nz * s23
            r21 = -nx * c1 * s23 - ny * s1 * s23 - nz * c23
            r31 = ny * c1 - nx * s1
            r12 = ox * c1 * c23 + oy * s1 * c23 - oz * s23
            r22 = -ox * c1 * s23 - oy * s1 * s23 - oz * c23
            r32 = oy * c1 - ox * s1
            r13 = ax * c1 * c23 + ay * s1 * c23 - az * s23
            r23 = -ax * c1 * s23 - ay * s1 * s23 - az * c23
            r33 = ay * c1 - ax * s1

            if r23 < 1.0 - tiny_err_ and r23 > -(1.0 - tiny_err_):
                wrist_sqrt = np.sqrt(r21 * r21 + r22 * r22)
                theta5_1 = np.arctan2(wrist_sqrt, r23)
                theta5_2 = np.arctan2(-wrist_sqrt, r23)
                theta4_1 = np.arctan2(
                    r33 / np.sin(theta5_1),
                    -r13 / np.sin(theta5_1),
                )
                theta4_2 = np.arctan2(
                    r33 / np.sin(theta5_2),
                    -r13 / np.sin(theta5_2),
                )
                theta6_1 = np.arctan2(
                    -r22 / np.sin(theta5_1),
                    r21 / np.sin(theta5_1),
                )
                theta6_2 = np.arctan2(
                    -r22 / np.sin(theta5_2),
                    r21 / np.sin(theta5_2),
                )
            elif r23 >= 1.0 - tiny_err_:
                theta5_1 = theta5_2 = 0.0
                theta4_1 = theta4_2 = ref_joint_controller[3]
                theta6_1 = np.arctan2(-r12, r11) - theta4_1
                theta6_2 = np.arctan2(-r12, r11) - theta4_2
            else:
                theta5_1 = theta5_2 = np.pi
                theta4_1 = theta4_2 = ref_joint_controller[3]
                theta6_1 = theta4_1 - np.arctan2(r12, -r11)
                theta6_2 = theta4_2 - np.arctan2(r12, -r11)

            # 为腕部角选择距离参考角最近的 2*pi 等价值。
            def closest_equivalent(angle, reference):
                if abs(angle) < tiny_err_:
                    equivalent = 2.0 * np.pi if reference >= 0.0 else -2.0 * np.pi
                elif angle <= 0.0:
                    equivalent = angle + 2.0 * np.pi
                else:
                    equivalent = angle - 2.0 * np.pi
                return angle if abs(angle - reference) < abs(equivalent - reference) else equivalent

            theta4_1 = closest_equivalent(theta4_1, ref_joint_controller[3])
            theta4_2 = closest_equivalent(theta4_2, ref_joint_controller[3])
            theta6_1 = closest_equivalent(theta6_1, ref_joint_controller[5])
            theta6_2 = closest_equivalent(theta6_2, ref_joint_controller[5])

            def normalize_theta6(theta6):
                if theta6 > np.pi:
                    theta6 -= 2.0 * np.pi
                elif theta6 < -np.pi:
                    theta6 += 2.0 * np.pi
                if ref_joint_controller[5] > np.pi / 2.0 and theta6 < 0.0:
                    theta6 += 2.0 * np.pi
                elif ref_joint_controller[5] < -np.pi / 2.0 and theta6 > 0.0:
                    theta6 -= 2.0 * np.pi
                return theta6

            theta6_1 = normalize_theta6(theta6_1)
            theta6_2 = normalize_theta6(theta6_2)

            tmp_vector_joints[2 * i, 3:] = [theta4_1, theta5_1, theta6_1]
            tmp_vector_joints[2 * i + 1, 3:] = [theta4_2, theta5_2, theta6_2]

        # C++ 的范围表对应控制器坐标，因此偏置也应用到范围上。
        limit_lower = np.array([
            -2.9671, 0.0, -3.1416, -2.6704, -1.5708, -2.8275
        ]) + joint_offset
        limit_upper = np.array([
            2.9671, 3.1416, 0.0, 2.6704, 1.5708, 2.8275
        ]) + joint_offset

        def enforce_joint_range(joints):
            joints = joints.copy()
            for index in range(6):
                if joints[index] < limit_lower[index] - 2.0 * np.pi:
                    joints[index] += 2.0 * np.pi
                elif joints[index] > limit_upper[index] + 2.0 * np.pi:
                    joints[index] -= 2.0 * np.pi
            return joints

        vector_states = []
        joint_range_tolerance = 1e-3
        for candidate in tmp_vector_joints:
            candidate = enforce_joint_range(candidate)
            if np.all(candidate >= limit_lower - joint_range_tolerance) and np.all(
                    candidate <= limit_upper + joint_range_tolerance):
                vector_states.append(candidate)

                # 保留 C++ 对关节1、关节6的等价角扩展逻辑。
                candidate_with_extra = candidate.copy()
                if candidate[0] - 2.0 * np.pi > limit_lower[0]:
                    candidate_with_extra[0] = candidate[0] - 2.0 * np.pi
                    vector_states.append(candidate_with_extra.copy())
                    if candidate[5] - 2.0 * np.pi > limit_lower[5]:
                        candidate_with_extra_2 = candidate_with_extra.copy()
                        candidate_with_extra_2[5] = candidate[5] - 2.0 * np.pi
                        vector_states.append(candidate_with_extra_2)
                    if candidate[5] + 2.0 * np.pi < limit_upper[5]:
                        candidate_with_extra_2 = candidate_with_extra.copy()
                        candidate_with_extra_2[5] = candidate[5] + 2.0 * np.pi
                        vector_states.append(candidate_with_extra_2)
                if candidate[0] + 2.0 * np.pi < limit_upper[0]:
                    candidate_with_extra = candidate.copy()
                    candidate_with_extra[0] = candidate[0] + 2.0 * np.pi
                    vector_states.append(candidate_with_extra.copy())
                    if candidate[5] - 2.0 * np.pi > limit_lower[5]:
                        candidate_with_extra_2 = candidate_with_extra.copy()
                        candidate_with_extra_2[5] = candidate[5] - 2.0 * np.pi
                        vector_states.append(candidate_with_extra_2)
                    if candidate[5] + 2.0 * np.pi < limit_upper[5]:
                        candidate_with_extra_2 = candidate_with_extra.copy()
                        candidate_with_extra_2[5] = candidate[5] + 2.0 * np.pi
                        vector_states.append(candidate_with_extra_2)

                if candidate[5] - 2.0 * np.pi > limit_lower[5]:
                    candidate_with_extra = candidate.copy()
                    candidate_with_extra[5] -= 2.0 * np.pi
                    vector_states.append(candidate_with_extra)
                if candidate[5] + 2.0 * np.pi < limit_upper[5]:
                    candidate_with_extra = candidate.copy()
                    candidate_with_extra[5] += 2.0 * np.pi
                    vector_states.append(candidate_with_extra)

        if not vector_states:
            return q_backup, False

        result_controller = min(
            vector_states,
            key=lambda candidate: np.sum(
                np.abs(candidate - ref_joint_controller)
            ),
        )

        # C++ 的 toExtraArmJointPos：返回用户/机械臂坐标。
        return result_controller - joint_offset, True
    def track_pose(self, targets: List[float], gripper_pos: float = -1.0) -> int:
        """
        笛卡尔空间位姿跟踪（非阻塞）
        接收目标位姿（法兰相对基座），解算关节角并发送给底层控制器。

        Args:
            targets: [x, y, z, qx, qy, qz, qw] 目标位姿
            gripper_pos: 夹爪间距（<0 表示不控制夹爪）

        Returns:
            1: 指令发送成功
            0: 指令发送失败（控制模式错误或 IK 无解）
        """
        # 检查控制模式
        if self.control_mode == 0:  # IDLE
            return 0

        if len(targets) != 7:
            return 0

        # 提取目标位置和姿态
        pos_des = np.array(targets[:3])
        quat_des = np.array(targets[3:])  # [qx, qy, qz, qw]

        # 调用逆运动学求解
        q_target, success = self._ik_solve(pos_des, quat_des)
        if not success:
            # print("IK 无解，无法跟踪目标位姿")
            return 0

        # 发送关节指令
        return self.track_joint(q_target.tolist(), gripper_pos)
    
    def move_pose(
        self,
        target_pos: List[float],
        desire_time=-1,
        is_sync=True,
    ) -> int:
        """
        笛卡尔空间点到点运动（法兰位姿 -> 关节空间执行）

        通过逆运动学求解目标关节角，然后调用关节空间轨迹规划执行。
        - desire_time > 0:  五次多项式插值 (时间精确)
        - desire_time == -1: 时间最优双 S 曲线 (利用关节速度/加速度/加加速度约束)

        Args:
            target_pos:  目标位姿 [x, y, z, qx, qy, qz, qw] (法兰相对基座)
            desire_time: 期望到达时间 (s)，-1 为时间最优
            is_sync:     True: 阻塞至运动完成; False: 非阻塞

        Returns:
            1: 指令发送成功
            0: 指令发送失败 (控制模式为 IDLE / IK 无解 / 参数错误)
        """
        # 1. 控制模式检查 (IDLE 不允许运动)
        if self.control_mode == 0:
            return 0

        # 2. 参数检查
        if len(target_pos) != 7:
            return 0

        # 3. 逆运动学求解
        pos_des = np.array(target_pos[:3])
        quat_des = np.array(target_pos[3:])  # [qx, qy, qz, qw]
        q_target, success = self._ik_solve(pos_des, quat_des)
        if not success:
            return 0

        # 4. 委托给关节空间运动
        return self.move_joint(
            q_target.tolist(),
            desire_time=desire_time,
            is_sync=is_sync,
        )
    
    def move_joint_traj(
        self,
        target_pos: List[List[float]],
        gripper_pos: float = -1.0,
        stamps: Optional[List[float]] = None,
        is_sync: bool = True
    ) -> int:
        """
        PT 运动：依次经过一系列关节空间路径点（复用 move_joint）

        Args:
            target_pos: 路径点列表，每个点 6 个关节角，至少 2 个点
            gripper_pos: 全程夹爪目标间距，<0 不控制
            stamps:     时间戳 (s)，长度与 target_pos 相同，stamps[0] 为 0，
                        若为 None 则各段自动采用时间最优双 S 曲线
            is_sync:    阻塞至全部运动完成
        """
        if self.control_mode == 0:
            return 0
        if len(target_pos) < 2:
            return 0
        for pt in target_pos:
            if len(pt) != 6:
                return 0

        # 构建队列
        queue = []
        if stamps is not None and len(stamps) == len(target_pos):
            for i in range(len(target_pos) - 1):
                dt = stamps[i + 1] - stamps[i]
                if dt < 1e-6:
                    continue
                queue.append((target_pos[i + 1], float(dt), gripper_pos))
        else:
            for i in range(1, len(target_pos)):
                queue.append((target_pos[i], -1.0, gripper_pos))

        if not queue:
            return 0

        # 保存队列，并启动第一段
        self._traj_queue = queue
        first_target, first_time, _ = queue.pop(0)
        # 非阻塞调用 move_joint
        if self.move_joint(first_target, desire_time=first_time, is_sync=False) == 0:
            self._traj_queue = []
            return 0

        # 同步等待
        if is_sync:
            while self._traj_queue or self._traj_active or self._ds_active:
                self.step(5)
        return 1
    
    def move_pose_traj(
        self,
        target_pos: List[List[float]],
        gripper_pos: float = -1.0,
        stamps: Optional[List[float]] = None,
        is_sync: bool = True
    ) -> int:
        """
        笛卡尔空间 PT 运动：依次经过一系列末端位姿点（法兰相对基座）

        Args:
            target_pos: 路径点位姿列表，每个点为 [x, y, z, qx, qy, qz, qw]，至少 2 个点
            gripper_pos: 全程夹爪目标间距，<0 不控制
            stamps:     时间戳列表 (s)，长度与 target_pos 相同，stamps[0] 应为 0，
                        若为 None 则各段采用时间最优双 S 曲线
            is_sync:    阻塞至全部运动完成 (True) / 非阻塞 (False)

        Returns:
            1: 指令发送成功
            0: 指令发送失败 (控制模式为 IDLE 或 IK 无解或参数错误)
        """
        if self.control_mode == 0:
            return 0
        if len(target_pos) < 2:
            return 0
        for pt in target_pos:
            if len(pt) != 7:
                return 0

        # 对所有目标位姿逐一求逆运动学，得到关节角序列
        joint_targets = []
        for pose in target_pos:
            pos_des = np.array(pose[:3])
            quat_des = np.array(pose[3:])
            q_sol, success = self._ik_solve(pos_des, quat_des)
            if not success:
                return 0
            joint_targets.append(q_sol.tolist())

        # 委托给关节空间多点轨迹执行，完全复用 move_joint_traj
        return self.move_joint_traj(
            joint_targets,
            gripper_pos=gripper_pos,
            stamps=stamps,
            is_sync=is_sync
        )
    
    def inverse_kine(
        self,
        tool_index: int = 0,
        quat_pose = None,
        ref_joint = None
    ) -> Tuple[int, List[float]]:
        """
        单点逆运动学求解（接口与 C++ 一致）

        Args:
            tool_index: 工具号（默认 0，暂只支持法兰盘末端）
            quat_pose:  目标位姿 [x, y, z, qx, qy, qz, qw]
            ref_joint:  参考关节角 (6,)，用于多解选优，若为 None 则使用当前关节角

        Returns:
            (ret, joint_angles)  ret=1 成功，ret=0 失败；失败时 joint_angles 为空列表
        """
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")

        if quat_pose is None or len(quat_pose) != 7:
            return 0, []

        pos_des = np.array(quat_pose[:3], dtype=float)
        quat_des = np.array(quat_pose[3:], dtype=float)

        if ref_joint is not None:
            if len(ref_joint) != 6:
                return 0, []
            ref = np.array(ref_joint, dtype=float)
        else:
            ref = None

        q_sol, success = self._ik_solve(pos_des, quat_des, ref_joint=ref)
        if success:
            return 1, q_sol.tolist()
        else:
            return 0, []
        
    def inverse_kine_array(
        self,
        tool_index: int = 0,
        quat_pose_list = None,
        ref_joint_list = None
    ) -> Tuple[int, List[List[float]]]:
        """
        批量逆运动学求解（接口与 C++ 一致）

        Args:
            tool_index:     工具号（默认 0，暂只支持法兰盘末端）
            quat_pose_list: 目标位姿列表，每个元素为 [x, y, z, qx, qy, qz, qw]
            ref_joint_list: 参考关节角列表，与 quat_pose_list 等长，每个元素为 (6,)；
                            若为 None，则每个点均使用当前关节角作为参考

        Returns:
            (ret, joint_angles_list)  ret=1 全部成功，ret=0 至少一个失败；
            失败时 joint_angles_list 中对应位置为 []，成功位置为关节角列表
        """
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")

        if not quat_pose_list:
            return 0, []

        n = len(quat_pose_list)
        results = []

        if ref_joint_list is None:
            refs = [None] * n
        else:
            if len(ref_joint_list) != n:
                return 0, []
            refs = ref_joint_list

        all_success = True
        for pose, ref in zip(quat_pose_list, refs):
            ret, joint = self.inverse_kine(tool_index=0, quat_pose=pose, ref_joint=ref)
            if ret == 1:
                results.append(joint)
            else:
                all_success = False
                results.append([])   # 占位

        return (1 if all_success else 0), results
    

    def move_line_joint(self, target_pos: List[float], is_sync: bool = True) -> int:
        """
        关节空间直线到点运动（路径约束的同步双 S 曲线）
        
        Args:
            target_pos: 关节目标位置 (6 个角度，单位 rad)
            is_sync:    是否阻塞至运动完成
        
        Returns:
            1: 成功
            0: 失败（控制模式为 IDLE 或参数错误）
        """
        if self.control_mode == 0:
            return 0
        if len(target_pos) != 6:
            return 0

        # 起点与终点
        q_cur = self.data.qpos[:6].copy()
        q_tar = np.array(target_pos, dtype=float)
        dq = q_tar - q_cur

        # 若位移为零，直接返回
        if np.all(np.abs(dq) < 1e-12):
            if is_sync:
                pass  # 已在目标点，无需运动
            return 1

        # ---------- 计算 λ 空间的合成约束 ----------
        v_lam = float('inf')
        a_lam = float('inf')
        j_lam = float('inf')
        for i in range(6):
            if abs(dq[i]) < 1e-12:
                continue
            v_lam = min(v_lam, self.joint_vel_limits[i] / abs(dq[i]))
            a_lam = min(a_lam, self.joint_acc_limits[i] / abs(dq[i]))
            j_lam = min(j_lam, self.joint_jerk_limits[i] / abs(dq[i]))

        if v_lam == float('inf') or a_lam == float('inf') or j_lam == float('inf'):
            # 所有关节位移都为零（已在上面处理），理论上不会走到这里
            return 0

        # ---------- 对 λ 进行双 S 曲线规划 ----------
        T_total, params = self._plan_double_s(
            D=1.0,          # λ 从 0 到 1
            v_max=v_lam,
            a_max=a_lam,
            j_max=j_lam
        )

        # ---------- 激活直线运动 ----------
        self._traj_active = False   # 关闭其他轨迹
        self._ds_active = False
        self._line_ds_active = True
        self._line_ds_start_time = self.data.time
        self._line_ds_duration = T_total
        self._line_ds_q0 = q_cur
        self._line_ds_dq = dq
        self._line_ds_params = params
        self._hold_target = None

        # 同步等待
        if is_sync:
            while self._line_ds_active:
                self.step(5)
        return 1
    
    def move_line_pose(self, target_pos: List[float], is_sync: bool = True) -> int:
        """
        笛卡尔空间直线到点运动（法兰相对基座）
        
        通过逆运动学求解目标关节角，然后调用关节空间直线运动执行。
        
        Args:
            target_pos: 目标位姿 [x, y, z, qx, qy, qz, qw]
            is_sync:    True: 阻塞至运动完成; False: 非阻塞
        
        Returns:
            1: 指令发送成功
            0: 指令发送失败 (控制模式为 IDLE / IK 无解 / 参数错误)
        """
        # 1. 控制模式检查 (IDLE 不允许运动)
        if self.control_mode == 0:
            return 0

        # 2. 参数检查
        if len(target_pos) != 7:
            return 0

        # 3. 逆运动学求解
        pos_des = np.array(target_pos[:3])
        quat_des = np.array(target_pos[3:])  # [qx, qy, qz, qw]
        q_target, success = self._ik_solve(pos_des, quat_des)
        if not success:
            return 0

        # 4. 委托给关节空间直线运动
        return self.move_line_joint(q_target.tolist(), is_sync=is_sync)
        

    def forward_kine(self, jnt_value: List[float], tool_index : int = 0) -> Tuple[int, List[float]]:
        """
        单点正运动学求解

        Args:
            tool_index: 工具号（目前仅支持 0，即法兰末端）
            jnt_value:  关节角度列表，长度为 6

        Returns:
            (ret, quat_pose)
                ret=1 成功，ret=0 失败
                quat_pose: [x, y, z, qx, qy, qz, qw] (失败时返回空列表)
        """
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if len(jnt_value) != 6:
            return 0, []

        # 备份当前状态
        qpos_backup = self.data.qpos.copy()

        # 设置临时关节角
        self.data.qpos[:6] = np.array(jnt_value, dtype=float)

        # 更新运动学
        mujoco.mj_forward(self.model, self.data)

        # 获取末端位置与旋转矩阵
        pos = self.data.site_xpos[self.ee_site_id].copy()
        rot = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)

        # 旋转矩阵 → 四元数 (MuJoCo 顺序: [w, x, y, z])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())

        # 恢复原始状态
        self.data.qpos[:] = qpos_backup

        # 返回 [x, y, z, qx, qy, qz, qw]
        quat_pose = [pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]]
        return 1, quat_pose


    def forward_kine_array(
        self,
        jnt_value_list: List[List[float]],
        tool_index: int = 0
    ) -> Tuple[int, List[List[float]]]:
        """
        批量正运动学求解

        Args:
            tool_index:     工具号（目前仅支持 0，即法兰末端）
            jnt_value_list: 关节角度列表，每个元素为长度为 6 的列表

        Returns:
            (ret, quat_pose_list)
                ret=1 全部成功，ret=0 至少一个失败
                quat_pose_list: 每个输入对应的位姿结果，失败位置为 []
        """
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")

        results = []
        all_success = True

        for jnt in jnt_value_list:
            ret, pose = self.forward_kine(jnt, tool_index)
            if ret == 1:
                results.append(pose)
            else:
                all_success = False
                results.append([])

        return (1 if all_success else 0), results
    
    def get_plan_joint_pos(self) -> List[float]:
        """获取当前控制指令的期望关节角度 (rad)"""
        return self._plan_q.tolist()

    def get_plan_joint_vel(self) -> List[float]:
        """获取当前控制指令的期望关节角速度 (rad/s)"""
        return self._plan_v.tolist()

    def get_plan_joint_tau(self) -> List[float]:
        """获取当前控制指令的期望关节力矩 (Nm)"""
        return self._plan_tau.tolist()
    

    def _save_teach_json(self, name: str) -> int:
        """将指定轨迹导出为 JSON 文件（无时间戳）"""
        if name not in self._teach_data:
            return 0

        data = self._teach_data[name]
        joint_pos = data['joint_pos']  # list of list

        export_data = {
            "eeff_name": "",
            "name": name,
            "path": {
                "joint_pos": joint_pos
            },
            "point_size": len(joint_pos),
            "record_angle_interval_": 0,
            "record_time_interval": 0
        }

        filepath = os.path.join(self._export_dir, f"{name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=3, ensure_ascii=False)
        print(f"[Trajectory] Exported to {filepath}")
        return 1
    
    def trajectory_teach(self, off_on: bool, name: str) -> int:
        if off_on:
            if self._recording:
                return 0
            if not name:
                return 0
            # 保存当前控制模式，并切换到拖动模式（3）
            self._prev_control_mode = self.control_mode
            if self.control_mode != 3:
                self.set_control_mode(3)
            self._recording = True
            self._rec_name = name
            self._rec_start_time = self.data.time
            self._rec_joint_pos = []
            self._rec_time_stamps = []
            return 1
        else:
            if not self._recording:
                return 0
            if len(self._rec_joint_pos) < 2:
                self._recording = False
                return 0
            # 保存数据
            self._teach_data[self._rec_name] = {
                'joint_pos': [pos.tolist() for pos in self._rec_joint_pos],
                'time_stamps': self._rec_time_stamps.copy()
            }
            self._save_teach_json(self._rec_name)
            self._recording = False
            # 恢复之前的控制模式
            if hasattr(self, '_prev_control_mode'):
                self.set_control_mode(self._prev_control_mode)
            return 1

    def trajectory_recorder(self, name: str, is_sync: bool = True) -> int:
        print(f"[Recorder] Starting replay for '{name}'")
        if name not in self._teach_data:
            print(f"[Recorder] Not in memory, trying to import from default dir...")
            if self._import_teach_from_json(name) == 0:
                print("[Recorder] Import failed.")
                return 0
            else:
                print("[Recorder] Import succeeded.")

        data = self._teach_data[name]
        joint_pos = np.array(data['joint_pos'])
        stamps = data.get('time_stamps', None)

        if len(joint_pos) < 2:
            print("[Recorder] Too few points.")
            return 0

        # ---------- 若没有有效时间戳，生成均匀时间戳（假设匀速） ----------
        if stamps is None or len(stamps) == 0:
            print("[Recorder] No valid time stamps, generating uniform time stamps (using simulation dt).")
            # 使用仿真步长作为采样间隔（或者你可以指定一个固定值，如 0.01s）
            dt = self.model.opt.timestep
            stamps = np.arange(len(joint_pos)) * dt
            # 也可以让用户指定一个速度缩放因子，这里仅做演示

        # ---------- 统一使用 track_joint 线性插值 ----------
        stamps = np.array(stamps)
        if stamps[0] != 0.0:
            stamps = stamps - stamps[0]
        total_duration = stamps[-1]
        print(f"[Recorder] Total duration = {total_duration:.3f}s, using track_joint interpolation.")
        if total_duration <= 0:
            print("[Recorder] Invalid duration.")
            return 0

        if self.control_mode == 0:
            print("[Recorder] Control mode is IDLE, setting to position control (mode 1).")
            self.set_control_mode(1)

        start_time = self.data.time
        step_count = 0
        while self.data.time - start_time < total_duration:
            t = self.data.time - start_time
            idx = np.searchsorted(stamps, t, side='right') - 1
            if idx < 0:
                idx = 0
            if idx >= len(joint_pos) - 1:
                q_des = joint_pos[-1]
            else:
                t0, t1 = stamps[idx], stamps[idx+1]
                if t1 - t0 < 1e-12:
                    q_des = joint_pos[idx]
                else:
                    alpha = (t - t0) / (t1 - t0)
                    q_des = joint_pos[idx] + alpha * (joint_pos[idx+1] - joint_pos[idx])
            if step_count % 100 == 0:
                print(f"[Recorder] t={t:.3f}/{total_duration:.3f}")
            self.track_joint(q_des.tolist(), gripper_pos=-1.0)
            self.step(1)
            step_count += 1

        print("[Recorder] Replay finished, finalizing...")
        self.track_joint(joint_pos[-1].tolist(), gripper_pos=-1.0)
        for _ in range(5):
            self.step(1)
        return 1

    def check_teach(self) -> List[str]:
        """返回内存中已加载的轨迹名称列表"""
        return list(self._teach_data.keys())

    # ==================== 私有辅助方法 ====================
    def _save_teach_json(self, name: str) -> int:
        """将轨迹导出为 JSON（带时间戳）到默认目录"""
        if name not in self._teach_data:
            return 0
        data = self._teach_data[name]
        export = {
            "name": name,
            "joint_pos": data['joint_pos'],
            "time_stamps": data['time_stamps'],
            "point_size": len(data['joint_pos'])
        }
        filepath = os.path.join(self._export_dir, f"{name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=3, ensure_ascii=False)
        return 1

    def _import_teach_from_json(self, name: str) -> int:
        """从默认目录导入 JSON 到内存（仅内部使用）"""
        filepath = os.path.join(self._export_dir, f"{name}.json")
        if not os.path.exists(filepath):
            return 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            return 0

        # 提取关节数据（兼容有/无 path 嵌套）
        joint_pos = data.get('joint_pos')
        if joint_pos is None:
            joint_pos = data.get('path', {}).get('joint_pos', [])
        if not joint_pos or len(joint_pos) < 2:
            return 0

        time_stamps = data.get('time_stamps', None)   # 可能不存在
        if time_stamps is not None and len(time_stamps) == 0:
            time_stamps = None

        self._teach_data[name] = {
            'joint_pos': joint_pos,
            'time_stamps': time_stamps
        }
        return 1
