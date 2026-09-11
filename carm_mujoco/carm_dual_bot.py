from typing import List
import os
import json
import mujoco
import mujoco.viewer
import numpy as np


class CArmDualBot:
    """
    CARM 双臂机器人 MuJoCo 仿真控制类。

    适配模型: carm_d3 (7-DOF 双臂 + 夹爪)
    左臂: qpos/qvel 索引 [0:7]，右臂: qpos/qvel 索引 [9:16]
    """

    LEFT_ARM_SLICE = slice(0, 7)
    RIGHT_ARM_SLICE = slice(9, 16)

    LEFT_GRIPPER_INDICES = [7, 8]
    RIGHT_GRIPPER_INDICES = [16, 17]

    LEFT_TAU_SENSOR_ADR = [2, 5, 8, 11, 14, 17, 20]       # joint1~7-LEFT_f
    RIGHT_TAU_SENSOR_ADR = [29, 32, 35, 38, 41, 44, 47]    # joint1~7-RIGHT_f

    def __init__(self, xml_path: str, render: bool = True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.control_mode = 0                     # 0: IDLE

        # 机械臂位置环 PI 增益 (参考 carm_single_col)
        self.kp_arm_pos = 70.0
        self.ki_arm_pos = 5.0
        # 机械臂速度环 PI 增益
        self.kp_arm_vel = 30.0
        self.ki_arm_vel = 2.0
        self.kd_arm_vel = 5.0

        # MIT 增益 (7x7)
        self.kp_mit = np.diag([500, 500, 500, 400, 400, 400, 400])
        self.kd_mit = np.diag([5, 5, 5, 5, 5, 5, 5])

        # 积分状态（左右臂各 7 维）
        pos0 = np.zeros(7)
        vel0 = np.zeros(7)
        self._pos_error_int = {"left": pos0.copy(), "right": pos0.copy()}
        self._vel_error_int = {"left": vel0.copy(), "right": vel0.copy()}

        # 积分限幅
        self.pos_int_limit = 10.0
        self.vel_int_limit = 10.0

        # 末端 link7 body (用于位姿获取)
        self._link7_left_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link7-LEFT")
        self._link7_right_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link7-RIGHT")

        # 臂体 ID 集合（用于外力计算）
        self._left_arm_body_ids = set()
        self._right_arm_body_ids = set()
        for i in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("link"):
                if name.endswith("-LEFT"):
                    self._left_arm_body_ids.add(i)
                elif name.endswith("-RIGHT"):
                    self._right_arm_body_ids.add(i)

        # 根 body (基坐标系原点)
        self._root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "root")

        # 阻尼和摩擦力系数 (从模型读取)
        self._damping = {
            "left": self.model.dof_damping[self.LEFT_ARM_SLICE].copy(),
            "right": self.model.dof_damping[self.RIGHT_ARM_SLICE].copy(),
        }
        self._frictionloss = {
            "left": self.model.dof_frictionloss[self.LEFT_ARM_SLICE].copy(),
            "right": self.model.dof_frictionloss[self.RIGHT_ARM_SLICE].copy(),
        }

        # 规划值记录
        self._plan_q = {"left": np.zeros(7), "right": np.zeros(7)}
        self._plan_v = {"left": np.zeros(7), "right": np.zeros(7)}
        self._plan_tau = {"left": np.zeros(7), "right": np.zeros(7)}

        # 夹爪目标
        self._gripper_target = {"left": 0.0, "right": 0.0}

        # 夹爪力限位置控制状态（左右夹爪独立）
        self._gripper_pos_pre = {"left": 0.0, "right": 0.0}
        self._gripper_err_int = {"left": 0.0, "right": 0.0}
        self._gripper_pending = {"left": None, "right": None}
        self.kp_grip_pi = 10.0
        self.ki_grip_pi = 5.0
        self.gripper_err_int_limit = 0.5

        # D3 模型没有单臂模型中的 tip site，使用与单臂 XML 相同的
        # 指尖局部位置，根据两指 body 位姿计算实际指尖间距。
        self._gripper_tip_local_pos = np.array([0.0, 0.0, -0.022])
        self._gripper_body_ids = {
            "left": (
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right-LEFT"),
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left-LEFT"),
            ),
            "right": (
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right-RIGHT"),
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left-RIGHT"),
            ),
        }

        # ========== 轨迹规划状态 ==========
        ARM_KEYS = ("left", "right")

        # 五次多项式轨迹
        self._traj_active = {k: False for k in ARM_KEYS}
        self._traj_q0 = {k: np.zeros(7) for k in ARM_KEYS}
        self._traj_qf = {k: np.zeros(7) for k in ARM_KEYS}
        self._traj_start_time = {k: 0.0 for k in ARM_KEYS}
        self._traj_duration = {k: 0.0 for k in ARM_KEYS}

        # 双 S 轨迹
        self._ds_active = {k: False for k in ARM_KEYS}
        self._ds_start_time = {k: 0.0 for k in ARM_KEYS}
        self._ds_duration = {k: 0.0 for k in ARM_KEYS}
        self._ds_target = {k: np.zeros(7) for k in ARM_KEYS}
        self._ds_params = {k: [] for k in ARM_KEYS}

        # 直线（标量 λ）双 S 轨迹
        self._line_ds_active = {k: False for k in ARM_KEYS}
        self._line_ds_start_time = {k: 0.0 for k in ARM_KEYS}
        self._line_ds_duration = {k: 0.0 for k in ARM_KEYS}
        self._line_ds_q0 = {k: np.zeros(7) for k in ARM_KEYS}
        self._line_ds_dq = {k: np.zeros(7) for k in ARM_KEYS}
        self._line_ds_params = {k: {} for k in ARM_KEYS}

        # 保持目标 & 轨迹队列
        self._hold_target = {k: None for k in ARM_KEYS}
        self._traj_queue = {k: [] for k in ARM_KEYS}

        # 示教录制
        self._teach_data = {}
        self._recording = {"left": False, "right": False}
        self._rec_name = {"left": "", "right": ""}
        self._rec_joint_pos = {"left": [], "right": []}
        self._rec_time_stamps = {"left": [], "right": []}
        self._rec_start_time = {"left": 0.0, "right": 0.0}
        self._prev_control_mode = {"left": 0, "right": 0}
        self._export_dir = "./traj_data"
        os.makedirs(self._export_dir, exist_ok=True)

        # 运动约束
        self.joint_vel_limits = np.array([4.2, 4.2, 3.14, 3.14, 3.14, 3.14, 3.14])
        self.joint_acc_limits = np.array([8.4, 8.4, 6.28, 6.28, 6.28, 6.28, 6.28])
        self.joint_jerk_limits = np.array([37.8, 37.8, 28.26, 28.26, 28.26, 28.26, 28.26])

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = [0, 0, 0.44]

    def step(self, nstep: int = 1):
        for _ in range(nstep):
            if self._recording["left"]:
                self._rec_joint_pos["left"].append(
                    self.data.qpos[self.LEFT_ARM_SLICE].copy())
                self._rec_time_stamps["left"].append(
                    self.data.time - self._rec_start_time["left"])
            if self._recording["right"]:
                self._rec_joint_pos["right"].append(
                    self.data.qpos[self.RIGHT_ARM_SLICE].copy())
                self._rec_time_stamps["right"].append(
                    self.data.time - self._rec_start_time["right"])
            self._update_traj_control()

            # 轨迹控制会刷新 position 执行器的 ctrl。set_*_gripper 的
            # 力矩控制必须最后写入，才能在夹持搬运期间保持生效。
            for arm in ("left", "right"):
                tau_grip = self._gripper_pending[arm]
                if tau_grip is not None:
                    self._set_gripper_torque_ctrl(arm, tau_grip)
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ---------- 辅助 ----------

    def _arm_slice(self, arm: str) -> slice:
        return self.LEFT_ARM_SLICE if arm == "left" else self.RIGHT_ARM_SLICE

    def _gripper_indices(self, arm: str) -> list:
        return self.LEFT_GRIPPER_INDICES if arm == "left" else self.RIGHT_GRIPPER_INDICES

    def _set_arm_ctrl(self, arm: str, tau: np.ndarray):
        if arm == "left":
            self.data.ctrl[self.LEFT_ARM_SLICE] = tau
        else:
            self.data.ctrl[self.RIGHT_ARM_SLICE] = tau

    def _set_gripper_ctrl(self, arm: str, pos: float):
        idx = self._gripper_indices(arm)
        self.data.ctrl[idx[0]] = pos
        self.data.ctrl[idx[1]] = pos

    def _get_gripper_pos(self, arm: str) -> float:
        """获取指定夹爪两指尖间距（米）。"""
        mujoco.mj_kinematics(self.model, self.data)
        right_id, left_id = self._gripper_body_ids[arm]
        right_tip = (
            self.data.xpos[right_id]
            + self.data.xmat[right_id].reshape(3, 3)
            @ self._gripper_tip_local_pos
        )
        left_tip = (
            self.data.xpos[left_id]
            + self.data.xmat[left_id].reshape(3, 3)
            @ self._gripper_tip_local_pos
        )
        return float(np.linalg.norm(
            right_tip - left_tip))

    def _get_gripper_vel(self, arm: str) -> float:
        """获取指定夹爪两指相对运动速度（米/秒）。"""
        # 两个手指通过 equality 约束同步，开合速度约等于 2*v_joint。
        gripper_joint = self._gripper_indices(arm)[0]
        return float(2.0 * self.data.qvel[gripper_joint])

    def _get_gripper_tau(self, arm: str) -> float:
        """获取指定夹爪第一根手指执行器的实际广义力。"""
        # 与单臂模型一致，返回第一根手指执行器的 qfrc_actuator。
        gripper_actuator = self._gripper_indices(arm)[0]
        return float(self.data.qfrc_actuator[gripper_actuator])

    def _set_gripper_torque_ctrl(self, arm: str, tau: float):
        """将目标力矩转换为 D3 position 执行器的等效 ctrl。"""
        for actuator_id in self._gripper_indices(arm):
            # position 执行器的力为 kp * (ctrl - length) - kv * velocity。
            # 反解 ctrl 后，data.qfrc_actuator 中的实际执行器力就是 tau。
            if self.model.actuator_biastype[actuator_id] == int(
                    mujoco.mjtBias.mjBIAS_AFFINE):
                kp = float(self.model.actuator_gainprm[actuator_id, 0])
                kv = -float(self.model.actuator_biasprm[actuator_id, 2])
                if abs(kp) < 1e-12:
                    self.data.ctrl[actuator_id] = 0.0
                    continue
                self.data.ctrl[actuator_id] = (
                    self.data.actuator_length[actuator_id]
                    + (tau + kv * self.data.actuator_velocity[actuator_id]) / kp
                )
            else:
                # 兼容使用 motor 力矩执行器的双臂模型。
                self.data.ctrl[actuator_id] = tau

    # ---------- 计算控制力矩 (参考 carm_single_col._compute_tau) ----------

    def _compute_tau(
        self,
        arm: str,
        q_des: np.ndarray,
        v_des: np.ndarray = None,
    ) -> np.ndarray:
        """
        根据当前 control_mode 计算控制力矩

        Args:
            arm: "left" 或 "right"
            q_des: 目标关节位置 (7,)
            v_des: 目标关节速度 (7,)

        Returns:
            tau_arm: (7,)
        """
        if v_des is None:
            v_des = np.zeros(7)

        sl = self._arm_slice(arm)
        q_arm = self.data.qpos[sl].copy()
        v_arm = self.data.qvel[sl].copy()

        tau_arm = np.zeros(7)

        # ---------- 摩擦补偿 ----------
        v_sign = np.tanh(v_arm)
        friction_est = -(
            self._damping[arm] * v_arm
            + self._frictionloss[arm] * v_sign
        )

        # ==================================================
        # MODE 0 — IDLE
        # ==================================================
        if self.control_mode == 0:
            return tau_arm

        # ==================================================
        # MODE 1 — 位置PI -> 速度PI
        # ==================================================
        elif self.control_mode == 1:
            pos_error = q_des - q_arm

            self._pos_error_int[arm] += (
                pos_error * self.model.opt.timestep
            )
            np.clip(
                self._pos_error_int[arm],
                -self.pos_int_limit,
                self.pos_int_limit,
                out=self._pos_error_int[arm],
            )

            v_cmd = (
                self.kp_arm_pos * pos_error
                + self.ki_arm_pos * self._pos_error_int[arm]
                + v_des
            )

            vel_error = v_cmd - v_arm

            self._vel_error_int[arm] += (
                vel_error * self.model.opt.timestep
            )
            np.clip(
                self._vel_error_int[arm],
                -self.vel_int_limit,
                self.vel_int_limit,
                out=self._vel_error_int[arm],
            )

            tau_arm = (
                self.kp_arm_vel * vel_error
                + self.ki_arm_vel * self._vel_error_int[arm]
            )

        # ==================================================
        # MODE 2 — MIT
        # ==================================================
        elif self.control_mode == 2:
            pos_error = q_des - q_arm
            vel_error = v_des - v_arm

            bias = self.data.qfrc_bias[sl]

            tau_arm = (
                self.kp_mit @ pos_error
                + self.kd_mit @ vel_error
                + bias
                + friction_est
            )

        # ==================================================
        # MODE 3 — 拖动模式
        # ==================================================
        elif self.control_mode == 3:
            tau_arm = self.data.qfrc_bias[sl] + friction_est

        # ==================================================
        # MODE 4 — 力矩限制模式
        # ==================================================
        elif self.control_mode == 4:
            pos_error = q_des - q_arm

            self._pos_error_int[arm] += (
                pos_error * self.model.opt.timestep
            )
            np.clip(
                self._pos_error_int[arm],
                -self.pos_int_limit,
                self.pos_int_limit,
                out=self._pos_error_int[arm],
            )

            v_cmd = (
                self.kp_arm_pos * pos_error
                + self.ki_arm_pos * self._pos_error_int[arm]
                + v_des
            )

            vel_error = v_cmd - v_arm

            self._vel_error_int[arm] += (
                vel_error * self.model.opt.timestep
            )
            np.clip(
                self._vel_error_int[arm],
                -self.vel_int_limit,
                self.vel_int_limit,
                out=self._vel_error_int[arm],
            )

            tau_arm = (
                self.kp_arm_vel * vel_error
                + self.ki_arm_vel * self._vel_error_int[arm]
            )

            mujoco.mj_rne(
                self.model,
                self.data,
                1,
                self.data.qfrc_inverse,
            )

            inv_tau = self.data.qfrc_inverse[sl]
            for i in range(7):
                limit = abs(inv_tau[i])
                if abs(tau_arm[i]) > limit:
                    tau_arm[i] = (
                        limit * np.sign(tau_arm[i])
                        + 4 * np.sign(tau_arm[i])
                    )

        return tau_arm

    @staticmethod
    def _get_qpos(data, slc: slice) -> List[float]:
        return data.qpos[slc].tolist()

    @staticmethod
    def _get_qvel(data, slc: slice) -> List[float]:
        return data.qvel[slc].tolist()

    def _get_tau(self, adrs: List[int]) -> List[float]:
        return [float(self.data.sensordata[adr]) for adr in adrs]

    # ================================================================
    # 接口
    # ================================================================

    def set_control_mode(self, mode: int) -> int:
        """
        设置控制模式
        0 - IDLE
        1 - 点位控制
        2 - MIT 控制
        3 - 关节拖动
        4 - PF 力位混合（力矩限制）
        """
        if mode not in (0, 1, 2, 3, 4):
            return 0
        if mode != self.control_mode:
            self._pos_error_int["left"].fill(0.0)
            self._pos_error_int["right"].fill(0.0)
            self._vel_error_int["left"].fill(0.0)
            self._vel_error_int["right"].fill(0.0)
        self.control_mode = mode
        print(f"控制模式切换到: {mode}")
        return 1

    def _set_gripper(self, arm: str, pos: float, tau: float = 10.0) -> int:
        """
        指定夹爪的力限位置控制（PI + 力矩限幅）。

        pos 是两指间距（米），tau 是每个 D3 夹爪执行器的力矩限幅。
        """
        if pos < 0:
            return 0

        e = pos - self._get_gripper_pos(arm)

        # PI 输出（限幅前）
        tau_raw = (
            self.kp_grip_pi * e
            + self.ki_grip_pi * self._gripper_err_int[arm]
        )

        # 条件积分，避免夹持时积分持续累积导致反向动作变慢。
        if abs(tau_raw) < tau:
            self._gripper_err_int[arm] += e * self.model.opt.timestep
            self._gripper_err_int[arm] = float(np.clip(
                self._gripper_err_int[arm],
                -self.gripper_err_int_limit,
                self.gripper_err_int_limit,
            ))

        tau_cmd = float(np.clip(tau_raw, -tau, tau))

        # 保存目标和力矩。step() 会在轨迹控制之后再次写入该力矩，
        # 因此左右夹爪可以各自独立锁存并同时控制。
        self._gripper_target[arm] = pos
        self._gripper_pos_pre[arm] = pos
        self._gripper_pending[arm] = tau_cmd
        self._set_gripper_torque_ctrl(arm, tau_cmd)
        return 1

    def set_left_gripper(self, pos: float, tau: float = 10.0) -> int:
        """左夹爪力限位置控制，pos 为两指间距（米）。"""
        return self._set_gripper("left", pos, tau)

    def set_right_gripper(self, pos: float, tau: float = 10.0) -> int:
        """右夹爪力限位置控制，pos 为两指间距（米）。"""
        return self._set_gripper("right", pos, tau)

    def get_left_gripper_pos(self) -> float:
        """获取左夹爪两指间距（米）。"""
        return self._get_gripper_pos("left")

    def get_left_gripper_vel(self) -> float:
        """获取左夹爪两指相对运动速度（米/秒）。"""
        return self._get_gripper_vel("left")

    def get_left_gripper_tau(self) -> float:
        """获取左夹爪第一根手指执行器的实际广义力。"""
        return self._get_gripper_tau("left")

    def get_right_gripper_pos(self) -> float:
        """获取右夹爪两指间距（米）。"""
        return self._get_gripper_pos("right")

    def get_right_gripper_vel(self) -> float:
        """获取右夹爪两指相对运动速度（米/秒）。"""
        return self._get_gripper_vel("right")

    def get_right_gripper_tau(self) -> float:
        """获取右夹爪第一根手指执行器的实际广义力。"""
        return self._get_gripper_tau("right")

    def track_left_joint(self, targets: List[float], gripper_pos: float = -1) -> int:
        if self.control_mode not in (1, 2, 3, 4):
            return 0
        if len(targets) != 7:
            return 0

        self._hold_target["left"] = None

        q_des = np.array(targets, dtype=float)
        tau_arm = self._compute_tau("left", q_des, np.zeros(7))

        self._set_arm_ctrl("left", tau_arm)
        if gripper_pos >= 0:
            self._gripper_target["left"] = gripper_pos
            self._gripper_pending["left"] = None
        self._set_gripper_ctrl("left", self._gripper_target["left"])

        self._plan_q["left"] = q_des.copy()
        self._plan_v["left"] = np.zeros(7)
        self._plan_tau["left"] = tau_arm.copy()
        return 1

    def track_right_joint(self, targets: List[float], gripper_pos: float = -1) -> int:
        if self.control_mode not in (1, 2, 3, 4):
            return 0
        if len(targets) != 7:
            return 0

        self._hold_target["right"] = None

        q_des = np.array(targets, dtype=float)
        tau_arm = self._compute_tau("right", q_des, np.zeros(7))

        self._set_arm_ctrl("right", tau_arm)
        if gripper_pos >= 0:
            self._gripper_target["right"] = gripper_pos
            self._gripper_pending["right"] = None
        self._set_gripper_ctrl("right", self._gripper_target["right"])

        self._plan_q["right"] = q_des.copy()
        self._plan_v["right"] = np.zeros(7)
        self._plan_tau["right"] = tau_arm.copy()
        return 1

    def track_left_pose(self, targets: List[float], gripper_pos: float = -1) -> int:
        """
        笛卡尔空间位姿跟踪（非阻塞）
        接收目标位姿（法兰相对基座），解算关节角并发送给底层控制器。

        Args:
            targets: [x, y, z, qx, qy, qz, qw]
            gripper_pos: 夹爪间距（<0 不控制）

        Returns:
            1: 成功，0: 失败
        """
        if self.control_mode == 0:
            return 0
        if len(targets) != 7:
            return 0

        pos_des = np.array(targets[:3])
        quat_des = np.array(targets[3:])  # [qx, qy, qz, qw]

        q_target, success = self._ik_solve("left", pos_des, quat_des,
            ref_joint=np.array(self.get_left_joint_pos(), dtype=float))
        if not success:
            return 0

        return self.track_left_joint(q_target.tolist(), gripper_pos)

    def track_right_pose(self, targets: List[float], gripper_pos: float = -1) -> int:
        """
        笛卡尔空间位姿跟踪（非阻塞）
        接收目标位姿（法兰相对基座），解算关节角并发送给底层控制器。

        Args:
            targets: [x, y, z, qx, qy, qz, qw]
            gripper_pos: 夹爪间距（<0 不控制）

        Returns:
            1: 成功，0: 失败
        """
        if self.control_mode == 0:
            return 0
        if len(targets) != 7:
            return 0

        pos_des = np.array(targets[:3])
        quat_des = np.array(targets[3:])  # [qx, qy, qz, qw]

        q_target, success = self._ik_solve("right", pos_des, quat_des,
            ref_joint=np.array(self.get_right_joint_pos(), dtype=float))
        if not success:
            return 0

        return self.track_right_joint(q_target.tolist(), gripper_pos)

    def get_left_joint_pos(self) -> List[float]:
        return self._get_qpos(self.data, self.LEFT_ARM_SLICE)

    def get_right_joint_pos(self) -> List[float]:
        return self._get_qpos(self.data, self.RIGHT_ARM_SLICE)

    def get_left_joint_vel(self) -> List[float]:
        return self._get_qvel(self.data, self.LEFT_ARM_SLICE)

    def get_right_joint_vel(self) -> List[float]:
        return self._get_qvel(self.data, self.RIGHT_ARM_SLICE)

    def get_left_joint_tau(self) -> List[float]:
        return self._get_tau(self.LEFT_TAU_SENSOR_ADR)

    def get_right_joint_tau(self) -> List[float]:
        return self._get_tau(self.RIGHT_TAU_SENSOR_ADR)

    # ---------- 规划值接口 ----------

    def get_left_plan_joint_pos(self) -> List[float]:
        return self._plan_q["left"].tolist()

    def get_right_plan_joint_pos(self) -> List[float]:
        return self._plan_q["right"].tolist()

    def get_left_plan_joint_vel(self) -> List[float]:
        return self._plan_v["left"].tolist()

    def get_right_plan_joint_vel(self) -> List[float]:
        return self._plan_v["right"].tolist()

    def get_left_plan_joint_tau(self) -> List[float]:
        return self._plan_tau["left"].tolist()

    def get_right_plan_joint_tau(self) -> List[float]:
        return self._plan_tau["right"].tolist()

    def get_left_plan_cart_pose(self) -> List[float]:
        return self._fk_pose_from_joints("left", self._plan_q["left"])

    def get_right_plan_cart_pose(self) -> List[float]:
        return self._fk_pose_from_joints("right", self._plan_q["right"])

    def _fk_pose_from_joints(self, arm: str, q: np.ndarray) -> List[float]:
        sl = self._arm_slice(arm)
        q_bak = self.data.qpos[sl].copy()
        self.data.qpos[sl] = q
        mujoco.mj_forward(self.model, self.data)
        root_pos = self.data.xpos[self._root_body_id].copy()
        ee_body = self._link7_left_body if arm == "left" else self._link7_right_body
        pos = self.data.xpos[ee_body].copy() - root_pos
        rot = self.data.xmat[ee_body].copy().reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())
        self.data.qpos[sl] = q_bak
        return [pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]]

    # ---------- 正运动学 ----------

    def forward_kine_left(self, jnt_value: List[float] = None,
                          tool_index: int = 0):
        """左臂单点正运动学求解，返回 (ret, quat_pose)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if jnt_value is None or len(jnt_value) != 7:
            return 0, []
        pose = self._fk_pose_from_joints("left", np.array(jnt_value, dtype=float))
        return 1, pose

    def forward_kine_right(self, jnt_value: List[float] = None,
                           tool_index: int = 0):
        """右臂单点正运动学求解，返回 (ret, quat_pose)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if jnt_value is None or len(jnt_value) != 7:
            return 0, []
        pose = self._fk_pose_from_joints("right", np.array(jnt_value, dtype=float))
        return 1, pose

    def forward_kine_left_array(self, jnt_value_list: List[List[float]] = None,
                                tool_index: int = 0):
        """左臂批量正运动学求解，返回 (ret, quat_pose_list)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if not jnt_value_list:
            return 0, []
        results = []
        all_ok = True
        for jnt in jnt_value_list:
            ret, pose = self.forward_kine_left(jnt_value=jnt, tool_index=0)
            if ret:
                results.append(pose)
            else:
                all_ok = False
                results.append([])
        return (1 if all_ok else 0), results

    def forward_kine_right_array(self, jnt_value_list: List[List[float]] = None,
                                 tool_index: int = 0):
        """右臂批量正运动学求解，返回 (ret, quat_pose_list)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if not jnt_value_list:
            return 0, []
        results = []
        all_ok = True
        for jnt in jnt_value_list:
            ret, pose = self.forward_kine_right(jnt_value=jnt, tool_index=0)
            if ret:
                results.append(pose)
            else:
                all_ok = False
                results.append([])
        return (1 if all_ok else 0), results

    # ---------- 逆运动学 ----------

    def inverse_kine_left(self, quat_pose: List[float] = None,
                          ref_joint: List[float] = None,
                          tool_index: int = 0):
        """左臂单点逆运动学求解，返回 (ret, jnt_value)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if quat_pose is None or len(quat_pose) != 7:
            return 0, []

        pos_des = np.array(quat_pose[:3], dtype=float)
        quat_des = np.array(quat_pose[3:], dtype=float)

        ref = None
        if ref_joint is not None:
            if len(ref_joint) != 7:
                return 0, []
            ref = np.array(ref_joint, dtype=float)

        q_sol, success = self._ik_solve("left", pos_des, quat_des, ref_joint=ref)
        if success:
            return 1, q_sol.tolist()
        return 0, []

    def inverse_kine_right(self, quat_pose: List[float] = None,
                           ref_joint: List[float] = None,
                           tool_index: int = 0):
        """右臂单点逆运动学求解，返回 (ret, jnt_value)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if quat_pose is None or len(quat_pose) != 7:
            return 0, []

        pos_des = np.array(quat_pose[:3], dtype=float)
        quat_des = np.array(quat_pose[3:], dtype=float)

        ref = None
        if ref_joint is not None:
            if len(ref_joint) != 7:
                return 0, []
            ref = np.array(ref_joint, dtype=float)

        q_sol, success = self._ik_solve("right", pos_des, quat_des, ref_joint=ref)
        if success:
            return 1, q_sol.tolist()
        return 0, []

    def inverse_kine_left_array(self, quat_pose_list: List[List[float]] = None,
                                ref_joint_list: List[List[float]] = None,
                                tool_index: int = 0):
        """左臂批量逆运动学求解，返回 (ret, jnt_value_list)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if not quat_pose_list:
            return 0, []

        n = len(quat_pose_list)
        if ref_joint_list is None:
            refs = [None] * n
        else:
            if len(ref_joint_list) != n:
                return 0, []
            refs = ref_joint_list

        results = []
        all_ok = True
        for pose, ref in zip(quat_pose_list, refs):
            ret, jnt = self.inverse_kine_left(quat_pose=pose, ref_joint=ref, tool_index=0)
            if ret:
                results.append(jnt)
            else:
                all_ok = False
                results.append([])

        return (1 if all_ok else 0), results

    def inverse_kine_right_array(self, quat_pose_list: List[List[float]] = None,
                                 ref_joint_list: List[List[float]] = None,
                                 tool_index: int = 0):
        """右臂批量逆运动学求解，返回 (ret, jnt_value_list)"""
        if tool_index != 0:
            print(f"Warning: tool_index {tool_index} not supported, using flange (0).")
        if not quat_pose_list:
            return 0, []

        n = len(quat_pose_list)
        if ref_joint_list is None:
            refs = [None] * n
        else:
            if len(ref_joint_list) != n:
                return 0, []
            refs = ref_joint_list

        results = []
        all_ok = True
        for pose, ref in zip(quat_pose_list, refs):
            ret, jnt = self.inverse_kine_right(quat_pose=pose, ref_joint=ref, tool_index=0)
            if ret:
                results.append(jnt)
            else:
                all_ok = False
                results.append([])

        return (1 if all_ok else 0), results

    # ---------- 外部力矩 ----------

    def get_left_joint_external_tau(self) -> List[float]:
        """获取左臂关节空间外部力矩 (Nm)，7 个关节"""
        return self._get_joint_external_tau("left")

    def get_right_joint_external_tau(self) -> List[float]:
        """获取右臂关节空间外部力矩 (Nm)，7 个关节"""
        return self._get_joint_external_tau("right")

    def _get_arm_body_ids(self, arm: str) -> set:
        return self._left_arm_body_ids if arm == "left" else self._right_arm_body_ids

    def _get_joint_external_tau(self, arm: str) -> List[float]:
        """
        获取指定臂关节空间外部力矩。
        包含：接触力 + xfrc_applied + qfrc_applied
        排除：重力、限位力、自碰撞
        """
        nv = self.model.nv
        tau = np.zeros(nv)
        arm_body_ids = self._get_arm_body_ids(arm)

        # 1. 接触力
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]

            in_arm1 = body1 in arm_body_ids
            in_arm2 = body2 in arm_body_ids

            if (in_arm1 and in_arm2) or (not in_arm1 and not in_arm2):
                continue

            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)

            if in_arm2:
                arm_body = body2
            else:
                arm_body = body1
                force = -force

            arm_pos = self.data.xpos[arm_body]
            arm_mat = self.data.xmat[arm_body].reshape(3, 3)
            local_point = arm_mat.T @ (con.pos - arm_pos)

            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, local_point, arm_body)
            tau += jacp.T @ force[:3] + jacr.T @ force[3:]

        # 2. xfrc_applied
        for body_id in range(1, self.model.nbody):
            xfrc = self.data.xfrc_applied[body_id]
            if np.all(xfrc == 0):
                continue
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, body_id)
            tau += jacp.T @ xfrc[:3] + jacr.T @ xfrc[3:]

        # 3. qfrc_applied
        tau += self.data.qfrc_applied

        arm_slice = self._arm_slice(arm)
        return tau[arm_slice].tolist()

    def get_left_cart_external_force(self) -> List[float]:
        """获取左臂末端外力旋量 [Fx, Fy, Fz, Mx, My, Mz]"""
        return self._get_cart_external_wrench("left")

    def get_right_cart_external_force(self) -> List[float]:
        """获取右臂末端外力旋量 [Fx, Fy, Fz, Mx, My, Mz]"""
        return self._get_cart_external_wrench("right")

    def _get_cart_external_wrench(self, arm: str) -> List[float]:
        """
        获取指定臂末端外力旋量。
        包含：接触力 + xfrc_applied + qfrc_applied 的等效末端力
        """
        ee_body = self._link7_left_body if arm == "left" else self._link7_right_body
        mujoco.mj_kinematics(self.model, self.data)
        ee_pos = self.data.xpos[ee_body].copy()
        arm_body_ids = self._get_arm_body_ids(arm)
        wrench_local = np.zeros(6)

        # 1. 接触力
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]

            if body1 == 0 and body2 == 0:
                continue
            if body1 in arm_body_ids and body2 in arm_body_ids:
                continue

            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if body1 == 0:
                force = -force

            r = con.pos - ee_pos
            wrench_local[:3] += force[:3]
            wrench_local[3:] += force[3:] + np.cross(r, force[:3])

        # 2. xfrc_applied
        for body_id in range(1, self.model.nbody):
            xfrc = self.data.xfrc_applied[body_id]
            if np.all(xfrc == 0):
                continue
            r = self.data.xpos[body_id] - ee_pos
            wrench_local[:3] += xfrc[:3]
            wrench_local[3:] += xfrc[3:] + np.cross(r, xfrc[:3])

        # 3. qfrc_applied → 末端等效力
        arm_sl = self._arm_slice(arm)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, np.zeros(3), ee_body)
        J = np.vstack((jacp[:, arm_sl], jacr[:, arm_sl]))
        tau_applied = self.data.qfrc_applied[arm_sl]
        if np.any(tau_applied):
            J_inv_T = np.linalg.pinv(J).T
            wq = J_inv_T @ tau_applied
            wrench_local[:3] += wq[:3]
            wrench_local[3:] += wq[3:]

        return wrench_local.tolist()

    def get_left_cart_pose(self) -> List[float]:
        """获取左臂末端实际笛卡尔位姿 [x, y, z, qx, qy, qz, qw]"""
        return self._get_ee_pose("left")

    def get_right_cart_pose(self) -> List[float]:
        """获取右臂末端实际笛卡尔位姿 [x, y, z, qx, qy, qz, qw]"""
        return self._get_ee_pose("right")

    def _get_ee_pose(self, arm: str) -> List[float]:
        """获取指定臂末端相对基坐标系的位姿"""
        mujoco.mj_forward(self.model, self.data)

        root_pos = self.data.xpos[self._root_body_id].copy()

        if arm == "left":
            ee_body = self._link7_left_body
        else:
            ee_body = self._link7_right_body

        pos = self.data.xpos[ee_body].copy() - root_pos
        rot = self.data.xmat[ee_body].copy().reshape(3, 3)

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())
        return [pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]]

    # ---------- 逆运动学 (IK) ----------

    def _ik_solve(
        self,
        arm: str,
        pos_des: np.ndarray,
        quat_des: np.ndarray,
        ref_joint: np.ndarray = None,
        max_iter: int = 400,
        tol: float = 2e-2,
        damp: float = 0.1,
    ):
        """
        数值逆运动学求解（阻尼最小二乘法）

        Args:
            arm: "left" / "right"
            pos_des: 目标位置 (3,) 相对基坐标系
            quat_des: 目标姿态四元数 (4,) [qx, qy, qz, qw]
            ref_joint: 参考关节角 (7,)，用于多解选优
            max_iter: 最大迭代次数
            tol: 收敛容差
            damp: 阻尼系数

        Returns:
            (q_joint, success) — success=True 时 q_joint 为解 (7,)
        """
        n_dof = 7
        arm_slice = self._arm_slice(arm)
        q_backup = self.data.qpos[arm_slice].copy()
        root_pos = self.data.xpos[self._root_body_id].copy()

        # 目标姿态旋转矩阵
        R_des = np.zeros(9)
        mujoco.mju_quat2Mat(R_des, np.array(
            [quat_des[3], quat_des[0], quat_des[1], quat_des[2]]))
        R_des = R_des.reshape(3, 3)

        # 初始关节角
        if ref_joint is not None:
            q = np.array(ref_joint, dtype=float).copy()
        else:
            q = self.data.qpos[arm_slice].copy()

        # 关节限位
        jnt_start = arm_slice.start if arm_slice.start is not None else 0
        q_min = self.model.jnt_range[jnt_start:jnt_start + n_dof, 0].copy()
        q_max = self.model.jnt_range[jnt_start:jnt_start + n_dof, 1].copy()
        q = np.clip(q, q_min, q_max)

        # 末端 body
        ee_body = self._link7_left_body if arm == "left" else self._link7_right_body

        # 记录最优解
        best_error = float('inf')
        best_q = q.copy()

        for _ in range(max_iter):
            self.data.qpos[arm_slice] = q
            mujoco.mj_forward(self.model, self.data)

            pos_cur = self.data.xpos[ee_body].copy() - root_pos
            R_cur = self.data.xmat[ee_body].copy().reshape(3, 3)

            e_p = pos_des - pos_cur
            R_err = R_des @ R_cur.T
            quat_err = np.zeros(4)
            mujoco.mju_mat2Quat(quat_err, R_err.ravel())
            if np.linalg.norm(quat_err) > 1e-12:
                quat_err /= np.linalg.norm(quat_err)
            w_err = np.zeros(3)
            mujoco.mju_quat2Vel(w_err, quat_err, 1.0)

            error = np.hstack([e_p, w_err])
            err_norm = np.linalg.norm(error)

            # 记录最优解
            if err_norm < best_error:
                best_error = err_norm
                best_q = q.copy()

            if np.linalg.norm(e_p) < tol and np.linalg.norm(w_err) < tol:
                self.data.qpos[arm_slice] = q_backup
                return q, True

            # 雅可比
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, np.zeros(3), ee_body)
            J = np.vstack((jacp[:, arm_slice], jacr[:, arm_slice]))

            # 阻尼最小二乘法 (Levenberg-Marquardt)
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + damp**2 * np.eye(6), error)

            q = q + dq
            q = np.clip(q, q_min, q_max)

        self.data.qpos[arm_slice] = q_backup
        if best_error < tol * 2:
            return best_q, True
        return best_q, False

    # ================================================================
    # 点到点运动
    # ================================================================

    def move_left_joint(
        self,
        target_pos: List[float],
        desire_time: float = -1,
        is_sync: bool = True,
    ) -> int:
        """
        左臂关节空间点到点运动
        desire_time >  0 : 五次多项式
        desire_time == -1: 时间最优双 S 曲线
        """
        return self._move_joint("left", target_pos, desire_time, is_sync)

    def move_right_joint(
        self,
        target_pos: List[float],
        desire_time: float = -1,
        is_sync: bool = True,
    ) -> int:
        """
        右臂关节空间点到点运动
        desire_time >  0 : 五次多项式
        desire_time == -1: 时间最优双 S 曲线
        """
        return self._move_joint("right", target_pos, desire_time, is_sync)

    def _move_joint(
        self,
        arm: str,
        target_pos: List[float],
        desire_time: float = -1,
        is_sync: bool = True,
    ) -> int:
        if self.control_mode == 0:
            return 0
        if len(target_pos) != 7:
            return 0

        self._hold_target[arm] = None
        q_cur = self.data.qpos[self._arm_slice(arm)].copy()
        q_tar = np.array(target_pos, dtype=float)

        # ----- 五次多项式 (desire_time > 0) -----
        if desire_time > 0:
            self._traj_q0[arm] = q_cur
            self._traj_qf[arm] = q_tar
            self._traj_start_time[arm] = self.data.time
            self._traj_duration[arm] = float(desire_time)
            self._traj_active[arm] = True
            self._ds_active[arm] = False
            self._line_ds_active[arm] = False

            if is_sync:
                while self._traj_active[arm]:
                    self.step(5)
            return 1

        # ----- 双 S 曲线 (desire_time == -1) -----
        if desire_time == -1:
            self._traj_active[arm] = False
            self._line_ds_active[arm] = False
            self._ds_active[arm] = True
            self._ds_start_time[arm] = self.data.time
            self._ds_target[arm] = q_tar.copy()

            Tsync, params_list = self._sync_double_s_plan(
                q_cur, q_tar,
                self.joint_vel_limits,
                self.joint_acc_limits,
                self.joint_jerk_limits,
            )
            self._ds_duration[arm] = Tsync
            self._ds_params[arm] = params_list

            if is_sync:
                while self._ds_active[arm]:
                    self.step(5)
            return 1

        return 0

    # ================================================================
    # 直线运动 (标量 λ 双 S)
    # ================================================================

    def move_left_line_joint(self, target_pos: List[float], is_sync: bool = True) -> int:
        """左臂关节空间直线到点运动"""
        return self._move_line_joint("left", target_pos, is_sync)

    def move_right_line_joint(self, target_pos: List[float], is_sync: bool = True) -> int:
        """右臂关节空间直线到点运动"""
        return self._move_line_joint("right", target_pos, is_sync)

    def _move_line_joint(self, arm: str, target_pos: List[float], is_sync: bool = True) -> int:
        if self.control_mode == 0:
            return 0
        if len(target_pos) != 7:
            return 0

        q_cur = self.data.qpos[self._arm_slice(arm)].copy()
        q_tar = np.array(target_pos, dtype=float)
        dq = q_tar - q_cur

        if np.all(np.abs(dq) < 1e-12):
            return 1

        # 计算 λ 空间的合成约束
        v_lam = a_lam = j_lam = float('inf')
        for i in range(7):
            if abs(dq[i]) < 1e-12:
                continue
            v_lam = min(v_lam, self.joint_vel_limits[i] / abs(dq[i]))
            a_lam = min(a_lam, self.joint_acc_limits[i] / abs(dq[i]))
            j_lam = min(j_lam, self.joint_jerk_limits[i] / abs(dq[i]))

        if v_lam == float('inf'):
            return 0

        # 对 λ 做双 S 规划
        T_total, params = self._plan_double_s(D=1.0, v_max=v_lam, a_max=a_lam, j_max=j_lam)

        # 激活直线运动
        self._traj_active[arm] = False
        self._ds_active[arm] = False
        self._line_ds_active[arm] = True
        self._line_ds_start_time[arm] = self.data.time
        self._line_ds_duration[arm] = T_total
        self._line_ds_q0[arm] = q_cur
        self._line_ds_dq[arm] = dq
        self._line_ds_params[arm] = params
        self._hold_target[arm] = None

        if is_sync:
            while self._line_ds_active[arm]:
                self.step(5)
        return 1

    # ================================================================
    # 笛卡尔空间直线运动
    # ================================================================

    def move_left_line_pose(self, target_pos: List[float], is_sync: bool = True) -> int:
        """左臂笛卡尔空间直线到点运动"""
        return self._move_line_pose("left", target_pos, is_sync)

    def move_right_line_pose(self, target_pos: List[float], is_sync: bool = True) -> int:
        """右臂笛卡尔空间直线到点运动"""
        return self._move_line_pose("right", target_pos, is_sync)

    def _move_line_pose(self, arm: str, target_pos: List[float], is_sync: bool = True) -> int:
        if self.control_mode == 0:
            return 0
        if len(target_pos) != 7:
            return 0

        pos_des = np.array(target_pos[:3])
        quat_des = np.array(target_pos[3:])
        q_target, success = self._ik_solve(
            arm, pos_des, quat_des,
            ref_joint=np.array(
                self.get_left_joint_pos() if arm == "left" else self.get_right_joint_pos(),
                dtype=float))
        if not success:
            return 0

        return self._move_line_joint(arm, q_target.tolist(), is_sync)

    # ================================================================
    # 笛卡尔空间点到点运动
    # ================================================================

    def move_left_pose(self, target_pos: List[float],
                       desire_time: float = -1,
                       is_sync: bool = True) -> int:
        """左臂笛卡尔空间点到点运动"""
        return self._move_pose("left", target_pos, desire_time, is_sync)

    def move_right_pose(self, target_pos: List[float],
                        desire_time: float = -1,
                        is_sync: bool = True) -> int:
        """右臂笛卡尔空间点到点运动"""
        return self._move_pose("right", target_pos, desire_time, is_sync)

    def _move_pose(self, arm: str, target_pos: List[float],
                   desire_time: float = -1,
                   is_sync: bool = True) -> int:
        if self.control_mode == 0:
            return 0
        if len(target_pos) != 7:
            return 0

        pos_des = np.array(target_pos[:3])
        quat_des = np.array(target_pos[3:])
        q_target, success = self._ik_solve(
            arm, pos_des, quat_des,
            ref_joint=np.array(
                self.get_left_joint_pos() if arm == "left" else self.get_right_joint_pos(),
                dtype=float))
        if not success:
            return 0

        return self._move_joint(arm, q_target.tolist(), desire_time, is_sync)

    # ================================================================
    # 轨迹队列 (PT 运动)
    # ================================================================

    def move_left_joint_traj(self,
                             target_pos: List[List[float]],
                             gripper_pos: float = -1.0,
                             stamps: List[float] = None,
                             is_sync: bool = True) -> int:
        return self._move_joint_traj("left", target_pos, gripper_pos, stamps, is_sync)

    def move_right_joint_traj(self,
                              target_pos: List[List[float]],
                              gripper_pos: float = -1.0,
                              stamps: List[float] = None,
                              is_sync: bool = True) -> int:
        return self._move_joint_traj("right", target_pos, gripper_pos, stamps, is_sync)

    def _move_joint_traj(self, arm: str,
                         target_pos: List[List[float]],
                         gripper_pos: float = -1.0,
                         stamps: List[float] = None,
                         is_sync: bool = True) -> int:
        if self.control_mode == 0:
            return 0
        if len(target_pos) < 2:
            return 0
        for pt in target_pos:
            if len(pt) != 7:
                return 0

        queue = []
        if stamps and len(stamps) == len(target_pos):
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

        self._traj_queue[arm] = queue
        first_target, first_time, _ = queue.pop(0)
        if self._move_joint(arm, first_target, desire_time=first_time, is_sync=False) == 0:
            self._traj_queue[arm] = []
            return 0

        if is_sync:
            while (self._traj_queue[arm] or self._traj_active[arm] or
                   self._ds_active[arm] or self._line_ds_active[arm]):
                self.step(1)
        return 1

    def move_left_pose_traj(self,
                            target_pos: List[List[float]],
                            gripper_pos: float = -1.0,
                            stamps: List[float] = None,
                            is_sync: bool = True) -> int:
        return self._move_pose_traj("left", target_pos, gripper_pos, stamps, is_sync)

    def move_right_pose_traj(self,
                             target_pos: List[List[float]],
                             gripper_pos: float = -1.0,
                             stamps: List[float] = None,
                             is_sync: bool = True) -> int:
        return self._move_pose_traj("right", target_pos, gripper_pos, stamps, is_sync)

    def _move_pose_traj(self, arm: str,
                        target_pos: List[List[float]],
                        gripper_pos: float = -1.0,
                        stamps: List[float] = None,
                        is_sync: bool = True) -> int:
        if self.control_mode == 0:
            return 0
        if len(target_pos) < 2:
            return 0
        for pt in target_pos:
            if len(pt) != 7:
                return 0

        joint_targets = []
        for pose in target_pos:
            pos_des = np.array(pose[:3])
            quat_des = np.array(pose[3:])
            q_sol, success = self._ik_solve(
                arm, pos_des, quat_des,
                ref_joint=np.array(
                    self.get_left_joint_pos() if arm == "left" else self.get_right_joint_pos(),
                    dtype=float))
            if not success:
                return 0
            joint_targets.append(q_sol.tolist())

        return self._move_joint_traj(arm, joint_targets,
                                     gripper_pos=gripper_pos,
                                     stamps=stamps,
                                     is_sync=is_sync)

    # ================================================================

    def _update_traj_control(self):
        """更新所有臂的轨迹控制"""
        for arm in ("left", "right"):
            self._update_single_arm_traj(arm)

    def _update_single_arm_traj(self, arm: str):
        """更新单臂轨迹控制"""
        arm_slice = self._arm_slice(arm)
        n_dof = 7

        # MODE 3: 拖动模式
        if self.control_mode == 3:
            v_arm = self.data.qvel[arm_slice].copy()
            friction_est = -(self._damping[arm] * v_arm
                             + self._frictionloss[arm] * np.tanh(v_arm))
            tau_arm = self.data.qfrc_bias[arm_slice] + friction_est
            self._set_arm_ctrl(arm, tau_arm)
            return

        # 直线运动（标量 λ 双 S）
        if self._line_ds_active[arm]:
            t = self.data.time - self._line_ds_start_time[arm]
            if t < 0:
                return
            if t >= self._line_ds_duration[arm]:
                self._line_ds_active[arm] = False
                self._hold_target[arm] = (self._line_ds_q0[arm]
                                          + self._line_ds_dq[arm]).copy()
                self._apply_control(arm, self._hold_target[arm])
                self._try_start_next_traj(arm)
                return

            s, ds, _ = self._double_s_interp(t, self._line_ds_params[arm])
            q_des = self._line_ds_q0[arm] + s * self._line_ds_dq[arm]
            v_des = ds * self._line_ds_dq[arm]
            self._apply_control(arm, q_des, v_des)
            return

        # 双 S 轨迹
        if self._ds_active[arm]:
            t = self.data.time - self._ds_start_time[arm]
            if t < 0:
                return
            if t >= self._ds_duration[arm]:
                self._ds_active[arm] = False
                self._hold_target[arm] = self._ds_target[arm].copy()
                self._apply_control(arm, self._hold_target[arm])
                self._try_start_next_traj(arm)
                return

            q_des = np.zeros(n_dof)
            v_des = np.zeros(n_dof)
            for j in range(n_dof):
                par = self._ds_params[arm][j]
                if par.get('is_static', False):
                    q_des[j] = par['q0']
                    v_des[j] = 0.0
                else:
                    q_rel, v_rel, _ = self._double_s_interp_7phase(t, par)
                    q_des[j] = par['q0'] + q_rel
                    v_des[j] = v_rel
            self._apply_control(arm, q_des, v_des)
            return

        # 五次多项式轨迹
        if self._traj_active[arm]:
            t = self.data.time - self._traj_start_time[arm]
            if t >= self._traj_duration[arm]:
                q_des = self._traj_qf[arm].copy()
                v_des = np.zeros(n_dof)
                self._traj_active[arm] = False
                self._hold_target[arm] = self._traj_qf[arm].copy()
                self._try_start_next_traj(arm)
            else:
                q_des = np.zeros(n_dof)
                v_des = np.zeros(n_dof)
                for i in range(n_dof):
                    q_des[i], v_des[i], _ = self._quintic_interp(
                        self._traj_q0[arm][i], self._traj_qf[arm][i],
                        self._traj_duration[arm], t)
            self._apply_control(arm, q_des, v_des)
            return

        # ========== 保持目标 ==========
        if self._hold_target[arm] is not None:
            self._apply_control(arm, self._hold_target[arm])

    def _apply_control(self, arm: str, q_des: np.ndarray,
                       v_des: np.ndarray = None, gripper_pos: float = -1.0):
        if v_des is None:
            v_des = np.zeros(7)
        tau_arm = self._compute_tau(arm, q_des, v_des)
        self._set_arm_ctrl(arm, tau_arm)
        if gripper_pos >= 0:
            self._gripper_target[arm] = gripper_pos
            self._gripper_pending[arm] = None
        self._set_gripper_ctrl(arm, self._gripper_target[arm])
        self._plan_q[arm] = q_des.copy()
        self._plan_v[arm] = v_des.copy()
        self._plan_tau[arm] = tau_arm.copy()

    def _try_start_next_traj(self, arm: str):
        """若队列非空，取出下一段并调用 move_joint"""
        if not self._traj_queue[arm]:
            return
        next_target, next_time, grip = self._traj_queue[arm].pop(0)
        if grip >= 0:
            self._gripper_target[arm] = grip
            self._gripper_pending[arm] = None
        self._move_joint(arm, next_target, desire_time=next_time, is_sync=False)

    # ================================================================
    # 五次多项式插值
    # ================================================================

    @staticmethod
    def _quintic_interp(q0, qf, T, t):
        """五次多项式插值，返回 (q, v, a)"""
        if T <= 0:
            raise ValueError("T must be positive")
        s = np.clip(t / T, 0.0, 1.0)
        h = 10 * s**3 - 15 * s**4 + 6 * s**5
        dh = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        ddh = (60 * s - 180 * s**2 + 120 * s**3) / (T**2)
        dq = qf - q0
        return q0 + dq * h, dq * dh, dq * ddh

    # ================================================================
    # 双 S 曲线规划与插值
    # ================================================================

    @staticmethod
    def _plan_double_s(D, v_max, a_max, j_max):
        """对称双 S 曲线规划"""
        if abs(D) < 1e-12:
            return 0.0, {
                'q0': 0.0, 'qf': 0.0, 'sign': 1, 'T': 0.0,
                't_j1': 0.0, 't_j2': 0.0, 't_v': 0.0,
                'v_lim': 0.0, 'a_lim': 0.0, 'j_max': j_max
            }
        sign = 1.0 if D > 0 else -1.0
        D_abs = abs(D)
        v_a = a_max**2 / j_max

        if v_max <= v_a:
            t_j1 = np.sqrt(v_max / j_max)
            t_j2 = 0.0
            a_lim = np.sqrt(v_max * j_max)
            v_lim = v_max
            s_a = t_j1 * v_lim
            D_req = 2.0 * s_a
            if D_abs >= D_req:
                t_v = (D_abs - D_req) / v_lim
            else:
                v_lim = (D_abs * np.sqrt(j_max) / 2.0) ** (2.0 / 3.0)
                t_j1 = np.sqrt(v_lim / j_max)
                a_lim = np.sqrt(v_lim * j_max)
                t_v = 0.0
        else:
            t_j1 = a_max / j_max
            a_lim = a_max
            t_j2_vmax = max((v_max - v_a) / a_max, 0.0)
            s_a_vmax = (2 * t_j1 + t_j2_vmax) * v_max / 2.0
            D_req = 2.0 * s_a_vmax
            if D_abs >= D_req:
                t_j2 = t_j2_vmax
                v_lim = v_max
                t_v = (D_abs - D_req) / v_max
            else:
                A = a_max
                B = 3.0 * a_max**2 / j_max
                C = 2.0 * a_max**3 / j_max**2 - D_abs
                disc = B**2 - 4.0 * A * C
                if disc < 0:
                    disc = 0.0
                t_j2 = max((-B + np.sqrt(disc)) / (2.0 * A), 0.0)
                v_lim = v_a + a_max * t_j2
                t_v = 0.0

        T_acc = 2.0 * t_j1 + t_j2
        T_total = 2.0 * T_acc + t_v
        return T_total, {
            'q0': 0.0, 'qf': D, 'sign': sign,
            'T': T_total, 't_j1': t_j1, 't_j2': t_j2,
            't_v': t_v, 'v_lim': v_lim, 'a_lim': a_lim, 'j_max': j_max
        }

    @staticmethod
    def _double_s_interp(t, params):
        """双 S 曲线插值，返回 (q, v, a)"""
        T = params['T']
        if T < 1e-12:
            return params['qf'], 0.0, 0.0

        sgn = params['sign']
        t1 = params['t_j1']; t2 = params['t_j2']
        tv = params['t_v']; vl = params['v_lim']
        al = params['a_lim']; jm = params['j_max']
        Tac = 2.0 * t1 + t2
        sac = (Tac * vl) / 2.0

        if t <= Tac:
            if t <= t1:
                a = jm * t
                v = 0.5 * jm * t**2
                s = jm * t**3 / 6.0
            elif t <= t1 + t2:
                dt = t - t1
                v1 = 0.5 * jm * t1**2
                s1 = jm * t1**3 / 6.0
                a = al
                v = v1 + al * dt
                s = s1 + v1 * dt + 0.5 * al * dt**2
            else:
                dt = t - (t1 + t2)
                v2 = 0.5 * jm * t1**2 + al * t2
                s2 = (jm * t1**3 / 6.0) + (0.5 * jm * t1**2) * t2 + 0.5 * al * t2**2
                a = al - jm * dt
                v = v2 + al * dt - 0.5 * jm * dt**2
                s = s2 + v2 * dt + 0.5 * al * dt**2 - jm * dt**3 / 6.0
        elif t <= Tac + tv:
            dt = t - Tac
            s = sac + vl * dt
            v = vl
            a = 0.0
        else:
            td = t - (Tac + tv)
            tr = Tac - td
            if tr <= t1:
                a_acc = jm * tr
                v_acc = 0.5 * jm * tr**2
                s_acc_rev = jm * tr**3 / 6.0
            elif tr <= t1 + t2:
                dt = tr - t1
                v1 = 0.5 * jm * t1**2
                s1 = jm * t1**3 / 6.0
                a_acc = al
                v_acc = v1 + al * dt
                s_acc_rev = s1 + v1 * dt + 0.5 * al * dt**2
            else:
                dt = tr - (t1 + t2)
                v2 = 0.5 * jm * t1**2 + al * t2
                s2 = (jm * t1**3 / 6.0) + (0.5 * jm * t1**2) * t2 + 0.5 * al * t2**2
                a_acc = al - jm * dt
                v_acc = v2 + al * dt - 0.5 * jm * dt**2
                s_acc_rev = s2 + v2 * dt + 0.5 * al * dt**2 - jm * dt**3 / 6.0
            v = v_acc
            s = sac + vl * tv + (sac - s_acc_rev)
            a = -a_acc

        return s * sgn, v * sgn, a * sgn

    def _sync_double_s_plan(self, q0, qf, vmax, amax, jmax):
        """多轴同步 Double-S 轨迹规划"""
        n = len(q0)
        D = qf - q0
        zm = np.abs(D) < 1e-12

        Ti = np.zeros(n)
        for i in range(n):
            Ti[i] = 0.0 if zm[i] else self._min_time_double_s(
                abs(D[i]), vmax[i], amax[i], jmax[i])
        Ts = max(np.max(Ti), 1e-8)

        pl = []
        for i in range(n):
            if zm[i]:
                pl.append({
                    'q0': q0[i], 'qf': qf[i], 'sign': 1.0,
                    'v_peak': 0.0, 'a_peak': 0.0, 'j_used': jmax[i],
                    't1': 0, 't2': 0, 't3': 0, 't4': Ts,
                    't5': Ts, 't6': Ts, 't7': Ts,
                    'Tsync': Ts, 'is_static': True
                })
            else:
                par = self._sync_joint_double_s(
                    abs(D[i]), vmax[i], amax[i], jmax[i], Ts, D[i])
                if par is None:
                    tj = np.cbrt(abs(D[i]) / (2.0 * jmax[i]))
                    Ti_i = 4.0 * tj
                    if Ti_i < Ts:
                        scale = Ts / Ti_i
                        tj *= scale
                    vp = jmax[i] * tj**2
                    ap = jmax[i] * tj
                    t1 = tj; t2 = tj; t3 = 2*tj; t4 = 2*tj
                    t5 = 3*tj; t6 = 3*tj; t7 = 4*tj
                    par = {
                        'sign': 1.0 if D[i] > 0 else -1.0,
                        'v_peak': vp, 'a_peak': ap, 'j_used': jmax[i],
                        't1': t1, 't2': t2, 't3': t3, 't4': t4,
                        't5': t5, 't6': t6, 't7': t7,
                        'Tsync': t7, 'is_static': False
                    }
                par['q0'] = q0[i]
                par['qf'] = qf[i]
                par['Tsync'] = Ts
                par['is_static'] = False
                pl.append(par)
        return Ts, pl

    @staticmethod
    def _min_time_double_s(Da, vm, am, jm):
        Da_am = 2.0 * am**3 / jm**2
        Dv_vm = vm * (vm / am + am / jm)
        if Da >= Dv_vm:
            return Da / vm + vm / am + am / jm
        if Da >= Da_am:
            coeff = [1.0 / am, am / jm, -Da]
            roots = np.roots(coeff)
            vp = float(np.max(roots[np.isreal(roots)]).real)
            return 2.0 * (vp / am + am / jm)
        return 4.0 * np.cbrt(Da / (2.0 * jm))

    @staticmethod
    def _sync_joint_double_s(Da, vm, am, jm, Ts, Dt):
        if Da < 1e-12 or Ts < 1e-12:
            return None

        def solve_ap(vp):
            ap = am
            if ap <= 0:
                ap = 0.1 * jm * (Ts / 2.0)
            for _ in range(50):
                if ap <= 0:
                    return None
                f = (vp * Ts - vp**2 / ap - ap * vp / jm +
                     ap**3 / (3 * jm**2) - Da)
                if abs(f) < 1e-12:
                    break
                df = vp**2 / ap**2 - vp / jm + ap**2 / jm**2
                if abs(df) < 1e-12:
                    return None
                delta = f / df
                if abs(delta) > 0.5 * am:
                    delta = 0.5 * am * np.sign(delta)
                ap -= delta
                if ap <= 0:
                    ap = 1e-6
                if ap > am:
                    ap = am
            if 2.0 * (vp / ap + ap / jm) > Ts + 1e-12 or ap > am + 1e-12:
                return None
            return ap

        vl, vh = 0.0, vm
        vp = 0.0
        ap = None
        for _ in range(60):
            vm_ = (vl + vh) / 2.0
            a = solve_ap(vm_)
            if a is not None:
                vp = vm_
                ap = a
                vl = vm_
            else:
                vh = vm_
            if vh - vl < 1e-8 * vm:
                break

        if ap is None:
            return None

        Tj1 = ap / jm
        Tj2 = max((vp - ap**2 / jm) / ap, 0.0) if ap > 0 else 0.0
        Tac = Tj1 + Tj2 + Tj1
        sa = vp**2 / (2 * ap) + (vp * ap) / (2 * jm) - ap**3 / (6 * jm**2)
        Tc = max((Da - 2 * sa) / vp, 0.0) if vp > 0 else 0.0

        t1 = Tj1; t2 = t1 + Tj2; t3 = t2 + Tj1
        t4 = t3 + Tc; t5 = t4 + Tj1; t6 = t5 + Tj2; t7 = t6 + Tj1
        return {
            'sign': 1.0 if Dt > 0 else -1.0,
            'v_peak': vp, 'a_peak': ap, 'j_used': jm,
            't1': t1, 't2': t2, 't3': t3, 't4': t4,
            't5': t5, 't6': t6, 't7': t7
        }

    @staticmethod
    def _double_s_interp_7phase(t, par):
        """七段式 Double-S 插值，返回 (q_rel, v, a)"""
        if par.get('is_static', False) or par['Tsync'] < 1e-12:
            return 0.0, 0.0, 0.0
        if t >= par['Tsync'] - 1e-12:
            return par['qf'] - par['q0'], 0.0, 0.0

        sgn = par['sign']
        vp = par['v_peak']; ap = par['a_peak']; jm = par['j_used']
        t1 = par['t1']; t2 = par['t2']; t3 = par['t3']
        t4 = par['t4']; t5 = par['t5']; t6 = par['t6']; t7 = par['t7']

        def p1(tau): return (jm * tau**3 / 6, 0.5 * jm * tau**2, jm * tau)
        def p2(tau):
            dt = tau - t1; s, v, _ = p1(t1)
            return (s + v * dt + 0.5 * ap * dt**2, v + ap * dt, ap)
        def p3(tau):
            dt = tau - t2; s, v, _ = p2(t2)
            a_ = ap - jm * dt
            return (s + v * dt + 0.5 * ap * dt**2 - jm * dt**3 / 6,
                    v + ap * dt - 0.5 * jm * dt**2, a_)
        def p4(tau):
            dt = tau - t3; s, v, _ = p3(t3)
            return (s + vp * dt, vp, 0.0)
        def p5(tau):
            dt = tau - t4; s, v, _ = p4(t4)
            a_ = -jm * dt
            return (s + vp * dt - jm * dt**3 / 6,
                    vp - 0.5 * jm * dt**2, a_)
        def p6(tau):
            dt = tau - t5; s, v, _ = p5(t5)
            return (s + v * dt - 0.5 * ap * dt**2,
                    v - ap * dt, -ap)
        def p7(tau):
            dt = tau - t6; s, v, _ = p6(t6)
            a_ = -ap + jm * dt
            return (s + v * dt - 0.5 * ap * dt**2 + jm * dt**3 / 6,
                    v - ap * dt + 0.5 * jm * dt**2, a_)

        if t <= t1:          s, v, a = p1(t)
        elif t <= t2:        s, v, a = p2(t)
        elif t <= t3:        s, v, a = p3(t)
        elif t <= t4:        s, v, a = p4(t)
        elif t <= t5:        s, v, a = p5(t)
        elif t <= t6:        s, v, a = p6(t)
        else:                s, v, a = p7(t)

        return s * sgn, v * sgn, a * sgn

    # ================================================================
    # 示教录制与轨迹回放
    # ================================================================

    def trajectory_teach_left(self, off_on: bool, name: str) -> int:
        """左臂开始/停止示教录制"""
        if off_on:
            if self._recording["left"]:
                return 0
            if not name:
                return 0
            self._prev_control_mode["left"] = self.control_mode
            if self.control_mode != 3:
                self.set_control_mode(3)
            self._recording["left"] = True
            self._rec_name["left"] = name
            self._rec_start_time["left"] = self.data.time
            self._rec_joint_pos["left"] = []
            self._rec_time_stamps["left"] = []
            return 1
        else:
            if not self._recording["left"]:
                return 0
            if len(self._rec_joint_pos["left"]) < 2:
                self._recording["left"] = False
                return 0
            key = name + "_left"
            self._teach_data[key] = {
                'joint_pos': [pos.tolist() for pos in self._rec_joint_pos["left"]],
                'time_stamps': self._rec_time_stamps["left"].copy()
            }
            self._save_teach_json(key)
            self._recording["left"] = False
            self.set_control_mode(self._prev_control_mode["left"])
            return 1

    def trajectory_teach_right(self, off_on: bool, name: str) -> int:
        """右臂开始/停止示教录制"""
        if off_on:
            if self._recording["right"]:
                return 0
            if not name:
                return 0
            self._prev_control_mode["right"] = self.control_mode
            if self.control_mode != 3:
                self.set_control_mode(3)
            self._recording["right"] = True
            self._rec_name["right"] = name
            self._rec_start_time["right"] = self.data.time
            self._rec_joint_pos["right"] = []
            self._rec_time_stamps["right"] = []
            return 1
        else:
            if not self._recording["right"]:
                return 0
            if len(self._rec_joint_pos["right"]) < 2:
                self._recording["right"] = False
                return 0
            key = name + "_right"
            self._teach_data[key] = {
                'joint_pos': [pos.tolist() for pos in self._rec_joint_pos["right"]],
                'time_stamps': self._rec_time_stamps["right"].copy()
            }
            self._save_teach_json(key)
            self._recording["right"] = False
            self.set_control_mode(self._prev_control_mode["right"])
            return 1

    def trajectory_recorder_left(self, name: str, is_sync: bool = True) -> int:
        """左臂复现指定名称的示教轨迹"""
        return self._trajectory_recorder("left", name + "_left", is_sync)

    def trajectory_recorder_right(self, name: str, is_sync: bool = True) -> int:
        """右臂复现指定名称的示教轨迹"""
        return self._trajectory_recorder("right", name + "_right", is_sync)

    def _trajectory_recorder(self, arm: str, key: str, is_sync: bool = True) -> int:
        if key not in self._teach_data and self._import_teach_from_json(key) == 0:
            return 0

        data = self._teach_data[key]
        joint_pos = np.array(data['joint_pos'])
        stamps = data.get('time_stamps', None)

        if len(joint_pos) < 2:
            return 0

        if stamps is None or len(stamps) == 0:
            stamps = np.arange(len(joint_pos)) * self.model.opt.timestep

        stamps = np.array(stamps)
        if stamps[0] != 0.0:
            stamps = stamps - stamps[0]
        total_duration = stamps[-1]
        if total_duration <= 0:
            return 0

        if self.control_mode == 0:
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
                t0, t1 = stamps[idx], stamps[idx + 1]
                if t1 - t0 < 1e-12:
                    q_des = joint_pos[idx]
                else:
                    alpha = (t - t0) / (t1 - t0)
                    q_des = joint_pos[idx] + alpha * (joint_pos[idx + 1] - joint_pos[idx])
            if step_count % 100 == 0:
                print(f"[Recorder {arm}] t={t:.3f}/{total_duration:.3f}")
            self.track_left_joint(q_des.tolist(), gripper_pos=-1.0) if arm == "left" else \
                self.track_right_joint(q_des.tolist(), gripper_pos=-1.0)
            self.step(1)
            step_count += 1

        last_joint = q_des.tolist()
        self.track_left_joint(last_joint, gripper_pos=-1.0) if arm == "left" else \
            self.track_right_joint(last_joint, gripper_pos=-1.0)
        for _ in range(5):
            self.step(1)
        return 1

    def check_teach(self, left_traj_list: List[str], right_traj_list: List[str]) -> int:
        """获取已记录的示教轨迹列表"""
        for k in self._teach_data:
            if k.endswith("_left"):
                left_traj_list.append(k[:-5])
            elif k.endswith("_right"):
                right_traj_list.append(k[:-6])
        return 1

    def _save_teach_json(self, key: str) -> int:
        if key not in self._teach_data:
            return 0
        data = self._teach_data[key]
        export = {
            "name": key,
            "joint_pos": data['joint_pos'],
            "time_stamps": data['time_stamps'],
            "point_size": len(data['joint_pos'])
        }
        fpath = os.path.join(self._export_dir, f"{key}.json")
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=3, ensure_ascii=False)
        return 1

    def _import_teach_from_json(self, key: str) -> int:
        fpath = os.path.join(self._export_dir, f"{key}.json")
        if not os.path.exists(fpath):
            return 0
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return 0

        jp = data.get('joint_pos') or data.get('path', {}).get('joint_pos')
        if not jp or len(jp) < 2:
            return 0
        ts = data.get('time_stamps')
        if ts is not None and len(ts) == 0:
            ts = None
        self._teach_data[key] = {'joint_pos': jp, 'time_stamps': ts}
        return 1
