import numpy as np
import mujoco
from carm_mujoco import CArmDualBot

# 初始化机械臂
arm = CArmDualBot("models/carm_d3_mjcf/carm_d3.xml", render=True)
arm.set_control_mode(2)  # 位置控制模式

# 获取左臂相关ID
link7_left = arm._link7_left_body
root_id = arm._root_body_id
arm_slice = arm.LEFT_ARM_SLICE

# 1. 运动到初始关节位置
target_joint = [np.pi/6, np.pi/4, 0.0, -np.pi/3, 0.0, 0.0, 0.0]
print("Moving to target joint position...")
arm.move_left_joint(target_joint, is_sync=True)

# 2. 记录当前末端位姿作为控制目标（固定）
# get_left_cart_pose 返回 [x, y, z, qx, qy, qz, qw]，转为 [x, y, z, qw, qx, qy, qz]
raw_pose = arm.get_left_cart_pose()
pose_des = np.zeros(7)
pose_des[:3] = raw_pose[:3]
pose_des[3] = raw_pose[6]   # qw 
pose_des[4] = raw_pose[3]   # qx
pose_des[5] = raw_pose[4]   # qy
pose_des[6] = raw_pose[5]   # qz

# 3. 初始关节位置（作为零空间积分基准）
q_init = np.array(arm.get_left_joint_pos())

# 4. 积分变量：零空间累积偏移量
q_null_integral = np.zeros(7)

# 5. 控制参数
kp_pos = 1.0      # 位置闭环增益
kp_ori = 0.5      # 姿态闭环增益（简化）
null_gain = 1   # 零空间运动速度增益
svd_rcond = 1e-4  # SVD 伪逆奇异值截断阈值
dt = arm.model.opt.timestep  # 仿真步长

# SVD 伪逆求解
def svd_pinv(A, rcond=1e-4):
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    cutoff = rcond * S[0] if S.size > 0 else 0.0
    S_inv = np.zeros_like(S)
    valid = S > cutoff
    S_inv[valid] = 1.0 / S[valid]
    return Vh.T @ np.diag(S_inv) @ U.T

# 姿态误差辅助函数（四元数误差转轴角）
def quat_error(q_des, q_curr):
    """
    计算期望四元数 q_des 与当前四元数 q_curr 之间的误差，
    返回轴角向量（3维），用于角速度反馈。
    四元数格式为 [w, x, y, z]
    """
    # 共轭
    q_curr_conj = np.array([q_curr[0], -q_curr[1], -q_curr[2], -q_curr[3]])
    # 相对四元数：q_des * conj(q_curr)
    q_rel = np.zeros(4)
    q_rel[0] = q_des[0]*q_curr_conj[0] - q_des[1]*q_curr_conj[1] - q_des[2]*q_curr_conj[2] - q_des[3]*q_curr_conj[3]
    q_rel[1] = q_des[0]*q_curr_conj[1] + q_des[1]*q_curr_conj[0] + q_des[2]*q_curr_conj[3] - q_des[3]*q_curr_conj[2]
    q_rel[2] = q_des[0]*q_curr_conj[2] - q_des[1]*q_curr_conj[3] + q_des[2]*q_curr_conj[0] + q_des[3]*q_curr_conj[1]
    q_rel[3] = q_des[0]*q_curr_conj[3] + q_des[1]*q_curr_conj[2] - q_des[2]*q_curr_conj[1] + q_des[3]*q_curr_conj[0]
    # 取虚部作为轴角（小角度近似）
    # 注意：q_rel[0] 应接近1，否则需标准化
    if q_rel[0] < 0:
        q_rel = -q_rel
    # 轴角 = 2 * 虚部 / 实部（简化，小角度直接取虚部*2）
    # 更准确：角度 = 2*acos(real)，轴 = 虚部/sin(angle/2)
    sin_half = np.linalg.norm(q_rel[1:])
    if sin_half > 1e-6:
        angle = 2 * np.arctan2(sin_half, q_rel[0])
        axis = q_rel[1:] / sin_half
        return angle * axis
    else:
        return np.zeros(3)

# 主循环
prev_direction = None  # 用于保持零空间方向一致性
while arm.viewer.is_running():

    # ---- 获取当前位姿 ----
    raw_curr = arm.get_left_cart_pose()     # [x, y, z, qx, qy, qz, qw]
    pos_curr = raw_curr[:3]
    quat_curr = np.array([raw_curr[6], raw_curr[3], raw_curr[4], raw_curr[5]])  # [qw, qx, qy, qz]

    # ---- 计算末端误差 ----
    pos_err = pose_des[:3] - pos_curr
    ori_err = quat_error(pose_des[3:], quat_curr)

    # ---- 计算雅可比矩阵 ----
    jacp = np.zeros((3, arm.model.nv))
    jacr = np.zeros((3, arm.model.nv))
    # 零向量作为点（相对于link7局部坐标原点）
    mujoco.mj_jac(arm.model, arm.data, jacp, jacr, np.zeros(3), link7_left)
    J = np.vstack((jacp[:, arm_slice], jacr[:, arm_slice]))  # (6, 7)

    # ---- 高优先级任务：跟踪固定末端位姿（带反馈） ----
    # 期望末端速度：位置和角速度均用比例反馈
    v_des = np.zeros(6)
    v_des[:3] = kp_pos * pos_err
    v_des[3:] = kp_ori * ori_err

    # SVD 伪逆: J+ = V Σ+ Uᵀ
    J_pinv = svd_pinv(J, svd_rcond)
    q_dot_task = J_pinv @ v_des

    # ---- 零空间投影矩阵 ----
    N = np.eye(7) - J_pinv @ J

    # ---- 沿零空间基向量持续运动 ----
    # 计算零空间的正交基（通过SVD）
    U, S, Vh = np.linalg.svd(N, full_matrices=True)
    # N 的特征值 ≈1 对应 J 的零空间，≈0 对应任务空间
    # S 从大到小排列，取 S >= 0.5 的右奇异向量（零空间基）
    null_dim = np.sum(S >= 0.5)  # 零空间维度（通常为 1）
    print(f"Null-space dimension: {null_dim}")
    null_gain = 2*np.sin(arm.data.time) 
    if null_dim > 0:
        # 取前 null_dim 个右奇异向量（对应最大的奇异值 ≈ J 的零空间）
        null_basis = Vh[:null_dim].T  # (7, null_dim)
        # 选择第一个方向
        direction = null_basis[:, 0]
        if prev_direction is not None:
            if np.dot(direction, prev_direction) < 0:
                direction = -direction

        prev_direction = direction.copy()
        # 持续沿该方向积分
        q_dot_null = null_gain * direction
    else:
        # 若零空间维度为0，则无法进行零空间运动
        q_dot_null = np.zeros(7)
        print("Warning: No null-space available for motion.")

    # 将零空间速度投影（保证纯零空间运动，但基向量已经在零空间内，无需再投影）
    # 但为保证精确，可以再投影一次
    q_dot_null_proj = N @ q_dot_null
    print(f"q_dot_null before projection: {np.round(q_dot_null, 3)}")
    print(f"q_dot_null after projection: {np.round(q_dot_null_proj, 3)}")
    # 积分零空间速度
    q_null_integral += q_dot_null_proj * dt

    # 总期望关节位置 = 初始位置 + 零空间累积偏移
    q_des = q_init + q_null_integral


    # 发送控制命令
    arm.track_left_joint(q_des.tolist())

    # 可选：打印末端位姿误差和关节位置
    print(f"Pos err: {np.linalg.norm(pos_err):.4f}, Ori err: {np.linalg.norm(ori_err):.4f}")
    # print(f"Joint: {np.round(q_des, 3)}")

    # 步进仿真
    arm.step()

# 关闭窗口后退出
arm.close()
