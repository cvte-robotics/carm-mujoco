#!/usr/bin/env python3
"""
carm 仿真交互式测试程序（对应 C++ test_sdk）
用法: python test_carm_sim.py <MJCF_XML_PATH> [--render]
"""

import sys
import os
import signal
import time
import json
from typing import Dict, Callable
import numpy as np
from carm_mujoco import CArmSingleCol

# ---------- 导入 CArmSingleCol ----------
# 若类定义在单独文件 carm_sim.py 中，请取消下面注释
# from carm_sim import CArmSingleCol
# 为方便展示，此处直接粘贴类定义（实际使用时请改为导入）
# 此处省略类定义，假设已正确导入
# ---------- 以下为交互程序 ----------

# 全局变量
carm = None
error_flag = False
ctrl_c_flag = False
speed_level = 5          # 默认速度等级，影响 desire_time 的计算（0 最快，10 最慢）
tool_index = 0           # 当前工具号（仅0有效）

# -----------------------------------------------------------
# 基础状态操作
# -----------------------------------------------------------
def set_control_mode():
    """设置控制模式"""
    try:
        mode = int(input("请输入mode (0-4): 0 空闲 1 位置 2 MIT 3 拖动 4 力矩限制: "))
        if 0 <= mode <= 4:
            carm.set_control_mode(mode)
            print(f"控制模式已切换为 {mode}")
        else:
            print("无效模式")
    except ValueError:
        print("输入格式错误")

def print_state():
    """打印当前状态"""
    joint_pos = carm.get_joint_pos()
    cart_pos = carm.get_cart_pose()
    print(f"关节角度 (rad): {[f'{q:.4f}' for q in joint_pos]}")
    print(f"末端位姿 (xyz+q): {[f'{v:.4f}' for v in cart_pos]}")
    print(f"夹爪间距: {carm.get_gripper_pos():.4f} m, "
          f"夹爪力: {carm.get_gripper_tau():.2f} N")

# -----------------------------------------------------------
# 运动指令
# -----------------------------------------------------------
def move_to_home():
    """回零位"""
    home = [0.0]*6
    carm.move_joint(home, desire_time=-1)
        # carm.step()
    print("已回到零位")

def move_to_test_joint():
    """测试关节移动"""
    target = [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0]
    carm.move_joint(target, desire_time=-1)
    # carm.step()
    print("move_joint 完成")

def move_joint_with_time():
    """带时间的关节移动"""
    target = [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0]
    t = 1.5
    carm.move_joint(target, desire_time=t, is_sync=True)
    # carm.step()
    print(f"move_joint (time={t}s) 完成")

def move_to_test_pose():
    """测试笛卡尔位姿移动"""
    pose = [-0.010669, 0.000073, 0.415677, 0.733466, 0.000187, 0.679726, 0.000029]
    carm.move_pose(pose, desire_time=-1, is_sync=True)
    # carm.step()
    print("move_pose 完成")

def move_pose_with_time():
    """带时间的笛卡尔移动"""
    pose = [-0.010669, 0.000073, 0.415677, 0.733466, 0.000187, 0.679726, 0.000029]
    t = 1.5
    carm.move_pose(pose, desire_time=t, is_sync=True)
    # carm.step()
    print(f"move_pose (time={t}s) 完成")

def move_line_joint():
    """关节空间直线运动"""
    target = [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0]
    carm.move_line_joint(target, is_sync=True)
    # carm.step()
    print("move_line_joint 完成")

def move_line_pose():
    """笛卡尔直线运动"""
    pose = [-0.010669, 0.000073, 0.415677, 0.733466, 0.000187, 0.679726, 0.000029]
    carm.move_line_pose(pose, is_sync=True)
    # carm.step()
    print("move_line_pose 完成")

def cycle_test():
    """循环测试：在两个点与零位之间往复运动"""
    global ctrl_c_flag, error_flag
    ctrl_c_flag = False
    error_flag = False
    try:
        times = int(input("循环次数: "))
    except:
        print("无效输入")
        return

    joint1 = [0.3, 0.361361, -0.646622, 0.0, 0.361361, 0.0]
    joint2 = [-0.3, 0.361361, -0.646622, 0.0, 0.361361, 0.0]
    home = [0.0]*6

    for i in range(times):
        if error_flag or ctrl_c_flag:
            break
        carm.move_joint(joint2, desire_time=-1, is_sync=True)
        carm.move_joint(home, desire_time=-1, is_sync=True)
        carm.move_joint(joint1, desire_time=-1, is_sync=True)
        carm.move_joint(home, desire_time=-1, is_sync=True)
        print(f"循环进度: {i+1}/{times}")

    print("循环测试结束" if not ctrl_c_flag else "循环被用户中断")

def set_end_effector():
    """设置夹爪（只支持位置，力矩忽略）"""
    pos = float(input("夹爪目标开合 (m): "))
    tau = float(input("加持力矩 (N): "))
    for _ in range(500):
        carm.set_gripper(pos, tau=tau)
        carm.step()
    print(f"夹爪已设置: 开合 {carm.get_gripper_pos():.4f} m, 力矩 {carm.get_gripper_tau():.2f} N")

def inverse_kine_test():
    """逆运动学测试"""
    pose = [-0.010669, 0.000073, 0.415677, 0.733466, 0.000187, 0.679726, 0.000029]
    ref_joint = [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0]
    ret, jnt = carm.inverse_kine(tool_index=0, quat_pose=pose, ref_joint=ref_joint)
    if ret == 1:
        print(f"逆解关节角: {[f'{q:.4f}' for q in jnt]}")
    else:
        print("逆运动学求解失败")

def forward_kine_test():
    """正运动学测试"""
    jnt = [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0]
    ret, pose = carm.forward_kine(jnt, tool_index=0)
    if ret == 1:
        print(f"末端位姿: {[f'{v:.4f}' for v in pose]}")
    else:
        print("正运动学求解失败")

def move_pvt_joint():
    """关节空间多点运动（PVT，近似）"""
    # 四个关节路径点
    waypoints = [
        [0.3, 0.361361, -0.646622, 0.0, 0.361361, 0.0],
        [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0],
        [-0.3, 0.361361, -0.646622, 0.0, 0.361361, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    # 仿真中使用自动时间最优规划（无速度控制）
    ret = carm.move_joint_traj(
        waypoints, gripper_pos=-1.0, stamps=[0, 2, 4, 6], is_sync=True
    )
    print(f"move_joint_traj (PVT) {'完成' if ret == 1 else '失败'}, ret={ret}")

def move_pvt_pose():
    """笛卡尔空间多点运动（PVT，近似）"""
    # 先获得四个关节路径点对应的笛卡尔位姿
    waypoints_j = [
        [0.3, 0.361361, -0.646622, 0.0, 0.361361, 0.0],
        [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0],
        [-0.3, 0.361361, -0.646622, 0.0, 0.361361, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    ret, pose_list = carm.forward_kine_array(waypoints_j, tool_index=0)
    if ret != 1:
        print("无法计算部分位姿")
        return
    ret = carm.move_pose_traj(
        pose_list, gripper_pos=-1.0, stamps=[0, 2, 4, 6], is_sync=True
    )
    print(f"move_pose_traj (PVT) {'完成' if ret == 1 else '失败'}, ret={ret}")

def grasp_cube():
    """抓取物体（set_gripper 力限抓取）: 夹取后搬运到位姿，joint1=π/4 处放开"""
    import math
    GRIP_OPEN  = 0.07    # 张开目标开合 (m)，最大约 0.0734
    GRIP_CLOSE = 0.01    # 夹紧目标开合 (m)
    TAU        = 3.0     # 夹持力限幅 (N)
    POSE_CARRY = [0.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0]              # 夹持搬运位姿
    POSE_DROP  = [math.pi/4.0, 0.361361, -0.646622, 0.0, 0.361361, 0.0]      # 放爪位姿 (joint1=π/4)

    # 1. 张开夹爪（每步调用 set_gripper 使 PI 收敛到目标开合）
    carm.set_gripper(GRIP_OPEN, tau=TAU)
    for _ in range(100):
        carm.step()
        carm.set_gripper(GRIP_OPEN, tau=TAU)
    print(f"夹爪已张开: {carm.get_gripper_pos():.4f} m")

    # 2. 移动机械臂到抓取位置（阻塞执行轨迹）
    carm.move_joint([-0.00351, 2.32026, -1.04787,
     0.05038, -0.54667, 0.02934],
                     desire_time= 3,is_sync=True)

    # 3. 力限夹紧物体（每步调用 set_gripper，输出限幅到 ±TAU，夹持力不会超过 TAU）
    for _ in range(100):
        carm.set_gripper(GRIP_CLOSE, tau=TAU)
        carm.step()

    print(f"夹紧后开合: {carm.get_gripper_pos():.4f} m, "
          f"夹持力: {carm.get_gripper_tau():.2f} N")

    # 4. 夹持物体运动到搬运位姿（夹持力由 set_gripper 力矩锁存保持）
    carm.move_joint(POSE_CARRY, desire_time=3, is_sync=True)
    print(f"已夹持运动到搬运位姿, joint1 = {carm.get_joint_pos()[0]:.4f} rad")
    print(f"搬运过程夹爪状态: {carm.get_gripper_pos():.4f} m, "
            f"夹持力: {carm.get_gripper_tau():.2f} N")
    # # 5. 夹持运动到放爪位姿
    # carm.move_joint(POSE_DROP, desire_time=3, is_sync=True)
    # print(f"已到放爪位姿, joint1 = {carm.get_joint_pos()[0]:.4f} rad")

    # # 6. 张开夹爪放开物体
    # for _ in range(1100):
    #     carm.set_gripper(GRIP_OPEN, tau=TAU)
    #     carm.step()

    # print(f"已张开夹爪放开物体, 开合: {carm.get_gripper_pos():.4f} m")


# -----------------------------------------------------------
# 命令映射表
# -----------------------------------------------------------
CMD_MAP: Dict[str, Callable] = {
    "cm":   set_control_mode,
    "p":    print_state,
    "mh":   move_to_home,
    "mj":   move_to_test_joint,
    "mjt":  move_joint_with_time,
    "mp":   move_to_test_pose,
    "mpt":  move_pose_with_time,
    "mlj":  move_line_joint,
    "mlp":  move_line_pose,
    "ct":   cycle_test,
    "sgg":  set_end_effector,
    "ik":   inverse_kine_test,
    "fk":   forward_kine_test,
    "pvtj": move_pvt_joint,
    "pvtp": move_pvt_pose,
    "gc":   grasp_cube,
}

def print_help():
    print("\n可用命令:")
    print("  cm     - 设置控制模式")
    print("  p      - 打印状态")
    print("  mh     - 回零位")
    print("  mj     - 测试关节移动")
    print("  mjt    - 带时间关节移动")
    print("  mp     - 测试笛卡尔移动")
    print("  mpt    - 带时间笛卡尔移动")
    print("  mlj    - 关节直线移动")
    print("  mlp    - 笛卡尔直线移动")
    print("  ct     - 循环测试")
    print("  sgg    - 设置夹爪")
    print("  ik     - 逆运动学测试")
    print("  fk     - 正运动学测试")
    print("  pvtj   - 关节多点运动")
    print("  pvtp   - 笛卡尔多点运动")
    print("  gc     - 抓取物体")
    print("  q      - 退出")
    print("  help   - 显示本帮助\n")

# -----------------------------------------------------------
# 主函数
# -----------------------------------------------------------
def main():
    global carm, ctrl_c_flag

    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <mjcf_xml_path> [--render]")
        sys.exit(1)

    xml_path = sys.argv[1]
    render = True
    if len(sys.argv) > 2 and sys.argv[2] == "--no-render":
        render = False

    # 信号处理

    try:
        carm = CArmSingleCol(xml_path, render=render)
    except Exception as e:
        print(f"无法初始化仿真: {e}")
        sys.exit(1)

    print("carm 仿真测试程序启动。输入 help 查看命令，q 退出。")
    try:
        while True:
            try:
                cmd = input(">> ").strip()
            except EOFError:   # 也处理意外输入结束
                break
            if not cmd:
                continue
            if cmd == 'q':
                break
            if cmd == 'help':
                print_help()
                continue
            if cmd in CMD_MAP:
                CMD_MAP[cmd]()
            else:
                print(f"未知命令: {cmd}")
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        carm.close()

    finally:
        carm.close()
        print("程序正常退出。")

if __name__ == "__main__":
    main()