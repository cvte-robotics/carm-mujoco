#!/usr/bin/env python3
"""CARM D3 双臂 MuJoCo 交互式 API 测试程序。

用法: python test_dual_api.py <MJCF_XML_PATH> [--no-render]
"""

import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from carm_mujoco import CArmDualBot


HOME_JOINT = [0.0] * 7

# D3 左右臂关节 2、关节 4 的限位方向相反，使用对称的测试姿态。
LEFT_TEST_JOINT = [np.pi / 6, np.pi / 4, 0.0, -np.pi / 3, 0.0, 0.0, 0.0]
RIGHT_TEST_JOINT = [np.pi / 6, -np.pi / 4, 0.0, np.pi / 3, 0.0, 0.0, 0.0]

carm: Optional[CArmDualBot] = None


def _arm() -> CArmDualBot:
    if carm is None:
        raise RuntimeError("仿真尚未初始化")
    return carm


def _print_result(name: str, left_ret: int, right_ret: int) -> int:
    print(
        f"{name}: 左臂 {'完成' if left_ret == 1 else '失败'}, "
        f"右臂 {'完成' if right_ret == 1 else '失败'}"
    )
    return int(left_ret == 1 and right_ret == 1)


def _test_poses() -> Optional[Tuple[List[float], List[float]]]:
    """根据测试关节角计算左右臂对应的末端位姿。"""
    arm = _arm()
    left_ret, left_pose = arm.forward_kine_left(LEFT_TEST_JOINT, tool_index=0)
    right_ret, right_pose = arm.forward_kine_right(RIGHT_TEST_JOINT, tool_index=0)
    if left_ret != 1 or right_ret != 1:
        print("无法计算左右臂测试位姿")
        return None
    return left_pose, right_pose


def _test_waypoints() -> Tuple[List[List[float]], List[List[float]]]:
    left_mid = [0.0, np.pi / 4, 0.0, -np.pi / 3, 0.0, 0.0, 0.0]
    right_mid = [0.0, -np.pi / 4, 0.0, np.pi / 3, 0.0, 0.0, 0.0]
    left_other = left_mid.copy()
    right_other = right_mid.copy()
    left_other[0] = -np.pi / 6
    right_other[0] = -np.pi / 6
    return (
        [HOME_JOINT.copy(), left_mid, left_other, HOME_JOINT.copy()],
        [HOME_JOINT.copy(), right_mid, right_other, HOME_JOINT.copy()],
    )


# ---------------------------------------------------------------------------
# 基础状态操作
# ---------------------------------------------------------------------------
def set_control_mode() -> None:
    arm = _arm()
    try:
        mode = int(input("请输入 mode (0-4): 0 空闲 1 位置 2 MIT 3 拖动 4 力矩限制: "))
    except ValueError:
        print("输入格式错误")
        return

    if arm.set_control_mode(mode) == 1:
        print(f"控制模式已切换为 {mode}")
    else:
        print("无效模式")


def print_state() -> None:
    arm = _arm()
    left_joint_pos = arm.get_left_joint_pos()
    right_joint_pos = arm.get_right_joint_pos()
    left_joint_vel = arm.get_left_joint_vel()
    right_joint_vel = arm.get_right_joint_vel()
    left_joint_tau = arm.get_left_joint_tau()
    right_joint_tau = arm.get_right_joint_tau()
    left_cart_pose = arm.get_left_cart_pose()
    right_cart_pose = arm.get_right_cart_pose()
    left_gripper_pos = arm.get_left_gripper_pos()
    right_gripper_pos = arm.get_right_gripper_pos()
    left_gripper_vel = arm.get_left_gripper_vel()
    right_gripper_vel = arm.get_right_gripper_vel()
    left_gripper_tau = arm.get_left_gripper_tau()
    right_gripper_tau = arm.get_right_gripper_tau()

    print(f"左臂关节角度 (rad): {[f'{q:.4f}' for q in left_joint_pos]}")
    print(f"右臂关节角度 (rad): {[f'{q:.4f}' for q in right_joint_pos]}")
    print(f"左臂关节速度 (rad/s): {[f'{v:.4f}' for v in left_joint_vel]}")
    print(f"右臂关节速度 (rad/s): {[f'{v:.4f}' for v in right_joint_vel]}")
    print(f"左臂关节力矩 (Nm): {[f'{v:.4f}' for v in left_joint_tau]}")
    print(f"右臂关节力矩 (Nm): {[f'{v:.4f}' for v in right_joint_tau]}")
    print(f"左臂末端位姿 (xyz+q): {[f'{v:.4f}' for v in left_cart_pose]}")
    print(f"右臂末端位姿 (xyz+q): {[f'{v:.4f}' for v in right_cart_pose]}")
    print(f"左臂夹爪状态: 间距 {left_gripper_pos:.4f} m, "
          f"速度 {left_gripper_vel:.4f} m/s, 力矩 {left_gripper_tau:.4f} N")
    print(f"右臂夹爪状态: 间距 {right_gripper_pos:.4f} m, "
          f"速度 {right_gripper_vel:.4f} m/s, 力矩 {right_gripper_tau:.4f} N")


# ---------------------------------------------------------------------------
# 运动指令
# ---------------------------------------------------------------------------
def move_to_home() -> None:
    arm = _arm()
    duration = 1.5

    # 先下发左右臂目标，再统一推进仿真，两个动作会同时开始。
    left_ret = arm.move_left_joint(
        HOME_JOINT.copy(), desire_time=duration, is_sync=False
    )
    right_ret = arm.move_right_joint(
        HOME_JOINT.copy(), desire_time=duration, is_sync=False
    )

    if left_ret == 1 and right_ret == 1:
        step_count = int(np.ceil(duration / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result("回零位", left_ret, right_ret)


def move_to_test_joint() -> None:
    arm = _arm()
    duration = 1.5

    # 先连续下发左右臂目标，中间不要调用 step()。
    left_ret = arm.move_left_joint(
        LEFT_TEST_JOINT, desire_time=duration, is_sync=False
    )
    right_ret = arm.move_right_joint(
        RIGHT_TEST_JOINT, desire_time=duration, is_sync=False
    )

    # 两个目标都下发完成后，再统一推进仿真。
    if left_ret == 1 and right_ret == 1:
        step_count = int(np.ceil(duration / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result("move_joint", left_ret, right_ret)


def move_joint_with_time() -> None:
    """同时控制左右臂的最小示例。"""
    arm = _arm()
    duration = 1.5

    # 关键：先连续下发左右臂目标，中间不要调用 step()。
    left_ret = arm.move_left_joint(
        LEFT_TEST_JOINT, desire_time=duration, is_sync=False
    )
    right_ret = arm.move_right_joint(
        RIGHT_TEST_JOINT, desire_time=duration, is_sync=False
    )

    # 两个目标都下发后，再统一推进仿真，左右臂会同时运动。
    if left_ret == 1 and right_ret == 1:
        step_count = int(np.ceil(duration / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()

    _print_result(f"move_joint (time={duration}s)", left_ret, right_ret)


def move_to_test_pose() -> None:
    arm = _arm()
    poses = _test_poses()
    if poses is None:
        return
    left_pose, right_pose = poses
    duration = 1.5
    left_ret = arm.move_left_pose(
        left_pose, desire_time=duration, is_sync=False
    )
    right_ret = arm.move_right_pose(
        right_pose, desire_time=duration, is_sync=False
    )

    if left_ret == 1 and right_ret == 1:
        step_count = int(np.ceil(duration / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result("move_pose", left_ret, right_ret)


def move_pose_with_time() -> None:
    arm = _arm()
    poses = _test_poses()
    if poses is None:
        return
    duration = 1.5
    left_pose, right_pose = poses
    left_ret = arm.move_left_pose(left_pose, duration, is_sync=False)
    right_ret = arm.move_right_pose(right_pose, duration, is_sync=False)

    if left_ret == 1 and right_ret == 1:
        step_count = int(np.ceil(duration / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result(f"move_pose (time={duration}s)", left_ret, right_ret)


def move_line_joint() -> None:
    arm = _arm()
    left_ret = arm.move_left_line_joint(LEFT_TEST_JOINT, is_sync=False)
    right_ret = arm.move_right_line_joint(RIGHT_TEST_JOINT, is_sync=False)

    if left_ret == 1 and right_ret == 1:
        # 该测试目标的直线轨迹约 1.07 秒，额外预留 5 个仿真步。
        step_count = int(np.ceil(2.0 / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result("move_line_joint", left_ret, right_ret)


def move_line_pose() -> None:
    arm = _arm()
    poses = _test_poses()
    if poses is None:
        return
    left_pose, right_pose = poses
    left_ret = arm.move_left_line_pose(left_pose, is_sync=False)
    right_ret = arm.move_right_line_pose(right_pose, is_sync=False)

    if left_ret == 1 and right_ret == 1:
        step_count = int(np.ceil(2.0 / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result("move_line_pose", left_ret, right_ret)


def cycle_test() -> None:
    try:
        times = int(input("循环次数: "))
    except ValueError:
        print("无效输入")
        return
    if times < 1:
        print("循环次数必须大于 0")
        return

    for i in range(times):
        move_to_test_joint()
        move_to_home()
        print(f"循环进度: {i + 1}/{times}")
    print("循环测试结束")


def set_end_effector() -> None:
    """同时设置左右夹爪的目标开合和力矩。"""
    arm = _arm()
    try:
        pos = float(input("左右夹爪目标开合 (m): "))
        tau = float(input("夹持力矩 (N): "))
    except ValueError:
        print("输入格式错误")
        return

    left_ret = 0
    right_ret = 0
    for _ in range(500):
        # 每个仿真步同时更新左右夹爪。
        left_ret = arm.set_left_gripper(pos, tau=tau)
        right_ret = arm.set_right_gripper(pos, tau=tau)
        arm.step()

    _print_result("夹爪设置", left_ret, right_ret)
    print(
        f"左臂夹爪间距: {arm.get_left_gripper_pos():.4f} m, "
        f"右臂夹爪间距: {arm.get_right_gripper_pos():.4f} m"
    )


# ---------------------------------------------------------------------------
# 运动学和多点轨迹
# ---------------------------------------------------------------------------
def inverse_kine_test() -> None:
    arm = _arm()
    poses = _test_poses()
    if poses is None:
        return
    left_pose, right_pose = poses
    left_ret, left_joint = arm.inverse_kine_left(
        quat_pose=left_pose, ref_joint=LEFT_TEST_JOINT, tool_index=0
    )
    right_ret, right_joint = arm.inverse_kine_right(
        quat_pose=right_pose, ref_joint=RIGHT_TEST_JOINT, tool_index=0
    )
    _print_result("逆运动学", left_ret, right_ret)
    if left_ret == 1:
        print(f"左臂逆解关节角: {[f'{q:.4f}' for q in left_joint]}")
    if right_ret == 1:
        print(f"右臂逆解关节角: {[f'{q:.4f}' for q in right_joint]}")


def forward_kine_test() -> None:
    arm = _arm()
    left_ret, left_pose = arm.forward_kine_left(LEFT_TEST_JOINT, tool_index=0)
    right_ret, right_pose = arm.forward_kine_right(RIGHT_TEST_JOINT, tool_index=0)
    _print_result("正运动学", left_ret, right_ret)
    if left_ret == 1:
        print(f"左臂末端位姿: {[f'{v:.4f}' for v in left_pose]}")
    if right_ret == 1:
        print(f"右臂末端位姿: {[f'{v:.4f}' for v in right_pose]}")


def move_pvt_joint() -> None:
    arm = _arm()
    left_waypoints, right_waypoints = _test_waypoints()
    stamps = [0.0, 2.0, 4.0, 6.0]
    # 先同时下发左右臂的整条轨迹，再统一推进仿真。
    left_ret = arm.move_left_joint_traj(
        left_waypoints, gripper_pos=-1.0, stamps=stamps, is_sync=False
    )
    right_ret = arm.move_right_joint_traj(
        right_waypoints, gripper_pos=-1.0, stamps=stamps, is_sync=False
    )

    if left_ret == 1 and right_ret == 1:
        step_count = int(np.ceil(stamps[-1] / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result("move_joint_traj (PVT)", left_ret, right_ret)


def move_pvt_pose() -> None:
    arm = _arm()
    left_joint_waypoints, right_joint_waypoints = _test_waypoints()
    left_ret, left_poses = arm.forward_kine_left_array(
        left_joint_waypoints, tool_index=0
    )
    right_ret, right_poses = arm.forward_kine_right_array(
        right_joint_waypoints, tool_index=0
    )
    if left_ret != 1 or right_ret != 1:
        print("无法计算左右臂 PVT 位姿路径")
        return

    stamps = [0.0, 2.0, 4.0, 6.0]
    # 左右臂位姿轨迹都设置完成后，step() 会在每个仿真步同时更新两侧。
    left_move_ret = arm.move_left_pose_traj(
        left_poses, gripper_pos=-1.0, stamps=stamps, is_sync=False
    )
    right_move_ret = arm.move_right_pose_traj(
        right_poses, gripper_pos=-1.0, stamps=stamps, is_sync=False
    )

    if left_move_ret == 1 and right_move_ret == 1:
        step_count = int(np.ceil(stamps[-1] / arm.model.opt.timestep)) + 5
        for _ in range(step_count):
            arm.step()
    _print_result("move_pose_traj (PVT)", left_move_ret, right_move_ret)


CMD_MAP: Dict[str, Callable[[], None]] = {
    "cm": set_control_mode,
    "p": print_state,
    "mh": move_to_home,
    "mj": move_to_test_joint,
    "mjt": move_joint_with_time,
    "mp": move_to_test_pose,
    "mpt": move_pose_with_time,
    "mlj": move_line_joint,
    "mlp": move_line_pose,
    "ct": cycle_test,
    "sgg": set_end_effector,
    "ik": inverse_kine_test,
    "fk": forward_kine_test,
    "pvtj": move_pvt_joint,
    "pvtp": move_pvt_pose,
}


def print_help() -> None:
    print("\n可用命令:")
    print("  cm     - 设置控制模式")
    print("  p      - 打印左右臂状态")
    print("  mh     - 左右臂回零位")
    print("  mj     - 左右臂测试关节移动")
    print("  mjt    - 左右臂带时间关节移动")
    print("  mp     - 左右臂测试笛卡尔移动")
    print("  mpt    - 左右臂带时间笛卡尔移动")
    print("  mlj    - 左右臂关节直线移动")
    print("  mlp    - 左右臂笛卡尔直线移动")
    print("  ct     - 左右臂循环测试")
    print("  sgg    - 同时设置左右夹爪")
    print("  ik     - 左右臂逆运动学测试")
    print("  fk     - 左右臂正运动学测试")
    print("  pvtj   - 左右臂关节多点运动")
    print("  pvtp   - 左右臂笛卡尔多点运动")
    print("  q      - 退出")
    print("  help   - 显示本帮助\n")


def main() -> int:
    global carm

    parser = argparse.ArgumentParser(description="CARM D3 双臂 MuJoCo API 测试")
    parser.add_argument("xml_path", help="MJCF XML 文件路径")
    parser.add_argument("--no-render", action="store_true", help="关闭 MuJoCo viewer")
    args = parser.parse_args()

    try:
        carm = CArmDualBot(args.xml_path, render=not args.no_render)
    except Exception as exc:
        print(f"无法初始化仿真: {exc}")
        return 1

    print("CARM D3 双臂仿真测试程序启动。输入 help 查看命令，q 退出。")
    try:
        while True:
            try:
                cmd = input(">> ").strip()
            except EOFError:
                break

            if not cmd:
                continue
            if cmd == "q":
                break
            if cmd == "help":
                print_help()
                continue
            if cmd in CMD_MAP:
                try:
                    CMD_MAP[cmd]()
                except KeyboardInterrupt:
                    print("\n当前测试被用户中断")
                except Exception as exc:
                    print(f"命令执行失败: {exc}")
            else:
                print(f"未知命令: {cmd}")
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    finally:
        if carm is not None:
            carm.close()
            carm = None
        print("程序正常退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
