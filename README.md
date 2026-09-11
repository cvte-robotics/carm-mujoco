# CARM 仿真交互式测试程序

## 简介
本仓库提供了 CARM A3 与 CARM D3 两代机械臂的 MuJoCo 仿真接口程序和演示程序：
- **A3**：单臂，仿真接口程序为 `carm_single_col.py`；
- **D3**：7 自由度双臂 + 夹爪，仿真接口程序为 `carm_dual_bot.py`。



---

## 环境依赖
- Python >= 3.8.10
- mujoco = 3.2.3
- numpy = 1.24.4
- 自定义模块 `carm_mujoco`（提供 `CArmSingleCol`、`CArmDualBot` 类）

---

## 文件说明

### 仿真接口程序（`carm_mujoco` 模块）
- `carm_mujoco/carm_single_col.py`：**A3 机械臂的 MuJoCo 仿真接口程序**，封装了 `CArmSingleCol` 类，提供关节运动、笛卡尔运动、正逆运动学求解、夹爪控制等接口，对应 A3 模型 `models/carm_a3_mjcf/carm_a3.xml`。
- `carm_mujoco/carm_dual_bot.py`：**D3 机械臂的 MuJoCo 仿真接口程序**，封装了 `CArmDualBot` 类（7 自由度双臂 + 夹爪），提供左右臂的关节/笛卡尔运动、正逆运动学求解等接口，对应 D3 模型 `models/carm_d3_mjcf/carm_d3.xml`。

### 演示程序
- `test_carm_api.py`：A3 交互测试主程序（基于 `CArmSingleCol`），对应 C++ SDK 中的 `test_carm_api.cpp`。用户可以通过命令行输入不同指令，实时控制仿真环境中的机械臂完成关节运动、笛卡尔运动、正逆运动学求解、夹爪控制等操作。
- `test_dual_api.py`：D3 双臂交互测试主程序（基于 `CArmDualBot`），支持左右臂同时进行关节运动、笛卡尔运动、直线运动、多点轨迹运动、正逆运动学测试、夹爪控制和循环测试。
- `null_space_control.py`：**D3 零空间控制仿真演示程序**（基于 `CArmDualBot`），演示末端位姿保持的同时利用冗余自由度进行零空间运动。

### 模型文件
- `models/carm_a3_mjcf/carm_a3.xml`：CARM A3 机械臂的 MuJoCo 场景描述文件。
- `models/carm_d3_mjcf/carm_d3.xml`：CARM D3 机械臂的 MuJoCo 场景描述文件。

### 逆运动学注意事项

`carm_dual_bot.py` 中的 `inverse_kine_left` 和 `inverse_kine_right` 函数基于**阻尼最小二乘**数值迭代求解，返回的是**最小二乘意义下的一个逆解**。

---

## 快速开始
在终端中执行以下命令启动 A3 仿真：
```bash
python3 test_carm_api.py ./models/carm_a3_mjcf/carm_a3.xml
```

启动后，在交互提示符 `>>` 中先输入 `cm` 设置控制模式，再输入其他控制命令。例如：

```text
>> cm
请输入mode (0-4): 0 空闲 1 位置 2 MIT 3 拖动 4 力矩限制: 1
>> p
>> mh
```

可输入 `help` 查看全部控制命令，输入 `q` 退出程序。

启动 D3 双臂交互测试：
```bash
python3 test_dual_api.py ./models/carm_d3_mjcf/carm_d3.xml
```

如果不需要打开 MuJoCo viewer，可添加 `--no-render` 参数：
```bash
python3 test_dual_api.py ./models/carm_d3_mjcf/carm_d3.xml --no-render
```

启动后，在交互提示符 `>>` 中输入 `help` 查看全部命令。常用命令如下：

```text
>> cm       # 设置控制模式
>> mh       # 左右臂回零位
>> mj       # 左右臂测试关节移动
>> mp       # 左右臂测试笛卡尔移动
>> mlj      # 左右臂关节直线移动
>> mlp      # 左右臂笛卡尔直线移动
>> sgg      # 同时设置左右夹爪
>> ik       # 左右臂逆运动学测试
>> fk       # 左右臂正运动学测试
>> pvtj     # 左右臂关节多点运动
>> pvtp     # 左右臂笛卡尔多点运动
>> p        # 打印左右臂状态
>> q        # 退出程序
```

执行 `cm` 后，根据提示输入控制模式（`0` 空闲、`1` 位置、`2` MIT、`3` 拖动、`4` 力矩限制）。执行 `ct` 和 `sgg` 时，还需要根据提示输入循环次数，或输入夹爪目标开合距离和夹持力矩。
