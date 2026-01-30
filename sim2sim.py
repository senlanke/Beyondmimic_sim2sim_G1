import argparse  # 解析命令行参数用于配置运行  
import time  # 提供计时与延时功能  

import mujoco.viewer  # MuJoCo 可视化查看器接口  
import mujoco  # MuJoCo 物理引擎接口  
import numpy as np  # 数值计算与数组处理  
import onnxruntime  # ONNX 运行时推理引擎  
import onnx  # ONNX 模型解析加载  
import torch  # PyTorch 张量与模型工具  

XML_PATH = "./unitree_description/mjcf/g1.xml"  # MuJoCo 模型 XML 路径  
SIMULATION_DURATION = 300.0  # 总仿真时长（秒）  
SIMULATION_DT = 0.002  # 物理仿真步长  
CONTROL_DECIMATION = 10  # 控制更新降采样倍数  
NUM_ACTIONS = 29  # 动作维度数量  
NUM_OBS = 160  # 观测维度数量  
MOTION_BODY_INDEX = 9  # 动作序列中参考刚体索引  
BODY_NAME = "torso_link"  # 机器人参考刚体名称  

JOINT_XML = [  # MuJoCo 关节顺序列表  
    "left_hip_pitch_joint",  # 左髋俯仰关节名称  
    "left_hip_roll_joint",  # 左髋滚转关节名称  
    "left_hip_yaw_joint",  # 左髋偏航关节名称  
    "left_knee_joint",  # 左膝关节名称  
    "left_ankle_pitch_joint",  # 左踝俯仰关节名称  
    "left_ankle_roll_joint",  # 左踝滚转关节名称  
    "right_hip_pitch_joint",  # 右髋俯仰关节名称  
    "right_hip_roll_joint",  # 右髋滚转关节名称  
    "right_hip_yaw_joint",  # 右髋偏航关节名称  
    "right_knee_joint",  # 右膝关节名称  
    "right_ankle_pitch_joint",  # 右踝俯仰关节名称  
    "right_ankle_roll_joint",  # 右踝滚转关节名称  
    "waist_yaw_joint",  # 腰部偏航关节名称  
    "waist_roll_joint",  # 腰部滚转关节名称  
    "waist_pitch_joint",  # 腰部俯仰关节名称  
    "left_shoulder_pitch_joint",  # 左肩俯仰关节名称  
    "left_shoulder_roll_joint",  # 左肩滚转关节名称  
    "left_shoulder_yaw_joint",  # 左肩偏航关节名称  
    "left_elbow_joint",  # 左肘关节名称  
    "left_wrist_roll_joint",  # 左腕滚转关节名称  
    "left_wrist_pitch_joint",  # 左腕俯仰关节名称  
    "left_wrist_yaw_joint",  # 左腕偏航关节名称  
    "right_shoulder_pitch_joint",  # 右肩俯仰关节名称  
    "right_shoulder_roll_joint",  # 右肩滚转关节名称  
    "right_shoulder_yaw_joint",  # 右肩偏航关节名称  
    "right_elbow_joint",  # 右肘关节名称  
    "right_wrist_roll_joint",  # 右腕滚转关节名称  
    "right_wrist_pitch_joint",  # 右腕俯仰关节名称  
    "right_wrist_yaw_joint",  # 右腕偏航关节名称  
]  # 关节顺序列表结束  


def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):  # 计算 A 到 B 的相对位姿  
    rotm_a = np.zeros(9)  # 初始化旋转矩阵扁平数组  
    mujoco.mju_quat2Mat(rotm_a, quat_a)  # 四元数转旋转矩阵  
    rotm_a = rotm_a.reshape(3, 3)  # 重塑为 3x3 矩阵  
    rel_pos = rotm_a.T @ (pos_b - pos_a)  # 计算相对位置向量  
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)  # 计算相对旋转四元数  
    rel_quat = rel_quat / np.linalg.norm(rel_quat)  # 归一化四元数  
    return rel_pos, rel_quat  # 返回相对位置和相对姿态  

def quaternion_conjugate(q):  # 计算四元数共轭  
    return np.array([q[0], -q[1], -q[2], -q[3]])  # 标量不变向量取负  

def quaternion_multiply(q1, q2):  # 计算四元数乘积  
    w1, x1, y1, z1 = q1  # 解包第一个四元数  
    w2, x2, y2, z2 = q2  # 解包第二个四元数  
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # 计算结果 w 分量  
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # 计算结果 x 分量  
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # 计算结果 y 分量  
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # 计算结果 z 分量  
    
    return np.array([w, x, y, z])  # 返回乘积四元数  


def pd_control(target_q, q, kp, target_dq, dq, kd):  # 计算 PD 控制力矩  
    return (target_q - q) * kp + (target_dq - dq) * kd  # 位置与速度误差加权求和  


def parse_args():  # 解析命令行参数  
    parser = argparse.ArgumentParser()  # 创建参数解析器  
    parser.add_argument("--motion_file", type=str, default="./motion.npz", help="motion npz file")  # 动作文件路径参数  
    parser.add_argument("--policy_path", type=str, default="./policy.onnx", help="onnx policy")  # 策略模型路径参数  
    return parser.parse_args()  # 返回解析后的参数  


def load_policy_metadata(policy_path, joint_xml):  # 读取策略元数据并重排关节  
    model = onnx.load(policy_path)  # 加载 ONNX 模型  
    joint_seq = []  # 初始化关节名称序列  
    joint_pos_array_seq = None  # 初始化默认关节角度序列  
    joint_pos_array = None  # 初始化重排后默认角度  
    stiffness_array = None  # 初始化重排后刚度  
    damping_array = None  # 初始化重排后阻尼  
    action_scale = None  # 初始化动作缩放  
    anchor_body_name = None  # 初始化锚点刚体名称  
    body_names = None  # 初始化刚体名称列表  
    for prop in model.metadata_props:  # 遍历模型元数据属性  
        values = prop.value.split(",")  # 解析属性字符串为列表  
        if prop.key == "joint_names":  # 处理关节名称字段  
            joint_seq = values  # 保存关节名称顺序  
        elif prop.key == "default_joint_pos":  # 处理默认关节角度  
            joint_pos_array_seq = np.array([float(x) for x in values])  # 转为浮点数组  
            joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "joint_stiffness":  # 处理关节刚度  
            stiffness_array = np.array([float(x) for x in values])  # 转为浮点数组  
            stiffness_array = np.array([stiffness_array[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "joint_damping":  # 处理关节阻尼  
            damping_array = np.array([float(x) for x in values])  # 转为浮点数组  
            damping_array = np.array([damping_array[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "action_scale":  # 处理动作缩放  
            action_scale = np.array([float(x) for x in values])  # 转为浮点数组  
        elif prop.key == "anchor_body_name":  # 处理锚点刚体名称  
            anchor_body_name = prop.value  # 记录锚点刚体名称  
        elif prop.key == "body_names":  # 处理刚体名称列表  
            body_names = values  # 记录刚体名称列表  
        print(f"{prop.key}: {prop.value}")  # 打印元数据方便检查  

    if not joint_seq:  # 检查关节名称是否缺失  
        raise ValueError("ONNX 元数据缺少 joint_names，无法对齐关节顺序")  # 抛出缺失错误  
    if joint_pos_array_seq is None or joint_pos_array is None:  # 检查默认关节角度是否缺失  
        raise ValueError("ONNX 元数据缺少 default_joint_pos，无法初始化关节角度")  # 抛出缺失错误  
    if stiffness_array is None or damping_array is None or action_scale is None:  # 检查控制参数是否缺失  
        raise ValueError("ONNX 元数据缺少关节参数或动作缩放，请检查 joint_stiffness/joint_damping/action_scale")  # 抛出缺失错误  

    return (  # 返回整理后的策略参数  
        joint_seq,  # 关节名称顺序  
        joint_pos_array_seq,  # 默认关节角度原序列  
        joint_pos_array,  # 默认关节角度 MuJoCo 顺序  
        stiffness_array,  # 关节刚度 MuJoCo 顺序  
        damping_array,  # 关节阻尼 MuJoCo 顺序  
        action_scale,  # 动作缩放系数  
        anchor_body_name,  # 锚点刚体名称  
        body_names,  # 刚体名称列表  
    )  # 返回结束  


def load_motion(motion_file):  # 读取动作数据文件  
    motion = np.load(motion_file)  # 加载 npz 数据  
    motion_pos = motion["body_pos_w"]  # 读取所有刚体位置序列  
    motion_quat = motion["body_quat_w"]  # 读取所有刚体姿态四元数序列  
    motion_input_pos = motion["joint_pos"]  # 读取关节位置序列  
    motion_input_vel = motion["joint_vel"]  # 读取关节速度序列  
    return motion_pos, motion_quat, motion_input_pos, motion_input_vel  # 返回动作数据  


if __name__ == "__main__":  # 主程序入口  
    args = parse_args()  # 解析命令行参数  
    motion_file = args.motion_file  # 获取动作文件路径  
    policy_path = args.policy_path  # 获取策略模型路径  

    motion_pos, motion_quat, motion_input_pos, motion_input_vel = load_motion(motion_file)  # 加载动作数据  

    (  # 解包策略元数据  
        joint_seq,  # 关节名称顺序  
        joint_pos_array_seq,  # 默认关节角度原序列  
        joint_pos_array,  # 默认关节角度 MuJoCo 顺序  
        stiffness_array,  # 关节刚度 MuJoCo 顺序  
        damping_array,  # 关节阻尼 MuJoCo 顺序  
        action_scale,  # 动作缩放系数  
        anchor_body_name,  # 锚点刚体名称  
        body_names,  # 刚体名称列表  
    ) = load_policy_metadata(policy_path, JOINT_XML)  # 加载策略元数据
    print(f"Loaded policy metadata with {len(joint_seq)} joints and body names: {body_names}")  # 打印加载信息

    obs = np.zeros(NUM_OBS, dtype=np.float32)  # 初始化观测向量  
    counter = 0  # 控制器计数器  

    m = mujoco.MjModel.from_xml_path(XML_PATH)  # 加载 MuJoCo 模型  
    d = mujoco.MjData(m)  # 创建 MuJoCo 数据对象  
    m.opt.timestep = SIMULATION_DT  # 设置仿真时间步长  

    # 使用ONNX Runtime库创建了一个推理会话，用于加载和执行预训练的神经网络策略模型
    policy = onnxruntime.InferenceSession(policy_path)  

    action_buffer = np.zeros((NUM_ACTIONS,), dtype=np.float32)  # 初始化上一动作缓存  
    timestep = 0  # 初始化动作序列索引  
    target_dof_pos = joint_pos_array.copy()  # 目标关节位置初始化为默认角度  
    d.qpos[2] = 0.8  # 设置初始基座高度  
    d.qpos[7:] = target_dof_pos  # 写入初始关节角度  

    body_name = anchor_body_name or BODY_NAME  # 使用元数据锚点名称或默认名称  
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)  # 获取锚点 ID  
    if body_id == -1:  # 检查锚点是否存在  
        raise ValueError(f"Body {body_name} not found in model")  # 抛出错误提示  
    motion_body_index = MOTION_BODY_INDEX

    with mujoco.viewer.launch_passive(m, d) as viewer:  # 启动被动可视化窗口  
        
        viewer.cam.type = 1  # mjCAMERA_TRACKING  # 跟踪相机
        viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")  # 跟踪pelvis身体
        viewer.cam.lookat[2] = 1.5  # 跟踪点高度
        viewer.cam.distance = 3  # 距离
        viewer.cam.azimuth = 135  # 角度
        viewer.cam.elevation = -20  # 仰角

    
        start = time.time()  # 记录仿真起始时间  
        while viewer.is_running() and time.time() - start < SIMULATION_DURATION:  # 主循环条件  
            step_start = time.time()  # 记录本步起始时间  

            mujoco.mj_step(m, d)  # 进行一步物理仿真  
            tau = pd_control(  # 计算关节控制力矩  
                target_dof_pos,  # 目标关节角度  
                d.qpos[7:],  # freejoint 后面的所有关节位置  
                stiffness_array,  # 关节刚度  
                np.zeros_like(damping_array),  # 目标关节速度置零  
                d.qvel[6:],  # freejoint 后面的所有关节角速度  
                damping_array,  # 关节阻尼  
            )  # 控制力矩计算结束  

            d.ctrl[:] = tau  # 写入力矩控制输入  
            counter += 1  # 更新控制器计数  
            if counter % CONTROL_DECIMATION == 0:  # 到达控制周期更新策略  
                position = d.xpos[body_id]  # 获取仿真中锚点刚体位置  
                quaternion = d.xquat[body_id]  # 获取仿真中锚点刚体姿态  
                motion_input = np.concatenate(  # 拼接目标关节位置与速度  
                    (motion_input_pos[timestep, :], motion_input_vel[timestep, :]),  # 使用 npz 原始关节顺序  
                    axis=0,  # 进行拼接  
                )  # 拼接结束  
                motion_pos_current = motion_pos[timestep, motion_body_index, :]  # 读取当前参考锚点动作位置  
                motion_quat_current = motion_quat[timestep, motion_body_index, :]  # 读取当前参考锚点动作姿态  
                anchor_pos, anchor_quat = subtract_frame_transforms_mujoco(  # 计算相对锚点位置与姿态  
                    position, quaternion, motion_pos_current, motion_quat_current  # 输入位姿参数  
                )  # 获取相对位置与旋转结果  
                anchor_ori = np.zeros(9)  # 初始化锚点旋转矩阵  
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat)  # 仿真锚点相对于参考锚点的相对姿态四元数转旋转矩阵  
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]  # 取旋转矩阵前两列  
                anchor_ori = anchor_ori.reshape(-1,)  # 展平为向量  
                base_rot = np.zeros(9)  # 初始化基座旋转矩阵  
                mujoco.mju_quat2Mat(base_rot, quaternion)  # 基座四元数转旋转矩阵  
                base_rot = base_rot.reshape(3, 3)  # 重塑为 3x3 矩阵  
                base_lin_vel = base_rot.T @ d.qvel[0:3]  # 把“基座root/freejoint线速度”从世界坐标系转换到机器人本体坐标系
                base_ang_vel = d.qvel[3:6]  # 读取基座坐标系角速度  

                offset = 0  # 观测向量写入偏移量  
                obs[offset : offset + 58] = motion_input  # 写入目标关节指令  
                offset += 58  # 更新偏移量  
                obs[offset : offset + 3] = anchor_pos  # 写入参考锚点相对位置  
                offset += 3  # 更新偏移量  
                obs[offset : offset + 6] = anchor_ori  # 写入参考锚点相对姿态
                offset += 6  # 更新偏移量  

                obs[offset : offset + 3] = base_lin_vel  # 写入基座root/freejoint线速度  
                offset += 3  # 更新偏移量  
                obs[offset : offset + 3] = base_ang_vel  # 写入基座角速度  
                offset += 3  # 更新偏移量  
                qpos_xml = d.qpos[7 : 7 + NUM_ACTIONS]  # 读取 MuJoCo 顺序关节角度  
                qpos_seq = np.array([qpos_xml[JOINT_XML.index(joint)] for joint in joint_seq])  # 变换到策略关节顺序  
                obs[offset : offset + NUM_ACTIONS] = qpos_seq - joint_pos_array_seq  # 写入关节角度偏差  
                offset += NUM_ACTIONS  # 更新偏移量  
                qvel_xml = d.qvel[6 : 6 + NUM_ACTIONS]  # 读取 MuJoCo 顺序关节速度  
                qvel_seq = np.array([qvel_xml[JOINT_XML.index(joint)] for joint in joint_seq])  # 变换到策略关节顺序  
                obs[offset : offset + NUM_ACTIONS] = qvel_seq  # 写入关节速度  
                offset += NUM_ACTIONS  # 更新偏移量  
                obs[offset : offset + NUM_ACTIONS] = action_buffer  # 写入上一动作  

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # 转为批量张量输入  
                action = policy.run(  # 执行 ONNX 推理  
                    ["actions"],  # 指定输出节点名称  
                    {"obs": obs_tensor.numpy(), "time_step": np.array([timestep], dtype=np.float32).reshape(1, 1)},  # 传入观测与时间步  
                )[0]  # 获取推理输出  
                action = np.asarray(action).reshape(-1)  # 转为一维数组  
                action_buffer = action.copy()  # 更新上一动作缓存  
                target_dof_pos = action * action_scale + joint_pos_array_seq  # 计算目标关节角度  
                target_dof_pos = target_dof_pos.reshape(-1,)  # 确保为一维向量  
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in JOINT_XML])  # 重排到 MuJoCo 顺序  
                timestep += 1  # 更新动作序列索引  

            viewer.sync()  # 同步可视化显示  
            time_until_next_step = m.opt.timestep - (time.time() - step_start)  # 计算剩余睡眠时间  
            if time_until_next_step > 0:  # 若剩余时间为正则休眠  
                time.sleep(time_until_next_step)  # 休眠保持实时步长  
