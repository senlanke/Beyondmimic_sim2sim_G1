# Beyondmimic_sim2sim_G1
说明：
1. 使用beyondmimic训练的模型进行sim2sim测试
2. 观测维度为160维
3. 运行指令：
   ```bash
   python sim2sim.py
   ```
   可选参数：
   ```bash
   --motion_file
   --policy_path
   ```
## 注意事项
  - 观测拼接顺序必须严格一致：command → motion_ref_pos_b → motion_ref_ori_b → base_lin_vel → base_ang_vel → joint_pos → joint_vel → previous_action，并匹配 obs=160。
  - command（motion joint_pos/vel）必须用 npz 原始顺序（来自 csv_to_npz.py 的 joint_names），不能按 ONNX joint_names 重排。
  - motion_anchor 的索引必须与生成 npz 的 body list 对齐；你确认 torso_link index=9，所以固定用 MOTION_BODY_INDEX = 9。
  - 速度 frame：线速度用 R.T @ qvel[0:3]（转换到基座坐标系），角速度直接用 qvel[3:6]（本体坐标系）。
  - 锚点索引固定为 MOTION_BODY_INDEX=9（与 npz 生成一致），修改为自己机器人时需要修改。

# 主要参考
https://github.com/oYYmYYo/Beyondmimic_Deploy_G1.git
https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic.git

# Acknowledgement：
[1] Beyondmimic训练源码：https://github.com/HybridRobotics/whole_body_tracking

[2] Unitree_rl_gym仓库： https://github.com/unitreerobotics/unitree_rl_gym

