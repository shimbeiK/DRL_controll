import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# モデル読み込み
m = mujoco.MjModel.from_xml_path('two_wheel_robot/scene.xml')
d = mujoco.MjData(m)

INTERVAL = 0.1
bef_pos, bef_euler = 0, 0
# ビューワー起動
with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    now = 0

    # アクチュエータのインデックス（名前から取得）
    left_idx = m.actuator('motor_l_wheel').id
    right_idx = m.actuator('motor_r_wheel').id

    time.sleep(2.0)  # 初期化待ち

    # 目標角速度（rad/s）設定
    target_left_velocity = 10.0     # 左車輪
    target_right_velocity = - 10.0   # 右車輪（逆回転で回転させる）
    d.qpos[0:2] = np.random.uniform(-0.1, 0.1, size=2)
    d.qvel = np.random.uniform(-0.1, 0.1, size=m.nv)
    # d.ctrl[0] = -40
    # d.ctrl[1] = 40
    # mujoco.mj_forward(m, d)

    while viewer.is_running():
        step_start = time.time()

        # モーターに角速度を指令する場合
        d.ctrl[left_idx] = target_left_velocity
        d.ctrl[right_idx] = target_right_velocity

        mujoco.mj_step(m, d)
        # print(time.time(), start)

         # 1秒ごとに位置と回転角を表示

        if (time.time() - start) - now > INTERVAL:
            pos = d.xpos[1]  # ルートボディの位置
            rotmat = d.xmat[1].reshape(3, 3)
            rot = R.from_matrix(rotmat)
            euler = rot.as_euler('xyz', degrees=True)
            print("位置:", pos) # 初期位置は原点, 移動は前後がｙ
            print("速度:", (pos - bef_pos)/INTERVAL) # 初期位置は原点, 移動は前後がｙ
            print("回転角 (Euler xyz):", euler, "unit_m/s") # ロボット座標系で見て前後がｘ、左右がｙ、旋回がｚ
            print("角速度 (Euler xyz):", (euler-bef_euler)/INTERVAL, "rad/s") # ロボット座標系で見て前後がｘ、左右がｙ、旋回がｚ
            bef_pos = pos
            bef_euler = euler
            now += INTERVAL
            print(d.qpos.shape)  # (7,)
            print(d.qpos)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # print(m.opt.timestep, time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
