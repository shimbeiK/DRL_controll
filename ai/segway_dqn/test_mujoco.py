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
    counter = 0

    # アクチュエータのインデックス（名前から取得）
    # left_idx = m.actuator('motor_l_wheel').id
    # right_idx = m.actuator('motor_r_wheel').id

    time.sleep(2.0)  # 初期化待ち

    # 目標角速度（rad/s）設定
    target_left_velocity = 1     # 左車輪
    target_right_velocity = 1   # 右車輪（逆回転で回転させる）
    # d.qpos[0:2] = np.random.uniform(-0.1, 0.1, size=2)
    # d.qvel = np.random.uniform(-0.1, 0.1, size=m.nv)
    # d.ctrl[0] = -40
    # d.ctrl[1] = 40
    # mujoco.mj_forward(m, d)

    while viewer.is_running():
        step_start = time.time()

        # モーターに角速度を指令する場合
        if(counter <= 3):
            d.ctrl[0] = target_left_velocity
            d.ctrl[1] = target_right_velocity
        #     print("a")
        # elif(counter <= 6):
        #     d.ctrl[0] = - target_left_velocity
        #     d.ctrl[1] = - target_right_velocity
        #     print("bb")
        else:
            counter = 0
        counter += 1

        mujoco.mj_step(m, d)
        # print(time.time(), start)

         # 1秒ごとに位置と回転角を表示
        # chassis_id = m.sensor('chassis_angle').id
        chassis_id = 1
        if (time.time() - start) - now > INTERVAL:
            pos = d.xpos[chassis_id]  # ルートボディの位置
            rotmat = d.xmat[chassis_id].reshape(3, 3)
            rot = R.from_matrix(rotmat)
            euler = rot.as_euler('xyz', degrees=False)
            print("")
            print("位置:", pos) # 初期位置は原点, 移動は前後がｙ
            # print("速度:", (pos - bef_pos)/INTERVAL, "unit_m/s") # 初期位置は原点, 移動は前後がｙ
            print("回転角 (Euler xyz):", euler) # ロボット座標系で見て前後がｘ、左右がｙ、旋回がｚ
            # print("角速度 (Euler xyz):", (euler-bef_euler)/INTERVAL, "rad/s") # ロボット座標系で見て前後がｘ、左右がｙ、旋回がｚ
            bef_pos = pos
            bef_euler = euler
            now += INTERVAL

            x = d.xpos[0, 0]  # カート位置
            x_dot = d.qvel[0]  # カート速度
            rotmat = d.xmat[1].reshape(3, 3)
            rot = R.from_matrix(rotmat)
            angle = rot.as_euler('xyz')[0]  # ロール角
            angle_dot = d.qvel[1]  # 回転の角速度
            print("観測値:", np.array([x, x_dot, angle, angle_dot], dtype=np.float32))




        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # print(m.opt.timestep, time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
